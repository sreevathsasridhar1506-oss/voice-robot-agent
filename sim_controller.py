"""
sim_controller.py
==================
High-level, thread-safe robot control API over the MuJoCo Panda simulation.

One dedicated background worker thread owns the MuJoCo model/data and the
mujoco.Renderer/OpenCV display loop (MuJoCo's Python bindings are not safe to
call from multiple threads concurrently). All public methods below dispatch
work to that thread and block for the result — except stop(), which is
special-cased to act instantly from ANY thread by setting a flag the worker
checks between simulation steps, without waiting behind whatever command is
currently queued. This is the same mechanism barge-in (M14) will use later.

Public API
----------
get_scene_state() -> dict   — object positions/colors, what's held, current action
move_to(position)           — move the end-effector to a raw [x, y, z]
pick(object_name)           — approach, descend, close gripper, grasp, lift
place(position)             — move above position, descend, release, lift
stop()                      — halt any in-progress motion immediately
close()                     — shut down the worker thread and display window
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cv2
import mink
import mujoco
import numpy as np

SCENE_PATH = "assets_mujoco_menagerie/franka_emika_panda/pick_place_scene.xml"
WINDOW_NAME = "Robot arm"

BLOCK_NAMES = ["block_red", "block_red2", "block_blue", "block_green", "block_yellow"]
# Deliberately two-to-one for "red" — block_red2 exists specifically to make
# "pick up the red block" genuinely ambiguous (see robot_tools.py's
# disambiguation handling and scripts/test_hri_behaviors.py).
BLOCK_COLORS = {
    "block_red": "red",
    "block_red2": "red",
    "block_blue": "blue",
    "block_green": "green",
    "block_yellow": "yellow",
}

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 120.0
APPROACH_HEIGHT = 0.15
GRASP_HEIGHT_OFFSET = 0.0  # pinch point targets the block's center height directly

# The IK task targets the "hand" BODY frame, but that's not where the fingers
# actually are — the fingertip pads (the real pinch point) sit ~10.3cm below
# the hand origin along the hand's local +Z axis (measured empirically via
# forward kinematics: left_finger body is 0.0584 below hand origin, and its
# fingertip_pad_collision geom is a further 0.0445 down, in world Z, when the
# gripper points straight down). Without compensating for this, "move to the
# block's position" actually moves the hand MESH there while the fingers end
# up ~10cm off — which is exactly why grasped blocks looked like they were
# floating below/beside the gripper instead of held between the fingers.
GRIPPER_PAD_OFFSET_Z = 0.1029


def _weld_at_current_pose(model, data, eq_id, body1_id, body2_id):
    """Recompute a weld constraint's relpose from the LIVE relative pose of the two
    bodies, so activating it freezes them where they currently are instead of
    snapping to whatever relpose got baked in at compile time (MuJoCo bakes
    weld eq_data from the XML-authored pose at compile time, not runtime)."""
    p1, q1 = data.xpos[body1_id].copy(), data.xquat[body1_id].copy()
    p2, q2 = data.xpos[body2_id].copy(), data.xquat[body2_id].copy()
    neg_q1 = np.zeros(4)
    mujoco.mju_negQuat(neg_q1, q1)
    relpos = np.zeros(3)
    mujoco.mju_rotVecQuat(relpos, p2 - p1, neg_q1)
    relquat = np.zeros(4)
    mujoco.mju_mulQuat(relquat, neg_q1, q2)
    model.eq_data[eq_id, 3:6] = relpos
    model.eq_data[eq_id, 6:10] = relquat


@dataclass
class _Command:
    fn: Callable
    args: tuple
    future: Future


class SimController:
    def __init__(self, show: bool = True, startup_timeout: float = 30.0):
        self._show = show
        self._cmd_queue: "queue.Queue[Optional[_Command]]" = queue.Queue()
        self._stop_flag = threading.Event()
        # Test-only knob: scripts/eval_halt_state.py sets this False to
        # reproduce the pre-fix behaviour for its before/after comparison.
        self._retarget_on_stop = True
        self._ready = threading.Event()
        self._current_action = "idle"
        self._held_object: Optional[str] = None
        self._latest_frame_jpeg: Optional[bytes] = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=startup_timeout):
            raise RuntimeError("SimController worker thread failed to start in time.")

    # ── Public API (all dispatch to the worker thread) ─────────────────────

    def get_scene_state(self) -> dict:
        return self._submit(self._impl_get_scene_state)

    def save_snapshot(self, path: str) -> None:
        """Render the current view to a PNG. Must run on the worker thread — the
        renderer's GL context is thread-affine (created in _run()), so touching
        sc.renderer directly from the caller's thread fails with a GLFW
        'context in use' error. Dispatch through the queue like everything else."""
        return self._submit(self._impl_save_snapshot, path)

    def get_current_action(self) -> str:
        """What the arm is doing RIGHT NOW. Deliberately does NOT go through the
        command queue (unlike get_scene_state(), which reads live MuJoCo data
        and must run on the worker thread) — a plain string attribute read is
        safe from any thread under the GIL, so this stays instant even while a
        long move_to()/pick()/place() is in progress on the worker thread. This
        is what makes "what are you doing?" queries actually reflect live state
        instead of blocking until the in-progress action finishes."""
        return self._current_action

    def get_current_frame_jpeg(self) -> Optional[bytes]:
        """Latest rendered frame as JPEG bytes, for the browser's live MJPEG
        stream (see api.py's /sim/stream). Same non-queued pattern as
        get_current_action() and for the same reason: a video stream reading
        through the command queue would freeze solid for the whole duration of
        every move_to()/pick()/place() call, since the worker thread processes
        one queued command at a time — the whole point of a *live* view is
        that it keeps updating precisely while the arm is mid-motion. The
        frame is refreshed as a side effect of _maybe_render(), which already
        runs on every physics step throughout any in-progress motion."""
        return self._latest_frame_jpeg

    def move_to(self, position) -> dict:
        return self._submit(self._impl_move_to, np.asarray(position, dtype=float))

    def pick(self, object_name: str) -> dict:
        return self._submit(self._impl_pick, object_name)

    def place(self, position) -> dict:
        return self._submit(self._impl_place, np.asarray(position, dtype=float))

    def stop(self) -> None:
        """Halt any in-progress motion immediately. Safe to call from any thread,
        at any time — does NOT wait behind whatever command is currently running."""
        self._stop_flag.set()

    def reset_object(self, object_name: str, position) -> None:
        """Teleport a free-body object to `position` (world xyz) with identity
        orientation and zero velocity. For evaluation scripts that need each
        trial to start from a known, controlled configuration instead of
        wherever a previous trial happened to leave the block — see
        eval_task_robustness.py, where a block left near a destination zone by
        a prior trial could otherwise make a later trial's success/failure
        scoring meaningless (a genuinely-successful placement registering as a
        "no real motion happened" failure, or vice versa)."""
        return self._submit(self._impl_reset_object, object_name, np.asarray(position, dtype=float))

    def close(self) -> None:
        self._cmd_queue.put(None)
        self._thread.join(timeout=5)

    # ── Command dispatch ─────────────────────────────────────────────────

    def _submit(self, fn: Callable, *args) -> Any:
        fut: Future = Future()
        self._cmd_queue.put(_Command(fn, args, fut))
        return fut.result()

    # ── Worker thread ────────────────────────────────────────────────────

    def _run(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = self.model.key_qpos[0, :9]
        # mj_resetData zeros data.ctrl, which does NOT match the home qpos we just
        # set — without this, the position actuators immediately start pulling the
        # arm toward ctrl=0 (a very different pose) the moment ANY _tick() runs
        # before the first move_to() (e.g. _set_gripper(), which pick() calls
        # first), corrupting the arm's starting point for every IK solve after it.
        self.data.ctrl[:7] = self.model.key_qpos[0, :7]
        mujoco.mj_forward(self.model, self.data)

        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.steps_per_frame = max(1, int((1 / 60) / self.model.opt.timestep))
        self._frame_counter = 0

        self.configuration = mink.Configuration(self.model)
        self.task = mink.FrameTask(
            frame_name="hand", frame_type="body",
            position_cost=1.0, orientation_cost=1.0, lm_damping=1e-2,
        )
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._down_quat = self.data.xquat[hand_id].copy()

        # Populate get_current_frame_jpeg() immediately — otherwise the browser's
        # video stream shows nothing until the first move_to()/pick()/place()
        # call happens to trigger a render via _tick().
        self._render_and_cache_frame()

        self._ready.set()

        while True:
            cmd = self._cmd_queue.get()
            if cmd is None:
                break
            try:
                result = cmd.fn(*cmd.args)
                cmd.future.set_result(result)
            except Exception as exc:  # noqa: BLE001 — must not kill the worker loop
                cmd.future.set_exception(exc)

        if self._show:
            cv2.destroyAllWindows()

    # ── Low-level simulation helpers (worker thread only) ───────────────

    def _render_and_cache_frame(self):
        """Render the current scene, cache it as JPEG for get_current_frame_jpeg(),
        and (if show=True) push it to the native OpenCV window. Runs on the
        worker thread only — the renderer's GL context is thread-affine."""
        self.renderer.update_scene(self.data)
        frame_bgr = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self._latest_frame_jpeg = encoded.tobytes()
        if self._show:
            cv2.imshow(WINDOW_NAME, frame_bgr)
            cv2.waitKey(1)

    def _maybe_render(self):
        self._frame_counter += 1
        if self._frame_counter % self.steps_per_frame != 0:
            return
        self._render_and_cache_frame()

    def _tick(self, n: int = 1) -> bool:
        """Step physics n times. Returns False early if stop() was called mid-loop."""
        for _ in range(n):
            if self._stop_flag.is_set():
                # Retarget the arm's position servos to where it is now.
                # _move_ee_to() writes the motion's FINAL IK target into
                # data.ctrl before stepping, so without this the next
                # command's first physics steps (e.g. pick()'s gripper-open
                # ticks, before it sets any new IK target) carry the arm the
                # rest of the way to the target the user just interrupted —
                # measured at a median 34mm of travel toward it
                # (scripts/eval_halt_state.py).
                if self._retarget_on_stop:
                    self.data.ctrl[:7] = self.data.qpos[:7]
                return False
            mujoco.mj_step(self.model, self.data)
            self._maybe_render()
        return True

    def _move_ee_to(self, gripper_target_pos, target_quat=None, max_iters=150, tol=1e-3, settle_steps=300) -> bool:
        """gripper_target_pos is where the GRIPPER PINCH POINT should end up — not
        the hand body frame the IK task actually solves for. We compensate by
        the measured pad offset here so every caller can think in terms of
        "where should the fingers be" rather than remembering the offset
        themselves. Only correct for the fixed straight-down orientation this
        project uses throughout (see GRIPPER_PAD_OFFSET_Z's docstring)."""
        target_quat = self._down_quat if target_quat is None else target_quat
        hand_target_pos = np.asarray(gripper_target_pos, dtype=float) + np.array([0, 0, GRIPPER_PAD_OFFSET_Z])
        self.configuration.update(self.data.qpos)
        target = mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), hand_target_pos)
        self.task.set_target(target)
        for _ in range(max_iters):
            vel = mink.solve_ik(self.configuration, [self.task], 0.01, solver="daqp", damping=1e-6)
            self.configuration.integrate_inplace(vel, 0.01)
            if np.linalg.norm(self.task.compute_error(self.configuration)) < tol:
                break
        self.data.ctrl[:7] = self.configuration.q[:7]
        return self._tick(settle_steps)

    def _set_gripper(self, value: float, steps: int = 100) -> bool:
        self.data.ctrl[7] = value
        return self._tick(steps)

    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    # ── Command implementations (worker thread only) ────────────────────

    def _impl_save_snapshot(self, path: str) -> None:
        self.renderer.update_scene(self.data)
        img = self.renderer.render()
        import PIL.Image
        PIL.Image.fromarray(img).save(path)

    def _impl_get_scene_state(self) -> dict:
        objects = []
        for name in BLOCK_NAMES:
            pos = self.data.xpos[self._body_id(name)].copy()
            objects.append({"name": name, "color": BLOCK_COLORS[name], "position": pos.tolist()})
        return {
            "objects": objects,
            "held_object": self._held_object,
            "current_action": self._current_action,
        }

    def _impl_move_to(self, position: np.ndarray) -> dict:
        self._stop_flag.clear()
        self._current_action = f"moving to {np.round(position, 3).tolist()}"
        completed = self._move_ee_to(position)
        self._current_action = "idle"
        return {"completed": completed}

    def _impl_pick(self, object_name: str) -> dict:
        if object_name not in BLOCK_NAMES:
            return {"completed": False, "error": f"Unknown object '{object_name}'."}
        if self._held_object is not None:
            return {"completed": False, "error": f"Already holding '{self._held_object}'."}

        self._stop_flag.clear()
        self._current_action = f"picking up {BLOCK_COLORS[object_name]} block"

        block_pos = self.data.xpos[self._body_id(object_name)].copy()
        steps = [
            lambda: self._set_gripper(GRIPPER_OPEN),
            lambda: self._move_ee_to(block_pos + np.array([0, 0, APPROACH_HEIGHT])),
            lambda: self._move_ee_to(block_pos + np.array([0, 0, GRASP_HEIGHT_OFFSET])),
            lambda: self._set_gripper(GRIPPER_CLOSED),
        ]
        for step in steps:
            if not step():
                self._current_action = "idle"
                return {"completed": False, "error": "Interrupted (stop() called)."}

        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{object_name}")
        _weld_at_current_pose(self.model, self.data, eq_id, self._body_id("hand"), self._body_id(object_name))
        self.data.eq_active[eq_id] = True
        self._held_object = object_name

        completed = self._move_ee_to(block_pos + np.array([0, 0, APPROACH_HEIGHT + 0.1]))
        self._current_action = "idle"
        return {"completed": completed, "held_object": self._held_object}

    def _impl_reset_object(self, object_name: str, position: np.ndarray) -> None:
        body_id = self._body_id(object_name)
        jnt_id = self.model.body_jntadr[body_id]
        qpos_adr = self.model.jnt_qposadr[jnt_id]
        dof_adr = self.model.jnt_dofadr[jnt_id]
        self.data.qpos[qpos_adr:qpos_adr + 3] = position
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]  # identity quat
        self.data.qvel[dof_adr:dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _impl_place(self, position: np.ndarray) -> dict:
        if self._held_object is None:
            return {"completed": False, "error": "Not holding anything."}

        self._stop_flag.clear()
        object_name = self._held_object
        self._current_action = f"placing {BLOCK_COLORS[object_name]} block"

        steps = [
            lambda: self._move_ee_to(position + np.array([0, 0, APPROACH_HEIGHT])),
            lambda: self._move_ee_to(position + np.array([0, 0, GRASP_HEIGHT_OFFSET])),
        ]
        for step in steps:
            if not step():
                self._current_action = "idle"
                return {"completed": False, "error": "Interrupted (stop() called)."}

        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{object_name}")
        self.data.eq_active[eq_id] = False
        self._set_gripper(GRIPPER_OPEN)
        self._held_object = None

        completed = self._move_ee_to(position + np.array([0, 0, APPROACH_HEIGHT + 0.1]))
        self._current_action = "idle"
        return {"completed": completed}
