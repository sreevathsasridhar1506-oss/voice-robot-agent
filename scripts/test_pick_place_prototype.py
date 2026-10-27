"""
Standalone check (M2): hardcoded pick-and-place on the raw MuJoCo scene.

No agent, no voice, no sim_controller abstraction yet — this is where the
IK/grasping approach gets proven out before M3 wraps it in a reusable API.

Sequence: pick up block_red, move it to a drop location, release it.

Visualization uses mujoco.Renderer + OpenCV (matching the pattern already
proven out in mujoco_stewart/run_stewart.py), NOT mujoco.viewer.launch_passive
— the latter has a known "sync busy" reliability issue on this machine.
"""

import time

import cv2
import mujoco
import numpy as np
import mink

SCENE_PATH = "assets_mujoco_menagerie/franka_emika_panda/pick_place_scene.xml"
DROP_POS = np.array([0.55, 0.0, 0.03])
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 120.0  # weld constraint does the real "holding" — this just looks right
WINDOW_NAME = "Panda pick-place prototype (M2)"


def weld_at_current_pose(model, data, eq_id, body1_id, body2_id):
    """Recompute a weld constraint's relpose from the LIVE relative pose of the two
    bodies, so activating it freezes them where they currently are instead of
    snapping to whatever relpose got baked in at compile time."""
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


class PandaRig:
    def __init__(self, model, data, renderer=None, show=True):
        self.model = model
        self.data = data
        self.renderer = renderer
        self.show = show
        self.steps_per_frame = max(1, int((1 / 60) / model.opt.timestep))
        self._step_count = 0

        self.configuration = mink.Configuration(model)
        self.task = mink.FrameTask(
            frame_name="hand", frame_type="body",
            position_cost=1.0, orientation_cost=1.0, lm_damping=1e-2,
        )
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.down_quat = data.xquat[hand_id].copy()  # captured at home pose — points straight down

    def _maybe_render(self):
        if self.renderer is None:
            return
        self._step_count += 1
        if self._step_count % self.steps_per_frame != 0:
            return
        self.renderer.update_scene(self.data)
        frame = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        if self.show:
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(1)

    def _tick(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self._maybe_render()

    def move_to(self, target_pos, target_quat=None, max_iters=150, tol=1e-3, settle_steps=300):
        target_quat = self.down_quat if target_quat is None else target_quat
        self.configuration.update(self.data.qpos)
        target = mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_pos)
        self.task.set_target(target)
        for _ in range(max_iters):
            vel = mink.solve_ik(self.configuration, [self.task], 0.01, solver="daqp", damping=1e-6)
            self.configuration.integrate_inplace(vel, 0.01)
            err = self.task.compute_error(self.configuration)
            if np.linalg.norm(err) < tol:
                break
        q_arm_target = self.configuration.q[:7]
        self.data.ctrl[:7] = q_arm_target
        self._tick(settle_steps)

    def set_gripper(self, value, steps=100):
        self.data.ctrl[7] = value
        self._tick(steps)

    def grasp(self, block_name):
        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{block_name}")
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, block_name)
        weld_at_current_pose(self.model, self.data, eq_id, hand_id, block_id)
        self.data.eq_active[eq_id] = True

    def release(self, block_name):
        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{block_name}")
        self.data.eq_active[eq_id] = False


def save_snapshot(renderer, data, path):
    renderer.update_scene(data)
    img = renderer.render()
    import PIL.Image
    PIL.Image.fromarray(img).save(path)


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:9] = model.key_qpos[0, :9]
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    save_snapshot(renderer, data, "scripts/_snapshot_before.png")

    rig = PandaRig(model, data, renderer=renderer, show=True)
    try:
        block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block_red")
        block_pos_before = data.xpos[block_id].copy()
        print(f"block_red starts at {np.round(block_pos_before, 3)}")

        rig.set_gripper(GRIPPER_OPEN)

        approach = block_pos_before + np.array([0, 0, 0.15])
        rig.move_to(approach)
        print("approached above block_red")

        grasp_point = block_pos_before + np.array([0, 0, 0.02])
        rig.move_to(grasp_point)
        print("descended to grasp height")

        rig.set_gripper(GRIPPER_CLOSED)
        rig.grasp("block_red")
        print("grasped block_red (weld active)")

        rig.move_to(grasp_point + np.array([0, 0, 0.2]))
        print("lifted")

        drop_approach = DROP_POS + np.array([0, 0, 0.2])
        rig.move_to(drop_approach)
        rig.move_to(DROP_POS + np.array([0, 0, 0.02]))
        print("moved to drop location")

        rig.release("block_red")
        rig.set_gripper(GRIPPER_OPEN)
        print("released block_red")

        rig.move_to(drop_approach)
        print("retreated up")

        block_pos_after = data.xpos[block_id].copy()
        print(f"block_red ends at {np.round(block_pos_after, 3)}")

        moved_distance = np.linalg.norm(block_pos_after[:2] - block_pos_before[:2])
        print(f"XY displacement: {moved_distance:.3f} m")
        if moved_distance > 0.03:
            print("SUCCESS — block was moved.")
        else:
            print("FAILED — block did not move appreciably.")

        save_snapshot(renderer, data, "scripts/_snapshot_after.png")

        print("Holding final frame for 3s (press any key to close early)...")
        renderer.update_scene(data)
        frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(3000)
    finally:
        renderer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
