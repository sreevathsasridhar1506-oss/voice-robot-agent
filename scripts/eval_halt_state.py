"""
ICAAIC paper evaluation — what state does a halt leave the arm in?

Physics only steps inside commands (_tick() returns as soon as the stop flag
is set, and the worker loop never steps while idle), so a halt freezes
simulated time with the arm mid-motion. This script interrupts move_to()
calls at random points and measures, per trial:

  - controller-side halt latency: wall time from sc.stop() to the
    interrupted move_to() returning to its caller (the simulator's share of
    the end-to-end barge-in latency, which also includes VAD confirmation,
    WebSocket transit and event-loop scheduling);
  - arm joint speed at the moment of the halt (the freeze does not model
    deceleration — this is the velocity the arm still has);
  - post-halt displacement: how far the hand moves during the first 100
    physics steps of the NEXT command, before it sets any new IK target.
    _impl_pick() begins with _set_gripper(GRIPPER_OPEN) == _tick(100), so
    this is exactly what happens if the user's next turn is a pick. The
    component of that displacement along (interrupted target - halt pose)
    shows whether the arm resumes the motion it was told to abandon.

Reproducing pick()'s opening ticks uses SimController internals (_submit,
_tick, data) — instrumentation only, run on the worker thread like every
other command.

Usage:
  python scripts/eval_halt_state.py                       -> halt_state.json (fix on)
  python scripts/eval_halt_state.py --before-fix          -> halt_state_before_fix.json
--before-fix sets SimController._retarget_on_stop = False, reproducing the
original behaviour (servo targets left at the interrupted motion's final IK
solution) without reverting code. Both runs use the same seed, so the
same targets and halt points are drawn.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from sim_controller import GRIPPER_PAD_OFFSET_Z, SimController

RESULTS_DIR = Path("../icaaic_paper/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BEFORE_FIX = "--before-fix" in sys.argv[1:]
OUT_NAME = "halt_state_before_fix.json" if BEFORE_FIX else "halt_state.json"

N_TRIALS = 50
NEXT_COMMAND_PRE_IK_STEPS = 100  # == _impl_pick()'s opening _set_gripper() call
SEED = 0

# Gripper pinch-point targets spread across the workspace, far enough from
# home that an interrupted move is genuinely mid-motion.
TARGETS = [
    [0.55, 0.25, 0.25],
    [0.55, -0.25, 0.25],
    [0.35, 0.0, 0.15],
    [0.60, 0.0, 0.30],
]


def hand_pos(sc: SimController) -> np.ndarray:
    return np.array(sc._submit(lambda: sc.data.xpos[sc._body_id("hand")].copy()))


def arm_speed(sc: SimController) -> float:
    return float(sc._submit(lambda: float(np.max(np.abs(sc.data.qvel[:7])))))


def summarize(xs: list[float]) -> dict:
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def main() -> None:
    rng = random.Random(SEED)
    sc = SimController(show=False)
    sc._retarget_on_stop = not BEFORE_FIX
    print(f"condition: {'BEFORE fix (no retarget)' if BEFORE_FIX else 'AFTER fix (retarget on stop)'}")
    home_pinch = hand_pos(sc) - np.array([0, 0, GRIPPER_PAD_OFFSET_Z])

    # Uninterrupted move once, to pick a halt window inside the motion.
    t0 = time.perf_counter()
    sc.move_to(TARGETS[0])
    full_move_s = time.perf_counter() - t0
    sc.move_to(home_pinch.tolist())
    print(f"uninterrupted move_to: {full_move_s * 1000:.0f} ms wall")

    trials = []
    for i in range(N_TRIALS):
        target = TARGETS[i % len(TARGETS)]
        halt_after_s = rng.uniform(0.15, 0.6) * full_move_s

        returned = {}

        def run_move():
            returned["result"] = sc.move_to(target)
            returned["t"] = time.perf_counter()

        th = threading.Thread(target=run_move)
        th.start()
        time.sleep(halt_after_s)
        t_stop = time.perf_counter()
        sc.stop()
        th.join()
        halt_latency_ms = (returned["t"] - t_stop) * 1000

        p_halt = hand_pos(sc)
        speed = arm_speed(sc)
        # Every command clears the stop flag first (_impl_pick line order:
        # _stop_flag.clear(), then _set_gripper()), so mirror that exactly —
        # ticking with the flag still set would take zero steps.
        sc._submit(lambda: (sc._stop_flag.clear(), sc._tick(NEXT_COMMAND_PRE_IK_STEPS)))
        p_after = hand_pos(sc)

        target_hand = np.array(target) + np.array([0, 0, GRIPPER_PAD_OFFSET_Z])
        to_target = target_hand - p_halt
        dist_to_target = float(np.linalg.norm(to_target))
        disp = p_after - p_halt
        disp_mm = float(np.linalg.norm(disp)) * 1000
        toward_mm = float(disp @ (to_target / dist_to_target)) * 1000 if dist_to_target > 1e-6 else 0.0

        trial = {
            "trial": i + 1,
            "target": target,
            "halt_after_ms": halt_after_s * 1000,
            "completed": returned["result"].get("completed"),
            "controller_halt_latency_ms": halt_latency_ms,
            "arm_joint_speed_at_halt_rad_s": speed,
            "remaining_distance_to_target_mm": dist_to_target * 1000,
            "post_halt_displacement_mm": disp_mm,
            "post_halt_toward_interrupted_target_mm": toward_mm,
        }
        trials.append(trial)
        print(f"  trial {i + 1:2d}: completed={trial['completed']}  "
              f"halt={halt_latency_ms:5.1f}ms  speed={speed:.2f}rad/s  "
              f"remaining={dist_to_target * 1000:5.1f}mm  "
              f"moved={disp_mm:5.1f}mm  toward_old_target={toward_mm:5.1f}mm")

        sc.move_to(home_pinch.tolist())

    sc.close()

    interrupted = [t for t in trials if t["completed"] is False]
    results = {
        "condition": "before_fix_no_retarget" if BEFORE_FIX else "after_fix_retarget_on_stop",
        "seed": SEED,
        "description": "Halts of move_to() at random points; post-halt displacement "
                       f"over the next command's first {NEXT_COMMAND_PRE_IK_STEPS} physics steps.",
        "n_trials": N_TRIALS,
        "n_interrupted": len(interrupted),
        "uninterrupted_move_ms": full_move_s * 1000,
        "controller_halt_latency_ms": summarize([t["controller_halt_latency_ms"] for t in interrupted]),
        "arm_joint_speed_at_halt_rad_s": summarize([t["arm_joint_speed_at_halt_rad_s"] for t in interrupted]),
        "remaining_distance_to_target_mm": summarize([t["remaining_distance_to_target_mm"] for t in interrupted]),
        "post_halt_displacement_mm": summarize([t["post_halt_displacement_mm"] for t in interrupted]),
        "post_halt_toward_interrupted_target_mm": summarize(
            [t["post_halt_toward_interrupted_target_mm"] for t in interrupted]),
        "trials": trials,
    }
    out = RESULTS_DIR / OUT_NAME
    out.write_text(json.dumps(results, indent=2))
    print(f"\ninterrupted {len(interrupted)}/{N_TRIALS}")
    for k in ["controller_halt_latency_ms", "arm_joint_speed_at_halt_rad_s",
              "post_halt_displacement_mm", "post_halt_toward_interrupted_target_mm"]:
        s = results[k]
        print(f"{k}: median={s['median']:.2f} mean={s['mean']:.2f} sd={s['stdev']:.2f} "
              f"[{s['min']:.2f}, {s['max']:.2f}]")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
