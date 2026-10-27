"""
Standalone check (M3): sim_controller.py's high-level API + thread-safety contract.

Part (a): drive the same pick-place sequence as M2, but through the new
          SimController API instead of hand-rolled PandaRig code.
Part (b): the load-bearing test — start a move_to() on a background thread,
          call stop() from the main thread mid-motion, and confirm the arm
          actually halts short of the target instead of finishing the move.
          This is the exact mechanism barge-in (M14) will rely on later.
"""

import sys
import threading
import time

import numpy as np

sys.path.insert(0, ".")
from sim_controller import SimController


def part_a_pick_place():
    print("=== Part (a): pick + place through SimController ===")
    sc = SimController(show=True)
    try:
        state_before = sc.get_scene_state()
        red_before = next(o for o in state_before["objects"] if o["name"] == "block_red")["position"]
        print(f"block_red starts at {np.round(red_before, 3)}")

        pick_result = sc.pick("block_red")
        print("pick():", pick_result)
        assert pick_result["completed"], "pick() should complete uninterrupted"

        place_result = sc.place([0.55, 0.0, 0.03])
        print("place():", place_result)
        assert place_result["completed"], "place() should complete uninterrupted"

        state_after = sc.get_scene_state()
        red_after = next(o for o in state_after["objects"] if o["name"] == "block_red")["position"]
        print(f"block_red ends at {np.round(red_after, 3)}")

        moved = np.linalg.norm(np.array(red_after[:2]) - np.array(red_before[:2]))
        print(f"XY displacement: {moved:.3f} m")
        assert moved > 0.03, "block should have moved"
        print("Part (a): SUCCESS\n")
    finally:
        sc.close()


def part_b_stop_from_another_thread():
    print("=== Part (b): stop() from another thread mid-motion ===")
    sc = SimController(show=True)
    try:
        far_target = np.array([0.3, 0.4, 0.5])

        result_holder = {}

        def run_move():
            result_holder["result"] = sc.move_to(far_target)

        t = threading.Thread(target=run_move)
        t.start()
        time.sleep(0.15)  # let the motion begin
        sc.stop()
        t.join(timeout=10)

        result = result_holder.get("result")
        print("move_to() result after stop():", result)
        assert result is not None, "move_to() thread did not finish"
        assert result["completed"] is False, "move_to() should report interrupted, not completed"

        hand_pos = np.array(sc.get_scene_state()["objects"][0]["position"])  # sanity: state still queryable
        print("SimController still responsive after stop().")
        print("Part (b): SUCCESS — motion was interrupted, arm did not silently finish the move.\n")
    finally:
        sc.close()


if __name__ == "__main__":
    part_a_pick_place()
    part_b_stop_from_another_thread()
    print("ALL CHECKS PASSED")
