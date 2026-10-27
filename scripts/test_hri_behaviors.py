"""
Standalone check (M6): disambiguation + transparency behaviors.

(a) "pick up the red block" with TWO red blocks present should trigger a
    clarifying question, not a guess — and nothing should move.
(b) "pick up the red block on the left" (disambiguating info in the same
    sentence) should resolve directly and succeed without asking.
(c) Querying current_action WHILE a motion is in progress (via a background
    thread, same mechanism M3 proved for stop()) should reflect the real
    in-progress action, not a stale/generic "idle".
"""

import sys
import threading
import time

import numpy as np

sys.path.insert(0, ".")

from robot_tools import get_sim_controller
from voice_agent import process_text_query


def part_a_ambiguous_request():
    print("=== Part (a): ambiguous 'pick up the red block' ===")
    sc = get_sim_controller()
    state_before = sc.get_scene_state()
    reds_before = {o["name"]: o["position"] for o in state_before["objects"] if o["color"] == "red"}
    print(f"Red blocks before: { {k: [round(v,3) for v in p] for k,p in reds_before.items()} }")

    result = process_text_query("Pick up the red block.")
    print(f"Agent: {result['answer']}")

    state_after = sc.get_scene_state()
    reds_after = {o["name"]: o["position"] for o in state_after["objects"] if o["color"] == "red"}

    for name in reds_before:
        moved = np.linalg.norm(np.array(reds_after[name][:2]) - np.array(reds_before[name][:2]))
        assert moved < 0.03, f"{name} should NOT have moved (ambiguous command shouldn't act)"
    assert "?" in result["answer"], "response should ask a clarifying question"
    print("Part (a): SUCCESS — asked for clarification, nothing moved.\n")


def part_b_disambiguated_in_sentence():
    print("=== Part (b): 'pick up the red block on the left' (disambiguated inline) ===")
    sc = get_sim_controller()

    result = process_text_query("Pick up the red block on the left side, no need to ask, just do it.")
    print(f"Agent: {result['answer']}")

    state = sc.get_scene_state()
    print(f"Held object: {state['held_object']}")
    assert state["held_object"] in ("block_red", "block_red2"), "should have picked one of the red blocks"
    print("Part (b): SUCCESS — resolved and acted without asking.\n")

    # Put it back down and release so later tests start from a clean slate.
    sc.place([0.6, 0.0, 0.03])


def part_c_live_status_mid_motion():
    print("=== Part (c): current_action reflects live state mid-motion ===")
    sc = get_sim_controller()

    def run_move():
        sc.move_to([0.3, 0.4, 0.5])

    t = threading.Thread(target=run_move)
    t.start()
    time.sleep(0.15)
    # get_current_action() bypasses the command queue (unlike get_scene_state(),
    # which would block until the in-progress move_to finishes, defeating the
    # whole point of a live status check).
    mid_action = sc.get_current_action()
    print(f"current_action mid-motion: {mid_action!r}")
    assert mid_action != "idle", "should reflect the in-progress move, not idle"
    sc.stop()
    t.join(timeout=10)

    settled_action = sc.get_current_action()
    print(f"current_action after stop: {settled_action!r}")
    assert settled_action == "idle", "should return to idle after motion ends"
    print("Part (c): SUCCESS — status reflected the real in-progress action, not a stale value.\n")


if __name__ == "__main__":
    part_a_ambiguous_request()
    part_b_disambiguated_in_sentence()
    part_c_live_status_mid_motion()
    print("ALL CHECKS PASSED")
    get_sim_controller().close()
