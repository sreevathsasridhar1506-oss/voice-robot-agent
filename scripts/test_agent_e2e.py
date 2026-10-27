"""
Standalone check (M5): automated end-to-end text-command smoke test.

Same underlying pipeline as cli_text_agent.py, but scripted so it can run
non-interactively: send a few commands through process_text_query() and
verify both the agent's reply AND the actual sim state changed as expected.
"""

import sys

sys.path.insert(0, ".")

from robot_tools import get_sim_controller
from voice_agent import process_text_query


def main():
    sc = get_sim_controller()

    state = sc.get_scene_state()
    blue_before = next(o for o in state["objects"] if o["name"] == "block_blue")["position"]
    print(f"block_blue starts at {[round(v, 3) for v in blue_before]}")

    r1 = process_text_query("What's on the table right now?")
    print(f"\n[describe] Agent: {r1['answer']}")
    assert "blue" in r1["answer"].lower(), "description should mention the blue block"

    r2 = process_text_query("Pick up the blue block and put it on the right side.")
    print(f"\n[pick+place] Agent: {r2['answer']}")

    state = sc.get_scene_state()
    blue_after = next(o for o in state["objects"] if o["name"] == "block_blue")["position"]
    print(f"block_blue ends at {[round(v, 3) for v in blue_after]}")

    import numpy as np
    moved = np.linalg.norm(np.array(blue_after[:2]) - np.array(blue_before[:2]))
    print(f"XY displacement: {moved:.3f} m")
    assert moved > 0.03, "the agent's tool calls should have actually moved the block"

    r3 = process_text_query("What are you doing right now?")
    print(f"\n[status] Agent: {r3['answer']}")

    print("\nSUCCESS — text command produced correct tool calls AND real sim motion.")
    sc.close()


if __name__ == "__main__":
    main()
