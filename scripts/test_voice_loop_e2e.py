"""
Standalone check (M10): full batch voice loop, audio in -> action -> audio out.

Same trick as M9: use Piper to synthesize a spoken command (standing in for
a microphone recording), feed it through process_audio_query(), and verify
all three things worked — correct transcript, real simulation motion, and
valid spoken audio back.
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from robot_tools import get_sim_controller
from tts import text_to_speech
from voice_agent import process_audio_query


def main():
    sc = get_sim_controller()

    state = sc.get_scene_state()
    green_before = next(o for o in state["objects"] if o["name"] == "block_green")["position"]
    print(f"block_green starts at {[round(v, 3) for v in green_before]}")

    spoken_command = text_to_speech("Pick up the green block and put it in the center.")
    print(f"Synthesized {len(spoken_command)} bytes of 'spoken' command audio.\n")

    result = process_audio_query(spoken_command)

    print(f"Transcript: {result['transcript']!r}")
    print(f"Agent:      {result['answer']}")
    print(f"Reply audio: {len(result['audio'])} bytes\n")

    assert "green" in result["transcript"].lower(), "transcript should mention 'green'"
    assert len(result["audio"]) > 1000, "reply audio should be substantial, not empty/silent"

    state = sc.get_scene_state()
    green_after = next(o for o in state["objects"] if o["name"] == "block_green")["position"]
    print(f"block_green ends at {[round(v, 3) for v in green_after]}")

    moved = np.linalg.norm(np.array(green_after[:2]) - np.array(green_before[:2]))
    print(f"XY displacement: {moved:.3f} m")
    assert moved > 0.03, "the block should have actually moved"

    print("\nSUCCESS — full audio-in -> action -> audio-out loop works end to end.")
    sc.close()


if __name__ == "__main__":
    main()
