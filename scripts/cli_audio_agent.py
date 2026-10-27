"""
Interactive CLI (M10): speak a command, watch the arm move, hear the reply.

    python scripts/cli_audio_agent.py

Press Enter to record a 4-second clip, then speak your command. No VAD yet
(that's M11) — recording is a fixed duration for now. Ctrl+C to exit.
"""

import io
import sys

sys.path.insert(0, ".")

import numpy as np
import sounddevice as sd
import soundfile as sf

from voice_agent import process_audio_query

SAMPLE_RATE = 16000
RECORD_SECONDS = 4


def record_clip() -> bytes:
    print(f"Recording for {RECORD_SECONDS}s...")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("Done recording.")
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def play_audio(audio_bytes: bytes) -> None:
    data, samplerate = sf.read(io.BytesIO(audio_bytes))
    sd.play(data, samplerate)
    sd.wait()


def main():
    print("Robot-arm voice agent — press Enter to record a command (Ctrl+C to quit).")
    print("Try: 'what's on the table?' / 'pick up the red block' / 'put it on the left' / 'stop'")
    while True:
        try:
            input("\nPress Enter to record...")
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        audio_bytes = record_clip()
        result = process_audio_query(audio_bytes)

        print(f"Heard: {result.get('transcript', '')!r}")
        print(f"Agent: {result['answer']}")

        if result.get("audio"):
            play_audio(result["audio"])


if __name__ == "__main__":
    main()
