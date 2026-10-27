"""
Interactive demo (M11): live speech start/stop detection from your actual mic.

    python scripts/live_vad_demo.py

Talk normally — you should see "speech started" shortly after you begin
talking and "speech ended" shortly after you stop. Ctrl+C to exit.
"""

import sys
import time

sys.path.insert(0, ".")

import numpy as np
import sounddevice as sd

from vad import CHUNK_SAMPLES, SAMPLE_RATE, UtteranceSegmenter


def main():
    segmenter = UtteranceSegmenter()
    print(f"Listening at {SAMPLE_RATE}Hz ({CHUNK_SAMPLES} samples/chunk = {CHUNK_SAMPLES/SAMPLE_RATE*1000:.0f}ms). "
          f"Talk normally. Ctrl+C to quit.\n")

    def callback(indata, frames, time_info, status):
        chunk = indata[:, 0]
        event = segmenter.process_chunk(chunk)
        if event == "start":
            print(f"[{time.strftime('%H:%M:%S')}] speech started")
        elif event == "end":
            print(f"[{time.strftime('%H:%M:%S')}] speech ended")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
        callback=callback,
    ):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nBye.")


if __name__ == "__main__":
    main()
