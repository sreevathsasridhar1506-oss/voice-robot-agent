"""
Standalone check (M8): does Piper TTS synthesize and play correctly?

A sentence in, WAV bytes out, played through the default speaker and also
saved to disk so it can be replayed/inspected manually.
"""

import io
import sys
import wave

sys.path.insert(0, ".")

import sounddevice as sd
import soundfile as sf

from tts import text_to_speech

TEST_SENTENCES = [
    "Hello! I picked up the red block and placed it on the left side of the table.",
    "There are two red blocks on the table. Which one do you mean?",
]


def main():
    for i, sentence in enumerate(TEST_SENTENCES):
        print(f"\nSynthesizing: {sentence!r}")
        audio_bytes = text_to_speech(sentence)
        print(f"Got {len(audio_bytes)} bytes of WAV audio.")

        out_path = f"scripts/_tts_sample_{i}.wav"
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        print(f"Saved to {out_path}")

        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        duration_s = len(data) / samplerate
        print(f"Duration: {duration_s:.2f}s at {samplerate}Hz — playing...")
        sd.play(data, samplerate)
        sd.wait()

    print("\nSUCCESS — synthesized and played audio for all test sentences.")


if __name__ == "__main__":
    main()
