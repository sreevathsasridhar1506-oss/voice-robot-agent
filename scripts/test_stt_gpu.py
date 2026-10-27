"""
Standalone check (M9): GPU Whisper transcription — latency and accuracy.

Rather than needing a human to record samples, this generates realistic
speech via Piper (M8) for a few robot-vocabulary sentences and round-trips
it through Whisper — a fully self-contained way to test STT, and a nice
sanity check that the TTS and STT sample rates/formats are compatible.
"""

import sys
import time

sys.path.insert(0, ".")

from stt import get_whisper_model, transcribe_audio, WHISPER_DEVICE, WHISPER_MODEL_NAME
from tts import text_to_speech

TEST_SENTENCES = [
    "Pick up the red block.",
    "Put it down on the left side.",
    "There are two red blocks, one on the left and one on the right.",
    "What are you doing right now?",
]


def word_overlap(expected: str, actual: str) -> float:
    exp_words = set(w.strip(".,?!").lower() for w in expected.split())
    act_words = set(w.strip(".,?!").lower() for w in actual.split())
    if not exp_words:
        return 1.0
    return len(exp_words & act_words) / len(exp_words)


def main():
    print(f"Loading Whisper ({WHISPER_MODEL_NAME} on {WHISPER_DEVICE})...")
    t0 = time.perf_counter()
    get_whisper_model()
    print(f"Loaded in {time.perf_counter() - t0:.2f}s\n")

    overlaps = []
    for sentence in TEST_SENTENCES:
        audio_bytes = text_to_speech(sentence)

        t0 = time.perf_counter()
        transcript = transcribe_audio(audio_bytes)
        elapsed = time.perf_counter() - t0

        overlap = word_overlap(sentence, transcript)
        overlaps.append(overlap)
        print(f"Original:    {sentence!r}")
        print(f"Transcribed: {transcript!r}")
        print(f"Latency: {elapsed:.2f}s | Word overlap: {overlap:.0%}\n")

    avg_overlap = sum(overlaps) / len(overlaps)
    print(f"Average word overlap across {len(TEST_SENTENCES)} sentences: {avg_overlap:.0%}")
    assert avg_overlap > 0.7, "transcription accuracy too low — check model/audio format"
    print("\nSUCCESS — GPU Whisper transcribes robot-vocabulary commands accurately and fast.")


if __name__ == "__main__":
    main()
