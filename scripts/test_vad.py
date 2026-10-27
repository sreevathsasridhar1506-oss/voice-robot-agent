"""
Standalone check (M11): Silero VAD start/stop detection accuracy.

Builds a synthetic stream — silence, then speech (via Piper, M8), then
silence again — and feeds it chunk-by-chunk through UtteranceSegmenter,
checking that "start" and "end" events land at roughly the right times.
No live mic needed for this automated check; scripts/live_vad_demo.py is
the real interactive version for trying it with your actual voice.
"""

import io
import sys

sys.path.insert(0, ".")

import numpy as np
import soundfile as sf
from scipy.signal import resample

from tts import text_to_speech
from vad import CHUNK_SAMPLES, SAMPLE_RATE, UtteranceSegmenter


def synth_at_16k(text: str) -> np.ndarray:
    audio_bytes = text_to_speech(text)
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if sr != SAMPLE_RATE:
        n_out = int(len(data) * SAMPLE_RATE / sr)
        data = resample(data, n_out).astype(np.float32)
    return data


def main():
    silence_lead = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    speech = synth_at_16k("Pick up the blue block.")
    silence_trail = np.zeros(int(1.5 * SAMPLE_RATE), dtype=np.float32)

    stream = np.concatenate([silence_lead, speech, silence_trail])
    speech_start_s = len(silence_lead) / SAMPLE_RATE
    speech_end_s = (len(silence_lead) + len(speech)) / SAMPLE_RATE
    print(f"Synthetic stream: {len(stream)/SAMPLE_RATE:.2f}s total, "
          f"speech from {speech_start_s:.2f}s to {speech_end_s:.2f}s")

    segmenter = UtteranceSegmenter()
    events = []
    n_chunks = len(stream) // CHUNK_SAMPLES
    for i in range(n_chunks):
        chunk = stream[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
        event = segmenter.process_chunk(chunk)
        if event:
            t = i * CHUNK_SAMPLES / SAMPLE_RATE
            events.append((event, t))
            print(f"  t={t:.2f}s: {event}")

    starts = [t for e, t in events if e == "start"]
    ends = [t for e, t in events if e == "end"]

    assert len(starts) == 1, f"expected exactly 1 start event, got {len(starts)}"
    assert len(ends) == 1, f"expected exactly 1 end event, got {len(ends)}"

    start_error = abs(starts[0] - speech_start_s)
    end_error = abs(ends[0] - speech_end_s)
    print(f"\nstart detected at {starts[0]:.2f}s (actual speech at {speech_start_s:.2f}s, error {start_error:.2f}s)")
    print(f"end detected at {ends[0]:.2f}s (actual speech ended {speech_end_s:.2f}s, error {end_error:.2f}s)")

    assert start_error < 0.3, "start detection should be within ~300ms of actual speech onset"
    assert end_error < 1.0, "end detection should fire within ~1s of actual speech end (hangover is expected)"

    print("\nSUCCESS — VAD correctly detected exactly one start and one end event, at sensible times.")


if __name__ == "__main__":
    main()
