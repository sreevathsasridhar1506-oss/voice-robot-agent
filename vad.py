"""
vad.py
======
Real-time speech start/stop detection via Silero VAD, run through ONNX
Runtime rather than the official `silero-vad` pip package — that package
pulls in a full PyTorch dependency just for VAD, which is unnecessary
weight given GPU memory is already earmarked for Whisper (M9). The raw
model file (`vad_model/silero_vad.onnx`) was fetched directly from the
Silero VAD GitHub repo instead.

Silero VAD is a streaming model: it processes fixed-size chunks (512
samples = 32ms at 16kHz) and carries a small recurrent state across calls,
so chunks must be fed in order from a live stream — this is NOT a
call-it-on-any-clip API like Whisper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

MODEL_PATH = Path("vad_model/silero_vad.onnx")
SAMPLE_RATE = 16000  # matches Whisper's expected input rate — no resampling needed between VAD and STT
CHUNK_SAMPLES = 512  # required chunk size for 16kHz per Silero VAD's streaming API


CONTEXT_SAMPLES = 64  # required lookback prefix for the 16kHz model — see below


class SileroVAD:
    """Low-level wrapper: raw audio chunk in, speech probability out.

    The ONNX graph's `input` isn't just the new 512-sample chunk — the
    reference implementation (silero_vad/utils_vad.py's OnnxWrapper) prepends
    a 64-sample "context" window carried over from the tail of the previous
    chunk (576 samples total), and the model's output shapes (`[None, None]`
    for input, `[None, None, None]` for stateN) don't reveal this on their
    own. Skipping it doesn't error — it just silently produces near-zero
    speech probabilities for real speech, which is what tipped this off."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = SAMPLE_RATE):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Silero VAD model not found at {MODEL_PATH}")
        self.session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._sr = np.array(sample_rate, dtype="int64")
        self.reset()

    def reset(self) -> None:
        """Clear recurrent state — call this at the start of a new stream/session."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray) -> float:
        """chunk: float32 array of exactly CHUNK_SAMPLES samples in [-1, 1].
        Returns the speech probability for this chunk."""
        if len(chunk) != CHUNK_SAMPLES:
            raise ValueError(f"Silero VAD expects exactly {CHUNK_SAMPLES} samples, got {len(chunk)}")
        x = np.concatenate([self._context, chunk[None, :].astype(np.float32)], axis=1)
        out, state = self.session.run(
            ["output", "stateN"],
            {
                "input": x,
                "state": self._state,
                "sr": self._sr,
            },
        )
        self._state = state
        self._context = x[:, -CONTEXT_SAMPLES:]
        return float(out[0][0])


class UtteranceSegmenter:
    """Wraps SileroVAD with hysteresis to emit clean speech-start/speech-end
    events instead of a raw per-chunk probability — a few consecutive
    speech chunks are required to trigger "start" (avoids a single noise
    spike registering as an utterance) and a longer run of silence is
    required to trigger "end" (tolerates brief pauses mid-sentence without
    cutting the utterance off early)."""

    def __init__(
        self,
        vad: Optional[SileroVAD] = None,
        speech_chunks_to_start: int = 3,   # ~96ms of consistent speech
        silence_chunks_to_end: int = 15,   # ~480ms of silence
    ):
        self.vad = vad or SileroVAD()
        self.speech_chunks_to_start = speech_chunks_to_start
        self.silence_chunks_to_end = silence_chunks_to_end
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0

    def reset(self) -> None:
        self.vad.reset()
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0

    def process_chunk(self, chunk: np.ndarray) -> Optional[str]:
        """Feed one CHUNK_SAMPLES-length chunk. Returns "start", "end", or None."""
        prob = self.vad.process_chunk(chunk)
        is_speech = prob >= self.vad.threshold

        if is_speech:
            self._speech_run += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            self._speech_run = 0

        if not self._in_speech and self._speech_run >= self.speech_chunks_to_start:
            self._in_speech = True
            return "start"
        if self._in_speech and self._silence_run >= self.silence_chunks_to_end:
            self._in_speech = False
            return "end"
        return None

    @property
    def in_speech(self) -> bool:
        return self._in_speech
