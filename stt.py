"""
stt.py
======
Speech-to-text via faster-whisper, GPU-accelerated.

This machine has an idle RTX 3060 — upgrades the learning repo's CPU/"base"
config to GPU/"small" for materially better accuracy on spoken object/color
names, at the cost of the Windows-specific DLL-path fix below (see M1's
scripts/test_whisper_gpu.py for how this was diagnosed).
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _add_nvidia_dll_dirs() -> None:
    """faster-whisper's ctranslate2 backend needs cuBLAS/cuDNN DLLs (from the
    nvidia-cublas-cu12/nvidia-cudnn-cu12 pip packages) on the loader search
    path. os.add_dll_directory() does NOT reliably work for ctranslate2's
    nested DLL dependencies on Windows — prepending PATH directly does. Must
    run before faster_whisper/ctranslate2 is imported, hence the placement
    here at module top, before the import below."""
    if sys.platform != "win32":
        return
    nvidia_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia_dir.exists():
        return
    bin_dirs = [str(p / "bin") for p in nvidia_dir.iterdir() if (p / "bin").is_dir()]
    os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ.get("PATH", "")


_add_nvidia_dll_dirs()

from faster_whisper import WhisperModel  # noqa: E402 — must follow the PATH fix above

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")

_MODEL: Optional[WhisperModel] = None
_MODEL_LOCK = threading.Lock()


def get_whisper_model() -> WhisperModel:
    """Lazy-loaded singleton — same pattern as get_agent_graph()/get_voice().

    Double-checked locking — see robot_tools.get_sim_controller()'s docstring
    for why this matters under this app's real thread concurrency (M13/M14)."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                logger.info("Loading Whisper model '%s' on %s ...", WHISPER_MODEL_NAME, WHISPER_DEVICE)
                try:
                    compute_type = "float16" if WHISPER_DEVICE == "cuda" else "int8"
                    _MODEL = WhisperModel(WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=compute_type)
                except Exception as exc:
                    logger.warning("GPU load failed (%s) — falling back to CPU per the documented fallback.", exc)
                    _MODEL = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
                logger.info("Whisper model ready.")
    return _MODEL


def transcribe_audio(audio_bytes: bytes, language: str = "en") -> str:
    """Transcribe raw audio bytes (WAV/MP3/any ffmpeg-readable format)."""
    model = get_whisper_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True,  # trims silence within the clip — distinct from the
                               # real-time Silero VAD segmentation planned for M11
        )
        text = " ".join(seg.text.strip() for seg in segments)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Transcribed in %.2fs (lang=%s, prob=%.2f): %r",
            elapsed, info.language, info.language_probability, text[:80],
        )
        return text.strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
