"""
tts.py
======
Text-to-speech via Piper — replaces the learning repo's gTTS.

Piper runs entirely locally (no network round-trip), which matters once
barge-in (M14) needs to be able to cut off audio instantly — you can't
reliably cancel a request that's already left the machine. It's also fast
enough on CPU for the sentence-chunked streaming planned in M13.

Outputs WAV (16-bit PCM), not MP3 — that's what Piper produces natively, and
skipping an MP3 encode step keeps latency down, which matters more here than
file size.
"""

from __future__ import annotations

import io
import logging
import os
import re
import threading
import wave
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from piper import PiperVoice

load_dotenv()
logger = logging.getLogger(__name__)

PIPER_VOICE_NAME = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
PIPER_VOICE_PATH = Path("piper_voices") / f"{PIPER_VOICE_NAME}.onnx"

_VOICE: Optional[PiperVoice] = None
_VOICE_LOCK = threading.Lock()


def get_voice() -> PiperVoice:
    """Lazy-loaded singleton — same pattern as get_whisper_model()/get_agent_graph().

    Double-checked locking — see robot_tools.get_sim_controller()'s docstring
    for why this matters: text_to_speech() is called via asyncio.to_thread()
    once per sentence (M13), which can land on different thread-pool
    threads, and concurrent WS sessions could race here too."""
    global _VOICE
    if _VOICE is None:
        with _VOICE_LOCK:
            if _VOICE is None:
                if not PIPER_VOICE_PATH.exists():
                    raise FileNotFoundError(
                        f"Piper voice model not found at {PIPER_VOICE_PATH}. Download it with:\n"
                        f"  python -m piper.download_voices --download-dir piper_voices {PIPER_VOICE_NAME}"
                    )
                logger.info("Loading Piper voice from %s ...", PIPER_VOICE_PATH)
                _VOICE = PiperVoice.load(str(PIPER_VOICE_PATH))
                logger.info("Piper voice loaded.")
    return _VOICE


def _clean_text_for_tts(text: str) -> str:
    """Strip markdown/URLs the LLM might emit before sending to TTS — the robot's
    answers are short and conversational, so this rarely matters, but it's a
    cheap guard against a stray '**' or URL being read out literally."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\*+([^*]+)\*+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def text_to_speech(text: str) -> bytes:
    """Synthesize `text` to WAV bytes (16-bit PCM, mono)."""
    voice = get_voice()
    clean_text = _clean_text_for_tts(text)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(clean_text, wav_file)
    audio_bytes = buf.getvalue()

    logger.info("TTS generated %d bytes for %d chars of text.", len(audio_bytes), len(clean_text))
    return audio_bytes
