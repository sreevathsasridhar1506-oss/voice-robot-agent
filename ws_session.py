"""
ws_session.py
=============
Per-connection state for the full-duplex /ws endpoint.

Buffers incoming raw PCM16 audio, runs it through Silero VAD (M11) to
auto-detect utterance boundaries (no manual "click to record" — the client
just streams continuously), and dispatches completed utterances through the
full voice pipeline (M10: STT -> agent -> TTS, sentence-streamed per M13)
without blocking the FastAPI event loop.

Barge-in (M14): while a turn is being processed/spoken, incoming audio is
NOT ignored — it keeps running through the same VAD, and a fresh "start"
event while _processing is True is treated as the user interrupting. This
required a real architectural fix, not just new logic: _process_utterance()
used to be awaited directly from the receive loop, which meant the loop
couldn't read any NEW incoming audio until the previous utterance finished
processing (often several seconds) — incoming barge-in audio would never
even reach _handle_chunk(). It now runs as a background asyncio.Task instead,
so the receive loop keeps pulling in audio (and therefore keeps running VAD)
the whole time a response is being generated/spoken.
"""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import threading

import numpy as np
import soundfile as sf
from fastapi import WebSocket

from metrics import get_session
from stt import transcribe_audio
from tts import text_to_speech
from vad import CHUNK_SAMPLES, SAMPLE_RATE, UtteranceSegmenter
from voice_agent import stream_text_query_sentences

logger = logging.getLogger(__name__)

FRAME_BYTES = CHUNK_SAMPLES * 2  # int16 mono = 2 bytes/sample

# VAD needs a few consecutive speech chunks (see UtteranceSegmenter's
# speech_chunks_to_start) before it confirms "start" — by then, real speech
# has already been happening for ~100ms. Keeping a short rolling pre-buffer
# and prepending it once "start" fires avoids clipping the first word.
PRE_BUFFER_CHUNKS = 10  # ~320ms at 32ms/chunk


class WSSession:
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.segmenter = UtteranceSegmenter()
        self._pcm_buffer = b""  # leftover bytes shorter than one full chunk
        self._pre_buffer: list[np.ndarray] = []
        self._utterance_chunks: list[np.ndarray] = []
        self._processing = False
        self._cancel_flag = threading.Event()
        self._active_task: "asyncio.Task | None" = None

    # ── Incoming audio ───────────────────────────────────────────────────

    async def handle_audio_bytes(self, raw_bytes: bytes) -> None:
        """Feed raw int16 PCM bytes in — any chunk size; re-framed internally
        to Silero VAD's required CHUNK_SAMPLES regardless of how the client
        happens to batch its sends."""
        self._pcm_buffer += raw_bytes
        while len(self._pcm_buffer) >= FRAME_BYTES:
            frame_bytes, self._pcm_buffer = self._pcm_buffer[:FRAME_BYTES], self._pcm_buffer[FRAME_BYTES:]
            chunk = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            await self._handle_chunk(chunk)

    async def _handle_chunk(self, chunk: np.ndarray) -> None:
        # Branch on the event value itself, not segmenter.in_speech checked
        # afterward — process_chunk() already flips in_speech to False before
        # returning "end", so checking in_speech post-hoc silently skips the
        # "end" handling on exactly the chunk where it matters most.
        event = self.segmenter.process_chunk(chunk)

        if event == "start":
            if self._processing:
                await self._barge_in()
            await self._send_json({"type": "speech_started"})
            self._utterance_chunks = self._pre_buffer + [chunk]
            self._pre_buffer = []
            return

        if self._processing:
            # Mid-turn, not a fresh "start": there's no in-progress utterance
            # of our own to accumulate into, but the pre-buffer must still be
            # kept warm with recent audio — otherwise, when a barge-in "start"
            # eventually fires, it gets prefixed with whatever stale audio was
            # in the buffer from BEFORE this turn even began (the early return
            # here used to skip this entirely, corrupting the first ~300ms of
            # the interrupting utterance with old audio — confirmed by a
            # barge-in transcript coming back mangled until this was fixed).
            if not self.segmenter.in_speech:
                self._pre_buffer.append(chunk)
                if len(self._pre_buffer) > PRE_BUFFER_CHUNKS:
                    self._pre_buffer.pop(0)
            return

        if event == "end":
            self._utterance_chunks.append(chunk)
            await self._send_json({"type": "speech_ended"})
            # Run as a background task, NOT awaited here — the receive loop
            # must keep pulling in audio (and running VAD on it) while this
            # turn is processed/spoken, or barge-in could never be detected.
            self._active_task = asyncio.create_task(self._process_utterance())
            return

        if self.segmenter.in_speech:
            self._utterance_chunks.append(chunk)
        else:
            self._pre_buffer.append(chunk)
            if len(self._pre_buffer) > PRE_BUFFER_CHUNKS:
                self._pre_buffer.pop(0)

    # ── Barge-in (M14) ───────────────────────────────────────────────────

    async def _barge_in(self) -> None:
        logger.info("Barge-in detected — cancelling in-flight response.")
        self._cancel_flag.set()

        # sim_controller.stop() just sets a threading.Event — instant, no
        # need for to_thread. Reuses the exact mechanism M3 proved halts an
        # in-progress motion from another thread.
        from robot_tools import get_sim_controller
        get_sim_controller().stop()

        get_session().record_interrupt()
        await self._send_json({"type": "barge_in"})
        self._processing = False

    # ── Utterance processing ────────────────────────────────────────────

    async def _process_utterance(self) -> None:
        audio = np.concatenate(self._utterance_chunks)
        self._utterance_chunks = []
        self._processing = True
        self._cancel_flag = threading.Event()
        cancel_flag = self._cancel_flag  # local ref: _barge_in() may replace self._cancel_flag before we're done
        try:
            await self._send_json({"type": "status", "content": "thinking"})
            wav_bytes = _to_wav_bytes(audio)

            # transcribe_audio() is blocking (Whisper inference) — offload it
            # so it doesn't stall the event loop for other connections.
            transcript = await asyncio.to_thread(transcribe_audio, wav_bytes)
            # Always report the transcript, even if a barge-in landed while
            # transcribing — it's just informational text (unlike audio
            # output, it doesn't need to stop "promptly"), and showing what
            # was heard before being cut off is more useful than silently
            # dropping it. Only the actual speaking (_stream_and_speak) needs
            # to respect cancellation.
            await self._send_json({"type": "transcript", "content": transcript})

            await self._stream_and_speak(transcript, cancel_flag)
            if not cancel_flag.is_set():
                await self._send_json({"type": "done"})
        except Exception as exc:
            logger.exception("Error processing utterance")
            await self._send_json({"type": "error", "content": str(exc)})
        finally:
            self._processing = False
            if not cancel_flag.is_set():
                # A normal (non-interrupted) turn ended -- safe to reset.
                # If cancelled, the segmenter is already mid-tracking the
                # barge-in utterance that interrupted us; resetting it here
                # would wipe that in-progress state.
                self.segmenter.reset()

    # ── Incoming text (handy for quick testing + M15's initial text-only UI) ──

    async def handle_text_command(self, text: str) -> None:
        if self._processing:
            await self._send_json({"type": "error", "content": "Still processing the previous command."})
            return
        self._processing = True
        self._cancel_flag = threading.Event()
        cancel_flag = self._cancel_flag
        try:
            await self._send_json({"type": "status", "content": "thinking"})
            await self._send_json({"type": "transcript", "content": text})
            await self._stream_and_speak(text, cancel_flag)
            if not cancel_flag.is_set():
                await self._send_json({"type": "done"})
        except Exception as exc:
            logger.exception("Error processing text command")
            await self._send_json({"type": "error", "content": str(exc)})
        finally:
            self._processing = False

    # ── Sentence-chunked streaming (M13), cancellable (M14) ─────────────

    async def _stream_and_speak(self, query: str, cancel_flag: threading.Event) -> None:
        """Bridges the synchronous stream_text_query_sentences() generator
        (which makes blocking network calls internally) into this coroutine:
        runs it in a background thread, which pushes each ready sentence onto
        a thread-safe queue.Queue; this coroutine drains that queue via
        asyncio.to_thread(q.get) so the event loop is never blocked waiting
        for the next sentence, and each sentence's audio is sent to the
        client as soon as it's synthesized rather than waiting for the whole
        answer (time-to-first-audio — see scripts/test_streaming_tts.py).

        cancel_flag is checked between every step so a barge-in stops new
        audio going out promptly — though note the underlying blocking call
        already in flight inside the producer thread (an LLM token, a Piper
        synthesis) can't be forcibly aborted mid-call, only its result
        discarded. Same documented limitation as Whisper/Piper in general."""
        q: "queue.Queue[tuple[str, str]]" = queue.Queue()

        def producer():
            try:
                for sentence in stream_text_query_sentences(query):
                    if cancel_flag.is_set():
                        break
                    q.put(("sentence", sentence))
            except Exception as exc:  # noqa: BLE001 — surface to the consumer, don't crash the thread silently
                q.put(("error", str(exc)))
            finally:
                q.put(("done", ""))

        threading.Thread(target=producer, daemon=True).start()

        while True:
            kind, payload = await asyncio.to_thread(q.get)
            if cancel_flag.is_set():
                return
            if kind == "done":
                return
            if kind == "error":
                await self._send_json({"type": "error", "content": payload})
                return
            await self._send_json({"type": "answer_chunk", "content": payload})
            if cancel_flag.is_set():
                return
            audio_bytes = await asyncio.to_thread(text_to_speech, payload)
            if cancel_flag.is_set():
                return
            await self.ws.send_bytes(audio_bytes)

    # ── Outgoing ─────────────────────────────────────────────────────────

    async def _send_json(self, payload: dict) -> None:
        await self.ws.send_json(payload)


def _to_wav_bytes(audio_float32: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio_float32, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()
