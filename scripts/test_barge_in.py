"""
Standalone check (M14): barge-in.

Streams a slow multi-tool-call command, waits until it's mid-processing
(thinking / speaking), then streams a second "Stop." command on the SAME
connection — simulating the user interrupting mid-response. Verifies:
- a barge_in event fires
- the interrupt is recorded in the metrics session
- the second utterance is still correctly transcribed/answered as a fresh
  turn (not lost / not merged with the first)
"""

import asyncio
import io
import json
import sys

sys.path.insert(0, ".")

import numpy as np
import requests
import soundfile as sf
import websockets
from scipy.signal import resample_poly

from tts import text_to_speech
from vad import SAMPLE_RATE, CHUNK_SAMPLES

WS_URL = "ws://127.0.0.1:8000/ws"
API_URL = "http://127.0.0.1:8000"


def synth_pcm16_at_16k(text: str) -> bytes:
    audio_bytes = text_to_speech(text)
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if sr != SAMPLE_RATE:
        data = resample_poly(data, SAMPLE_RATE, sr).astype(np.float32)
    return (np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes()


async def stream_utterance(ws, text: str, trail_s: float = 1.0):
    silence_lead = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.int16).tobytes()
    speech = synth_pcm16_at_16k(text)
    silence_trail = np.zeros(int(trail_s * SAMPLE_RATE), dtype=np.int16).tobytes()
    stream = silence_lead + speech + silence_trail
    send_chunk_bytes = CHUNK_SAMPLES * 2 * 2
    for i in range(0, len(stream), send_chunk_bytes):
        await ws.send(stream[i:i + send_chunk_bytes])
        await asyncio.sleep(send_chunk_bytes / 2 / SAMPLE_RATE)


async def main():
    metrics_before = requests.get(f"{API_URL}/metrics").json()
    interrupts_before = metrics_before.get("interrupt_count", 0)
    print(f"interrupt_count before: {interrupts_before}")

    events = []
    audio_chunk_count = 0
    barge_in_seen = False
    done_event = asyncio.Event()

    async def collector(ws):
        nonlocal audio_chunk_count, barge_in_seen
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=30)
            if isinstance(msg, bytes):
                audio_chunk_count += 1
                print(f"  <- [audio chunk #{audio_chunk_count}]")
            else:
                data = json.loads(msg)
                events.append(data)
                print(f"  <- {data}")
                if data["type"] == "barge_in":
                    barge_in_seen = True
                if data["type"] == "done":
                    # Only the SECOND (interrupting) turn should ever reach
                    # "done" — the first is cancelled and signals that via
                    # "barge_in" instead, by design (see ws_session.py).
                    done_event.set()
                    return

    async with websockets.connect(WS_URL) as ws:
        collector_task = asyncio.create_task(collector(ws))

        print("Streaming utterance 1: 'Pick up the green block and put it on the right side.'")
        await stream_utterance(ws, "Pick up the green block and put it on the right side.")

        print("Waiting 2.5s (letting utterance 1 get into processing)...")
        await asyncio.sleep(2.5)

        print("Streaming utterance 2 (the interruption): 'Please stop right now.'")
        await stream_utterance(ws, "Please stop right now.")

        await asyncio.wait_for(done_event.wait(), timeout=30)
        collector_task.cancel()

    print(f"\nTotal audio chunks received: {audio_chunk_count}")
    print(f"barge_in event seen: {barge_in_seen}")
    assert barge_in_seen, "expected a barge_in event when utterance 2 interrupted utterance 1"

    transcripts = [e["content"] for e in events if e["type"] == "transcript"]
    print(f"Transcripts (should be 2, one per utterance): {transcripts}")
    assert len(transcripts) == 2, f"expected 2 transcripts, got {len(transcripts)}"
    assert "green" in transcripts[0].lower()
    assert "stop" in transcripts[1].lower(), f"expected 'stop' somewhere in the second transcript, got {transcripts[1]!r}"

    metrics_after = requests.get(f"{API_URL}/metrics").json()
    interrupts_after = metrics_after.get("interrupt_count", 0)
    print(f"interrupt_count after: {interrupts_after}")
    # At least +1 from the barge-in itself. It's legitimate for this to be +2
    # instead of +1 if the LLM's response to "please stop right now" ALSO
    # calls the stop_arm tool (which records its own interrupt) — both are
    # real "an interrupt happened" signals, just from different mechanisms.
    assert interrupts_after >= interrupts_before + 1, "expected at least one interrupt to be recorded"

    print("\nSUCCESS — barge-in cancelled the first turn and the second command was captured correctly.")


if __name__ == "__main__":
    asyncio.run(main())
