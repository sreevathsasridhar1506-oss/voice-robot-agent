"""
Standalone check (M12), part 2: no event-loop stalls under concurrent clients.

Client A sends a slow multi-tool-call command (several seconds of agent
processing). WHILE that's in flight, client B sends a quick text command and
we time how long client B waits for its reply. If process_audio_query()
weren't offloaded via asyncio.to_thread(), client B would be stuck behind
client A's entire turn — this proves it isn't.
"""

import asyncio
import io
import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly

from tts import text_to_speech
from vad import SAMPLE_RATE, CHUNK_SAMPLES

WS_URL = "ws://127.0.0.1:8000/ws"


def synth_pcm16_at_16k(text: str) -> bytes:
    audio_bytes = text_to_speech(text)
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if sr != SAMPLE_RATE:
        data = resample_poly(data, SAMPLE_RATE, sr).astype(np.float32)
    return (np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes()


async def stream_utterance(ws, text: str):
    silence_lead = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.int16).tobytes()
    speech = synth_pcm16_at_16k(text)
    silence_trail = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.int16).tobytes()
    stream = silence_lead + speech + silence_trail
    send_chunk_bytes = CHUNK_SAMPLES * 2 * 2
    for i in range(0, len(stream), send_chunk_bytes):
        await ws.send(stream[i:i + send_chunk_bytes])
        await asyncio.sleep(send_chunk_bytes / 2 / SAMPLE_RATE)


async def collect_until_done(ws, label, timeout=30):
    events = []
    while True:
        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
        if isinstance(msg, str):
            data = json.loads(msg)
            events.append((time.perf_counter(), data))
            if data.get("type") == "done":
                break
    return events


async def client_a():
    t0 = time.perf_counter()
    async with websockets.connect(WS_URL) as ws:
        await asyncio.gather(
            stream_utterance(ws, "Pick up the green block and put it on the right side."),
            collect_until_done(ws, "A"),
        )
    print(f"[A] slow voice command finished in {time.perf_counter() - t0:.2f}s")


async def client_b():
    # Give A a head start so its slow agent turn is actively in flight
    # (thinking / mid-motion) when B connects and asks its quick question.
    await asyncio.sleep(3.0)
    t0 = time.perf_counter()
    async with websockets.connect(WS_URL) as ws:
        await ws.send("What's on the table?")
        events = await collect_until_done(ws, "B")
    elapsed = time.perf_counter() - t0
    print(f"[B] quick text command finished in {elapsed:.2f}s (while A was still processing)")
    return elapsed


async def main():
    _, b_elapsed = await asyncio.gather(client_a(), client_b())

    print(f"\nClient B's response time while A was mid-turn: {b_elapsed:.2f}s")
    assert b_elapsed < 6.0, (
        "client B should get a timely response even while client A's slower "
        "command is being processed — a long wait here would mean "
        "process_audio_query() is blocking the event loop instead of running "
        "in a worker thread via asyncio.to_thread()"
    )
    print("\nSUCCESS — client B was not stalled behind client A's in-flight processing.")


if __name__ == "__main__":
    asyncio.run(main())
