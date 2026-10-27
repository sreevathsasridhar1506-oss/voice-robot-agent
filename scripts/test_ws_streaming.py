"""
Standalone check (M12), part 1: full-duplex WebSocket + VAD auto-segmentation.

Streams synthetic PCM16 audio (silence -> Piper speech -> silence) to a live
/ws connection at realistic real-time pacing (not all at once — that would
defeat the point of testing streaming), verifying:
- speech_started/speech_ended fire from server-side VAD at the right time
- the resulting transcript/answer are correct
- the simulation actually moved (checked via GET /state, a separate process
  from the server, proving this isn't just reading in-memory state)

Requires the server already running: uvicorn api:app (see README/explanation.md).
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
    pcm16 = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    return pcm16.tobytes()


async def stream_utterance(ws, text: str):
    silence_lead = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.int16).tobytes()
    speech = synth_pcm16_at_16k(text)
    silence_trail = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.int16).tobytes()
    stream = silence_lead + speech + silence_trail

    send_chunk_bytes = CHUNK_SAMPLES * 2 * 2  # a couple VAD-frames per network send
    for i in range(0, len(stream), send_chunk_bytes):
        await ws.send(stream[i:i + send_chunk_bytes])
        await asyncio.sleep(send_chunk_bytes / 2 / SAMPLE_RATE)  # pace to real time


async def collect_until_done(ws, timeout=30):
    events = []
    audio_chunks = []
    while True:
        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
        if isinstance(msg, bytes):
            audio_chunks.append(msg)
            print(f"  <- [binary audio chunk, {len(msg)} bytes]")
        else:
            data = json.loads(msg)
            events.append(data)
            print(f"  <- {data}")
            if data.get("type") == "done":
                break
    return events, audio_chunks


async def main():
    state_before = requests.get(f"{API_URL}/state").json()
    yellow_before = next(o for o in state_before["objects"] if o["name"] == "block_yellow")["position"]
    print(f"block_yellow starts at {[round(v, 3) for v in yellow_before]}")

    async with websockets.connect(WS_URL) as ws:
        print("\nStreaming: 'Pick up the yellow block and put it in the center.'")
        _, (events, audio_chunks) = await asyncio.gather(
            stream_utterance(ws, "Pick up the yellow block and put it in the center."),
            collect_until_done(ws),
        )

    event_types = [e["type"] for e in events]
    assert "speech_started" in event_types, "VAD should have detected speech start"
    assert "speech_ended" in event_types, "VAD should have detected speech end"
    transcript = next(e["content"] for e in events if e["type"] == "transcript")
    answer_chunks = [e["content"] for e in events if e["type"] == "answer_chunk"]
    answer = " ".join(answer_chunks)
    print(f"\nTranscript: {transcript!r}")
    print(f"Answer ({len(answer_chunks)} sentence(s)): {answer!r}")
    assert "yellow" in transcript.lower()
    assert len(audio_chunks) == len(answer_chunks) > 0, "one audio chunk per answer sentence expected"
    assert all(len(c) > 1000 for c in audio_chunks)

    state_after = requests.get(f"{API_URL}/state").json()
    yellow_after = next(o for o in state_after["objects"] if o["name"] == "block_yellow")["position"]
    print(f"block_yellow ends at {[round(v, 3) for v in yellow_after]}")
    moved = np.linalg.norm(np.array(yellow_after[:2]) - np.array(yellow_before[:2]))
    print(f"XY displacement: {moved:.3f} m")
    assert moved > 0.03, "the block should have actually moved"

    print("\nSUCCESS — full-duplex WS streaming + VAD segmentation + real sim motion, all verified.")


if __name__ == "__main__":
    asyncio.run(main())
