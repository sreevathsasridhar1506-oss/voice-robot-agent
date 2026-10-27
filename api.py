"""
api.py
======
FastAPI backend.

Full-duplex /ws endpoint: the client streams continuous raw PCM16 audio (or
text commands), Silero VAD auto-segments utterances (no manual "click to
record"), and status/transcript/answer/audio stream back as they're ready.
Replaces the learning repo's one-shot request/response /query/audio pattern.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ws_session import WSSession

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Robot Voice Agent API...")
    try:
        from voice_agent import build_llm, LLM_PROVIDER
        build_llm()
        logger.info("LLM reachable (provider=%s).", LLM_PROVIDER)
    except Exception as exc:
        logger.warning("LLM connectivity check failed: %s", exc)
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Voice-Directed Collaborative Assembly Assistant", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/state")
async def get_state():
    """Current scene state — handy for tests/debugging and the React frontend."""
    from robot_tools import get_sim_controller
    import asyncio as _asyncio
    sc = get_sim_controller()
    return await _asyncio.to_thread(sc.get_scene_state)


@app.get("/metrics")
async def get_metrics():
    """Current session's interaction metrics — handy for tests/debugging."""
    from metrics import get_session
    return get_session().as_dict()


MJPEG_BOUNDARY = b"frame"


@app.get("/sim/stream")
async def sim_stream():
    """
    Live MJPEG feed of the simulation for the React frontend — an <img
    src="/sim/stream"> tag renders multipart/x-mixed-replace natively with no
    client-side JS needed. Reads sim_controller's non-queued
    get_current_frame_jpeg() (see its docstring) so the stream keeps updating
    throughout an in-progress move_to()/pick()/place() instead of freezing
    for the whole duration of every command.
    """
    from robot_tools import get_sim_controller

    async def frame_generator():
        sc = get_sim_controller()
        while True:
            frame = sc.get_current_frame_jpeg()
            if frame is not None:
                yield (
                    b"--" + MJPEG_BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                    frame + b"\r\n"
                )
            await asyncio.sleep(1 / 20)  # ~20fps — plenty for a status-style view, keeps bandwidth low

    return StreamingResponse(
        frame_generator(),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY.decode()}",
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = WSSession(websocket)
    logger.info("WS connected: %s", websocket.client)
    try:
        while True:
            message = await websocket.receive()
            # Starlette's low-level receive() returns a raw ASGI message dict
            # with type "websocket.disconnect" rather than raising — it only
            # raises WebSocketDisconnect from the receive_bytes()/receive_text()
            # convenience wrappers. Not checking this explicitly means calling
            # receive() again after a disconnect, which raises a RuntimeError
            # instead of exiting cleanly.
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await session.handle_audio_bytes(message["bytes"])
            elif message.get("text") is not None:
                await session.handle_text_command(message["text"])
    except WebSocketDisconnect:
        pass
    finally:
        logger.info("WS disconnected: %s", websocket.client)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
        log_level="info",
    )
