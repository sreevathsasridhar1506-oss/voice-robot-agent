# Voice-Directed Collaborative Assembly Assistant

A voice-controlled simulated robot arm (Franka Panda, MuJoCo) driven by a LangGraph AI agent — built as an HRI-focused portfolio project. Speak a command, watch the arm pick up and place colored blocks, and get a spoken reply back — with real-time streaming, automatic speech detection (no push-to-talk), and the ability to interrupt the agent mid-sentence (barge-in).

See [`explanation.md`](explanation.md) for a detailed narrative of the design decisions and bugs found along the way (covers the simulation/agent core, M1–M7). This README covers setup and the full system (M1–M17).

## Architecture

```
Browser (React)                    Backend (FastAPI, /ws)
┌─────────────────┐                ┌──────────────────────────────────┐
│ mic → AudioWorklet│──PCM16 audio──▶│ ws_session.py                     │
│ (16kHz)          │                │  Silero VAD → utterance detection │
│                  │◀─status/audio──│  → stt.py (Whisper, GPU)          │
│ chat UI          │                │  → voice_agent.py (LangGraph/Groq)│
└─────────────────┘                │     → robot_tools.py               │
                                    │        → sim_controller.py         │
                                    │           → MuJoCo (Panda + blocks)│
                                    │  → tts.py (Piper) — streamed       │
                                    │     per sentence back to client    │
                                    └──────────────────────────────────┘
```

## Prerequisites

- Python 3.11, Node.js 18+
- (Optional but recommended) NVIDIA GPU for Whisper — falls back to CPU automatically if unavailable
- A free [Groq API key](https://console.groq.com)

## Setup

### 1. Python backend

```powershell
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

Download the required model assets (not committed to git — see `.gitignore`):

```powershell
# Piper TTS voice (~63MB)
.\venv\Scripts\python.exe -m piper.download_voices --download-dir piper_voices en_US-lessac-medium

# Silero VAD (~2.3MB) — fetched directly, not via the silero-vad pip package (avoids a PyTorch dependency)
curl -L -o vad_model/silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

# MuJoCo Menagerie's Franka Panda model (sparse-checkout, not the whole repo)
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git assets_mujoco_menagerie
cd assets_mujoco_menagerie && git sparse-checkout set franka_emika_panda && cd ..

# This project's pick-and-place scene (Panda + blocks + grasp constraints) — must sit next to panda.xml
copy scenes\pick_place_scene.xml assets_mujoco_menagerie\franka_emika_panda\
```

Copy `.env.example` to `.env` and add your `GROQ_API_KEY`.

**GPU Whisper on Windows:** if you hit a `cublas64_12.dll is not found` error, this is a known `ctranslate2` DLL-resolution quirk — `stt.py` already works around it, but it requires the `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` pip packages (in `requirements.txt`) to actually be installed. If GPU load fails anyway, it automatically falls back to CPU.

### 2. React frontend

```powershell
cd frontend
npm install
```

## Running it

**Backend:**
```powershell
.\venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

**Frontend:**
```powershell
cd frontend
npm run dev
```
Open `http://localhost:5173`, click **Start Mic**, and speak. A live view of the simulated arm opens in a separate OpenCV window on the machine running the backend.

**Or, without the frontend at all** — a terminal-only version of the same experience:
```powershell
.\venv\Scripts\python.exe scripts\cli_audio_agent.py
```

## Project structure

| File | Role |
|---|---|
| `sim_controller.py` | Thread-safe MuJoCo simulation API (`pick`, `place`, `move_to`, `stop`) |
| `robot_tools.py` | LangChain tools over the simulation, incl. disambiguation logic |
| `voice_agent.py` | LangGraph agent (Groq), batch and sentence-streamed query paths |
| `stt.py` / `tts.py` | Whisper (GPU) / Piper speech-to-text and text-to-speech |
| `vad.py` | Silero VAD — real-time speech start/stop detection |
| `metrics.py` | Per-session interaction metrics (turns, clarifications, interrupts) |
| `api.py` / `ws_session.py` | FastAPI backend, full-duplex `/ws` streaming endpoint |
| `frontend/` | React chat UI with live mic streaming and barge-in |
| `assets_mujoco_menagerie/franka_emika_panda/pick_place_scene.xml` | The scene (Panda + 5 colored blocks + grasp constraints) |
| `scripts/` | One standalone, independently-runnable script per component — see below |

## Testing

Every component has its own standalone verification script in `scripts/` — run any of them directly to sanity-check that piece in isolation (several need `uvicorn api:app` running first; the WS-based ones say so):

```powershell
.\venv\Scripts\python.exe scripts\test_mujoco_panda.py       # sim loads/steps
.\venv\Scripts\python.exe scripts\test_sim_controller.py     # pick/place + stop() thread-safety
.\venv\Scripts\python.exe scripts\test_robot_tools.py        # LLM selects correct tool/args
.\venv\Scripts\python.exe scripts\test_agent_e2e.py          # text command -> real sim motion
.\venv\Scripts\python.exe scripts\test_hri_behaviors.py      # disambiguation + live status
.\venv\Scripts\python.exe scripts\test_metrics.py            # session metrics logging
.\venv\Scripts\python.exe scripts\test_tts.py                # Piper synthesis + playback
.\venv\Scripts\python.exe scripts\test_stt_gpu.py             # GPU Whisper accuracy
.\venv\Scripts\python.exe scripts\test_voice_loop_e2e.py     # full audio-in -> audio-out
.\venv\Scripts\python.exe scripts\test_vad.py                 # VAD start/end detection
.\venv\Scripts\python.exe scripts\test_streaming_tts.py       # streaming vs. batch latency
.\venv\Scripts\python.exe scripts\test_ws_streaming.py        # needs server running
.\venv\Scripts\python.exe scripts\test_ws_concurrency.py      # needs server running
.\venv\Scripts\python.exe scripts\test_barge_in.py            # needs server running
```

Frontend:
```powershell
cd frontend
npm run lint
npm run build
node smoke_test_ws.mjs   # needs server running — protocol check from a JS WebSocket client
```

## Known limitations

- **Groq daily token quota**: heavy testing can exhaust `llama-3.3-70b-versatile`'s free-tier daily limit (100k tokens/day), returning `429` errors until it resets. Swap `GROQ_MODEL` in `.env` for a lighter model (e.g. `llama-3.1-8b-instant`) if you need to keep testing.
- **Single session at a time**: the simulation is one shared `SimController` instance — this is a single-user demo, not a multi-tenant deployment.
- **`asyncio.to_thread` cancellation is partial**: barge-in stops the arm and outgoing audio essentially instantly, but an already-in-flight Whisper/Piper/LLM call inside a worker thread can't be forcibly aborted mid-call — only its result is discarded.
