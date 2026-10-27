# Design Notes

This document explains how the system is built and why: the technology choices, what each file does, and the non-obvious bugs and design decisions that shaped the implementation. The [README](README.md) covers setup and results.

## 1. What this project is

A voice-controlled robot arm: a simulated Franka Panda, driven by a LangGraph AI agent, that picks up and places colored blocks on a table in response to spoken or typed natural-language commands. It grew out of an earlier voice-search agent I built (LangGraph tool-calling on Groq, a FastAPI backend, Whisper STT), with those same patterns pointed at robot control instead of web search.

I designed it around Human-Robot Interaction themes rather than as a pure demo:

- **Grounding**: resolving ambiguous references ("the red block" when there are two)
- **Transparency**: "what are you doing?" is answered from real live state, not a canned response
- **Mixed-initiative safety**: the person can interrupt the arm mid-motion and mid-sentence (barge-in)
- **Measurable interaction quality**: clarification-turn count, interrupt count and completion time are logged per session, and a set of evaluation scripts measures latency, disambiguation and task success

## 2. Technology stack

| Layer | Choice | Why |
|---|---|---|
| Physics simulation | **MuJoCo** (`mujoco` Python bindings) | I chose it over PyBullet because I had already used MuJoCo for a Stewart-platform project (PID/MPC/SMC/ADRC controllers and custom IK), so I could focus on the interaction problem rather than learning a new simulator. |
| Robot model | **MuJoCo Menagerie**'s `franka_emika_panda` | Official, high-quality MJCF model (Apache-2.0), sparse-cloned from `google-deepmind/mujoco_menagerie`. |
| Inverse kinematics | **`mink`** | A small, actively maintained differential-IK library built directly on the `mujoco` bindings. The Stewart platform's closed-form parallel-robot IK doesn't transfer to the Panda's serial chain, and `mink` keeps the effort on robot-control logic rather than IK math. |
| Visualization | **`mujoco.Renderer`**, shown in an OpenCV window or streamed to the browser | *Not* `mujoco.viewer.launch_passive()`, which had a "sync busy" reliability issue on my machine. |
| LLM | **Groq-hosted `llama-3.3-70b-versatile`** by default, or a **local model via Ollama** | Groq gives fast, free inference with native tool-calling. Ollama runs fully locally with no rate limits, which I needed for clean latency measurement (see §5). |
| Agent orchestration | **LangGraph** (`create_react_agent`) | A ReAct tool-calling loop. |
| Backend | **FastAPI** + WebSocket | Full-duplex audio streaming in both directions. |
| Frontend | **React** + Vite | Needed for real-time mic streaming via an AudioWorklet. |
| STT | **faster-whisper** `small`, GPU-accelerated | Better accuracy on spoken object and color names than CPU/`base`; falls back to CPU automatically. |
| TTS | **Piper** | Free, open-source and fully local. No network round-trip matters because barge-in has to cut audio off instantly. |
| VAD | **Silero VAD** via ONNX Runtime | Real-time speech start/stop detection for automatic utterance segmentation and barge-in. Run through ONNX to avoid pulling in PyTorch just for VAD. |
| Interaction logging | Plain **JSONL** | Deliberately simple: one record per session, enough for a few concrete numbers without a database. |

## 3. Architecture overview

```
                    ┌─────────────────────────────────────────┐
                    │   frontend/ (React)  ◀──/ws──▶  api.py   │
                    │   ws_session.py: VAD → STT → agent →     │
                    │   sentence-streamed TTS, barge-in        │
                    └───────────────────┬───────────────────────┘
                                        │ calls
                    ┌───────────────────▼───────────────────────┐
                    │           voice_agent.py                   │
                    │   LangGraph ReAct agent (Groq / Ollama)     │
                    │   retry loop for malformed tool calls,      │
                    │   batch + sentence-streaming entry points   │
                    └───────────────────┬───────────────────────┘
                                        │ binds
                    ┌───────────────────▼───────────────────────┐
                    │            robot_tools.py                  │
                    │  5 @tool functions: describe_scene,         │
                    │  move_to_object, pick_object, place_object,  │
                    │  stop_arm — disambiguation logic lives here  │
                    └───────────────────┬───────────────────────┘
                                        │ calls
                    ┌───────────────────▼───────────────────────┐
                    │           sim_controller.py                │
                    │  SimController — one dedicated worker       │
                    │  thread owns MuJoCo model/data + renderer;   │
                    │  get_scene_state/move_to/pick/place/stop     │
                    └───────────────────┬───────────────────────┘
                                        │ drives
                    ┌───────────────────▼───────────────────────┐
                    │     MuJoCo simulation (Panda + blocks)      │
                    │     scenes/pick_place_scene.xml             │
                    └─────────────────────────────────────────────┘
```

**Why this layering?** Each layer only knows about the one below it:
- The **agent** doesn't know MuJoCo exists; it only sees tool names and descriptions.
- **`robot_tools.py`** doesn't know how the simulation works internally; it only calls `SimController`'s public methods.
- **`sim_controller.py`** is the only file that touches MuJoCo and `mink` directly.

That let me build and test each layer independently, bottom-up (see the `scripts/` table below), rather than as one big integration.

## 4. File-by-file breakdown

### `scenes/pick_place_scene.xml`
An MJCF scene that `<include>`s Menagerie's official `panda.xml` (so it has to sit next to it; the setup step copies it there) and adds:
- A checkered floor plane
- **5 colored blocks** as free-jointed bodies: red, a *second* red (`block_red2`, deliberately, to create a genuine disambiguation case), blue, green and yellow. Sized and positioned following Menagerie's own `mjx_single_cube.xml` example.
- **5 weld equality constraints** (`grasp_block_<name>`), one per block, all `active="false"` by default. This is how grasping is simulated; see §5.

### `sim_controller.py`
The only file that talks to MuJoCo directly. Exposes a thread-safe `SimController` class:

- **One dedicated background thread** owns the MuJoCo `model`/`data` and the renderer. MuJoCo's Python bindings aren't safe to call from several threads at once, so every public method (`get_scene_state`, `move_to`, `pick`, `place`) puts a command on a `queue.Queue` and blocks on a `concurrent.futures.Future` for the result. The worker processes commands one at a time.
- **`stop()` deliberately bypasses that queue.** It sets a `threading.Event` that the worker's inner step loop (`_tick`) checks every physics step, so it interrupts a long-running motion *immediately* instead of waiting behind whatever is queued. Barge-in uses exactly this.
- **`get_current_action()` also bypasses the queue.** It's a plain string attribute read, safe from any thread. This is what lets "what are you doing?" reflect the *live* action instead of blocking until the motion finishes.
- **`_move_ee_to(gripper_target_pos, ...)`** runs the IK: builds a `mink.SE3` target, iterates `mink.solve_ik()` + `configuration.integrate_inplace()` until convergence, writes the solved joint angles into `data.ctrl[:7]` (position-actuator targets), then steps physics until the arm settles.
- **`pick()` / `place()`** are scripted sequences on top of `_move_ee_to` / `_set_gripper`: approach above → descend → close/open gripper → activate/deactivate the weld constraint → lift.

### `robot_tools.py`
LangChain `@tool` functions the agent can call, exported as `AGENT_TOOLS`.

- **`describe_scene()`** reports objects, colors, positions, what's held and the current action.
- **`move_to_object(color, which="")`** hovers over an object without picking it up (a pointing gesture).
- **`pick_object(color, which="")`** / **`place_object(zone)`** are the core actions. `place_object` takes a *named zone* ("left" / "right" / "center") rather than raw coordinates, because LLMs are unreliable at producing precise floats for spatial tasks.
- **`stop_arm()`** calls `sc.stop()` and records an interrupt in the session metrics.
- **Disambiguation logic** (`_matches_for_color`, `_side_hint`, `_resolve_with_disambiguation`): when a color matches more than one block, the tool returns a clarifying question describing each match as "left" / "right" instead of guessing. Both `move_to_object` and `pick_object` take an optional `which` argument, so the LLM can resolve inline when the phrasing already disambiguates ("the red block on the left").

### `voice_agent.py`
The agent itself.

- **`build_llm()` / `build_agent_graph()`** build a `ChatGroq` or `ChatOllama` instance (per `LLM_PROVIDER`) wrapped in a `create_react_agent` bound to `AGENT_TOOLS`.
- **`SYSTEM_PROMPT`** sets a robot-control persona: ask clarifying questions on ambiguous references (and how to use `which`), narrate actions taken, answer status questions from real state, and admit when a command doesn't make sense rather than pretending to comply.
- **`process_text_query()`** is the batch entry point. It includes a **retry-with-temperature-bump loop**: Groq's tool-calling models occasionally emit a malformed textual function call instead of a structured one (`400 tool_use_failed`), and it can get "stuck" reproducing the same broken output at a given temperature. Each retry rebuilds the agent at a higher temperature (0.3 → 0.5 → 0.7 → 0.9) instead of resending the identical request.
- **`stream_text_query_sentences()`** streams the reply sentence by sentence, so TTS can start speaking the first sentence while the LLM is still generating the rest.

### `stt.py`, `tts.py`, `vad.py`
Thin wrappers around faster-whisper, Piper and Silero VAD (ONNX). Each model is a lazily loaded singleton with double-checked locking, because sentence streaming and barge-in call them from several threads at once.

### `api.py`, `ws_session.py`
The FastAPI backend. The browser streams continuous raw PCM16 audio over `/ws`; `ws_session.py` runs it through VAD to find utterance boundaries (no push-to-talk), then runs STT → agent → sentence-streamed TTS. Processing runs as a background `asyncio.Task`, so the receive loop keeps reading audio. That's what makes barge-in possible: a new speech start while a turn is in progress cancels the turn, stops the arm and cuts outgoing audio.

### `metrics.py`
`SessionMetrics` dataclass: turn count, clarification-turn count, interrupt count and total duration. `.flush()` appends one JSON record per session to `session_metrics.jsonl`, and the backend exposes the live session at `/metrics`.

### `scripts/` — standalone, independently runnable checks
Every component was built *and proven* in isolation before being wired into the next layer. None of these need each other to run:

| Script | Proves |
|---|---|
| `test_mujoco_panda.py` | The Panda MJCF model loads and simulates stably (no LLM, no agent). |
| `test_pick_place_prototype.py` | Hard-coded pick-and-place works via raw `mink`/MuJoCo calls (the prototype before `sim_controller.py` existed). |
| `test_sim_controller.py` | `SimController`'s API works, **and** `stop()` genuinely interrupts a motion called from a different thread. |
| `test_robot_tools.py` | The LLM selects the correct tool and arguments for natural-language commands. |
| `test_agent_e2e.py` | A full text command → tool call → real simulation motion, end to end. |
| `test_hri_behaviors.py` | Disambiguation (asks when ambiguous, resolves inline when specified) and live status transparency (mid-motion, not stale). |
| `test_metrics.py` | Session logging produces correct counts for a scripted session. |
| `test_whisper_gpu.py` / `test_stt_gpu.py` | GPU Whisper loads and transcribes robot vocabulary accurately. |
| `test_tts.py` / `test_streaming_tts.py` | Piper synthesis, and streaming vs. batch time-to-first-audio. |
| `test_vad.py` / `live_vad_demo.py` | Silero VAD start/stop detection, on synthetic audio and a live mic. |
| `test_voice_loop_e2e.py` | Full batch voice loop: audio in → action → audio out. |
| `test_ws_streaming.py` / `test_ws_concurrency.py` / `test_barge_in.py` | The WebSocket pipeline, no event-loop stalls under concurrent clients, and barge-in. |
| `verify_grasp_visual.py` | Renders a snapshot after a pick, to confirm the block is held between the fingers (not floating; see §5). |
| `eval_*.py` | The quantitative evaluation reported in the README. |

## 5. Key design decisions and bugs found along the way

None of these were obvious going in.

**Weld constraints bake their relative pose at compile time, not activation time.** MuJoCo's weld equality constraint (used to simulate grasping, which is much simpler than friction-based grasp physics) stores its target relative pose in `model.eq_data`, computed from the poses the two bodies had *in the XML*. Naively activating a weld pulls the bodies toward that stale offset, not wherever they actually are. My fix, `_weld_at_current_pose()` in `sim_controller.py`, recomputes `eq_data` from the *live* gripper-to-object transform immediately before activating.

**The IK task targets the wrong point for grasping.** `mink`'s `FrameTask` targets the hand body's origin, but the fingertip pads (where a grasp actually happens) sit about 10.3 cm below it, which I measured via forward kinematics. Without compensating, "move to the block" put the *hand mesh* there and the fingers 10 cm off, so grasped blocks floated near the gripper. `_move_ee_to()` now always treats its argument as the gripper pinch point and adds `GRIPPER_PAD_OFFSET_Z` internally, so every caller gets correct behavior without knowing about the offset.

**A more fundamental bug that investigation exposed.** `mujoco.mj_resetData()` zeroes `data.ctrl`, but startup only set `data.qpos` to the home pose. Any physics step *before* the first `move_to()` (for example `pick()`'s opening `_set_gripper()` call) ran with the position actuators commanding `ctrl = 0`, silently dragging the arm away from home and corrupting the starting point for everything after. Fixed by initializing `data.ctrl[:7]` to the home keyframe alongside `data.qpos`.

**A halt left the arm drifting toward the target it was told to abandon.** Physics only steps inside commands, so `stop()` freezes the arm mid-motion, but the position actuators still held the interrupted motion's final IK solution. As soon as the next command stepped physics, the arm resumed moving toward the old target. I measured this with `eval_halt_state.py` and fixed it by re-targeting the actuators to the current pose on stop. Mean post-halt drift fell from 39.5 mm to 8.3 mm over 50 random halts.

**Groq's tool-calling can fail with a malformed function call.** Occasionally the model emits `<function=name>{...}</function>` as plain text instead of a structured tool call, which Groq's API rejects. Retrying at a higher temperature reliably breaks out of it; resending the identical request can repeat the same failure.

**Hosted-API rate limits contaminated latency measurements.** Under Groq's free-tier per-minute limit, the SDK silently retries with backoff, which inflated measured LLM latency by up to ten times. I added `LLM_PROVIDER=ollama` so the evaluation could run a local model with no rate limit. It's also a legitimate deployment option: no per-request cost and no dependency on a third-party API.

**Measurement methodology mattered as much as the system.** My first barge-in latency harness timed from the start of the synthesized audio clip, which silently included Piper's variable leading silence (17–244 ms), and it sent each audio chunk at the *start* of its real-time interval, faster than a real microphone could. The current harness times from actual speech onset and paces chunks on an absolute clock so they can only arrive late, never early. That makes the reported numbers conservative.

**GPU Whisper needs a Windows-specific DLL path fix.** `faster-whisper`'s `ctranslate2` backend needs cuBLAS/cuDNN from the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` pip packages, but `os.add_dll_directory()` doesn't reliably work for its nested DLL dependencies. The working fix (`_add_nvidia_dll_dirs()` in `stt.py`) prepends those packages' `bin/` directories to `PATH` before the import.

**The renderer's GL context is thread-affine.** It belongs to `SimController`'s worker thread, so any snapshot or render has to go through the same command queue as everything else. `SimController.save_snapshot()` exists for this; calling `sc.renderer.render()` from another thread fails with a GLFW "context in use" error.

## 6. Limitations

- **No conversation memory across turns.** Each `process_text_query()` call is a fresh turn. When the agent asks "which red block?", the person's answer has to restate the object ("the red one on the left"). The disambiguation evaluation therefore only tests same-utterance qualification.
- **The agent sometimes invents ambiguity.** In the evaluation it asked an unnecessary clarifying question for 28% of unambiguous commands.
- **Single session at a time.** There is one shared `SimController`; this is a single-user system, not a multi-tenant deployment.
- **Simulated grasping.** Grasps are weld constraints, not friction-based contact, so the system says nothing about grasp robustness on real hardware.
- **Partial cancellation.** Barge-in stops the arm and outgoing audio essentially instantly, but a Whisper, Piper or LLM call already running in a worker thread can't be aborted mid-call; only its result is discarded.
