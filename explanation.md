# Voice-Directed Collaborative Assembly Assistant — Explanation

This document explains what's been built so far, why it's built this way, and how the pieces fit together. It covers the technology stack, every code file, the logic behind each one, and the non-obvious bugs/design decisions that shaped the implementation.

## 1. What this project is

A voice-controlled robot arm — a simulated Franka Panda, driven by a LangGraph AI agent — that picks up and places colored blocks on a table in response to spoken/typed natural-language commands. It's being built as a portfolio project for a Masters application (Human-Robot Interaction focus), evolving from an earlier "voice search agent" learning project (`voice_agent` folder) that established the core patterns — LangGraph tool-calling on Groq, FastAPI backend, Whisper STT — now redirected at robot control instead of web search.

The project is deliberately designed around HRI research themes rather than just being a demo:

- **Grounding** — resolving ambiguous references ("the red block" when there are two)
- **Transparency** — "what are you doing?" answered from real live state, not a canned response
- **Mixed-initiative safety** — the arm can be interrupted mid-motion (barge-in, planned)
- **Measurable interaction quality** — clarification-turn count, interrupt count, completion time, logged per session

## 2. Technology stack

| Layer | Choice | Why |
|---|---|---|
| Physics simulation | **MuJoCo** (`mujoco` Python bindings) | Chosen over PyBullet specifically because the user already has real MuJoCo experience (a Stewart platform project with PID/MPC/SMC/ADRC controllers, plus custom IK work) — building on a known tool is faster and reads as deeper expertise for the application. |
| Robot model | **MuJoCo Menagerie**'s `franka_emika_panda` | Official, high-quality MJCF model (Apache-2.0), sparse-cloned from `google-deepmind/mujoco_menagerie` into `assets_mujoco_menagerie/`. |
| Inverse kinematics | **`mink`** | A small, actively-maintained differential IK library built directly on the `mujoco` bindings. Chosen over hand-rolling a Jacobian solver — the Stewart platform's closed-form parallel-robot IK doesn't transfer to the Panda's serial kinematic chain, and `mink` keeps the project's effort on the robot-control logic rather than IK math. |
| Visualization | **`mujoco.Renderer` + OpenCV** (`cv2.imshow`) | *Not* `mujoco.viewer.launch_passive()` — that has a known "sync busy" reliability issue on this machine. Mirrors the pattern already proven in the user's own `mujoco_stewart/run_stewart.py`. |
| LLM | **Groq-hosted `llama-3.3-70b-versatile`** via `langchain-groq` | Reused from the learning project; free/fast inference, native tool-calling. |
| Agent orchestration | **LangGraph** (`create_react_agent`) | Reused pattern from the learning project — a ReAct tool-calling loop. |
| Backend (planned, M12+) | **FastAPI** | Reused pattern from the learning project. |
| Frontend (planned, M15+) | **React** | Replaces the learning project's Streamlit UI — needed for real-time mic streaming, which Streamlit can't do. |
| STT (planned, M9) | **faster-whisper**, GPU-accelerated | Reused from the learning project, upgraded from CPU/`base` to GPU/`small` (this machine has an idle RTX 3060) for better accuracy on spoken object/color names. |
| TTS (planned, M8) | **Piper** | Replaces the learning project's gTTS — free, fully open-source, runs locally (no network round-trip, which matters once barge-in needs to cut off audio instantly). |
| VAD (planned, M11) | **Silero VAD** via ONNX Runtime | Real-time speech start/stop detection for automatic utterance segmentation and barge-in — chosen via ONNX specifically to avoid pulling in a full PyTorch dependency. |
| Interaction logging | Plain **JSONL** | Deliberately not a production analytics system — just enough structured data for an application write-up. |

## 3. Architecture overview

```
                    ┌─────────────────────────────────────────┐
                    │           voice_agent.py                 │
                    │   LangGraph ReAct agent (Groq LLM)        │
                    │   process_text_query() — retry loop for   │
                    │   malformed tool-calls, records metrics   │
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
                    │     assets_mujoco_menagerie/.../             │
                    │     pick_place_scene.xml                    │
                    └─────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │              metrics.py                   │
                    │  SessionMetrics — turn/clarification/       │
                    │  interrupt counts, written as JSONL          │
                    │  (called from voice_agent.py + robot_tools)  │
                    └─────────────────────────────────────────────┘
```

**Why this layering?** Each layer only knows about the one below it, mirroring the learning project's structure (`mcp_config.py` → `voice_agent.py` → `api.py`):
- The **agent** doesn't know MuJoCo exists — it only sees tool names/descriptions.
- **`robot_tools.py`** doesn't know how the simulation works internally — it only calls `SimController`'s public methods.
- **`sim_controller.py`** is the only file that touches MuJoCo/`mink` directly.

This means each layer was built and tested independently, bottom-up (see §5), rather than as one big integration.

## 4. File-by-file breakdown

### `assets_mujoco_menagerie/franka_emika_panda/pick_place_scene.xml`
A new MJCF scene file (sibling to Menagerie's own `scene.xml`, which is left untouched) that `<include>`s the official `panda.xml` and adds:
- A checkered floor plane (from the original `scene.xml`)
- **5 colored blocks** as free-jointed bodies: red, a *second* red (`block_red2` — deliberately, to create a genuine disambiguation test case), blue, green, yellow. Sized and positioned following the convention in Menagerie's own `mjx_single_cube.xml` example.
- **5 weld equality constraints** (`grasp_block_<name>`), one per block, all `active="false"` by default. These are how "grasping" is simulated — see §5.

### `sim_controller.py`
The only file that talks to MuJoCo directly. Exposes a thread-safe `SimController` class:

- **One dedicated background thread** owns the MuJoCo `model`/`data` and the renderer/display loop, spawned in `__init__`. MuJoCo's Python bindings aren't safe to call from multiple threads at once, so every public method (`get_scene_state`, `move_to`, `pick`, `place`) puts a command on a `queue.Queue` and blocks on a `concurrent.futures.Future` for the result — the worker thread processes commands one at a time.
- **`stop()` deliberately bypasses that queue.** It just sets a `threading.Event` that the worker's inner step-loop (`_tick`) checks every physics step, so it can interrupt a long-running motion *immediately* instead of waiting behind whatever's currently queued. This is the same mechanism planned for barge-in (M14).
- **`get_current_action()` also bypasses the queue** (added during M6) — it's a plain string attribute read, safe from any thread under Python's GIL. This is what lets "what are you doing?" reflect the *live* action instead of blocking until an in-progress motion finishes (which `get_scene_state()` would do, since it needs to run on the worker thread to safely read live MuJoCo data).
- **`_move_ee_to(gripper_target_pos, ...)`** runs the actual IK: builds an `mink.SE3` target, iterates `mink.solve_ik()` + `configuration.integrate_inplace()` until convergence, then writes the solved joint angles into `data.ctrl[:7]` (position-actuator targets) and steps physics (`_tick`) until the arm physically settles there.
- **`pick()`/`place()`** are scripted sequences built on top of `_move_ee_to`/`_set_gripper`: approach above → descend → close/open gripper → activate/deactivate the relevant weld constraint → lift.

### `robot_tools.py`
LangChain `@tool`-decorated functions the agent can call, mirroring the learning project's `mcp_config.py` pattern exactly (a list called `AGENT_TOOLS`).

- **`get_sim_controller()`** — lazy singleton, same pattern as the learning project's `get_whisper_model()`.
- **`describe_scene()`** — reports objects, colors, positions, what's held, and current action.
- **`move_to_object(color, which="")`** — hovers over an object without picking it up (a "pointing" gesture).
- **`pick_object(color, which="")`** / **`place_object(zone)`** — the core actions. `place_object` takes a *named zone* ("left"/"right"/"center") rather than raw coordinates, because LLMs are unreliable at producing precise floats for spatial tasks.
- **`stop_arm()`** — calls `sc.stop()` and records an interrupt in the metrics session.
- **Disambiguation logic** (`_matches_for_color`, `_side_hint`, `_resolve_with_disambiguation`): when a color matches more than one block, instead of guessing, the tool returns a clarifying question describing each match as "left"/"right" (reusing the same vocabulary as `place_object`'s zones). Both `move_to_object` and `pick_object` accept an optional `which` argument so the LLM can resolve inline if the user's phrasing already disambiguates ("the red block on the left").

### `voice_agent.py`
The agent itself — mirrors the learning project's `voice_agent.py` structure closely.

- **`build_llm()` / `build_agent_graph()`** — builds a `ChatGroq` instance and wraps it in a `create_react_agent` bound to `AGENT_TOOLS`.
- **`SYSTEM_PROMPT`** — a robot-control persona: ask clarifying questions on ambiguous references (and how to use the `which` argument), narrate actions taken, answer status questions from real state, admit when a command doesn't make sense rather than pretending to comply.
- **`process_text_query()`** — the entry point. Injects the current date, invokes the agent, and extracts the final answer. Carries over the learning project's **retry-with-temperature-bump loop**: Groq's tool-calling models occasionally emit a malformed textual function call instead of a structured tool call (`400 tool_use_failed`) — this isn't always random noise (it can get "stuck" reproducing the same broken output at a given temperature), so each retry rebuilds the agent at a higher temperature (0.3→0.5→0.7→0.9) instead of just resending the identical request. Also calls `get_session().record_turn(...)` to log the interaction.

### `metrics.py`
`SessionMetrics` dataclass — turn count, clarification-turn count (heuristic: does the answer contain "?"), interrupt count, total duration. `.flush()` appends one JSON record per session to `session_metrics.jsonl`. Deliberately simple (no database, no server) — this exists to produce a few concrete numbers for the application write-up, not to be a real analytics pipeline.

### `scripts/` — standalone, independently-runnable verification scripts
Every component above was built *and proven* in isolation before being wired into the next layer, matching a pattern already established in the learning project (`test_groq_tools.py`). None of these need each other to run:

| Script | Proves |
|---|---|
| `test_mujoco_panda.py` | The Panda MJCF model loads and simulates stably (no LLM, no agent). |
| `test_pick_place_prototype.py` | Hardcoded pick-and-place works via raw `mink`/MuJoCo calls (pre-`sim_controller.py`; kept as a historical reference of the M2 prototyping step). |
| `test_sim_controller.py` | `SimController`'s API works, **and** `stop()` genuinely interrupts a motion called from a different thread. |
| `test_robot_tools.py` | The LLM selects the correct tool + arguments for natural-language commands (binding a tool doesn't execute it — this test never touches the simulation). |
| `test_agent_e2e.py` | A full text command → tool call → real simulation motion, end to end. |
| `test_hri_behaviors.py` | Disambiguation (asks when ambiguous, resolves inline when specified) and live status transparency (mid-motion, not stale). |
| `test_metrics.py` | Session logging produces correct counts for a scripted mixed session. |
| `test_whisper_gpu.py` | GPU-accelerated Whisper loads and runs (includes the DLL-path fix — see §5). |
| `verify_grasp_visual.py` | Renders a snapshot after a pick, for visually confirming the block is actually held between the fingers (not floating — see §5). |
| `cli_text_agent.py` | Interactive REPL — the actual way to try the system by hand. |

## 5. Key design decisions & bugs found along the way

These are worth understanding if this code gets extended — none of them were obvious going in.

**Weld constraints bake their relative pose at compile time, not activation time.** MuJoCo's weld equality constraint (used to simulate "grasping" — much simpler than friction-based grasp physics) stores its target relative pose in `model.eq_data`, computed from whatever pose the two bodies had *in the XML*. Naively activating a weld pulls the two bodies toward that stale compile-time offset, not wherever they actually are when you activate it. The fix — `_weld_at_current_pose()` in `sim_controller.py` — recomputes `eq_data`'s relative position/orientation from the *live* transform between the gripper and object immediately before activating.

**The IK task targets the wrong point for grasping.** `mink`'s `FrameTask` targets the "hand" body's origin — but the actual fingertip pads (where a grasp really happens) sit about 10.3cm below that origin (measured empirically via forward kinematics). Without compensating, commanding "move to the block" moved the *hand mesh* there while the fingers ended up 10cm off, so grasped blocks visually floated near the gripper instead of sitting between the fingers. Fixed by having `_move_ee_to()` always treat its argument as "where the gripper pinch point should be" and internally add `GRIPPER_PAD_OFFSET_Z` to compute the real hand-frame IK target — every caller (approach, grasp, place, hover) gets correct behavior without needing to know about the offset.

**A more fundamental bug the above investigation exposed:** `mujoco.mj_resetData()` zeroes `data.ctrl`, but startup only ever set `data.qpos` to the home pose — never `data.ctrl` to match. Any physics-stepping call *before* the first `move_to()` (specifically `pick()`'s opening `_set_gripper()` call) would step the simulation with the position actuators still commanding `ctrl=0`, silently dragging the arm away from home before the very first real IK move ever ran — corrupting the starting point for everything downstream. Fixed by initializing `data.ctrl[:7]` to match the home keyframe alongside `data.qpos` at startup.

**Groq's tool-calling can fail with a malformed function call.** Occasionally the model emits `<function=name>{...}</function>` as plain text instead of a structured tool call, which Groq's API rejects (`400 tool_use_failed`). Retrying at a higher temperature reliably breaks out of it (see `process_text_query()`'s retry loop above) — simply resending the identical request can repeat the same failure, since the model can get "stuck" at a given temperature.

**GPU Whisper needs a Windows-specific DLL path fix.** `faster-whisper`'s `ctranslate2` backend needs cuBLAS/cuDNN, which the `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` pip packages provide — but `os.add_dll_directory()` (the "normal" fix) doesn't reliably work for `ctranslate2`'s nested DLL dependencies on Windows. The working fix (in `test_whisper_gpu.py`'s `_add_nvidia_dll_dirs()`) prepends those packages' `bin/` directories directly onto `PATH` at import time.

**`mujoco.viewer.launch_passive()` was avoided entirely.** It has a known "sync busy" reliability issue on this machine. Visualization instead uses `mujoco.Renderer` (offscreen render) + `cv2.imshow`, matching a pattern already proven in the user's own prior MuJoCo project (`mujoco_stewart/run_stewart.py`). One consequence: the renderer's GL context is thread-affine (owned by `SimController`'s worker thread), so any snapshot/render call must be dispatched through the same command queue as everything else — `SimController.save_snapshot()` exists specifically for this; calling `sc.renderer.render()` directly from another thread fails with a GLFW "context in use" error.

## 6. Status

**Done (M1–M7)** — the simulation, agent, and HRI-behavior core, verified independently at each step:
1. Project scaffold + GPU Whisper + MuJoCo model verified
2. Raw MuJoCo pick-and-place prototype (`mink` IK + weld grasping)
3. `sim_controller.py` — thread-safe API, `stop()` proven to interrupt mid-motion
4. `robot_tools.py` — LLM correctly selects tools/arguments
5. `voice_agent.py` — full text→action pipeline works end to end
6. Disambiguation + live transparency (plus the two bugs above, found and fixed)
7. Interaction-metrics logging

**Not yet built (M8–M17):**
- M8–M10: Piper TTS, GPU Whisper wrapper, full batch voice loop (record → transcribe → agent → speak)
- M11–M14: Silero VAD, full-duplex WebSocket streaming, sentence-chunked speech synthesis, barge-in
- M15–M16: React frontend (text chat, then live audio streaming)
- M17: Demo prep, cleanup, README

Full milestone-by-milestone detail (including verification methods and sizing) lives in the plan document at `C:\Users\LENOVO\.claude\plans\yeah-we-need-a-soft-grove.md`.

## 7. How to run it

```powershell
cd "c:\Users\LENOVO\OneDrive\Desktop\voice_robot_agent"
.\venv\Scripts\python.exe scripts\cli_text_agent.py
```

Opens a live OpenCV window showing the simulated arm; type commands at the `>` prompt (e.g. `what's on the table?`, `pick up the red block`, `put it on the left`, `stop`). Requires `GROQ_API_KEY` in `.env` (already configured, reused from the learning project).

Any individual component can also be re-verified on its own via the scripts in `scripts/` (see the table in §4) — e.g. `.\venv\Scripts\python.exe scripts\test_hri_behaviors.py`.
