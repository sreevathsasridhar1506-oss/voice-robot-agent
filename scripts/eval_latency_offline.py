"""
INDICON paper evaluation — pipeline latency (offline components).

Measures, without needing the WS server running:
  1. VAD utterance-detection latency (audio-domain delay + per-chunk compute time)
  2. STT latency: GPU vs CPU Whisper, directly instantiated (bypasses the
     app's device-fixed-at-import singleton so both can be compared in one run)
  3. LLM response latency: streaming (time-to-first-sentence) vs batch (total)
  4. TTS latency: sentence-streamed (time-to-first-chunk) vs batch (total)

Barge-in latency is NOT here — that's a full-system, real-time measurement
and lives in eval_barge_in_latency.py (needs the WS server running).

Writes results/latency_offline.json (raw per-trial numbers, for the paper's
LaTeX tables) plus a human-readable summary to stdout.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

RESULTS_DIR = Path("../icaaic_paper/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ICAAIC revision (2026-10-20): every section 5-8 -> 30 trials, each on a
# DISTINCT input (the lists below), not the old 8 repeated -- the old code
# sliced fixed 8-item lists, so raising N alone would have changed nothing.
# The first 8 items of each list are the INDICON ones, unchanged. The
# INDICON run is kept as results/latency_offline_indicon.json.
N_VAD_TRIALS = 30
N_STT_TRIALS = 30
N_LLM_TRIALS = 30
N_TTS_TRIALS = 30
N_STOP_TRIALS = 30

TEST_SENTENCES_FOR_STT = [
    "What's on the table right now?",
    "Pick up the red block on the left and place it in the center.",
    "Stop right there, please.",
    "Move to hover above the blue block without picking it up.",
    "Can you tell me what you're currently doing?",
    "Put the block down on the right side of the table.",
    "I'd like you to pick up the yellow block now.",
    "Why did you stop moving just now?",
    "Grab the green block and move it to the left.",
    "Are you holding anything at the moment?",
    "Wait, hold on a second.",
    "Place the blue block in the center of the table.",
    "Which blocks can you see from where you are?",
    "Please pick up the red block on the right.",
    "Move the yellow block over to the left side.",
    "That's the wrong one, put it back.",
    "How many red blocks are on the table?",
    "Lift the green block a little higher.",
    "Go back to where you started.",
    "Can you put that down on the right?",
    "Pick up whichever block is closest to you.",
    "Don't move yet, I'm still thinking.",
    "Take the blue block and set it on the left.",
    "What color is the block you're holding?",
    "Hover over the green block for a moment.",
    "Stop, that's not what I meant.",
    "Put everything back the way it was.",
    "Is the yellow block on the left or the right?",
    "Please pick up the red block next to the blue one.",
    "Thank you, that's all for now.",
]

TEST_QUERIES_FOR_LLM = [
    "What's on the table?",
    "Pick up the blue block.",
    "What are you doing right now?",
    "Put it on the right side.",
    "Pick up the green block and place it in the center.",
    "Stop.",
    "Are you holding anything?",
    "Move to hover above the yellow block.",
    "How many blocks are there?",
    "Pick up the yellow block.",
    "Place it on the left.",
    "Which blocks are on the right side of the table?",
    "Hover above the green block.",
    "Is anything in your gripper?",
    "Put the block you're holding in the center.",
    "Grab the green block.",
    "Set it down on the right.",
    "What color blocks can you see?",
    "Pick up the blue block and put it on the left.",
    "Stop moving.",
    "Describe the scene for me.",
    "Pick up the yellow block and place it on the right.",
    "What did you just do?",
    "Move above the blue block.",
    "Take the green block to the left side.",
    "Are there any red blocks?",
    "Put it down in the center.",
    "Pick up the blue block, please.",
    "Where is the yellow block?",
    "Place the block on the right side.",
]

TEST_SENTENCES_FOR_TTS = [
    "I picked up the blue block and placed it on the right side of the table.",
    "There are two red blocks on the table, one on the left and one on the right. Which one do you mean?",
    "I'm currently idle and not holding anything.",
    "I moved to hover above the yellow block.",
    "Stopped.",
    "There are five blocks on the table: two red, one blue, one green, and one yellow.",
    "I picked up the red block that was on the left side of the table.",
    "I'm not currently holding any objects, and the arm is idle.",
    "I've placed the green block in the center.",
    "I'm holding the yellow block right now.",
    "Okay, I've stopped the arm.",
    "The blue block is on the right side of the table.",
    "I couldn't find a purple block on the table. Could you pick another color?",
    "I'm moving to the green block now.",
    "Done. The red block is on the left.",
    "I can't place anything because I'm not holding a block.",
    "I stopped because you started speaking. What would you like me to do?",
    "Sure, picking up the yellow block.",
    "The green block is closest to the left edge of the table.",
    "I've put the blue block down in the center of the table.",
    "There is one green block, and it's on the left side.",
    "I'm already holding the red block. Should I place it first?",
    "Got it.",
    "I moved the yellow block to the right side, as you asked.",
    "Both red blocks are still on the table.",
    "I'm hovering above the blue block and haven't picked it up.",
    "Which side would you like me to place it on: left, right, or center?",
    "I've released the green block on the right.",
    "Nothing is in my gripper at the moment.",
    "Okay. I'll wait for your next instruction.",
]

# Section 5 (new): how long the LLM takes to DECIDE to stop, for stop-type
# utterances -- the term a semantic-first halt would wait on. Replaces the
# paper's single "Stop." trial. Deliberately includes indirect phrasings
# ("Wait.", "Hold on.", "Cancel that.") since a semantic-first halt must
# recognize those too; whether stop_arm is called at all is recorded.
STOP_UTTERANCES = [
    "Stop.",
    "Stop now.",
    "Stop moving.",
    "Stop the arm.",
    "Stop right there.",
    "Stop, please.",
    "Wait, stop.",
    "Hold on, stop.",
    "Halt.",
    "Freeze.",
    "Don't move.",
    "Stop that.",
    "Hey, stop!",
    "Stop immediately.",
    "Please stop moving.",
    "Stop what you're doing.",
    "Wait.",
    "Hold on.",
    "Hold it.",
    "Pause.",
    "Stop, stop, stop.",
    "No, stop.",
    "Cancel that.",
    "Stop the motion.",
    "Stop right now.",
    "Whoa, stop.",
    "Wait a second, stop.",
    "Stop there.",
    "Quit moving.",
    "Stop the robot.",
]

for _name, _lst, _n in [("STT", TEST_SENTENCES_FOR_STT, N_STT_TRIALS),
                        ("LLM", TEST_QUERIES_FOR_LLM, N_LLM_TRIALS),
                        ("TTS", TEST_SENTENCES_FOR_TTS, N_TTS_TRIALS),
                        ("STOP", STOP_UTTERANCES, N_STOP_TRIALS)]:
    assert len(_lst) >= _n and len(set(_lst)) == len(_lst), f"{_name}: need {_n} distinct inputs"
assert N_VAD_TRIALS <= len(TEST_SENTENCES_FOR_STT)


def summarize(name, values_ms):
    values_ms = list(values_ms)
    return {
        "name": name,
        "n": len(values_ms),
        "mean_ms": round(statistics.mean(values_ms), 1),
        "median_ms": round(statistics.median(values_ms), 1),
        "stdev_ms": round(statistics.stdev(values_ms), 1) if len(values_ms) > 1 else 0.0,
        "min_ms": round(min(values_ms), 1),
        "max_ms": round(max(values_ms), 1),
        "raw_ms": [round(v, 1) for v in values_ms],
    }


def print_summary(s):
    print(f"  {s['name']:<40s} mean={s['mean_ms']:>8.1f}ms  std={s['stdev_ms']:>7.1f}ms  "
          f"min={s['min_ms']:>8.1f}ms  max={s['max_ms']:>8.1f}ms  n={s['n']}")


# ── 1. VAD detection latency ────────────────────────────────────────────────

def eval_vad():
    print("\n=== 1. VAD utterance-detection latency ===")
    from vad import CHUNK_SAMPLES, SAMPLE_RATE, UtteranceSegmenter
    from tts import text_to_speech
    import io
    import soundfile as sf
    from scipy.signal import resample_poly

    seg = UtteranceSegmenter()

    detection_delays_ms = []  # audio-domain: how much speech elapsed before "start" fired
    compute_times_ms = []     # wall-clock: per-chunk inference time

    for text in TEST_SENTENCES_FOR_STT[:N_VAD_TRIALS]:
        audio_bytes = text_to_speech(text)
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if sr != SAMPLE_RATE:
            data = resample_poly(data, SAMPLE_RATE, sr).astype(np.float32)
        # Prepend silence, then measure delay from TRUE speech onset: the
        # first sample above 0.5% of full scale. (The INDICON version measured
        # from the start of the Piper clip, which includes Piper's own
        # variable leading silence -- 17-244 ms -- in the "delay".)
        data = np.clip(data, -1, 1)
        lead_silence = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
        stream = np.concatenate([lead_silence, data])
        # 0.005 = just above Piper's noise floor; same threshold as
        # eval_barge_in_latency.ONSET_THRESHOLD.
        onset_ms = (len(lead_silence) + int(np.argmax(np.abs(data) > 0.005))) / SAMPLE_RATE * 1000

        seg.reset()
        n_chunks = len(stream) // CHUNK_SAMPLES
        chunk_ms = CHUNK_SAMPLES / SAMPLE_RATE * 1000
        start_fired_at_chunk = None
        for i in range(n_chunks):
            chunk = stream[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
            t0 = time.perf_counter()
            event = seg.process_chunk(chunk)
            compute_times_ms.append((time.perf_counter() - t0) * 1000)
            if event == "start" and start_fired_at_chunk is None:
                start_fired_at_chunk = i
                break

        if start_fired_at_chunk is not None:
            # "start" can only fire once chunk i has fully arrived, i.e. at
            # the END of that chunk's audio.
            delay_ms = (start_fired_at_chunk + 1) * chunk_ms - onset_ms
            detection_delays_ms.append(delay_ms)
            print(f"  '{text[:40]}...' -> detected after {delay_ms:.0f}ms of speech")
        else:
            print(f"  '{text[:40]}...' -> WARNING: no 'start' event fired")

    s_delay = summarize("VAD detection delay (audio-domain)", detection_delays_ms)
    s_compute = summarize("VAD per-chunk compute time", compute_times_ms)
    print_summary(s_delay)
    print_summary(s_compute)
    print(f"  (chunk duration is {CHUNK_SAMPLES/SAMPLE_RATE*1000:.1f}ms — compute time must stay "
          f"well under this to keep up in real time)")
    return {"detection_delay": s_delay, "per_chunk_compute": s_compute}


# ── 2. STT latency: GPU vs CPU ──────────────────────────────────────────────

def eval_stt():
    print("\n=== 2. STT latency: GPU vs CPU (directly instantiated, bypassing the app singleton) ===")
    from stt import _add_nvidia_dll_dirs
    _add_nvidia_dll_dirs()
    from faster_whisper import WhisperModel
    from tts import text_to_speech
    import tempfile

    print("  Loading GPU model...")
    gpu_model = WhisperModel("small", device="cuda", compute_type="float16")
    print("  Loading CPU model...")
    cpu_model = WhisperModel("small", device="cpu", compute_type="int8")

    # Pre-synthesize test audio once (not timed) — timing measures STT only.
    wav_paths = []
    for text in TEST_SENTENCES_FOR_STT[:N_STT_TRIALS]:
        audio_bytes = text_to_speech(text)
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.write(audio_bytes)
        f.close()
        wav_paths.append(f.name)

    def time_model(model, label):
        times_ms = []
        for path in wav_paths:
            t0 = time.perf_counter()
            segments, info = model.transcribe(path, language="en", beam_size=5, vad_filter=True)
            _ = " ".join(seg.text for seg in segments)  # consume the generator
            times_ms.append((time.perf_counter() - t0) * 1000)
        s = summarize(f"STT ({label})", times_ms)
        print_summary(s)
        return s

    s_gpu = time_model(gpu_model, "GPU/float16")
    s_cpu = time_model(cpu_model, "CPU/int8")

    for p in wav_paths:
        Path(p).unlink(missing_ok=True)

    return {"gpu": s_gpu, "cpu": s_cpu}


# ── 3. LLM latency: streaming (time-to-first-sentence) vs batch ────────────

def eval_llm():
    print("\n=== 3. LLM response latency: streaming vs batch ===")
    import os
    from voice_agent import process_text_query, stream_text_query_sentences

    # LLM_PROVIDER=ollama has no rate limit, so no inter-call pacing is
    # needed here (Groq's free tier previously required 15s between calls
    # to avoid its SDK's silent retry-with-backoff inflating latency —
    # see README.md / paper.tex Section V-F for the full story). What
    # Ollama DOES need is a warmup: the first request after `ollama serve`
    # starts pays a one-time cost to load the model into VRAM (~140s for
    # qwen2.5:7b on this machine), which would otherwise wreck the mean.
    is_ollama = os.getenv("LLM_PROVIDER", "groq").strip().lower() == "ollama"
    INTER_CALL_DELAY_S = 0.0 if is_ollama else 15.0

    if is_ollama:
        print("  Warming up local model (loading into VRAM, untimed)...")
        process_text_query("Hello.")

    batch_times_ms = []
    stream_first_sentence_ms = []
    stream_total_ms = []
    # A call can fail outright -- e.g. Ollama aborting a degenerate
    # generation with "token repeat limit reached", which killed the first
    # ICAAIC run at query 29. That is a real failure the user would see, so
    # it is recorded and counted, not retried away; latency statistics use
    # the successful calls only.
    failures = []

    for i, query in enumerate(TEST_QUERIES_FOR_LLM[:N_LLM_TRIALS]):
        if i > 0:
            time.sleep(INTER_CALL_DELAY_S)

        t0 = time.perf_counter()
        try:
            process_text_query(query)
            batch_times_ms.append((time.perf_counter() - t0) * 1000)
            batch_txt = f"{batch_times_ms[-1]:.0f}ms"
        except Exception as exc:  # noqa: BLE001 -- record any failure, keep going
            failures.append({"query": query, "phase": "batch", "error": str(exc)[:300]})
            batch_txt = "FAILED"

        time.sleep(INTER_CALL_DELAY_S)

        t0 = time.perf_counter()
        first_sentence_t = None
        n_sentences = 0
        try:
            for _sentence in stream_text_query_sentences(query):
                n_sentences += 1
                if first_sentence_t is None:
                    first_sentence_t = time.perf_counter()
            total_t = time.perf_counter()
            if first_sentence_t is not None:
                stream_first_sentence_ms.append((first_sentence_t - t0) * 1000)
            stream_total_ms.append((total_t - t0) * 1000)
            stream_txt = (f"stream_first={(first_sentence_t - t0) * 1000 if first_sentence_t else float('nan'):.0f}ms  "
                          f"stream_total={stream_total_ms[-1]:.0f}ms  ({n_sentences} sentences)")
        except Exception as exc:  # noqa: BLE001
            failures.append({"query": query, "phase": "stream", "error": str(exc)[:300]})
            stream_txt = "stream FAILED"
        print(f"  '{query[:45]}' -> batch={batch_txt}  {stream_txt}")

    s_batch = summarize("LLM batch (full response)", batch_times_ms)
    s_stream_first = summarize("LLM streaming (time-to-first-sentence)", stream_first_sentence_ms)
    s_stream_total = summarize("LLM streaming (total)", stream_total_ms)
    print_summary(s_batch)
    print_summary(s_stream_first)
    print_summary(s_stream_total)
    n_calls = 2 * len(TEST_QUERIES_FOR_LLM[:N_LLM_TRIALS])
    print(f"  failed LLM calls: {len(failures)}/{n_calls}")
    for f in failures:
        print(f"    {f['phase']:6s} '{f['query']}': {f['error'][:90]}")
    return {"batch": s_batch, "stream_first_sentence": s_stream_first, "stream_total": s_stream_total,
            "n_calls_attempted": n_calls, "failures": failures}


# ── 4. TTS latency: sentence-streamed (time-to-first-chunk) vs batch ───────

def eval_tts():
    print("\n=== 4. TTS latency: sentence-streamed vs batch ===")
    from tts import text_to_speech
    from voice_agent import _split_complete_sentences

    batch_times_ms = []
    stream_first_chunk_ms = []
    stream_total_ms = []

    for text in TEST_SENTENCES_FOR_TTS[:N_TTS_TRIALS]:
        t0 = time.perf_counter()
        text_to_speech(text)
        batch_times_ms.append((time.perf_counter() - t0) * 1000)

        # Simulate the real sentence-chunked pipeline's synthesis pattern:
        # split into sentences, synthesize each independently, time the first.
        sentences, tail = _split_complete_sentences(text + " ")
        if tail.strip():
            sentences.append(tail.strip())
        if not sentences:
            sentences = [text]

        t0 = time.perf_counter()
        first_chunk_t = None
        for sentence in sentences:
            text_to_speech(sentence)
            if first_chunk_t is None:
                first_chunk_t = time.perf_counter()
        total_t = time.perf_counter()
        stream_first_chunk_ms.append((first_chunk_t - t0) * 1000)
        stream_total_ms.append((total_t - t0) * 1000)
        print(f"  '{text[:45]}' -> batch={batch_times_ms[-1]:.0f}ms  "
              f"stream_first={stream_first_chunk_ms[-1]:.0f}ms  "
              f"stream_total={stream_total_ms[-1]:.0f}ms  ({len(sentences)} sentences)")

    s_batch = summarize("TTS batch (full text)", batch_times_ms)
    s_stream_first = summarize("TTS streaming (time-to-first-chunk)", stream_first_chunk_ms)
    s_stream_total = summarize("TTS streaming (total)", stream_total_ms)
    print_summary(s_batch)
    print_summary(s_stream_first)
    print_summary(s_stream_total)
    return {"batch": s_batch, "stream_first_chunk": s_stream_first, "stream_total": s_stream_total}


# ── 5. LLM stop-decision latency (semantic-first halt) ─────────────────────

STOP_PROMPT_RULE = (
    "\n6. If the user asks you to stop, halt, freeze, wait, pause or hold, call "
    "stop_arm immediately, before replying -- never just say that you have stopped.\n"
)


def eval_llm_stop_decision(with_prompt_rule: bool = False):
    """Time from submitting a stop-type utterance's transcript to the moment
    the agent's streamed output first names the stop_arm tool -- i.e. when a
    semantic-first system would issue its halt. Streams the same agent graph
    with the same message format as voice_agent.process_text_query(); the
    stream is abandoned once stop_arm appears (execution time is not part of
    the decision).

    with_prompt_rule=True is an ABLATION run only inside this measurement:
    the deployed system prompt gets STOP_PROMPT_RULE appended, to test
    whether the missed stops (the agent saying it stopped without calling
    stop_arm) are fixed by simply instructing it. voice_agent.SYSTEM_PROMPT
    itself is NOT modified, so no other experiment is affected."""
    label = " + explicit stop rule in prompt (ablation)" if with_prompt_rule else ""
    print(f"\n=== 5. LLM stop-decision latency (time to stop_arm tool call){label} ===")
    from datetime import datetime
    from langchain_core.messages import HumanMessage, SystemMessage
    from voice_agent import get_agent_graph, process_text_query

    print("  Warming up local model (untimed)...")
    process_text_query("Hello.")
    if with_prompt_rule:
        from langgraph.prebuilt import create_react_agent
        from robot_tools import AGENT_TOOLS
        from voice_agent import SYSTEM_PROMPT, build_llm
        agent = create_react_agent(build_llm(), AGENT_TOOLS, prompt=SYSTEM_PROMPT + STOP_PROMPT_RULE)
    else:
        agent = get_agent_graph()

    decision_ms = []
    per_trial = []
    for utt in STOP_UTTERANCES[:N_STOP_TRIALS]:
        state = {"messages": [
            SystemMessage(content=f"Today's date is {datetime.now():%A, %B %d, %Y}."),
            HumanMessage(content=utt),
        ]}
        t0 = time.perf_counter()
        t_decision = None
        tools_called = []
        error = None
        reply = ""  # the agent's spoken text, to see what it said when it did NOT stop
        try:
            for chunk, _meta in agent.stream(state, stream_mode="messages"):
                if type(chunk).__name__ == "AIMessageChunk" and isinstance(chunk.content, str):
                    reply += chunk.content
                for tc in (getattr(chunk, "tool_call_chunks", None) or []):
                    name = tc.get("name")
                    if name and name not in tools_called:
                        tools_called.append(name)
                    if name == "stop_arm" and t_decision is None:
                        t_decision = time.perf_counter()
                if t_decision is not None:
                    break
        except Exception as exc:  # noqa: BLE001 -- a failed call is a missed stop
            error = str(exc)[:300]
        total_ms = (time.perf_counter() - t0) * 1000
        ms = (t_decision - t0) * 1000 if t_decision is not None else None
        if ms is not None:
            decision_ms.append(ms)
        per_trial.append({"utterance": utt, "stop_arm_called": ms is not None,
                          "decision_ms": round(ms, 1) if ms is not None else None,
                          "tools_called": tools_called, "elapsed_ms": round(total_ms, 1),
                          "reply": reply.strip(), "error": error})
        print(f"  '{utt:<24s}' -> " + (f"stop_arm after {ms:.0f}ms" if ms is not None
              else f"NO stop_arm (tools: {tools_called or 'none'}, {total_ms:.0f}ms"
                   + (f", ERROR: {error[:60]}" if error else "") + ")"))

    s = summarize("LLM stop decision (time to stop_arm call)", decision_ms)
    s["median_ms"] = round(statistics.median(decision_ms), 1) if decision_ms else None
    s["n_attempted"] = len(per_trial)
    s["n_stop_arm_called"] = len(decision_ms)
    s["per_trial"] = per_trial
    print_summary(s)
    print(f"  stop_arm called on {len(decision_ms)}/{len(per_trial)} stop-type utterances")
    return s


def main():
    # Optional: run/refresh a single section (e.g. after fixing rate-limit
    # pacing) without re-running the slow GPU+CPU model loads for the others.
    # Merges into the existing results JSON rather than overwriting it.
    only = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = RESULTS_DIR / "latency_offline.json"
    results = {}
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            results = json.load(f)

    def save():
        # Written after EVERY section, so a crash in a later section can
        # never discard sections that already completed.
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    for key, section, fn in [("vad", "vad", eval_vad), ("stt", "stt", eval_stt),
                             ("llm", "llm", eval_llm), ("tts", "tts", eval_tts),
                             ("llm_stop_decision", "stop", eval_llm_stop_decision),
                             ("llm_stop_decision_prompt_rule", "stop_rule",
                              lambda: eval_llm_stop_decision(with_prompt_rule=True))]:
        if only in (None, section):
            results[key] = fn()
            save()
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
