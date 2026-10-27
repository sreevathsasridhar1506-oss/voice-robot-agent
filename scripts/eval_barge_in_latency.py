"""
INDICON paper evaluation — physical barge-in response latency.

Needs the WS server running: uvicorn api:app --host 127.0.0.1 --port 8000

For N trials: streams a long-running command, waits until it's confirmed
mid-processing, then streams an interrupting "Stop right now." utterance and
measures elapsed time from SPEECH ONSET in the interrupt (first sample above
ONSET_THRESHOLD, located exactly within the paced stream) to when the
barge_in event is received. This includes network send time, VAD hysteresis
delay, and server-side processing.

DEFINITION CHANGE (ICAAIC revision, 2026-10-20): the INDICON version timed
from the start of the interrupt CLIP, which silently included Piper's
variable leading silence (17-244 ms) -- and it synthesized one clip per run,
so that silence shifted whole runs by a constant. See ONSET_THRESHOLD's
comment. The clip-start figure is still recorded per trial for traceability.

WARMUP: the first run of this script showed trial 1 at 1553ms vs. a tight
234-239ms cluster for trials 2-10, consistent with one-time WebSocket-
connection/asyncio-event-loop warmup rather than representative barge-in
cost -- but that was never independently confirmed, just a plausible
explanation for an outlier. To settle it properly rather than assert it:
N_WARMUP_TRIALS uncounted trials now run FIRST, using the exact same
run_trial() function (same code path, same measurement definition, nothing
different about how they're measured) to establish the connection/event loop,
and are explicitly excluded from the reported statistics. Their raw values
are still saved in the output JSON (as "warmup_raw_ms") for transparency --
not discarded, just not counted. The N_TRIALS=10 MEASURED trials that follow
are reported in full, with no post-hoc outlier removal.

Writes results/barge_in_latency.json.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly

from tts import text_to_speech
from vad import CHUNK_SAMPLES, SAMPLE_RATE

WS_URL = "ws://127.0.0.1:8000/ws"
RESULTS_DIR = Path("../icaaic_paper/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = 30  # ICAAIC revision: 10 -> 30 (INDICON run kept as barge_in_latency_indicon.json)
N_WARMUP_TRIALS = 2  # uncounted; establishes the WS connection/event loop before measuring
LONG_COMMAND = "Pick up the green block and put it on the right side."
INTERRUPT_COMMAND = "Stop right now."


# Speech onset = first sample above 2% of full scale. Piper clips begin with
# a variable stretch of near-silence (measured 17-244 ms for "Stop right
# now." across 12 syntheses), so timing from the CLIP start -- what the
# INDICON version did -- silently added that silence to the latency, and
# because one clip was synthesized per run and reused for every trial, it
# shifted whole runs by a constant (238 / 318 / 393 ms medians across three
# sessions). Latency is now measured from this onset instead.
# 0.005 sits just above Piper's measured noise floor (<= 0.0024) and marks
# onset at most ~10 ms earlier than 0.02 -- the earlier, conservative choice
# (an earlier onset can only make the measured latency longer).
ONSET_THRESHOLD = 0.005
LEAD_SILENCE_S = 0.3
# PACING (fixed 2026-10-20): the INDICON harness sent each 64 ms chunk at the
# START of its real-time interval and then slept, so the server received
# audio up to 64 ms before a real microphone could have delivered it -- which
# produced impossible sub-VAD-window latencies once timing was referenced to
# speech onset. Now: 128-sample (8 ms) chunks, the browser AudioWorklet's
# render quantum, each sent only AFTER its interval has elapsed on an
# absolute clock. Windows' coarse timer can only delay a chunk, never send
# it early, so any residual pacing error lengthens the measured latency.
SEND_SAMPLES = 128


def synth_pcm16_at_16k(text: str) -> tuple[bytes, int]:
    """Returns (PCM16 bytes, index of the first sample above ONSET_THRESHOLD)."""
    audio_bytes = text_to_speech(text)
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if sr != SAMPLE_RATE:
        data = resample_poly(data, SAMPLE_RATE, sr).astype(np.float32)
    data = np.clip(data, -1, 1)
    onset_idx = int(np.argmax(np.abs(data) > ONSET_THRESHOLD))
    return (data * 32767).astype(np.int16).tobytes(), onset_idx


async def stream_utterance(ws, pcm16_bytes: bytes, trail_s: float = 0.6):
    """Streams audio as a live microphone would: after LEAD_SILENCE_S of
    silence, each SEND_SAMPLES chunk is sent only once its real-time interval
    has fully elapsed (absolute schedule from t0; chunks that came due during
    a coarse timer tick are sent together, never before they are due).
    Returns (t0, lead_samples): stream time zero, so sample i "occurs" at
    t0 + i / SAMPLE_RATE."""
    silence_lead = np.zeros(int(LEAD_SILENCE_S * SAMPLE_RATE), dtype=np.int16).tobytes()
    silence_trail = np.zeros(int(trail_s * SAMPLE_RATE), dtype=np.int16).tobytes()
    stream = silence_lead + pcm16_bytes + silence_trail
    n_samples = len(stream) // 2

    def due(k):  # wall time at which chunk k's audio has fully "happened"
        return t0 + (k + 1) * SEND_SAMPLES / SAMPLE_RATE

    t0 = time.perf_counter()
    k = 0
    while k * SEND_SAMPLES < n_samples:
        wait = due(k) - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)
        now = time.perf_counter()
        while k * SEND_SAMPLES < n_samples and due(k) <= now:
            await ws.send(stream[k * SEND_SAMPLES * 2:(k + 1) * SEND_SAMPLES * 2])
            k += 1
    return t0, len(silence_lead) // 2


async def run_trial(trial_num, long_pcm: bytes, interrupt: tuple[bytes, int]):
    """Returns a dict with latency from speech ONSET (the reported figure)
    and from CLIP start (the INDICON definition, kept for traceability),
    or None if no barge_in event arrived."""
    interrupt_pcm, onset_in_clip = interrupt
    events = []
    barge_in_time = {"t": None}
    done = asyncio.Event()

    async def collector(ws):
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=30)
            if isinstance(msg, bytes):
                continue
            data = json.loads(msg)
            events.append((time.perf_counter(), data))
            if data["type"] == "barge_in" and barge_in_time["t"] is None:
                barge_in_time["t"] = time.perf_counter()
            if data["type"] == "done":
                done.set()
                return

    async with websockets.connect(WS_URL) as ws:
        collector_task = asyncio.create_task(collector(ws))

        await stream_utterance(ws, long_pcm)
        await asyncio.sleep(2.2)  # let the first command get into processing/speaking

        t0, lead_samples = await stream_utterance(ws, interrupt_pcm)

        try:
            await asyncio.wait_for(done.wait(), timeout=20)
        finally:
            collector_task.cancel()

    if barge_in_time["t"] is None:
        print(f"  Trial {trial_num}: FAILED — no barge_in event observed")
        return None

    t_clip_start = t0 + lead_samples / SAMPLE_RATE
    t_onset = t0 + (lead_samples + onset_in_clip) / SAMPLE_RATE
    from_onset_ms = (barge_in_time["t"] - t_onset) * 1000
    from_clip_ms = (barge_in_time["t"] - t_clip_start) * 1000
    lead_in_clip_ms = onset_in_clip / SAMPLE_RATE * 1000
    print(f"  Trial {trial_num}: from speech onset = {from_onset_ms:.0f}ms  "
          f"(from clip start = {from_clip_ms:.0f}ms; clip's leading silence {lead_in_clip_ms:.0f}ms)")
    return {"from_onset_ms": from_onset_ms, "from_clip_start_ms": from_clip_ms,
            "clip_leading_silence_ms": lead_in_clip_ms}


async def main():
    # The interrupt is re-synthesized for EVERY trial (Piper varies timing
    # between syntheses), so trials sample that variation instead of one
    # fixed clip. Synthesis happens before streaming, outside the timing.
    print(f"Synthesizing test audio ({N_WARMUP_TRIALS} warmup + {N_TRIALS} measured trials, "
          f"fresh interrupt clip per trial)...")
    long_pcm, _ = synth_pcm16_at_16k(LONG_COMMAND)
    interrupts = [synth_pcm16_at_16k(INTERRUPT_COMMAND) for _ in range(N_WARMUP_TRIALS + N_TRIALS)]

    import os
    is_ollama = os.getenv("LLM_PROVIDER", "groq").strip().lower() == "ollama"
    inter_trial_delay_s = 2.0 if is_ollama else 15.0
    # Groq-tier pacing rationale unchanged from before -- see
    # eval_latency_offline.py's eval_llm(). Applies identically to warmup and
    # measured trials; the barge_in event itself is VAD-triggered, not
    # LLM-triggered, so this never affected the measured latency directly,
    # only whether trials queued up behind Groq's retry backoff.

    print(f"\nRunning {N_WARMUP_TRIALS} uncounted warmup trial(s) against {WS_URL} "
          f"(establishes the WS connection/event loop; excluded from reported stats)...")
    warmup = []
    for i in range(1, N_WARMUP_TRIALS + 1):
        result = await run_trial(f"warmup-{i}", long_pcm, interrupts[i - 1])
        if result is not None:
            warmup.append(result)
        await asyncio.sleep(inter_trial_delay_s)

    print(f"\nRunning {N_TRIALS} MEASURED barge-in trials against {WS_URL}...")
    trials = []
    for i in range(1, N_TRIALS + 1):
        result = await run_trial(i, long_pcm, interrupts[N_WARMUP_TRIALS + i - 1])
        if result is not None:
            trials.append(result)
        await asyncio.sleep(inter_trial_delay_s)

    if not trials:
        print("\nNo successful measured trials — check the server is running and reachable.")
        return

    import statistics

    def stats(values):
        q = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
        return {
            "mean_ms": round(statistics.mean(values), 1),
            "median_ms": round(statistics.median(values), 1),
            "stdev_ms": round(statistics.stdev(values), 1) if len(values) > 1 else 0.0,
            "q1_ms": round(q[0], 1), "q3_ms": round(q[2], 1),
            "min_ms": round(min(values), 1), "max_ms": round(max(values), 1),
            "raw_ms": [round(v, 1) for v in values],
        }

    summary = {
        "n": len(trials),
        "n_attempted": N_TRIALS,
        "definition": f"barge_in event received minus nominal arrival of the first interrupt sample "
                      f"above {ONSET_THRESHOLD} of full scale (speech onset)",
        "from_onset": stats([t["from_onset_ms"] for t in trials]),
        "from_clip_start": stats([t["from_clip_start_ms"] for t in trials]),
        "clip_leading_silence": stats([t["clip_leading_silence_ms"] for t in trials]),
        "n_warmup": len(warmup),
        "n_warmup_attempted": N_WARMUP_TRIALS,
        "warmup_from_onset_ms": [round(t["from_onset_ms"], 1) for t in warmup],
        "trials": [{k: round(v, 1) for k, v in t.items()} for t in trials],
    }
    o, c = summary["from_onset"], summary["from_clip_start"]
    print(f"\n=== Barge-in latency summary ({summary['n']}/{summary['n_attempted']} successful MEASURED trials"
          f", {summary['n_warmup']}/{summary['n_warmup_attempted']} warmup trials excluded from stats) ===")
    print(f"  FROM SPEECH ONSET: median={o['median_ms']}ms  IQR=[{o['q1_ms']}, {o['q3_ms']}]  "
          f"mean={o['mean_ms']}ms  std={o['stdev_ms']}ms  min={o['min_ms']}ms  max={o['max_ms']}ms")
    print(f"  from clip start:   median={c['median_ms']}ms  min={c['min_ms']}ms  max={c['max_ms']}ms")
    print(f"  warmup (from onset): {summary['warmup_from_onset_ms']}")

    out_path = RESULTS_DIR / "barge_in_latency.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
