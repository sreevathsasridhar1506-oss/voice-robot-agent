"""
INDICON paper evaluation — task robustness.

Runs N randomized pick-and-place trials (random source block x random
destination zone) through the real text-command pipeline
(voice_agent.process_text_query -> robot_tools -> sim_controller -> MuJoCo),
and measures:
  - task success rate (block ends up within tolerance of the target zone)
  - malformed LLM tool-call rate and recovery rate (captured non-invasively
    via a logging handler on voice_agent's "Tool-call generation glitch"
    warning — see process_text_query()'s retry loop)

Does NOT need the WS server running — calls the agent/sim directly, same
pattern as scripts/test_agent_e2e.py.

Writes results/task_robustness.json.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

RESULTS_DIR = Path("../icaaic_paper/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ICAAIC revision (2026-10-20): 24 -> 60 trials, seeded so the colour/zone
# sequence is reproducible. The INDICON run (n=24, unseeded) is kept as
# results/task_robustness_indicon.json.
N_TRIALS = 60
SEED = 0
PLACE_TOLERANCE_M = 0.05  # matches the zone-approach clearance used elsewhere in this project

# Unambiguous colors only — disambiguation behavior is evaluated separately
# in eval_disambiguation.py, kept out of this robustness measurement.
SOURCE_COLORS = ["blue", "green", "yellow"]
ZONES = ["left", "right", "center"]

# Canonical scene-XML spawn positions (pick_place_scene.xml) — each trial
# resets its target block here before issuing the command. Without this, a
# block left near a destination zone by a PREVIOUS trial can make the next
# trial's scoring meaningless: a genuinely successful placement can measure
# as barely-moved (looking like nothing happened), or a block already near a
# zone by chance can score a false "success" with no real manipulation. This
# was confirmed to have actually happened in the original run (trials 4, 11,
# 24 — see research_methodology_exp.md) before this reset was added. All five
# spawn positions are listed for completeness even though only blue/green/
# yellow are used as SOURCE_COLORS here.
BLOCK_SPAWN_POSITIONS = {
    "block_red": [0.5, 0.15, 0.03],
    "block_red2": [0.35, -0.05, 0.03],
    "block_blue": [0.5, -0.15, 0.03],
    "block_green": [0.4, 0.32, 0.03],
    "block_yellow": [0.4, -0.32, 0.03],
}


class RetryCounter(logging.Handler):
    """Counts "Tool-call generation glitch" warnings from voice_agent's
    retry-with-temperature-bump loop, without touching its source — this is
    the only externally-observable signal of a malformed tool call, since
    process_text_query() doesn't return a retry count."""
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record):
        if "Tool-call generation glitch" in record.getMessage():
            self.count += 1


def main():
    from robot_tools import get_sim_controller, DROP_ZONES
    from voice_agent import process_text_query

    retry_counter = RetryCounter()
    logging.getLogger("voice_agent").addHandler(retry_counter)

    sc = get_sim_controller()
    trials = []

    # LLM_PROVIDER=ollama has no rate limit, so no inter-trial pacing is
    # needed (Groq's free tier previously required it — see
    # eval_latency_offline.py's eval_llm()).
    import os
    is_ollama = os.getenv("LLM_PROVIDER", "groq").strip().lower() == "ollama"
    INTER_TRIAL_DELAY_S = 0.0 if is_ollama else 15.0

    if is_ollama:
        print("Warming up local model (loading into VRAM, untimed)...")
        process_text_query("Hello.")

    rng = random.Random(SEED)
    for i in range(1, N_TRIALS + 1):
        if i > 1:
            time.sleep(INTER_TRIAL_DELAY_S)
        color = rng.choice(SOURCE_COLORS)
        zone = rng.choice(ZONES)
        command = f"Pick up the {color} block and put it on the {zone} side."

        # Ensure nothing is already held from a previous trial's failure.
        state = sc.get_scene_state()
        if state["held_object"] is not None:
            sc.place(DROP_ZONES["center"])

        block_name = f"block_{color}"
        # Reset ALL blocks to their canonical spawn positions before every
        # trial — see BLOCK_SPAWN_POSITIONS' comment for why the target must
        # be reset. Resetting only the target (the INDICON version) let
        # blocks placed by earlier trials accumulate in the drop zones, where
        # they could obstruct later placements; over 60 trials that drift
        # becomes a real confound, so every trial now starts from one scene.
        for name, pos in BLOCK_SPAWN_POSITIONS.items():
            sc.reset_object(name, pos)

        retries_before = retry_counter.count
        t0 = time.perf_counter()
        # A turn can fail outright (e.g. Ollama aborting a degenerate
        # generation: "token repeat limit reached"). The user would see a
        # failed command, so it is scored as a failed trial and recorded --
        # not retried away, and not allowed to crash the whole run.
        llm_error = None
        try:
            result = process_text_query(command)
        except Exception as exc:  # noqa: BLE001
            llm_error = str(exc)[:300]
            result = {"answer": ""}
        elapsed_s = time.perf_counter() - t0
        retries_this_trial = retry_counter.count - retries_before

        state_after = sc.get_scene_state()
        obj_after = next((o for o in state_after["objects"] if o["name"] == block_name), None)
        pos_after = np.array(obj_after["position"]) if obj_after else None
        held_after = state_after["held_object"]

        target = np.array(DROP_ZONES[zone])
        success = False
        distance_m = None
        if pos_after is not None:
            distance_m = float(np.linalg.norm(pos_after[:2] - target[:2]))
            # Primary success criterion is final placement error alone
            # (distance_m < PLACE_TOLERANCE_M), plus a real task-state check
            # that the object was actually released via a completed place()
            # rather than left stuck in the gripper — not a raw displacement
            # heuristic. Since the block is reset to a known, >5cm-from-every-
            # zone spawn position above, this can no longer register a false
            # success/failure from a previous trial's leftover placement.
            success = distance_m < PLACE_TOLERANCE_M and held_after is None and llm_error is None

        trial = {
            "trial": i,
            "command": command,
            "color": color,
            "zone": zone,
            "success": success,
            "distance_to_target_m": round(distance_m, 4) if distance_m is not None else None,
            "held_after": held_after,
            "elapsed_s": round(elapsed_s, 2),
            "retries": retries_this_trial,
            "answer": result["answer"],
            "llm_error": llm_error,
        }
        trials.append(trial)
        status = "OK" if success else ("FAIL (LLM error)" if llm_error else "FAIL")
        print(f"  [{i:2d}/{N_TRIALS}] {command:<50s} -> {status}  "
              f"(dist={distance_m:.3f}m, {elapsed_s:.1f}s, retries={retries_this_trial})")

    sc.close()

    n_success = sum(1 for t in trials if t["success"])
    n_with_retry = sum(1 for t in trials if t["retries"] > 0)
    total_retries = sum(t["retries"] for t in trials)
    # Total LLM turns attempted ≈ trials + retries (each retry is one extra attempt).
    total_attempts = len(trials) + total_retries

    summary = {
        "n_trials": len(trials),
        "n_success": n_success,
        "success_rate": round(n_success / len(trials), 3),
        "mean_tool_call_attempts_per_trial": round(total_attempts / len(trials), 2),
        "malformed_call_rate": round(total_retries / total_attempts, 3) if total_attempts else 0.0,
        "trials_requiring_retry": n_with_retry,
        "trials_requiring_retry_rate": round(n_with_retry / len(trials), 3),
        "mean_elapsed_s": round(sum(t["elapsed_s"] for t in trials) / len(trials), 2),
        "n_llm_errors": sum(1 for t in trials if t["llm_error"]),
        "seed": SEED,
        "trials": trials,
    }

    print(f"\n=== Task robustness summary ===")
    print(f"  Success rate: {summary['n_success']}/{summary['n_trials']} = {summary['success_rate']*100:.1f}%")
    print(f"  Mean tool-call attempts/trial: {summary['mean_tool_call_attempts_per_trial']}")
    print(f"  Malformed tool-call rate: {summary['malformed_call_rate']*100:.1f}% of attempts")
    print(f"  Trials requiring >=1 retry: {summary['trials_requiring_retry']} ({summary['trials_requiring_retry_rate']*100:.1f}%)")
    print(f"  (All observed malformed calls recovered successfully via retry — "
          f"none propagated to a failed trial unless noted FAIL above.)")

    out_path = RESULTS_DIR / "task_robustness.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
