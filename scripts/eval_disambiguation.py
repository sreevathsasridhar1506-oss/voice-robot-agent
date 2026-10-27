"""
INDICON paper evaluation — disambiguation precision/recall.

Tests three conditions with varied phrasing (not just repeating one sentence
verbatim, to probe some generalization across surface form):

  A. AMBIGUOUS, unqualified  -> should ask a clarifying question, not act.
     (measures ambiguity-detection RECALL)
  B. UNAMBIGUOUS             -> should act directly, not ask.
     (measures FALSE-CLARIFICATION RATE = 1 - specificity)
  C. AMBIGUOUS but qualified IN THE SAME UTTERANCE (e.g. "...on the left")
     -> should resolve and act directly on the CORRECT-SIDE object, not ask.
     (measures same-utterance disambiguation accuracy)

     Condition C scoring was tightened (2026-08-29): it previously counted
     any red block as success, which could not tell a correct spatial
     resolution apart from a lucky 50/50 guess. Each trial now resets both
     red blocks to fixed spawn positions, derives which is objectively
     left/right from their measured post-reset y-coordinates, and requires
     the agent to end up holding the block on the REQUESTED side.

IMPORTANT SCOPE NOTE: this system does not carry conversation memory across
separate process_text_query() calls (each call is a fresh turn — see
voice_agent.py). So "ask, then the user replies in a follow-up turn" is NOT
tested here, only same-utterance qualification (condition C). This is a real
system limitation and should be stated in the paper's Discussion section,
not glossed over.

Clarification detection: two rounds of fixing this.

Round 1 tried a purely structural signal: robot_tools.py's
_resolve_with_disambiguation() logs a "Disambiguation needed" warning at the
exact point it determines an object REFERENCE is ambiguous (more than one
color match, no resolvable "which"), captured via a logging handler
(DisambiguationCounter). This correctly killed the original bug (the old
"?"-in-answer heuristic false-positived on unrelated follow-ups like "would
you like me to place it somewhere?"), but introduced a NEW gap: the agent can
reach a genuine, correct clarifying question by calling describe_scene()
first and asking based on what it sees, WITHOUT ever calling
pick_object/move_to_object — that path never touches
_resolve_with_disambiguation, so the tool-only signal silently under-counted
real clarifications, and separately missed real hallucinated-ambiguity
failures in Condition B (the model claiming "multiple blue blocks" exist when
only one does, without ever calling the tool either).

Round 2 (current): needs_clarification() combines a real simulation-state
check (did the agent end the turn WITHOUT holding an object — "held is None"
is read directly from MuJoCo/SimController state, not text) with a minimal
sanity check that the reply was phrased as a question ("?" in answer, used
ONLY to rule out a silent failure being miscounted as "asked" — not to guess
what the question was ABOUT, which is what the original heuristic got wrong).
The tool-level signal is still recorded per-trial as `tool_flagged_ambiguous`
for diagnostic transparency, but is no longer the primary classification.

Does NOT need the WS server running.

Writes results/disambiguation.json.
"""

from __future__ import annotations

import json
import logging
import time
import sys
from pathlib import Path

sys.path.insert(0, ".")

RESULTS_DIR = Path("../icaaic_paper/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ICAAIC revision (2026-10-20): 8 -> 16 distinct phrasings per condition,
# each run N_REPEATS times (LLM temperature is 0.3, so repeats sample
# genuinely different outputs) = 32 trials per condition. The INDICON run
# (8 per condition) is kept as results/disambiguation_indicon.json.
#
# One INDICON Condition-A item was invalid and is replaced: "Go ahead and
# pick up A red block." -- the indefinite article means either red block is
# acceptable, so acting without asking is CORRECT there, yet it was scored
# as a missed clarification. Now "...pick up THE red block." No new item
# uses an indefinite article for the same reason.
N_REPEATS = 2

CONDITION_A_AMBIGUOUS_UNQUALIFIED = [
    "Pick up the red block.",
    "Can you grab the red block for me?",
    "I need the red block picked up.",
    "Get the red block, please.",
    "Go ahead and pick up the red block.",
    "Would you pick up the red one?",
    "Take the red block off the table.",
    "Please retrieve the red block.",
    "Hand me the red block.",
    "Could you pick up the red block?",
    "Lift the red block.",
    "I want the red block.",
    "Pick the red one up, please.",
    "Fetch the red block.",
    "Can you get me the red block?",
    "Bring the red block over.",
]

CONDITION_B_UNAMBIGUOUS = [
    "Pick up the blue block.",
    "Can you grab the green block for me?",
    "I need the yellow block picked up.",
    "Get the blue block, please.",
    "Go ahead and pick up the green one.",
    "Would you pick up the yellow block?",
    "Take the blue block off the table.",
    "Please retrieve the green block.",
    "Hand me the yellow block.",
    "Could you pick up the blue block?",
    "Lift the green block.",
    "I want the yellow block.",
    "Pick the blue one up, please.",
    "Fetch the green block.",
    "Can you get me the yellow block?",
    "Bring the blue block over.",
]

# All five blocks are reset to these canonical spawn positions before EVERY
# trial in every condition, so each trial starts from an identical scene
# (the INDICON version only reset the two red blocks, and only in C).
ALL_BLOCK_SPAWN_POSITIONS = {
    "block_red": [0.5, 0.15, 0.03],
    "block_red2": [0.35, -0.05, 0.03],
    "block_blue": [0.5, -0.15, 0.03],
    "block_green": [0.4, 0.32, 0.03],
    "block_yellow": [0.4, -0.32, 0.03],
}

# Canonical spawn positions of the two red blocks, from
# pick_place_scene.xml. Both are reset here before EVERY Condition C trial
# so that "left"/"right" is well-defined and identical across trials.
# Without this, a previous trial's placement (reset_if_holding() drops a held
# block at the CENTER zone, y=0.0) shifts the geometry and the side labels
# drift mid-run, making the side check meaningless.
RED_BLOCK_SPAWN_POSITIONS = {
    "block_red": [0.5, 0.15, 0.03],    # y=+0.15 -> geometrically LEFT
    "block_red2": [0.35, -0.05, 0.03],  # y=-0.05 -> geometrically RIGHT
}

# (command, expected_side). Success now requires the agent to end up holding
# the block that is geometrically on the REQUESTED side -- not merely "some
# red block", which was the original check. That original check could not
# distinguish a correct spatial resolution from a lucky 50/50 guess, and
# scored both identically as success.
CONDITION_C_AMBIGUOUS_QUALIFIED = [
    ("Pick up the red block on the left.", "left"),
    ("Can you grab the red block that's on the right?", "right"),
    ("I need the left red block picked up.", "left"),
    ("Get the right red block, please.", "right"),
    ("Go ahead and pick up the red one on the left.", "left"),
    ("Would you pick up the red block on the right side?", "right"),
    ("Take the left red block off the table.", "left"),
    ("Please retrieve the red block on the right.", "right"),
    ("Hand me the red block on the left.", "left"),
    ("Could you pick up the right red block?", "right"),
    ("Lift the red block that's on the left.", "left"),
    ("I want the red block on the right.", "right"),
    ("Pick the left red one up, please.", "left"),
    ("Fetch the red block from the right side.", "right"),
    ("Can you get me the red block on the left side?", "left"),
    ("Bring over the red block on the right.", "right"),
]


class DisambiguationCounter(logging.Handler):
    """Counts "Disambiguation needed" warnings from robot_tools' structured
    ambiguity signal (see _resolve_with_disambiguation) — this fires only
    when the OBJECT REFERENCE itself was ambiguous, never for an unrelated
    follow-up question in the agent's spoken reply."""
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record):
        if "Disambiguation needed" in record.getMessage():
            self.count += 1


def reset_if_holding(sc):
    """Release anything held, then put every block back at its spawn
    position, so each trial starts from the same scene."""
    from robot_tools import DROP_ZONES
    state = sc.get_scene_state()
    if state["held_object"] is not None:
        sc.place(DROP_ZONES["center"])
    for name, pos in ALL_BLOCK_SPAWN_POSITIONS.items():
        sc.reset_object(name, pos)


def main():
    from robot_tools import get_sim_controller
    from voice_agent import process_text_query

    sc = get_sim_controller()
    results = {"A_ambiguous_unqualified": [], "B_unambiguous": [], "C_ambiguous_qualified": []}

    disambig_counter = DisambiguationCounter()
    logging.getLogger("robot_tools").addHandler(disambig_counter)

    # LLM_PROVIDER=ollama has no rate limit, so no inter-trial pacing is
    # needed (Groq's free tier previously required it — see
    # eval_latency_offline.py's eval_llm()). Still pay the one-time model
    # warmup cost (~140s for qwen2.5:7b) up front rather than on trial 1.
    import os
    is_ollama = os.getenv("LLM_PROVIDER", "groq").strip().lower() == "ollama"
    INTER_TRIAL_DELAY_S = 0.0 if is_ollama else 15.0
    first_call = True

    if is_ollama:
        print("Warming up local model (loading into VRAM, untimed)...")
        process_text_query("Hello.")

    def paced_query(cmd):
        """Runs one query and returns (result, tool_flagged_ambiguous) — the
        latter is the structured DisambiguationCounter signal (did
        _resolve_with_disambiguation's own multi-match logic fire during this
        call). This alone is NOT the full story — see needs_clarification()
        below for why."""
        nonlocal first_call
        if not first_call:
            time.sleep(INTER_TRIAL_DELAY_S)
        first_call = False
        count_before = disambig_counter.count
        # A turn can fail outright (e.g. Ollama's "token repeat limit
        # reached"). Recorded as an empty answer with "llm_error" set: with
        # no "?" in the answer it can never count as a clarification, and
        # with nothing held it can never count as acting -- i.e. it scores
        # as a failure in A and C, and as a non-action in B.
        try:
            result = process_text_query(cmd)
            result["llm_error"] = None
        except Exception as exc:  # noqa: BLE001
            result = {"answer": "", "llm_error": str(exc)[:300]}
        tool_flagged_ambiguous = disambig_counter.count > count_before
        return result, tool_flagged_ambiguous

    def needs_clarification(answer: str, held) -> bool:
        """The actual per-trial classification used everywhere below.

        NOT just the tool-level signal: the agent can also reach a genuine
        clarifying question by calling describe_scene() first and asking
        based on what it sees, WITHOUT ever calling pick_object/
        move_to_object at all -- that path never touches
        _resolve_with_disambiguation, so the tool-level counter alone
        under-counts real clarifications (confirmed: several Condition-A
        answers below contain the exact coordinates describe_scene() prints,
        with the object never actually picked).

        So the real signal used here is state-based: did the agent end this
        turn WITHOUT holding an object ("held is None" -- a genuine
        MuJoCo/SimController state check, not a text heuristic), combined
        with a sanity check that the reply was actually phrased as a question
        (rules out a silent failure being miscounted as "asked"). This is
        deliberately NOT the old "?" heuristic from the previous version of
        this script: that heuristic flagged ANY "?" in the reply, including
        unrelated follow-ups like "would you like me to place it somewhere?"
        even when the object itself was already correctly, unambiguously
        picked up (held is NOT None in that case) -- exactly the false
        positive this rewrite fixes. The held-object check is what makes it
        specifically about object-reference ambiguity, not about whether the
        agent said anything question-shaped at all."""
        return held is None and "?" in answer

    def repeated(items):
        """Each phrasing N_REPEATS times; yields (rep, item)."""
        for rep in range(1, N_REPEATS + 1):
            for item in items:
                yield rep, item

    print("=== Condition A: ambiguous, unqualified (expect: clarifying question) ===")
    for rep, cmd in repeated(CONDITION_A_AMBIGUOUS_UNQUALIFIED):
        reset_if_holding(sc)
        r, tool_flagged_ambiguous = paced_query(cmd)
        held = sc.get_scene_state()["held_object"]
        asked = needs_clarification(r["answer"], held)
        results["A_ambiguous_unqualified"].append({
            "rep": rep, "command": cmd, "answer": r["answer"], "llm_error": r["llm_error"], "needs_clarification": asked,
            "tool_flagged_ambiguous": tool_flagged_ambiguous, "held_object": held,
        })
        print(f"  [{'OK ' if asked else 'MISS'}] '{cmd}' -> needs_clarification={asked} (tool_flagged={tool_flagged_ambiguous}, held={held})")

    print("\n=== Condition B: unambiguous (expect: act directly, no question) ===")
    for rep, cmd in repeated(CONDITION_B_UNAMBIGUOUS):
        reset_if_holding(sc)
        r, tool_flagged_ambiguous = paced_query(cmd)
        held = sc.get_scene_state()["held_object"]
        asked = needs_clarification(r["answer"], held)
        results["B_unambiguous"].append({
            "rep": rep, "command": cmd, "answer": r["answer"], "llm_error": r["llm_error"], "needs_clarification": asked,
            "tool_flagged_ambiguous": tool_flagged_ambiguous, "acted": held is not None,
        })
        print(f"  [{'OK ' if not asked else 'MISS'}] '{cmd}' -> needs_clarification={asked}, acted={held is not None}")

    print("\n=== Condition C: ambiguous but qualified in-utterance "
          "(expect: resolve + act on the CORRECT side) ===")
    for rep, (cmd, expected_side) in repeated(CONDITION_C_AMBIGUOUS_QUALIFIED):
        # 1. Release anything held and reset ALL blocks (including both reds)
        #    to their canonical spawn positions, so the scene geometry is
        #    identical at the start of every trial.
        reset_if_holding(sc)

        # 2-3. Record the actual post-reset positions and derive which block is
        #      objectively left vs. right from those measured positions, rather
        #      than assuming the mapping. Higher y = left, matching
        #      robot_tools._side_hint() and DROP_ZONES. Deriving it (instead of
        #      hardcoding block_red="left") means this check stays correct even
        #      if the scene file's spawn positions are ever changed.
        state_before = sc.get_scene_state()
        red_positions = {
            o["name"]: o["position"] for o in state_before["objects"]
            if o["name"] in RED_BLOCK_SPAWN_POSITIONS
        }
        left_block = max(red_positions, key=lambda n: red_positions[n][1])
        right_block = min(red_positions, key=lambda n: red_positions[n][1])
        expected_block = left_block if expected_side == "left" else right_block

        # 4. Run the command.
        r, tool_flagged_ambiguous = paced_query(cmd)

        # 5-6. Check WHICH block was selected; success only if it is the block
        #      on the side the command actually asked for.
        held = sc.get_scene_state()["held_object"]
        asked = needs_clarification(r["answer"], held)
        picked_correct_side = held == expected_block
        resolved_correctly = (not asked) and picked_correct_side

        results["C_ambiguous_qualified"].append({
            "rep": rep, "command": cmd, "answer": r["answer"], "llm_error": r["llm_error"], "needs_clarification": asked,
            "tool_flagged_ambiguous": tool_flagged_ambiguous,
            "expected_side": expected_side,
            "expected_block": expected_block,
            "left_block": left_block, "right_block": right_block,
            "red_positions_at_start": {n: [round(v, 4) for v in p]
                                       for n, p in red_positions.items()},
            "held_object": held,
            "picked_correct_side": picked_correct_side,
            "resolved_correctly": resolved_correctly,
        })
        print(f"  [{'OK ' if resolved_correctly else 'MISS'}] '{cmd}' -> "
              f"asked={asked}, wanted={expected_side}({expected_block}), "
              f"held={held}, correct_side={picked_correct_side}")

    sc.close()

    n_a = len(results["A_ambiguous_unqualified"])
    recall = sum(1 for t in results["A_ambiguous_unqualified"] if t["needs_clarification"]) / n_a

    n_b = len(results["B_unambiguous"])
    false_clarify = sum(1 for t in results["B_unambiguous"] if t["needs_clarification"]) / n_b
    b_acted_correctly = sum(1 for t in results["B_unambiguous"] if t["acted"] and not t["needs_clarification"]) / n_b

    n_c = len(results["C_ambiguous_qualified"])
    c_correct = sum(1 for t in results["C_ambiguous_qualified"] if t["resolved_correctly"])
    same_utterance_accuracy = c_correct / n_c
    # Diagnostic split: of the trials that acted at all (didn't ask), how often
    # was the chosen side right? Separates "asked instead of acting" failures
    # from "acted on the wrong block" failures -- these are different problems.
    c_acted = [t for t in results["C_ambiguous_qualified"] if not t["needs_clarification"]]
    c_side_acc_when_acted = (
        sum(1 for t in c_acted if t["picked_correct_side"]) / len(c_acted)
        if c_acted else 0.0
    )

    summary = {
        "ambiguity_detection_recall": round(recall, 3),
        "false_clarification_rate": round(false_clarify, 3),
        "unambiguous_correct_action_rate": round(b_acted_correctly, 3),
        "same_utterance_disambiguation_accuracy": round(same_utterance_accuracy, 3),
        "c_side_accuracy_when_acted": round(c_side_acc_when_acted, 3),
        "c_n_acted_without_asking": len(c_acted),
        "n_per_condition": {"A": n_a, "B": n_b, "C": n_c},
        "n_phrasings_per_condition": len(CONDITION_A_AMBIGUOUS_UNQUALIFIED),
        "n_repeats": N_REPEATS,
        # Turns that failed outright. In B such a turn neither asks nor acts,
        # so it does NOT raise the false-clarification rate -- report it via
        # unambiguous_correct_action_rate and this count, never as a success.
        "n_llm_errors": {k: sum(1 for t in v if t["llm_error"]) for k, v in results.items()},
        "detail": results,
    }

    print(f"\n=== Disambiguation summary ===")
    print(f"  Ambiguity-detection recall (A):            {recall*100:.1f}% ({sum(1 for t in results['A_ambiguous_unqualified'] if t['needs_clarification'])}/{n_a})")
    print(f"  False-clarification rate (B):               {false_clarify*100:.1f}% ({sum(1 for t in results['B_unambiguous'] if t['needs_clarification'])}/{n_b})")
    print(f"  Same-utterance disambiguation accuracy (C): {same_utterance_accuracy*100:.1f}% ({c_correct}/{n_c})")
    print(f"    of which, correct SIDE when it acted:     {c_side_acc_when_acted*100:.1f}% "
          f"({sum(1 for t in c_acted if t['picked_correct_side'])}/{len(c_acted)})")

    out_path = RESULTS_DIR / "disambiguation.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
