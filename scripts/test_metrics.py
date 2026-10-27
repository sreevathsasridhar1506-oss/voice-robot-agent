"""
Standalone check (M7): interaction-metrics logging.

Runs a small scripted session — a normal command, an ambiguous one (should
count as a clarification), and a stop (should count as an interrupt) — then
flushes and inspects the resulting log record for sane values.
"""

import sys
import threading
import time

sys.path.insert(0, ".")

from metrics import get_session, reset_session
from robot_tools import get_sim_controller
from voice_agent import process_text_query


def main():
    reset_session()
    sc = get_sim_controller()

    process_text_query("What's on the table?")
    process_text_query("Pick up the red block.")  # ambiguous -> clarification

    # An interrupt: start a move in the background, stop it via the tool path
    # (going through process_text_query so it's a realistic "user says stop").
    def run_move():
        sc.move_to([0.3, 0.4, 0.5])

    t = threading.Thread(target=run_move)
    t.start()
    time.sleep(0.15)
    process_text_query("Stop!")
    t.join(timeout=10)

    record = get_session().flush()
    print("Session record:")
    for k, v in record.items():
        if k != "turns":
            print(f"  {k}: {v}")
    print("  turns:")
    for turn in record["turns"]:
        print(f"    - {turn['query']!r} -> clarification={turn['was_clarification']}, {turn['elapsed_s']}s")

    assert record["turn_count"] == 3
    assert record["clarification_count"] >= 1, "the ambiguous pick should have counted as a clarification"
    assert record["interrupt_count"] == 1, "the stop command should have counted as one interrupt"
    print("\nSUCCESS — metrics captured turn count, clarification, and interrupt correctly.")
    sc.close()


if __name__ == "__main__":
    main()
