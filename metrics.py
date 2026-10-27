"""
metrics.py
==========
Lightweight interaction-quality logging for the HRI narrative: per-session
turn count, clarification-turn count, interrupt count, and total duration.
Plain JSONL — this is data for an application write-up, not a production
analytics system.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("session_metrics.jsonl")


@dataclass
class SessionMetrics:
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    started_at: float = field(default_factory=time.perf_counter)
    turns: list = field(default_factory=list)
    clarification_count: int = 0
    interrupt_count: int = 0

    def record_turn(self, query: str, answer: str, elapsed_s: float) -> None:
        was_clarification = "?" in answer
        self.turns.append({
            "query": query,
            "answer": answer,
            "elapsed_s": round(elapsed_s, 3),
            "was_clarification": was_clarification,
        })
        if was_clarification:
            self.clarification_count += 1

    def record_interrupt(self) -> None:
        self.interrupt_count += 1

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_duration_s": round(time.perf_counter() - self.started_at, 3),
            "turn_count": len(self.turns),
            "clarification_count": self.clarification_count,
            "interrupt_count": self.interrupt_count,
            "turns": self.turns,
        }

    def flush(self) -> dict:
        record = self.as_dict()
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record


_CURRENT_SESSION: SessionMetrics | None = None


def get_session() -> SessionMetrics:
    global _CURRENT_SESSION
    if _CURRENT_SESSION is None:
        _CURRENT_SESSION = SessionMetrics()
    return _CURRENT_SESSION


def reset_session() -> SessionMetrics:
    global _CURRENT_SESSION
    _CURRENT_SESSION = SessionMetrics()
    return _CURRENT_SESSION
