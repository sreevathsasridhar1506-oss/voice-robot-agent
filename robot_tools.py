"""
robot_tools.py
==============
LangChain tool wrappers over sim_controller.py's SimController.

Mirrors the learning repo's mcp_config.py pattern: @tool-decorated functions
exported as AGENT_TOOLS for the LangGraph agent to bind.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from langchain_core.tools import tool

from metrics import get_session
from sim_controller import BLOCK_COLORS, BLOCK_NAMES, SimController

logger = logging.getLogger(__name__)

_CONTROLLER: Optional[SimController] = None
_CONTROLLER_LOCK = threading.Lock()

# The native OpenCV window is redundant once the React frontend embeds the
# same feed via /sim/stream — default it off so starting the API server
# doesn't pop up a second, confusing view of the same simulation. Standalone
# CLI scripts (cli_text_agent.py etc.) have no browser involved at all, so
# set SIM_SHOW_WINDOW=true in .env if you're using those without the frontend.
SHOW_NATIVE_WINDOW = os.getenv("SIM_SHOW_WINDOW", "false").strip().lower() == "true"

# Named drop zones instead of raw coordinates — LLMs are unreliable at
# producing precise floats for spatial tasks, so place_object takes a
# category the model can pick from instead.
DROP_ZONES = {
    "left": [0.55, 0.25, 0.03],
    "right": [0.55, -0.25, 0.03],
    "center": [0.6, 0.0, 0.03],
}

APPROACH_HEIGHT = 0.15


def get_sim_controller() -> SimController:
    """Lazy-loaded singleton — mirrors get_whisper_model()/get_agent_graph()
    in the learning repo's voice_agent.py.

    Double-checked locking: barge-in (M14) is what actually exposed the need
    for this — _barge_in() calls this directly from the event-loop thread
    while a tool call in the streaming producer thread can call it at nearly
    the same instant. Without the lock, both threads can see _CONTROLLER as
    None simultaneously and each construct their own SimController (each
    spawning its own MuJoCo worker thread + GL context) — confirmed by
    "Starting simulation worker thread" logging three times in one run,
    followed by the connection dying from the resulting GL/resource
    contention between the orphaned duplicate instances."""
    global _CONTROLLER
    if _CONTROLLER is None:
        with _CONTROLLER_LOCK:
            if _CONTROLLER is None:
                logger.info("Starting simulation worker thread (native window: %s)...", SHOW_NATIVE_WINDOW)
                _CONTROLLER = SimController(show=SHOW_NATIVE_WINDOW)
    return _CONTROLLER


def _matches_for_color(color_or_name: str) -> list[str]:
    """Accepts either a full body name ('block_red') or a bare color ('red').
    Returns every matching object — may be more than one (e.g. two red blocks),
    which is the deliberate disambiguation test case."""
    name = color_or_name.strip().lower()
    if name in BLOCK_NAMES:
        return [name]
    return [n for n, c in BLOCK_COLORS.items() if c == name]


def _side_hint(sc: SimController, name: str, siblings: list[str]) -> str:
    """Describe `name` relative to same-colored `siblings` using the same
    left/right vocabulary as place_object's zones, so the agent and user can
    talk about disambiguation in the same terms as placement."""
    positions = {o["name"]: o["position"] for o in sc.get_scene_state()["objects"]}
    y = positions[name][1]
    other_ys = [positions[s][1] for s in siblings if s != name]
    if all(y > oy for oy in other_ys):
        return "left"
    if all(y < oy for oy in other_ys):
        return "right"
    return "middle"


def _resolve_with_disambiguation(sc: SimController, color: str, which: str) -> tuple[Optional[str], Optional[str]]:
    """Returns (object_name, clarifying_message). Exactly one is None."""
    matches = _matches_for_color(color)
    if not matches:
        return None, f"I don't see a '{color}' object on the table."
    if len(matches) == 1:
        return matches[0], None
    if which:
        which_key = which.strip().lower()
        hinted = [m for m in matches if _side_hint(sc, m, matches) == which_key]
        if len(hinted) == 1:
            return hinted[0], None
        logger.warning(
            "Disambiguation needed: '%s' still ambiguous among %s even with which=%r",
            color, matches, which,
        )
        return None, f"I still can't tell which {color} block you mean by '{which}'."
    hints = ", ".join(_side_hint(sc, m, matches) for m in matches)
    # This log line is the structured, code-level signal that the OBJECT
    # REFERENCE itself was ambiguous — used by eval_disambiguation.py instead
    # of scanning the agent's spoken reply for a "?", which false-positives on
    # unrelated follow-up questions like "would you like me to place it now?"
    # (see research_methodology_exp.md for the specific case this fixed).
    logger.warning("Disambiguation needed: '%s' has %d matches (%s)", color, len(matches), hints)
    return None, (
        f"There are {len(matches)} {color} blocks on the table ({hints}). Which one do you mean?"
    )


@tool
def describe_scene() -> str:
    """
    Look at the table and report what objects are there, their colors and
    positions, whether anything is currently being held, and what the arm
    is doing right now.

    Returns:
        A human-readable description of the current scene.
    """
    sc = get_sim_controller()
    state = sc.get_scene_state()
    lines = ["Objects on the table:"]
    for obj in state["objects"]:
        x, y, z = obj["position"]
        lines.append(f"- {obj['color']} block at ({x:.2f}, {y:.2f}, {z:.2f})")
    if state["held_object"]:
        lines.append(f"Currently holding: {BLOCK_COLORS[state['held_object']]} block")
    else:
        lines.append("Not currently holding anything.")
    lines.append(f"Current action: {state['current_action']}")
    return "\n".join(lines)


@tool
def move_to_object(color: str, which: str = "") -> str:
    """
    Move the end-effector to hover above the named object WITHOUT picking it
    up. Useful for pointing at or indicating an object.

    Args:
        color: The object's color (e.g. "red", "blue", "green", "yellow").
        which: If there's more than one object of that color, which one —
               "left" or "right". Leave empty if there's only one match.
    """
    sc = get_sim_controller()
    object_name, clarification = _resolve_with_disambiguation(sc, color, which)
    if clarification:
        return clarification
    state = sc.get_scene_state()
    obj = next(o for o in state["objects"] if o["name"] == object_name)
    x, y, z = obj["position"]
    result = sc.move_to([x, y, z + APPROACH_HEIGHT])
    if not result["completed"]:
        return f"Movement toward the {color} block was interrupted."
    return f"Moved to hover above the {color} block."


@tool
def pick_object(color: str, which: str = "") -> str:
    """
    Pick up the named object from the table.

    Args:
        color: The object's color (e.g. "red", "blue", "green", "yellow").
        which: If there's more than one object of that color, which one —
               "left" or "right". Leave empty if there's only one match; if
               there IS more than one and this is empty, you'll get back a
               clarifying question to relay to the user instead of a pick.
    """
    sc = get_sim_controller()
    object_name, clarification = _resolve_with_disambiguation(sc, color, which)
    if clarification:
        return clarification
    result = sc.pick(object_name)
    if not result.get("completed"):
        return result.get("error", "Pick was interrupted before completing.")
    return f"Picked up the {color} block."


@tool
def place_object(zone: str) -> str:
    """
    Place the currently-held object at a named location on the table.

    Args:
        zone: One of "left", "right", or "center".
    """
    sc = get_sim_controller()
    zone_key = zone.strip().lower()
    if zone_key not in DROP_ZONES:
        return f"'{zone}' isn't a location I know — use left, right, or center."
    result = sc.place(DROP_ZONES[zone_key])
    if not result.get("completed"):
        return result.get("error", "Place was interrupted before completing.")
    return f"Placed the block at the {zone_key}."


@tool
def stop_arm() -> str:
    """
    Immediately halt any in-progress arm motion. Use this if asked to stop,
    wait, or freeze.
    """
    sc = get_sim_controller()
    sc.stop()
    get_session().record_interrupt()
    return "Stopped."


AGENT_TOOLS = [describe_scene, move_to_object, pick_object, place_object, stop_arm]
