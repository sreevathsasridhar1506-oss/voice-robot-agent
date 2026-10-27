"""
voice_agent.py
==============
LangGraph-powered robot-control agent.

Pipeline (text-only for now — audio arrives in M8-M10)
-------------------------------------------------------
Text query ──► LangGraph ReAct Agent ──► robot_tools.py ──► sim_controller.py
                     │
             LLM: langchain_groq.ChatGroq

Mirrors the learning repo's voice_agent.py build_llm()/build_agent_graph()/
process_text_query() pattern, including the retry-with-temperature-bump loop
that works around Groq/Llama occasionally emitting a malformed textual
function call instead of a structured tool call (400 tool_use_failed) — a
robot-control tool call failing this way is worse than a search failing, so
this matters even more here.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Iterator

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from metrics import get_session
from robot_tools import AGENT_TOOLS
from stt import transcribe_audio
from tts import text_to_speech

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# LLM_PROVIDER=ollama runs entirely locally (no external API, no rate limits)
# — added specifically to get valid, uncontaminated latency measurements for
# the INDICON evaluation after Groq's free-tier per-minute rate limit proved
# stricter than any pacing scheme could reliably absorb (see
# scripts/eval_latency_offline.py's eval_llm() for the full story). Also a
# legitimate deployment option in its own right: no per-request cost, no
# dependency on a third-party API being reachable/within quota.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = """You are a helpful voice assistant controlling a real robot arm \
(currently a physics simulation) that can pick up and place colored blocks on a table.

You have tools to look at the table (describe_scene), hover over an object without \
picking it up (move_to_object), pick up an object (pick_object), place whatever \
you're holding at a named location (place_object: "left", "right", or "center"), \
and stop immediately (stop_arm).

Rules:
1. If a command is ambiguous given the current scene — e.g. the object referenced \
isn't clear, or there's more than one plausible match (there can be more than one \
block of the same color) — ask a clarifying question instead of guessing. \
move_to_object/pick_object accept an optional "which" argument ("left"/"right") for \
exactly this situation: if a tool comes back telling you there are multiple matches, \
relay that clarifying question to the user in your own words rather than picking one \
yourself. If the user's phrasing already specifies which one (e.g. "the one on the \
left"), pass that directly as the "which" argument instead of asking.
2. Always narrate what you did in plain, conversational language after acting — the \
person can't see your tool calls, only your spoken response.
3. If a command doesn't make sense given the current state (e.g. asked to pick up \
something that isn't on the table, or place something down while not holding \
anything), say so honestly rather than pretending to comply.
4. If asked what you're doing, why you stopped, or for the current state, answer \
based on the ACTUAL current state (check describe_scene if unsure) — never make \
something up.
5. Keep responses concise and conversational, since this will be spoken aloud.
"""


def build_llm(temperature: float = 0.3):
    """Instantiate and return the configured LLM (Groq-hosted or local Ollama,
    per LLM_PROVIDER). Both support LangChain tool-calling via .bind_tools(),
    which create_react_agent relies on — the rest of the agent code is
    provider-agnostic."""
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
        )

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file.\n"
            "Get a free key at https://console.groq.com"
        )
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=temperature,
        max_tokens=1024,
    )


def build_agent_graph(temperature: float = 0.3):
    """Build a LangGraph ReAct agent bound to the robot-control tools."""
    llm = build_llm(temperature=temperature)
    return create_react_agent(llm, AGENT_TOOLS, prompt=SYSTEM_PROMPT)


_AGENT_GRAPH = None
_AGENT_GRAPH_LOCK = threading.Lock()


def get_agent_graph():
    """Double-checked locking — see robot_tools.get_sim_controller()'s
    docstring for why this matters now: multiple WS sessions' producer
    threads (M13/M14) can call this concurrently on first use."""
    global _AGENT_GRAPH
    if _AGENT_GRAPH is None:
        with _AGENT_GRAPH_LOCK:
            if _AGENT_GRAPH is None:
                _AGENT_GRAPH = build_agent_graph()
    return _AGENT_GRAPH


def process_text_query(query: str) -> dict[str, Any]:
    """
    Run the LangGraph agent on a plain-text command.

    Returns
    -------
    dict with keys:
        query  – original command
        answer – the agent's spoken/text response
    """
    agent = get_agent_graph()
    date_context = SystemMessage(
        content=f"Today's date is {datetime.now():%A, %B %d, %Y}."
    )
    initial_state = {"messages": [date_context, HumanMessage(content=query)]}

    logger.info("Running agent for command: %r", query)
    t0 = time.perf_counter()

    # See learning repo's voice_agent.py for the full rationale — Groq's
    # tool-calling models occasionally emit a malformed function call as
    # plain text instead of a structured tool call (400 tool_use_failed).
    # Retrying at a bumped temperature reliably breaks out of it.
    max_attempts = 4
    final_state = None
    current_agent = agent
    for attempt in range(1, max_attempts + 1):
        try:
            final_state = current_agent.invoke(initial_state)
            break
        except Exception as exc:
            transient = "tool_use_failed" in str(exc) or "Failed to call a function" in str(exc)
            if transient and attempt < max_attempts:
                bumped_temperature = min(0.9, 0.3 + 0.2 * attempt)
                logger.warning(
                    "Tool-call generation glitch (attempt %d/%d), retrying at temperature=%.1f: %s",
                    attempt, max_attempts, bumped_temperature, exc,
                )
                current_agent = build_agent_graph(temperature=bumped_temperature)
                continue
            raise

    elapsed = time.perf_counter() - t0
    logger.info("Agent finished in %.2fs", elapsed)

    answer = ""
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not answer:
        answer = "Sorry, I couldn't generate a response."

    get_session().record_turn(query, answer, elapsed)

    return {"query": query, "answer": answer}


def process_audio_query(audio_bytes: bytes, language: str = "en") -> dict[str, Any]:
    """
    Full batch pipeline: audio in -> STT -> agent -> TTS -> audio out.

    Still fully batch/blocking (no streaming yet — that's M11-M13) but this
    is the "everything works, just not fast yet" checkpoint the streaming
    work gets built on top of.

    Returns
    -------
    dict with keys: query, transcript, answer, audio (WAV bytes)
    """
    transcript = transcribe_audio(audio_bytes, language=language)
    if not transcript:
        answer = "I couldn't understand the audio. Please try again."
        return {
            "query": "",
            "transcript": "",
            "answer": answer,
            "audio": text_to_speech(answer),
        }

    result = process_text_query(transcript)
    result["transcript"] = transcript
    result["audio"] = text_to_speech(result["answer"])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# M13: sentence-chunked streaming
# ──────────────────────────────────────────────────────────────────────────────

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

# The known malformed-tool-call failure mode (see the retry loop above) emits
# a literal string starting like this instead of a structured tool call. If
# streamed content starts this way, it's not a real answer — abort and fall
# back rather than speaking it.
_MALFORMED_PREFIXES = ("<function", "<|")


def _split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Split into complete sentences + a trailing partial tail. A sentence
    only counts as "complete" once whitespace follows its ending punctuation
    (i.e. more text came after it) — the final piece is always kept as the
    tail since we can't yet tell if it's finished."""
    parts = _SENTENCE_BOUNDARY_RE.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    complete, tail = parts[:-1], parts[-1]
    return [p.strip() for p in complete if p.strip()], tail


def stream_text_query_sentences(query: str) -> Iterator[str]:
    """
    Generator: yields complete sentences of the agent's answer as soon as
    they're ready, instead of waiting for the full response — the point is
    to let a caller start synthesizing/speaking sentence 1 while sentence 2
    is still being generated.

    Important interaction with the retry logic in process_text_query(): only
    text-content chunks belonging to a "final answer" LLM turn are ever
    considered speakable. Empirically (verified via direct inspection of
    agent.stream(..., stream_mode="messages")), Groq/OpenAI-style streaming
    never interleaves real text content with tool_call_chunks on the same
    message id — a turn that's deciding to call a tool streams empty content
    with populated tool_call_chunks; a turn producing the final answer
    streams real content with empty tool_call_chunks. So skipping any chunk
    with non-empty tool_call_chunks is sufficient to avoid ever speaking a
    tool-call-in-progress.

    That still leaves the malformed-tool-call failure mode (the model emits
    the tool call as literal text instead of using tool_call_chunks at all)
    — checked for explicitly via _MALFORMED_PREFIXES. If detected, or if
    anything else goes wrong mid-stream, this falls back to the non-streaming
    process_text_query() (which has the actual retry-with-temperature-bump
    logic) rather than risk speaking a broken partial response.
    """
    agent = get_agent_graph()
    date_context = SystemMessage(content=f"Today's date is {datetime.now():%A, %B %d, %Y}.")
    initial_state = {"messages": [date_context, HumanMessage(content=query)]}

    buffer = ""
    full_answer = ""
    current_id = None
    t0 = time.perf_counter()

    try:
        for chunk, metadata in agent.stream(initial_state, stream_mode="messages"):
            if metadata.get("langgraph_node") != "agent":
                continue
            if getattr(chunk, "tool_call_chunks", None):
                continue  # this turn is calling a tool, not speaking — not speakable content
            if not chunk.content:
                continue

            if chunk.id != current_id:
                current_id = chunk.id
                buffer = ""

            buffer += chunk.content
            full_answer += chunk.content

            if buffer.lstrip().startswith(_MALFORMED_PREFIXES):
                raise RuntimeError("Detected malformed tool-call-like content mid-stream.")

            sentences, buffer = _split_complete_sentences(buffer)
            yield from sentences

        if buffer.strip():
            yield buffer.strip()

        elapsed = time.perf_counter() - t0
        get_session().record_turn(query, full_answer.strip(), elapsed)

    except Exception as exc:
        logger.warning("Streaming aborted (%s) — falling back to non-streaming process_text_query.", exc)
        result = process_text_query(query)  # records its own metrics turn
        yield result["answer"]
