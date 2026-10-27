"""
Standalone check (M4): does the LLM select the right robot tool + arguments?

Mirrors the learning repo's test_groq_tools.py exactly: bind AGENT_TOOLS to a
bare ChatGroq and invoke a natural-language command, then inspect the
resulting tool_calls. No LangGraph agent loop, no need for the sim to even
be running — binding a tool does not execute it, only a ToolNode/agent loop
would actually call the underlying Python function.
"""

import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq

from robot_tools import AGENT_TOOLS

print("=" * 60)
print("Tool information")
print("=" * 60)
for t in AGENT_TOOLS:
    print(f"- {t.name}: {t.description.strip().splitlines()[0]}")

print("\n" + "=" * 60)
print("Creating LLM + binding tools...")
print("=" * 60)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(AGENT_TOOLS)
print("Bound successfully!")

TEST_COMMANDS = [
    "What's on the table right now?",
    "Pick up the red block.",
    "Put it down on the left side.",
    "Stop!",
]

for cmd in TEST_COMMANDS:
    print("\n" + "=" * 60)
    print(f"Command: {cmd!r}")
    print("=" * 60)
    response = llm_with_tools.invoke(cmd)
    if not response.tool_calls:
        print("  (no tool call — model answered directly)")
        print("  content:", response.content)
        continue
    for call in response.tool_calls:
        print(f"  tool={call['name']}  args={call['args']}")

print("\nSUCCESS!")
