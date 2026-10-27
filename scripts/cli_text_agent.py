"""
Interactive CLI (M5): type a command, watch the arm move, read the agent's reply.

    python scripts/cli_text_agent.py

Type commands like "pick up the red block" or "what's on the table?".
Ctrl+C to exit.
"""

import sys

sys.path.insert(0, ".")

from voice_agent import process_text_query

def main():
    print("Robot-arm text agent — type a command (Ctrl+C to quit).")
    print("Try: 'what's on the table?' / 'pick up the red block' / 'put it on the left' / 'stop'")
    while True:
        try:
            query = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not query:
            continue
        result = process_text_query(query)
        print(f"Agent: {result['answer']}")


if __name__ == "__main__":
    main()
