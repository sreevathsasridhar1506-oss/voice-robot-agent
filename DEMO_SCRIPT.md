# Demo Script

A scripted interaction of about two minutes that shows the system's key HRI behaviors: grounding/disambiguation, transparency, and mixed-initiative safety (barge-in).

## Setup before recording

1. Start the backend: `python -m uvicorn api:app --host 127.0.0.1 --port 8000` (with the venv's Python).
2. Start the frontend: `cd frontend && npm run dev`, then open `http://localhost:5173`. The page shows the live simulation next to the chat, so recording the browser window captures both.
3. Click **Start Mic**.
4. If using Groq, check you have quota left: don't run the full test suite right beforehand. With `LLM_PROVIDER=ollama` there's no quota to worry about.

## The script

**1. Scene awareness**
> "What's on the table?"

Expect: the agent lists all 5 blocks (two red, one each of blue, green and yellow) by color and position, and says it isn't holding anything. This shows the agent reads real state, not a script.

**2. Grounding / disambiguation**
> "Pick up the red block."

Expect: since there are two red blocks, the agent asks a clarifying question instead of guessing, e.g. "There are two red blocks, one on the left and one on the right. Which one?"

> "The red one on the left."

Expect: it picks up the correct one. The agent doesn't carry memory between turns, so the reply names the object again ("the red one") rather than just "the left one". The point of this step: when a reference is ambiguous, the agent asks rather than guesses, the way a careful collaborator would.

**3. Transparency mid-action**
> "Put it on the right side."

While the arm is moving, ask:
> "What are you doing right now?"

Expect: an accurate answer describing the action in progress, not a canned "I'm idle". This comes from `SimController.get_current_action()`'s non-blocking status read, checked in `scripts/test_hri_behaviors.py`.

**4. Barge-in (mixed-initiative safety)**
> "Pick up the blue block and put it in the center."

While it's responding (you hear it start speaking, or see the arm moving), **talk over it**:
> "Wait, stop."

Expect: the audio cuts off immediately, the arm halts, and the UI marks the interrupted message. It then handles "wait, stop" as a new command and confirms.

**5. Wrap-up**
> "Thanks, that's all for now."

## After recording

The session's interaction metrics (turns, clarifications, interrupts, per-turn latency) for the exact session you recorded:

```bash
curl http://127.0.0.1:8000/metrics
```
