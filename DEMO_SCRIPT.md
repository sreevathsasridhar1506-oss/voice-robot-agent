# Demo Script

A ~2-minute scripted interaction hitting the project's key HRI points — grounding/disambiguation, transparency, and mixed-initiative safety (barge-in) — for recording as part of a Masters application.

## Setup before recording

1. Start the backend: `uvicorn api:app --host 127.0.0.1 --port 8000` (leave its OpenCV sim window visible — screen-record that window, or the React UI, or both side by side).
2. Start the frontend: `cd frontend && npm run dev`, open `http://localhost:5173`.
3. Click **Start Mic**.
4. Have a stable network connection (Groq API) and check you have quota headroom (`GET /metrics` won't tell you this — just don't run the full test suite immediately beforehand).

## The script

**1. Opening / scene awareness**
> "What's on the table?"

Expect: the agent lists all 5 blocks (two red, one each of blue/green/yellow) by color and position, and confirms it's not holding anything. *(Establishes the agent perceives real state, not a script.)*

**2. Grounding / disambiguation — the key HRI moment**
> "Pick up the red block."

Expect: since there are two red blocks, the agent asks a clarifying question instead of guessing — e.g. "There are two red blocks — one on the left, one on the right. Which one?"

> "The one on the left."

Expect: it picks up the correct one. *(This is the moment to narrate in your application: the agent doesn't guess when a reference is ambiguous — it asks, the way a careful collaborator would.)*

**3. Transparency mid-action**
> "Put it on the right side."

While the arm is mid-motion, ask:
> "What are you doing right now?"

Expect: an accurate answer reflecting the actual in-progress action (not a canned "I'm idle" response) — this is `SimController.get_current_action()`'s non-blocking status read, proven in `test_hri_behaviors.py`.

**4. Barge-in — mixed-initiative safety**
> "Pick up the blue block and put it in the center."

While it's mid-response (you'll hear it start speaking, or see the arm moving), **talk over it**:
> "Wait, stop."

Expect: the audio cuts off immediately, the arm halts, and the UI marks the interrupted message. Then it processes "wait, stop" as a fresh command and confirms.

**5. Wrap-up**
> "Thanks, that's all for now."

## After recording

Grab the session's interaction metrics for your application write-up:
```powershell
curl http://127.0.0.1:8000/metrics
```
This gives you real numbers — turn count, clarification count, interrupt count, per-turn latency — from the exact session you just recorded.
