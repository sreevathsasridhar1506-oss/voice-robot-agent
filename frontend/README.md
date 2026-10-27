# Frontend

React + Vite single-page UI for the voice robot agent. It streams microphone audio to the backend's `/ws` WebSocket and plays back the agent's spoken replies as they arrive.

## What's here

| File | Role |
|---|---|
| `public/pcm-worklet.js` | AudioWorklet that forwards raw mic samples (128-sample blocks) off the audio thread |
| `src/useAgentSocket.js` | WebSocket hook: opens the mic in a 16 kHz `AudioContext`, converts samples to PCM16 and streams them; handles `transcript` / `answer_chunk` / audio / `barge_in` events, and clears queued playback the moment a barge-in arrives |
| `src/components/AgentPanel.jsx` | Chat transcript, mic toggle and text-command input |
| `src/components/SimulationView.jsx` | Live view of the MuJoCo simulation, streamed from the backend's `/sim/stream` |
| `src/components/Dashboard.jsx`, `StatCard.jsx` | Project overview and live session metrics (turns, clarifications, interrupts) polled from `/metrics` |
| `smoke_test_ws.mjs` | Node script that checks the WebSocket protocol end to end against a running backend |

## Running

The backend must be running first (see the [main README](../README.md)).

```bash
npm install
npm run dev      # http://localhost:5173
```

Other commands:

```bash
npm run lint             # oxlint
npm run build            # production build into dist/
node smoke_test_ws.mjs   # protocol check, needs the backend running
```

The mic needs a secure context, so open the page on `localhost`, not a LAN IP.
