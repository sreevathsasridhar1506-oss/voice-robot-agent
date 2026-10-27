// Smoke test (M15): confirm a JS WebSocket client (Node's built-in, same
// API as the browser) can talk to the backend the way App.jsx does — a
// different client implementation than the Python tests already used
// elsewhere, catching any client-library-specific protocol quirks.
const ws = new WebSocket("ws://127.0.0.1:8000/ws");
ws.binaryType = "arraybuffer"; // matches useAgentSocket.js — Node defaults to
                                // Blob otherwise, unlike what App.jsx actually uses

const events = [];
let audioChunks = 0;

ws.onopen = () => {
  console.log("connected, sending text command...");
  ws.send("What's on the table?");
};

ws.onmessage = (event) => {
  if (event.data instanceof ArrayBuffer) {
    audioChunks++;
    console.log(`<- [binary audio chunk, ${event.data.byteLength} bytes]`);
    return;
  }
  const data = JSON.parse(event.data);
  events.push(data);
  console.log("<-", data);
  if (data.type === "done") {
    const types = events.map((e) => e.type);
    const ok =
      types.includes("status") &&
      types.includes("transcript") &&
      types.includes("answer_chunk") &&
      audioChunks > 0;
    console.log(ok ? "\nSUCCESS" : "\nFAILED");
    ws.close();
    process.exit(ok ? 0 : 1);
  }
};

ws.onerror = (err) => {
  console.error("WS error:", err.message || err);
  process.exit(1);
};

setTimeout(() => {
  console.error("Timed out waiting for done");
  process.exit(1);
}, 30000);
