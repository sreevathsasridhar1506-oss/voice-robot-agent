// pcm-worklet.js
//
// AudioWorkletProcessor: runs on the audio rendering thread, receives raw
// mic samples in fixed 128-sample render quanta, and forwards each block to
// the main thread unchanged (as Float32). Chosen over MediaRecorder because
// MediaRecorder only produces compressed containers (WebM/Opus) — this
// machine has no ffmpeg on PATH to decode that server-side, and the backend
// (ws_session.py) already expects raw PCM16 samples directly, re-framing
// whatever chunk size arrives into its own fixed windows internally, so this
// side doesn't need to buffer into larger blocks before sending.
class PCMCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]
    if (input && input[0] && input[0].length > 0) {
      // .slice() copies out of the engine's reused buffer before posting —
      // the original Float32Array is only valid for this one process() call.
      this.port.postMessage(input[0].slice())
    }
    return true
  }
}

registerProcessor('pcm-capture-processor', PCMCaptureProcessor)
