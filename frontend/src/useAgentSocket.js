import { useCallback, useEffect, useRef, useState } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://127.0.0.1:8000/ws'
const MIC_SAMPLE_RATE = 16000 // must match vad.py/stt.py's SAMPLE_RATE

// Sequential audio playback — sentence-chunked TTS sends one audio chunk per
// sentence; playing them back-to-back instead of overlapping keeps speech
// intelligible. clear() is what barge-in uses to cut off whatever's playing.
class AudioQueue {
  constructor() {
    this.queue = []
    this.playing = false
    this.currentAudio = null
  }
  push(blobUrl) {
    this.queue.push(blobUrl)
    if (!this.playing) this._playNext()
  }
  _playNext() {
    const url = this.queue.shift()
    if (!url) {
      this.playing = false
      return
    }
    this.playing = true
    const audio = new Audio(url)
    this.currentAudio = audio
    audio.onended = () => {
      URL.revokeObjectURL(url)
      this._playNext()
    }
    audio.onerror = () => this._playNext()
    audio.play().catch(() => this._playNext())
  }
  clear() {
    this.queue.forEach((url) => URL.revokeObjectURL(url))
    this.queue = []
    if (this.currentAudio) {
      this.currentAudio.pause()
      this.currentAudio = null
    }
    this.playing = false
  }
}

function floatTo16BitPCM(float32Array) {
  const int16 = new Int16Array(float32Array.length)
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]))
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
  }
  return int16
}

let nextId = 1

/**
 * Encapsulates the /ws connection, chat message state, and mic capture —
 * everything AgentPanel needs, kept out of the component tree so the UI
 * layer (App/Dashboard/AgentPanel) stays purely presentational.
 */
export function useAgentSocket() {
  const [messages, setMessages] = useState([])
  const [status, setStatus] = useState('connecting')
  const [connected, setConnected] = useState(false)
  const [micActive, setMicActive] = useState(false)
  const [micError, setMicError] = useState(null)

  const wsRef = useRef(null)
  const audioQueueRef = useRef(new AudioQueue())
  const currentAgentIdRef = useRef(null)
  const micRef = useRef(null)

  const appendMessage = useCallback((role, text) => {
    const id = nextId++
    setMessages((prev) => [...prev, { id, role, text }])
    return id
  }, [])

  const appendToAgentMessage = useCallback((chunk) => {
    if (currentAgentIdRef.current == null) {
      currentAgentIdRef.current = appendMessage('agent', chunk)
    } else {
      const id = currentAgentIdRef.current
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, text: `${m.text} ${chunk}` } : m))
      )
    }
  }, [appendMessage])

  useEffect(() => {
    const ws = new WebSocket(WS_URL)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      setStatus('idle')
    }
    ws.onclose = () => {
      setConnected(false)
      setStatus('disconnected')
    }
    ws.onerror = () => setStatus('error')

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const blob = new Blob([event.data], { type: 'audio/wav' })
        audioQueueRef.current.push(URL.createObjectURL(blob))
        return
      }

      const data = JSON.parse(event.data)
      switch (data.type) {
        case 'speech_started':
          setStatus('hearing')
          break
        case 'speech_ended':
          setStatus('thinking')
          break
        case 'transcript':
          currentAgentIdRef.current = null
          appendMessage('user', data.content)
          break
        case 'status':
          setStatus(data.content)
          break
        case 'answer_chunk':
          setStatus('speaking')
          appendToAgentMessage(data.content)
          break
        case 'barge_in':
          audioQueueRef.current.clear()
          if (currentAgentIdRef.current != null) {
            const id = currentAgentIdRef.current
            setMessages((prev) =>
              prev.map((m) => (m.id === id ? { ...m, text: `${m.text} [interrupted]`, interrupted: true } : m))
            )
          }
          currentAgentIdRef.current = null
          setStatus(micRef.current ? 'listening' : 'idle')
          break
        case 'done':
          currentAgentIdRef.current = null
          setStatus(micRef.current ? 'listening' : 'idle')
          break
        case 'error':
          appendMessage('error', data.content)
          setStatus(micRef.current ? 'listening' : 'idle')
          break
        default:
          break
      }
    }

    return () => ws.close()
  }, [appendMessage, appendToAgentMessage])

  const sendText = useCallback((text) => {
    const trimmed = text.trim()
    if (!trimmed || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return false
    wsRef.current.send(trimmed)
    return true
  }, [])

  const startMic = useCallback(async () => {
    setMicError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      })
      const audioContext = new AudioContext({ sampleRate: MIC_SAMPLE_RATE })
      await audioContext.audioWorklet.addModule('/pcm-worklet.js')
      const source = audioContext.createMediaStreamSource(stream)
      const workletNode = new AudioWorkletNode(audioContext, 'pcm-capture-processor')

      workletNode.port.onmessage = (event) => {
        const int16 = floatTo16BitPCM(event.data)
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(int16.buffer)
        }
      }
      // Deliberately not connected to audioContext.destination — capturing
      // the mic to send over the network, not to play it back locally.
      source.connect(workletNode)

      micRef.current = { stream, audioContext, workletNode }
      setMicActive(true)
      setStatus('listening')
    } catch (err) {
      setMicError(err.message || String(err))
    }
  }, [])

  const stopMic = useCallback(() => {
    if (micRef.current) {
      micRef.current.stream.getTracks().forEach((t) => t.stop())
      micRef.current.workletNode.port.onmessage = null
      micRef.current.audioContext.close()
      micRef.current = null
    }
    setMicActive(false)
    setStatus('idle')
  }, [])

  useEffect(() => () => stopMic(), [stopMic])

  return { messages, status, connected, micActive, micError, sendText, startMic, stopMic }
}
