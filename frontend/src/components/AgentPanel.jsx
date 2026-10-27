import { useState } from 'react'
import { useAgentSocket } from '../useAgentSocket'

const STATUS_LABELS = {
  connecting: 'connecting…',
  idle: 'idle',
  listening: '🎙️ listening',
  hearing: '👂 hearing you…',
  thinking: '🤔 thinking…',
  speaking: '🔊 speaking',
  disconnected: 'disconnected',
  error: 'error',
}

export default function AgentPanel() {
  const { messages, status, connected, micActive, micError, sendText, startMic, stopMic } = useAgentSocket()
  const [input, setInput] = useState('')

  const onSubmit = (e) => {
    e.preventDefault()
    if (sendText(input)) setInput('')
  }

  return (
    <aside className="agent-panel">
      <header className="agent-header">
        <div>
          <span className="agent-header-icon">🤖</span>
          <span className="agent-header-title">Agent</span>
        </div>
        <span className={`status-pill status-${connected ? status : 'disconnected'}`}>
          {connected ? (STATUS_LABELS[status] || status) : 'disconnected'}
        </span>
      </header>

      <div className="chat">
        {messages.length === 0 && (
          <div className="empty-state">
            Try &ldquo;what&rsquo;s on the table?&rdquo; or &ldquo;pick up the red block&rdquo; —
            type below, or start the mic and speak.
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`bubble bubble-${m.role}${m.interrupted ? ' bubble-interrupted' : ''}`}>
            {m.text}
          </div>
        ))}
      </div>

      <div className="mic-row">
        <button
          type="button"
          className={`mic-button ${micActive ? 'mic-active' : ''}`}
          onClick={micActive ? stopMic : startMic}
          disabled={!connected}
        >
          {micActive ? '🛑 Stop Mic' : '🎙️ Start Mic'}
        </button>
        {micError && <span className="mic-error">{micError}</span>}
      </div>
      {micActive && (
        <p className="mic-hint">Speak normally — talking again mid-response interrupts it.</p>
      )}

      <form className="input-row" onSubmit={onSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="...or type a command"
          disabled={!connected}
        />
        <button type="submit" disabled={!connected || !input.trim()}>Send</button>
      </form>
    </aside>
  )
}
