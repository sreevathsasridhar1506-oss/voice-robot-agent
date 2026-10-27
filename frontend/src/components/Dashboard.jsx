import { useEffect, useState } from 'react'
import SimulationView from './SimulationView'
import StatCard from './StatCard'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const TECH_STACK = [
  'MuJoCo', 'Franka Panda', 'LangGraph', 'Groq', 'Whisper (GPU)',
  'Piper TTS', 'Silero VAD', 'FastAPI', 'React',
]

const HRI_FEATURES = [
  { icon: '🎯', title: 'Grounding', text: 'Resolves ambiguous references — asks "which one?" when two blocks share a color, instead of guessing.' },
  { icon: '💬', title: 'Transparency', text: '"What are you doing?" is answered from live state, even mid-motion — never a canned response.' },
  { icon: '🛑', title: 'Mixed-initiative safety', text: 'Talk over the agent mid-response and it stops immediately (barge-in) — no wake word required.' },
]

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null)

  useEffect(() => {
    let cancelled = false
    const fetchMetrics = () => {
      fetch(`${API_URL}/metrics`)
        .then((r) => r.json())
        .then((data) => { if (!cancelled) setMetrics(data) })
        .catch(() => {})
    }
    fetchMetrics()
    const id = setInterval(fetchMetrics, 4000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  return (
    <div className="dashboard">
      <section className="hero-card">
        <p className="eyebrow">Human-Robot Interaction · Portfolio Project</p>
        <h2>Voice-Directed Collaborative Assembly Assistant</h2>
        <p className="hero-copy">
          A simulated Franka Panda robot arm, directed entirely by voice through an LLM agent,
          collaborating on a pick-and-place task in real time — designed around grounding,
          transparency, and safety, the core research themes of Human-Robot Interaction.
        </p>
        <div className="tech-pills">
          {TECH_STACK.map((t) => (
            <span key={t} className="tech-pill">{t}</span>
          ))}
        </div>
      </section>

      <SimulationView />

      <section className="stats-row">
        <StatCard label="Turns this session" value={metrics?.turn_count ?? '—'} />
        <StatCard label="Clarifications asked" value={metrics?.clarification_count ?? '—'} />
        <StatCard label="Interrupts handled" value={metrics?.interrupt_count ?? '—'} />
        <StatCard
          label="Session duration"
          value={metrics ? `${Math.round(metrics.total_duration_s)}s` : '—'}
        />
      </section>

      <section className="features-card">
        <h3>How this reads as HRI research, not just a demo</h3>
        <div className="features-grid">
          {HRI_FEATURES.map((f) => (
            <div className="feature-item" key={f.title}>
              <span className="feature-icon">{f.icon}</span>
              <div>
                <div className="feature-title">{f.title}</div>
                <p className="feature-text">{f.text}</p>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
