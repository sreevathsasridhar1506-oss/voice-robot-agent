import './App.css'
import Dashboard from './components/Dashboard'
import AgentPanel from './components/AgentPanel'

export default function App() {
  return (
    <div className="app-shell">
      <div className="app-layout">
        <main className="main-section">
          <Dashboard />
        </main>
        <AgentPanel />
      </div>
    </div>
  )
}
