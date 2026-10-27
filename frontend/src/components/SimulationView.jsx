import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

export default function SimulationView() {
  // Bumping a query param forces the <img> to open a fresh connection to the
  // MJPEG endpoint — used for the manual "reconnect" affordance if the
  // stream ever drops (network hiccup, backend restart) without requiring a
  // full page reload.
  const [reloadKey, setReloadKey] = useState(0)
  const [loaded, setLoaded] = useState(false)
  const [errored, setErrored] = useState(false)

  const reconnect = () => {
    setErrored(false)
    setLoaded(false)
    setReloadKey((k) => k + 1)
  }

  return (
    <div className="sim-view">
      <div className="sim-view-header">
        <span className="sim-view-title">Live Simulation</span>
        <span className={`sim-view-dot ${loaded && !errored ? 'sim-view-dot-live' : ''}`} />
      </div>
      <div className="sim-view-frame">
        {!loaded && !errored && (
          <div className="sim-view-placeholder">Connecting to simulation feed…</div>
        )}
        {errored && (
          <div className="sim-view-placeholder sim-view-error">
            Lost connection to the simulation stream.
            <button type="button" onClick={reconnect}>Reconnect</button>
          </div>
        )}
        {/* Browsers render multipart/x-mixed-replace streams natively in <img> —
            no client-side JS needed for the video itself. */}
        <img
          key={reloadKey}
          src={`${API_URL}/sim/stream`}
          alt="Live MuJoCo simulation of the Franka Panda arm"
          className="sim-view-img"
          style={{ display: loaded && !errored ? 'block' : 'none' }}
          onLoad={() => setLoaded(true)}
          onError={() => setErrored(true)}
        />
      </div>
    </div>
  )
}
