import { useNavigate } from 'react-router-dom'

export default function HomePage() {
  const navigate = useNavigate()

  return (
    <main className="page-shell home">
      <section className="hero-card">
        <div className="brand">mosaic</div>
        <h1>Film Poster Blend Studio</h1>
        <p>
          Combine reference images into poster concept drafts for your film. Mosaic helps you explore
          tone, color, and composition before full design production.
        </p>
        <button className="btn btn-primary" onClick={() => navigate('/wizard/step-1')}>
          Get Started
        </button>
      </section>
    </main>
  )
}
