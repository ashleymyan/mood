import StepProgress from './StepProgress'

export default function StepLayout({ step, title, description, children, actions }) {
  return (
    <main className="page-shell">
      <section className="card">
        <div className="brand">mosaic</div>
        <StepProgress currentStep={step} />
        <h1>{title}</h1>
        <p className="subtitle">{description}</p>
        <div className="content">{children}</div>
        <div className="actions">{actions}</div>
      </section>
    </main>
  )
}
