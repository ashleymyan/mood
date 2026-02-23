export default function StepProgress({ currentStep, totalSteps = 3 }) {
  return (
    <div className="step-progress" aria-label="Progress">
      {Array.from({ length: totalSteps }, (_, idx) => {
        const step = idx + 1
        const state = step < currentStep ? 'done' : step === currentStep ? 'active' : 'todo'
        return (
          <div key={step} className={`step-dot ${state}`}>
            {step}
          </div>
        )
      })}
    </div>
  )
}
