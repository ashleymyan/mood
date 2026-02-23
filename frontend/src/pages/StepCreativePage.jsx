import { useNavigate } from 'react-router-dom'
import StepLayout from '../components/StepLayout'
import { useWizard } from '../context/WizardContext'

export default function StepCreativePage() {
  const navigate = useNavigate()
  const {
    imageA,
    imageB,
    direction,
    setDirection,
    blendStart,
    setBlendStart,
    blendEnd,
    setBlendEnd,
    draftCount,
    setDraftCount,
  } = useWizard()

  const missingRequiredImages = !(imageA && imageB)

  return (
    <StepLayout
      step={2}
      title="Step 2: Creative Controls"
      description="Set the style direction and how many poster drafts you want generated."
      actions={
        <>
          <button className="btn" onClick={() => navigate('/wizard/step-1')}>Back</button>
          <button
            className="btn btn-primary"
            disabled={missingRequiredImages}
            onClick={() => navigate('/wizard/step-3')}
          >
            Generate Images
          </button>
        </>
      }
    >
      <div className="form-grid">
        <label className="field field-full">
          <span>Creative direction</span>
          <textarea
            rows="4"
            placeholder="Example: Neo-noir thriller, heavy contrast, central silhouette, room for title at top"
            value={direction}
            onChange={(e) => setDirection(e.target.value)}
          />
        </label>

        <label className="field">
          <span>Blend start ({Number(blendStart).toFixed(1)})</span>
          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={blendStart}
            onChange={(e) => setBlendStart(Number(e.target.value))}
          />
        </label>

        <label className="field">
          <span>Blend end ({Number(blendEnd).toFixed(1)})</span>
          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={blendEnd}
            onChange={(e) => setBlendEnd(Number(e.target.value))}
          />
        </label>

        <label className="field">
          <span>Number of drafts</span>
          <input
            type="number"
            min="2"
            max="16"
            value={draftCount}
            onChange={(e) => setDraftCount(Math.min(16, Math.max(2, Number(e.target.value) || 2)))}
          />
        </label>
      </div>
    </StepLayout>
  )
}
