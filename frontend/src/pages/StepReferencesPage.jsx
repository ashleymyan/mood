import { useNavigate } from 'react-router-dom'
import StepLayout from '../components/StepLayout'
import { useWizard } from '../context/WizardContext'

function previewList(images) {
  if (!images?.length) return null
  return (
    <div className="thumb-grid">
      {images.map((src, idx) => (
        <img key={`${src}-${idx}`} src={src} alt={`upload ${idx + 1}`} className="thumb" />
      ))}
    </div>
  )
}

async function filesToDataUrls(fileList) {
  const files = Array.from(fileList || [])
  const toDataUrl = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result)
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  return Promise.all(files.map(toDataUrl))
}

export default function StepReferencesPage() {
  const navigate = useNavigate()
  const {
    imageA,
    setImageA,
    imageB,
    setImageB,
    inspirationImages,
    setInspirationImages,
    avoidImages,
    setAvoidImages,
  } = useWizard()

  const canContinue = Boolean(imageA && imageB)

  return (
    <StepLayout
      step={1}
      title="Step 1: Add Reference Images"
      description="Upload two main reference images, then add optional inspiration images and looks to avoid."
      actions={
        <>
          <button className="btn" onClick={() => navigate('/')}>Back</button>
          <button className="btn btn-primary" disabled={!canContinue} onClick={() => navigate('/wizard/step-2')}>
            Continue
          </button>
        </>
      }
    >
      <div className="form-grid">
        <label className="field">
          <span>Main reference image A</span>
          <input
            type="file"
            accept="image/*"
            onChange={async (e) => {
              const [url] = await filesToDataUrls(e.target.files)
              setImageA(url || null)
            }}
          />
          {imageA ? <img src={imageA} alt="reference A" className="preview" /> : null}
        </label>

        <label className="field">
          <span>Main reference image B</span>
          <input
            type="file"
            accept="image/*"
            onChange={async (e) => {
              const [url] = await filesToDataUrls(e.target.files)
              setImageB(url || null)
            }}
          />
          {imageB ? <img src={imageB} alt="reference B" className="preview" /> : null}
        </label>

        <label className="field">
          <span>More inspiration images (optional)</span>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={async (e) => setInspirationImages(await filesToDataUrls(e.target.files))}
          />
          {previewList(inspirationImages)}
        </label>

        <label className="field">
          <span>Looks to avoid (optional)</span>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={async (e) => setAvoidImages(await filesToDataUrls(e.target.files))}
          />
          {previewList(avoidImages)}
        </label>
      </div>
    </StepLayout>
  )
}
