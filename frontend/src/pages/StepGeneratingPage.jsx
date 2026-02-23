import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import StepLayout from '../components/StepLayout'
import { useWizard } from '../context/WizardContext'

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = reject
    image.src = src
  })
}

async function createBlendedDrafts(imageA, imageB, count, blendStart, blendEnd) {
  const [a, b] = await Promise.all([loadImage(imageA), loadImage(imageB)])
  const width = 768
  const height = 1152

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')

  const drawCover = (img) => {
    const imageRatio = img.width / img.height
    const canvasRatio = width / height
    let drawWidth = width
    let drawHeight = height
    let x = 0
    let y = 0

    if (imageRatio > canvasRatio) {
      drawWidth = height * imageRatio
      x = (width - drawWidth) / 2
    } else {
      drawHeight = width / imageRatio
      y = (height - drawHeight) / 2
    }

    ctx.drawImage(img, x, y, drawWidth, drawHeight)
  }

  const drafts = []
  for (let i = 0; i < count; i += 1) {
    const ratio = count === 1 ? 0.5 : i / (count - 1)
    const blendWeight = clamp(blendStart + (blendEnd - blendStart) * ratio, 0, 1)

    ctx.clearRect(0, 0, width, height)
    ctx.globalAlpha = 1
    drawCover(a)
    ctx.globalAlpha = blendWeight
    drawCover(b)

    ctx.globalAlpha = 0.14
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, width, height)

    ctx.globalAlpha = 1
    drafts.push(canvas.toDataURL('image/jpeg', 0.92))
  }

  return drafts
}

export default function StepGeneratingPage() {
  const navigate = useNavigate()
  const {
    imageA,
    imageB,
    direction,
    draftCount,
    blendStart,
    blendEnd,
    isGenerating,
    setIsGenerating,
    generatedImages,
    setGeneratedImages,
  } = useWizard()

  useEffect(() => {
    if (!imageA || !imageB) {
      navigate('/wizard/step-1', { replace: true })
      return
    }

    let active = true
    async function run() {
      setIsGenerating(true)
      try {
        const drafts = await createBlendedDrafts(imageA, imageB, draftCount, blendStart, blendEnd)
        if (active) {
          setGeneratedImages(drafts)
        }
      } finally {
        if (active) {
          setIsGenerating(false)
        }
      }
    }

    run()
    return () => {
      active = false
    }
  }, [imageA, imageB, draftCount, blendStart, blendEnd, navigate, setGeneratedImages, setIsGenerating])

  return (
    <StepLayout
      step={3}
      title="Step 3: Images Being Generated"
      description="Your poster drafts are being rendered below."
      actions={
        <>
          <button className="btn" onClick={() => navigate('/wizard/step-2')}>Back</button>
          <button className="btn btn-primary" onClick={() => navigate('/wizard/step-1')}>
            Start New Blend
          </button>
        </>
      }
    >
      {direction ? <p className="direction-note"><strong>Direction:</strong> {direction}</p> : null}
      {isGenerating ? <p className="status-pill">Generating poster concepts...</p> : null}
      {!isGenerating && generatedImages.length === 0 ? <p>No drafts generated yet.</p> : null}
      <div className="masonry-grid">
        {generatedImages.map((src, idx) => (
          <img key={`${idx}-${src.slice(0, 32)}`} src={src} alt={`Generated draft ${idx + 1}`} className="result-image" />
        ))}
      </div>
    </StepLayout>
  )
}
