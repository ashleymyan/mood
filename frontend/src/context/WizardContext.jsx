import { createContext, useContext, useMemo, useState } from 'react'

const WizardContext = createContext(null)

export function WizardProvider({ children }) {
  const [imageA, setImageA] = useState(null)
  const [imageB, setImageB] = useState(null)
  const [inspirationImages, setInspirationImages] = useState([])
  const [avoidImages, setAvoidImages] = useState([])
  const [direction, setDirection] = useState('')
  const [blendStart, setBlendStart] = useState(0.0)
  const [blendEnd, setBlendEnd] = useState(1.0)
  const [draftCount, setDraftCount] = useState(8)
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedImages, setGeneratedImages] = useState([])

  const value = useMemo(
    () => ({
      imageA,
      setImageA,
      imageB,
      setImageB,
      inspirationImages,
      setInspirationImages,
      avoidImages,
      setAvoidImages,
      direction,
      setDirection,
      blendStart,
      setBlendStart,
      blendEnd,
      setBlendEnd,
      draftCount,
      setDraftCount,
      isGenerating,
      setIsGenerating,
      generatedImages,
      setGeneratedImages,
    }),
    [
      imageA,
      imageB,
      inspirationImages,
      avoidImages,
      direction,
      blendStart,
      blendEnd,
      draftCount,
      isGenerating,
      generatedImages,
    ]
  )

  return <WizardContext.Provider value={value}>{children}</WizardContext.Provider>
}

export function useWizard() {
  const context = useContext(WizardContext)
  if (!context) {
    throw new Error('useWizard must be used inside WizardProvider')
  }
  return context
}
