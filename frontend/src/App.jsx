import { Navigate, Route, Routes } from 'react-router-dom'
import { WizardProvider } from './context/WizardContext'
import HomePage from './pages/HomePage'
import StepReferencesPage from './pages/StepReferencesPage'
import StepCreativePage from './pages/StepCreativePage'
import StepGeneratingPage from './pages/StepGeneratingPage'

function App() {
  return (
    <WizardProvider>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/wizard/step-1" element={<StepReferencesPage />} />
        <Route path="/wizard/step-2" element={<StepCreativePage />} />
        <Route path="/wizard/step-3" element={<StepGeneratingPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </WizardProvider>
  )
}

export default App
