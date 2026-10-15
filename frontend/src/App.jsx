import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import History from './pages/History'
import Chatbot from './pages/Chatbot'
import Admin from './pages/Admin'
import Auth from './pages/Auth'
import { AuthProvider, useAuth } from './services/AuthContext'
import { FarmProvider } from './services/FarmContext'

function RequireAuth({ children }) {
  const { user, loading } = useAuth()
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-xs text-slate-500 animate-pulse">
        Loading...
      </div>
    )
  }
  return user ? <FarmProvider>{children}</FarmProvider> : <Navigate to="/auth" replace />
}

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/auth" element={<Auth />} />
        <Route path="/" element={<RequireAuth><Layout /></RequireAuth>}>
          <Route index element={<Dashboard />} />
          <Route path="upload" element={<Upload />} />
          <Route path="history" element={<History />} />
          <Route path="chatbot" element={<Chatbot />} />
          <Route path="admin" element={<Admin />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AuthProvider>
  )
}
