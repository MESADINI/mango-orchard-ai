import React, { useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import { useAuth } from '../services/AuthContext'
import { errorMessage } from '../services/api'

export default function Auth() {
  const [isSignUp, setIsSignUp] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const { user, signIn, signUp } = useAuth()
  const navigate = useNavigate()

  if (user) return <Navigate to="/" replace />

  const handleSubmit = async (e) => {
    e.preventDefault()
    setSubmitting(true)
    try {
      if (isSignUp) {
        await signUp(email, password, fullName)
        toast.success('Account created.')
      } else {
        await signIn(email, password)
        toast.success('Signed in.')
      }
      navigate('/')
    } catch (err) {
      toast.error(errorMessage(err, 'Authentication failed'))
    } finally {
      setSubmitting(false)
    }
  }

  const inputClass =
    'w-full bg-slate-900 border border-slate-700 rounded-xl px-3.5 py-2.5 text-white focus:outline-none focus:border-emerald-500'

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-slate-950">
      <div className="max-w-md w-full glass-panel rounded-2xl p-8 border-slate-800 shadow-2xl space-y-6">
        <div className="text-center space-y-2">
          <span className="text-4xl">🥭</span>
          <h2 className="text-xl font-bold text-white tracking-wide">
            {isSignUp ? 'Create Grower Account' : 'Sign in to Mango Health AI'}
          </h2>
          <p className="text-xs text-slate-400">Precision agriculture surveillance & decision support system</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4 text-xs">
          {isSignUp && (
            <div className="space-y-1">
              <label htmlFor="fullName" className="text-slate-300 font-semibold block">Full Name (optional)</label>
              <input id="fullName" type="text" value={fullName} onChange={(e) => setFullName(e.target.value)}
                className={inputClass} maxLength={120} />
            </div>
          )}
          <div className="space-y-1">
            <label htmlFor="email" className="text-slate-300 font-semibold block">Email Address</label>
            <input id="email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)}
              placeholder="grower@orchard.com" autoComplete="email" className={inputClass} />
          </div>
          <div className="space-y-1">
            <label htmlFor="password" className="text-slate-300 font-semibold block">Password</label>
            <input id="password" type="password" required minLength={isSignUp ? 8 : 1} value={password}
              onChange={(e) => setPassword(e.target.value)} placeholder="••••••••"
              autoComplete={isSignUp ? 'new-password' : 'current-password'} className={inputClass} />
            {isSignUp && <p className="text-[10px] text-slate-500">At least 8 characters.</p>}
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white font-medium text-xs shadow-lg shadow-emerald-500/20 disabled:opacity-50 transition-all mt-2"
          >
            {submitting ? 'Please wait...' : isSignUp ? 'Create Account' : 'Sign In'}
          </button>
        </form>

        {!isSignUp && (
          <p className="text-[11px] text-slate-500 text-center">
            Local demo account: <span className="font-mono text-slate-300">grower@mango.ai</span> /{' '}
            <span className="font-mono text-slate-300">orchard123</span>
          </p>
        )}

        <div className="text-center pt-2 border-t border-slate-800">
          <button
            type="button"
            onClick={() => setIsSignUp(!isSignUp)}
            className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
          >
            {isSignUp ? 'Already have an account? Sign In' : "Don't have an account? Sign Up"}
          </button>
        </div>
      </div>
    </div>
  )
}
