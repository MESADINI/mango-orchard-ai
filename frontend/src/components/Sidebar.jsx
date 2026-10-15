import React, { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { checkHealth } from '../services/api'

const navigation = [
  { name: 'Dashboard & OHI', href: '/', icon: '📊' },
  { name: 'Scan & Diagnose', href: '/upload', icon: '📸' },
  { name: 'Scan History', href: '/history', icon: '🗂️' },
  { name: 'Advisor Chatbot', href: '/chatbot', icon: '💬', badge: 'EN / தமிழ்' },
  { name: 'Farm Admin', href: '/admin', icon: '🚜' },
]

function ModelStatus({ label, trained }) {
  return (
    <div className="flex items-center justify-between text-[11px]">
      <span className="text-slate-400">{label}</span>
      <span className={trained ? 'text-emerald-400' : 'text-amber-400'}>{trained ? 'Trained' : 'Not installed'}</span>
    </div>
  )
}

export default function Sidebar() {
  const [health, setHealth] = useState(null)
  const [offline, setOffline] = useState(false)

  useEffect(() => {
    checkHealth()
      .then((res) => setHealth(res.data))
      .catch(() => setOffline(true))
  }, [])

  return (
    <aside className="w-64 bg-slate-950 border-r border-slate-800 flex flex-col justify-between shrink-0">
      <div>
        {/* Brand Header */}
        <div className="h-16 flex items-center px-6 border-b border-slate-800 gap-3">
          <span className="text-2xl">🥭</span>
          <div>
            <h1 className="font-bold text-sm text-white tracking-wide">Mango Health AI</h1>
            <p className="text-[10px] text-emerald-400 font-mono tracking-wider uppercase">Orchard Intelligence</p>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="p-4 space-y-1.5">
          {navigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive }) =>
                `flex items-center justify-between px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-emerald-600/20 text-emerald-300 border border-emerald-500/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`
              }
            >
              <div className="flex items-center gap-3">
                <span className="text-base">{item.icon}</span>
                <span>{item.name}</span>
              </div>
              {item.badge && (
                <span className="text-[10px] font-semibold bg-mango-500/20 text-mango-300 px-2 py-0.5 rounded-full border border-mango-500/30">
                  {item.badge}
                </span>
              )}
            </NavLink>
          ))}
        </nav>
      </div>

      {/* Footer System Status (from /health) */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-950/40 space-y-1.5">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${offline ? 'bg-rose-500' : health ? 'bg-emerald-500' : 'bg-slate-500 animate-pulse'}`}></span>
            {offline ? 'Backend offline' : 'Backend'}
          </span>
          <span className="font-mono text-[11px] text-slate-500">{health ? `v${health.version}` : ''}</span>
        </div>
        {health && (
          <>
            <ModelStatus label="Disease model" trained={health.models.disease_model_trained} />
            <ModelStatus label="Pest model" trained={health.models.pest_model_trained} />
          </>
        )}
      </div>
    </aside>
  )
}
