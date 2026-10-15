import React from 'react'
import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import { useAuth } from '../services/AuthContext'
import { useFarm } from '../services/FarmContext'

const initials = (user) =>
  (user?.full_name || user?.email || '?')
    .split(/[\s@.]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((s) => s[0].toUpperCase())
    .join('')

export default function Layout() {
  const { user, signOut } = useAuth()
  const { farms, farm, selectFarm } = useFarm()

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-900 text-slate-100">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <header className="h-16 border-b border-slate-800 bg-slate-950/60 backdrop-blur px-8 flex items-center justify-between shrink-0 gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 shrink-0">Orchard:</span>
            {farms.length > 1 ? (
              <select
                value={farm?.id || ''}
                onChange={(e) => selectFarm(e.target.value)}
                className="bg-slate-800 border border-slate-700 rounded-md text-xs text-emerald-400 px-2 py-1 focus:outline-none"
              >
                {farms.map((f) => (
                  <option key={f.id} value={f.id}>{f.name}</option>
                ))}
              </select>
            ) : (
              <span className="px-2.5 py-1 bg-slate-800 rounded-md text-xs font-medium text-emerald-400 border border-slate-700 truncate">
                {farm ? farm.name : 'No orchard registered'}
              </span>
            )}
            {farm && (
              <span className="text-[11px] text-slate-500 font-mono hidden md:inline">
                {farm.latitude.toFixed(2)}°, {farm.longitude.toFixed(2)}°
              </span>
            )}
          </div>

          <div className="flex items-center gap-4">
            {farm && (
              <div className="hidden lg:flex items-center gap-2 bg-amber-500/10 border border-amber-500/20 px-3 py-1 rounded-full text-xs text-amber-300">
                <span>🌾</span>
                <span>Cultivar: <strong>{farm.cultivar}</strong></span>
              </div>
            )}
            <div
              className="h-8 w-8 rounded-full bg-emerald-600/30 border border-emerald-500/40 flex items-center justify-center text-xs font-bold text-emerald-300"
              title={user?.email}
            >
              {initials(user)}
            </div>
            <button
              type="button"
              onClick={signOut}
              className="text-xs text-slate-400 hover:text-white border border-slate-700 rounded-md px-2.5 py-1 transition-colors"
            >
              Sign out
            </button>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto p-8">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
