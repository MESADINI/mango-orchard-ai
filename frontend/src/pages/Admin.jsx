import React, { useState } from 'react'

export default function Admin() {
  const [cultivar, setCultivar] = useState('Alphonso')
  const [totalTrees, setTotalTrees] = useState(350)
  const [acreage, setAcreage] = useState(12.5)

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">Farm & Orchard Administration</h2>
        <p className="text-sm text-slate-400 mt-1">
          Configure cultivar parameters, tree tag registry, and geo-coordinates for microclimate Open-Meteo weather matching.
        </p>
      </div>

      <div className="glass-panel rounded-2xl p-6 border-slate-700/60 space-y-6">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider border-b border-slate-800 pb-3">
          Orchard Metadata
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
          <div className="space-y-1.5">
            <label className="text-slate-300 font-semibold block">Orchard Name</label>
            <input
              type="text"
              defaultValue="Krishnagiri Mango Valley"
              className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2.5 text-white focus:outline-none focus:border-emerald-500"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-slate-300 font-semibold block">Primary Cultivar</label>
            <select
              value={cultivar}
              onChange={(e) => setCultivar(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2.5 text-white focus:outline-none focus:border-emerald-500"
            >
              <option value="Alphonso">Alphonso (Hapus)</option>
              <option value="Banganapalli">Banganapalli (Benishan)</option>
              <option value="Dasheri">Dasheri</option>
              <option value="Totapuri">Totapuri</option>
              <option value="Kesar">Kesar</option>
            </select>
          </div>

          <div className="space-y-1.5">
            <label className="text-slate-300 font-semibold block">Total Acreage</label>
            <input
              type="number"
              value={acreage}
              onChange={(e) => setAcreage(parseFloat(e.target.value))}
              className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2.5 text-white focus:outline-none focus:border-emerald-500"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-slate-300 font-semibold block">Tree Inventory Count</label>
            <input
              type="number"
              value={totalTrees}
              onChange={(e) => setTotalTrees(parseInt(e.target.value))}
              className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2.5 text-white focus:outline-none focus:border-emerald-500"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-slate-300 font-semibold block">Latitude Coordinate</label>
            <input
              type="text"
              defaultValue="12.5266"
              className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2.5 text-white font-mono focus:outline-none focus:border-emerald-500"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-slate-300 font-semibold block">Longitude Coordinate</label>
            <input
              type="text"
              defaultValue="78.2146"
              className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2.5 text-white font-mono focus:outline-none focus:border-emerald-500"
            />
          </div>
        </div>

        <div className="pt-4 border-t border-slate-800 flex justify-end">
          <button
            type="button"
            className="px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs shadow-lg shadow-emerald-600/20 transition-all"
          >
            Save Farm Settings
          </button>
        </div>
      </div>
    </div>
  )
}
