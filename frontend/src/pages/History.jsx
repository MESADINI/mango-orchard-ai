import React, { useState } from 'react'
import { getReportUrl } from '../services/api'

const sampleHistory = [
  {
    id: 'scan-8a71b2',
    date: '2026-09-08 14:22',
    treeTag: 'T-042',
    cultivar: 'Alphonso',
    diagnosis: 'Anthracnose',
    severity: '14.5%',
    pest: 'None',
    ohiScore: 82.5,
    status: 'Mild Stress / Warning'
  },
  {
    id: 'scan-6c19f4',
    date: '2026-09-07 10:15',
    treeTag: 'T-108',
    cultivar: 'Banganapalli',
    diagnosis: 'Powdery Mildew',
    severity: '8.2%',
    pest: 'Fruit Fly (2)',
    ohiScore: 76.0,
    status: 'Mild Stress / Warning'
  },
  {
    id: 'scan-3d92e1',
    date: '2026-09-05 16:40',
    treeTag: 'T-015',
    cultivar: 'Alphonso',
    diagnosis: 'Healthy',
    severity: '0.0%',
    pest: 'None',
    ohiScore: 96.0,
    status: 'Optimal Health'
  }
]

export default function History() {
  const [filter, setFilter] = useState('all')

  const filteredData = sampleHistory.filter((item) => {
    if (filter === 'all') return true
    if (filter === 'healthy') return item.diagnosis === 'Healthy'
    if (filter === 'diseased') return item.diagnosis !== 'Healthy'
    return true
  })

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Scan History & Audits</h2>
          <p className="text-sm text-slate-400 mt-1">
            Historical repository of tree scans, deterministic OHI ratings, and generated WeasyPrint PDF reports.
          </p>
        </div>

        {/* Filter Tabs */}
        <div className="flex items-center gap-1.5 bg-slate-950/60 p-1.5 rounded-xl border border-slate-800">
          {[
            { id: 'all', label: 'All Scans' },
            { id: 'diseased', label: 'Afflicted Only' },
            { id: 'healthy', label: 'Healthy Only' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setFilter(tab.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                filter === tab.id
                  ? 'bg-emerald-600/30 text-emerald-300 border border-emerald-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* History Table */}
      <div className="glass-panel rounded-2xl border-slate-700/60 overflow-hidden shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 text-slate-400 uppercase tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3.5 px-5">Timestamp</th>
                <th className="py-3.5 px-4">Tree Tag</th>
                <th className="py-3.5 px-4">Foliar Diagnosis</th>
                <th className="py-3.5 px-4">Lesion Severity</th>
                <th className="py-3.5 px-4">Canopy Pests</th>
                <th className="py-3.5 px-4">OHI Score</th>
                <th className="py-3.5 px-5 text-right">PDF Report</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredData.map((row) => (
                <tr key={row.id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-3.5 px-5 text-slate-300 font-mono text-[11px]">{row.date}</td>
                  <td className="py-3.5 px-4">
                    <span className="font-semibold text-white">{row.treeTag}</span>
                    <span className="text-[10px] text-slate-500 block">{row.cultivar}</span>
                  </td>
                  <td className="py-3.5 px-4">
                    <span className={row.diagnosis === 'Healthy' ? 'text-emerald-400 font-medium' : 'text-rose-400 font-medium'}>
                      {row.diagnosis}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 text-slate-300">{row.severity}</td>
                  <td className="py-3.5 px-4 text-slate-300">{row.pest}</td>
                  <td className="py-3.5 px-4">
                    <span className={`px-2 py-0.5 rounded-full font-semibold text-[11px] ${
                      row.ohiScore >= 85 ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'
                    }`}>
                      {row.ohiScore}
                    </span>
                  </td>
                  <td className="py-3.5 px-5 text-right">
                    <a
                      href={getReportUrl(row.id)}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 text-[11px] transition-colors"
                    >
                      <span>📥</span>
                      <span>PDF</span>
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
