import React, { useEffect, useState } from 'react'
import toast from 'react-hot-toast'
import { errorMessage, listScans, openReport } from '../services/api'
import ScanResult from '../components/ScanResult'
import { StatusBadge, diagnosisSummary, formatDate, pestSummary } from '../components/ScanWidgets'

const FILTERS = [
  { id: 'all', label: 'All Scans' },
  { id: 'diseased', label: 'Afflicted Only' },
  { id: 'healthy', label: 'Healthy Only' },
]

const isHealthy = (scan) => scan.disease_results?.[0]?.is_healthy === true
const isDiseased = (scan) => scan.disease_results?.[0]?.is_healthy === false || scan.pest_results?.length > 0

export default function History() {
  const [scans, setScans] = useState(null)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('all')
  const [selected, setSelected] = useState(null)

  useEffect(() => {
    listScans(200)
      .then((res) => setScans(res.data))
      .catch((err) => setError(errorMessage(err, 'Could not load scan history')))
  }, [])

  const filtered = (scans || []).filter((s) =>
    filter === 'healthy' ? isHealthy(s) : filter === 'diseased' ? isDiseased(s) : true
  )

  const handleReport = async (scanId) => {
    try {
      await openReport(scanId)
    } catch (err) {
      toast.error(errorMessage(err, 'Could not generate report'))
    }
  }

  if (selected) {
    return (
      <div className="max-w-4xl mx-auto space-y-4">
        <button type="button" onClick={() => setSelected(null)} className="text-xs text-emerald-400 hover:text-emerald-300">
          ← Back to history
        </button>
        <p className="text-xs text-slate-500">Scanned {formatDate(selected.created_at)}</p>
        <ScanResult result={selected} />
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Scan History & Reports</h2>
          <p className="text-sm text-slate-400 mt-1">Your stored scans, OHI ratings and downloadable reports.</p>
        </div>
        <div className="flex items-center gap-1.5 bg-slate-950/60 p-1.5 rounded-xl border border-slate-800">
          {FILTERS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setFilter(tab.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                filter === tab.id ? 'bg-emerald-600/30 text-emerald-300 border border-emerald-500/30' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="glass-panel rounded-2xl border-slate-700/60 overflow-hidden shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 text-slate-400 uppercase tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3.5 px-5">Date</th>
                <th className="py-3.5 px-4">Type</th>
                <th className="py-3.5 px-4">Foliar Diagnosis</th>
                <th className="py-3.5 px-4">Lesion Area</th>
                <th className="py-3.5 px-4">Canopy Pests</th>
                <th className="py-3.5 px-4">OHI</th>
                <th className="py-3.5 px-5 text-right">Report</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {error && (
                <tr><td colSpan={7} className="py-8 text-center text-rose-300">{error}</td></tr>
              )}
              {!error && scans === null && (
                <tr><td colSpan={7} className="py-8 text-center text-slate-500 animate-pulse">Loading...</td></tr>
              )}
              {!error && scans !== null && filtered.length === 0 && (
                <tr><td colSpan={7} className="py-8 text-center text-slate-500">No scans to show.</td></tr>
              )}
              {filtered.map((row) => (
                <tr key={row.scan_id} onClick={() => setSelected(row)} className="hover:bg-slate-800/40 transition-colors cursor-pointer">
                  <td className="py-3.5 px-5 text-slate-300 font-mono text-[11px]">{formatDate(row.created_at)}</td>
                  <td className="py-3.5 px-4 text-slate-400 capitalize">{row.scan_type}</td>
                  <td className="py-3.5 px-4">
                    <span className={isHealthy(row) ? 'text-emerald-400 font-medium' : row.disease_results?.length ? 'text-rose-400 font-medium' : 'text-slate-500'}>
                      {diagnosisSummary(row)}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 text-slate-300">
                    {row.lesion_severity_percentage != null ? `${row.lesion_severity_percentage}%` : '-'}
                  </td>
                  <td className="py-3.5 px-4 text-slate-300">{pestSummary(row)}</td>
                  <td className="py-3.5 px-4 space-x-2 whitespace-nowrap">
                    <span className="font-semibold" style={{ color: row.badge_color }}>{row.ohi_score.toFixed(1)}</span>
                    <StatusBadge status={row.health_status} />
                  </td>
                  <td className="py-3.5 px-5 text-right">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleReport(row.scan_id)
                      }}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 text-[11px] transition-colors"
                    >
                      📥 Report
                    </button>
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
