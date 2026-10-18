import React from 'react'

/** OHI tier styling keyed by the backend's health_status label. */
const STATUS_STYLES = {
  'Optimal Health': 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
  'Mild Stress / Warning': 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  'High Infestation Risk': 'bg-orange-500/15 text-orange-300 border-orange-500/30',
  'Severe Outbreak / Critical': 'bg-rose-500/15 text-rose-300 border-rose-500/30',
}

export function StatusBadge({ status }) {
  const cls = STATUS_STYLES[status] || 'bg-slate-700/40 text-slate-300 border-slate-600'
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border ${cls}`}>
      {status}
    </span>
  )
}

export function OhiScore({ scan, size = 'text-4xl' }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className={`${size} font-extrabold`} style={{ color: scan.badge_color }}>
        {scan.ohi_score == null ? '—' : scan.ohi_score.toFixed(1)}
      </span>
      <span className="text-slate-500 text-xs font-semibold">
        {scan.ohi_score == null ? 'not enough data' : '/ 100'}
      </span>
      {scan.ohi_is_partial && (
        <span
          className="text-[10px] font-semibold text-amber-300 bg-amber-500/10 border border-amber-500/30 px-1.5 py-0.5 rounded"
          title={`Not included: ${scan.ohi_missing_components.join(', ')}`}
        >
          PARTIAL
        </span>
      )}
    </div>
  )
}

export function Caveats({ caveats }) {
  if (!caveats?.length) return null
  return (
    <div className="space-y-1.5">
      {caveats.map((c) => (
        <div
          key={c}
          className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-200 flex items-start gap-2"
        >
          <span aria-hidden>⚠️</span>
          <span>{c}</span>
        </div>
      ))}
    </div>
  )
}

/** Short text summary of a scan's diagnosis for lists and cards. */
export function diagnosisSummary(scan) {
  if (scan.disease_results?.length) return scan.disease_results[0].disease_name
  if (scan.scan_type === 'pest') return 'Not scanned'
  return 'Model unavailable'
}

export function pestSummary(scan) {
  if (!scan.pest_results?.length) {
    if (scan.scan_type === 'disease') return 'Not scanned'
    return scan.model_status?.pest_model_trained ? 'None detected' : 'Model unavailable'
  }
  const counts = {}
  scan.pest_results.forEach((p) => {
    counts[p.pest_name] = (counts[p.pest_name] || 0) + 1
  })
  return Object.entries(counts).map(([name, n]) => `${name} (${n})`).join(', ')
}

export const formatDate = (iso) =>
  iso ? new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : ''
