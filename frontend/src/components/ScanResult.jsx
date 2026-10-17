import React, { useState } from 'react'
import toast from 'react-hot-toast'
import { errorMessage, mediaUrl, openReport } from '../services/api'
import { Caveats, OhiScore, StatusBadge } from './ScanWidgets'

/** Full scan result view: OHI, caveats, detections with overlays, recommendations, report link. */
export default function ScanResult({ result }) {
  const [opening, setOpening] = useState(false)
  const disease = result.disease_results?.[0]

  const handleReport = async () => {
    setOpening(true)
    try {
      await openReport(result.scan_id)
    } catch (err) {
      toast.error(errorMessage(err, 'Could not generate report'))
    } finally {
      setOpening(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="glass-panel p-6 rounded-2xl border-slate-700/80 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Orchard Health Index</span>
            <div className="mt-1"><OhiScore scan={result} /></div>
          </div>
          <div className="text-right space-y-1">
            <StatusBadge status={result.health_status} />
            <p className="text-[10px] text-slate-500 font-mono">Scan ID: {result.scan_id.slice(0, 8)}</p>
          </div>
        </div>
        <div className="grid grid-cols-4 gap-2 text-center text-[11px]">
          {[
            ['Disease', result.penalty_breakdown.disease_severity_penalty],
            ['Pest', result.penalty_breakdown.pest_pressure_penalty],
            ['Weather', result.penalty_breakdown.weather_vulnerability_index],
            ['Total penalty', result.penalty_breakdown.weighted_total_penalty],
          ].map(([label, value]) => (
            <div key={label} className="bg-slate-800/60 rounded-lg py-2 border border-slate-700/60">
              <div className="text-slate-400">{label}</div>
              <div className="text-white font-semibold">{value == null ? 'n/a' : value.toFixed(1)}</div>
            </div>
          ))}
        </div>
        <Caveats caveats={result.caveats} />
      </div>

      <div className="glass-panel p-6 rounded-2xl border-slate-700/80 space-y-4">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider">Vision Detection Breakdown</h3>

        {disease?.low_confidence && (
          <div role="alert" className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-xs text-amber-200">
            <strong>Unsure — please retake the photo.</strong> Photograph a single leaf filling the frame, in daylight,
            without heavy shadow. The result below is not used for the health index or treatment advice.
          </div>
        )}
        {disease ? (
          <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className={`font-bold text-base ${disease.is_healthy ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {disease.disease_name}
                </span>
                <span className="text-[10px] bg-slate-700 text-slate-200 px-2 py-0.5 rounded">
                  {Math.round(disease.confidence * 100)}% confidence
                </span>
              </div>
              <span className="text-xs text-slate-400">
                Lesion area: <strong className="text-white">{disease.severity_percentage}%</strong>
              </span>
            </div>
            {disease.gradcam_overlay_url && (
              <div>
                <img src={mediaUrl(disease.gradcam_overlay_url)} alt="Grad-CAM heatmap"
                  className="max-h-64 rounded-lg mx-auto" />
                <p className="text-[10px] text-slate-500 text-center mt-1">
                  Grad-CAM: warmer regions contributed most to the prediction
                </p>
              </div>
            )}
          </div>
        ) : result.scan_type !== 'pest' && (
          <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700 text-xs text-slate-400">
            Disease classification unavailable (no trained model installed).
            {result.lesion_severity_percentage != null && (
              <> Colour-based lesion area estimate: <strong className="text-white">{result.lesion_severity_percentage}%</strong>.</>
            )}
          </div>
        )}

        {result.pest_results?.length > 0 ? (
          <>
            {result.pest_results.map((p, i) => (
              <div key={i} className="p-3 rounded-xl bg-slate-800/60 border border-slate-700 flex items-center justify-between">
                <span className="text-mango-400 font-bold">{p.pest_name}</span>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-400">{Math.round(p.confidence * 100)}%</span>
                  {p.is_unreliable && (
                    <span className="text-[10px] bg-amber-500/20 text-amber-300 px-2 py-0.5 rounded border border-amber-500/30">
                      ⚠️ UNVERIFIED CLASS (&lt;100 samples)
                    </span>
                  )}
                  {!p.is_unreliable && !p.host_verified && (
                    <span className="text-[10px] bg-amber-500/20 text-amber-300 px-2 py-0.5 rounded border border-amber-500/30" title={p.warning || ''}>
                      ⚠️ UNVERIFIED: model not trained on mango
                    </span>
                  )}
                </div>
              </div>
            ))}
            {result.pest_overlay_url && (
              <img src={mediaUrl(result.pest_overlay_url)} alt="Pest detections" className="max-h-64 rounded-lg mx-auto" />
            )}
          </>
        ) : result.scan_type !== 'disease' && (
          <p className="text-xs text-slate-500 italic">
            {result.model_status.pest_model_trained
              ? 'No canopy pests detected in this scan.'
              : 'Pest detection unavailable (no trained model installed).'}
          </p>
        )}
      </div>

      <div className="glass-panel p-6 rounded-2xl border-slate-700/80 space-y-4">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
          <span>📜</span><span>ICAR/TNAU Treatment Protocol</span>
        </h3>
        {result.recommendations?.length ? (
          <div className="space-y-3">
            {result.recommendations.map((rec, i) => (
              <div key={i} className="p-4 rounded-xl bg-slate-800/80 border border-slate-700/60 space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold text-emerald-400">{rec.condition}: {rec.title}</h4>
                  <span className="text-[10px] bg-slate-700 text-slate-300 px-2 py-0.5 rounded">{rec.treatment_type}</span>
                </div>
                <p className="text-xs text-slate-300"><strong>Product:</strong> {rec.product_name}</p>
                <p className="text-xs text-slate-300"><strong>Dosage:</strong> {rec.dosage_rate}</p>
                <p className="text-xs text-slate-400">{rec.application_guidance}</p>
                <div className="border-t border-slate-700/60 pt-2 flex items-center justify-between text-[11px] text-slate-500">
                  <a href={rec.citation_url} target="_blank" rel="noreferrer" className="italic hover:text-slate-300">
                    Citation: {rec.citation_source}
                  </a>
                  <span>
                    {rec.waiting_period_days != null
                      ? `Last spray ≥ ${rec.waiting_period_days} days before harvest`
                      : 'Check the product label (pre-harvest interval, mango on label)'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-slate-500 italic">No treatment protocol for this scan (nothing confirmed was detected).</p>
        )}

        <button
          type="button"
          onClick={handleReport}
          disabled={opening}
          className="w-full py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-medium text-xs flex items-center justify-center gap-2 border border-slate-600 transition-colors disabled:opacity-50"
        >
          <span>📄</span><span>{opening ? 'Generating...' : 'Open Scan Report'}</span>
        </button>
      </div>
    </div>
  )
}
