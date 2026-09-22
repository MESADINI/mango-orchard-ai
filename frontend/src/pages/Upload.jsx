import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'
import { analyzeImage, getReportUrl } from '../services/api'

export default function Upload() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [scanType, setScanType] = useState('dual')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const selected = acceptedFiles[0]
      setFile(selected)
      setPreview(URL.createObjectURL(selected))
      setResult(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
    multiple: false,
  })

  const handleAnalyze = async () => {
    if (!file) {
      toast.error('Please select an orchard image first.')
      return
    }

    setLoading(true)
    const formData = new FormData()
    formData.append('file', file)
    formData.append('scan_type', scanType)

    try {
      const response = await analyzeImage(formData)
      setResult(response.data)
      toast.success('Diagnosis and OHI computation complete!')
    } catch (err) {
      console.error(err)
      toast.error('Failed to complete inference. Check backend service.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">Scan & Diagnose</h2>
        <p className="text-sm text-slate-400 mt-1">
          Upload leaf or canopy images for EfficientNet-B3 disease classification, YOLOv8 pest detection, and deterministic OHI computation.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Upload & Configuration Column */}
        <div className="lg:col-span-5 space-y-6">
          {/* Scan Type Selector */}
          <div className="glass-panel p-4 rounded-xl space-y-3">
            <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider block">
              Diagnostic Mode
            </label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { id: 'dual', label: 'Dual Scan', icon: '⚡' },
                { id: 'disease', label: 'Disease Only', icon: '🍃' },
                { id: 'pest', label: 'Pests Only', icon: '🦗' },
              ].map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setScanType(m.id)}
                  className={`py-2 px-3 rounded-lg text-xs font-medium border flex flex-col items-center gap-1 transition-all ${
                    scanType === m.id
                      ? 'bg-emerald-600/30 text-emerald-300 border-emerald-500 shadow-sm'
                      : 'bg-slate-800/60 text-slate-400 border-slate-700 hover:text-slate-200'
                  }`}
                >
                  <span className="text-base">{m.icon}</span>
                  <span>{m.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Dropzone Area */}
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all ${
              isDragActive
                ? 'border-emerald-500 bg-emerald-500/10'
                : 'border-slate-700 bg-slate-800/40 hover:border-slate-500 hover:bg-slate-800/60'
            }`}
          >
            <input {...getInputProps()} />
            {preview ? (
              <div className="space-y-3">
                <img
                  src={preview}
                  alt="Selected Leaf"
                  className="max-h-60 mx-auto rounded-xl shadow-md object-contain"
                />
                <p className="text-xs text-slate-400">Click or drag another image to replace</p>
              </div>
            ) : (
              <div className="py-8 space-y-3">
                <div className="text-4xl">📸</div>
                <div className="text-sm font-medium text-slate-200">
                  Drag and drop mango leaf or canopy image here
                </div>
                <p className="text-xs text-slate-400">Supports JPG, PNG, WEBP (Max 15MB)</p>
              </div>
            )}
          </div>

          {/* Trigger Button */}
          <button
            onClick={handleAnalyze}
            disabled={!file || loading}
            className={`w-full py-3 px-4 rounded-xl font-medium text-sm flex items-center justify-center gap-2 shadow-lg transition-all ${
              !file || loading
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                : 'bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white shadow-emerald-500/20'
            }`}
          >
            {loading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                </svg>
                <span>Analyzing Pathogens & OHI...</span>
              </>
            ) : (
              <>
                <span>🚀</span>
                <span>Run Precision Diagnostic</span>
              </>
            )}
          </button>
        </div>

        {/* Results Column */}
        <div className="lg:col-span-7">
          {result ? (
            <div className="space-y-6">
              {/* OHI Score Card */}
              <div className="glass-panel p-6 rounded-2xl border-slate-700/80 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Orchard Health Index</span>
                    <div className="flex items-baseline gap-2 mt-1">
                      <span className="text-4xl font-extrabold text-amber-400">{result.ohi_score}</span>
                      <span className="text-slate-500 text-xs font-semibold">/ 100</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                      {result.health_status}
                    </span>
                    <p className="text-[10px] text-slate-500 mt-1 font-mono">Scan ID: {result.scan_id.slice(0, 8)}</p>
                  </div>
                </div>

                {/* Disclosed Nutrient Deficiency Policy Flag */}
                {result.nutrient_input_was_defaulted && (
                  <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-300 flex items-center gap-2">
                    <span>⚠️</span>
                    <span>Nutrient deficiency defaulted to 0; penalty weight reallocated per OHI protocol.</span>
                  </div>
                )}
              </div>

              {/* Diagnosis Findings */}
              <div className="glass-panel p-6 rounded-2xl border-slate-700/80 space-y-4">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider">Vision Detection Breakdown</h3>

                {result.disease_results?.map((d, i) => (
                  <div key={i} className="p-4 rounded-xl bg-slate-800/60 border border-slate-700 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-rose-400 font-bold text-base">{d.disease_name}</span>
                        <span className="text-[10px] bg-rose-500/20 text-rose-300 px-2 py-0.5 rounded border border-rose-500/30">
                          {Math.round(d.confidence * 100)}% Confidence
                        </span>
                      </div>
                      <span className="text-xs text-slate-400">
                        Affected Area: <strong className="text-white">{d.severity_percentage}%</strong>
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400">Model: EfficientNet-B3 (Trained on MangoLeafBD)</p>
                  </div>
                ))}

                {/* Pest Detections */}
                {result.pest_results?.length > 0 ? (
                  result.pest_results.map((p, i) => (
                    <div key={i} className="p-4 rounded-xl bg-slate-800/60 border border-slate-700 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-mango-400 font-bold text-base">{p.pest_name}</span>
                        {p.is_unreliable && (
                          <span className="text-[10px] bg-amber-500/20 text-amber-300 px-2 py-0.5 rounded border border-amber-500/30">
                            ⚠️ UNRELIABLE CLASS (&lt;100 samples)
                          </span>
                        )}
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-slate-500 italic">No canopy pest infestations detected in this scan.</p>
                )}
              </div>

              {/* Verified Treatment Recommendations */}
              <div className="glass-panel p-6 rounded-2xl border-slate-700/80 space-y-4">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <span>📜</span>
                  <span>Verified ICAR/TNAU Treatment Protocol</span>
                </h3>

                <div className="space-y-3">
                  {result.recommendations?.map((rec, i) => (
                    <div key={i} className="p-4 rounded-xl bg-slate-800/80 border border-slate-700/60 space-y-2">
                      <div className="flex items-center justify-between">
                        <h4 className="text-xs font-bold text-emerald-400">{rec.title}</h4>
                        <span className="text-[10px] bg-slate-700 text-slate-300 px-2 py-0.5 rounded">{rec.treatment_type}</span>
                      </div>
                      <p className="text-xs text-slate-300"><strong>Product:</strong> {rec.product_name}</p>
                      <p className="text-xs text-slate-300"><strong>Dosage:</strong> {rec.dosage_rate}</p>
                      <p className="text-xs text-slate-400">{rec.application_guidance}</p>
                      <div className="border-t border-slate-700/60 pt-2 flex items-center justify-between text-[11px] text-slate-500">
                        <span className="italic">Citation: {rec.citation_source}</span>
                        <span>Waiting period: {rec.waiting_period_days} days</span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* PDF Report Generation Button */}
                <div className="pt-2">
                  <a
                    href={getReportUrl(result.scan_id)}
                    target="_blank"
                    rel="noreferrer"
                    className="w-full py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-medium text-xs flex items-center justify-center gap-2 border border-slate-600 transition-colors"
                  >
                    <span>📄</span>
                    <span>Download WeasyPrint Executive PDF Audit</span>
                  </a>
                </div>
              </div>
            </div>
          ) : (
            <div className="glass-panel rounded-2xl p-12 border-slate-800 text-center flex flex-col items-center justify-center space-y-3 h-full min-h-[400px]">
              <div className="text-5xl opacity-40">🍃</div>
              <h3 className="text-sm font-semibold text-slate-300">No Active Diagnosis Loaded</h3>
              <p className="text-xs text-slate-500 max-w-sm">
                Upload a mango leaf or canopy image on the left to initiate real-time AI inference and view deterministic OHI metrics.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
