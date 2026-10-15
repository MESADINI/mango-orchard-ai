import React, { useCallback, useEffect, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'
import { analyzeImage, errorMessage } from '../services/api'
import { useFarm } from '../services/FarmContext'
import ScanResult from '../components/ScanResult'

const MAX_MB = 15
const ACCEPT = { 'image/jpeg': ['.jpg', '.jpeg'], 'image/png': ['.png'], 'image/webp': ['.webp'] }

const SCAN_MODES = [
  { id: 'dual', label: 'Dual Scan', icon: '⚡' },
  { id: 'disease', label: 'Disease Only', icon: '🍃' },
  { id: 'pest', label: 'Pests Only', icon: '🦗' },
]

function ImageDrop({ label, hint, file, onFile }) {
  const [preview, setPreview] = useState(null)

  useEffect(() => {
    if (!file) {
      setPreview(null)
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const onDrop = useCallback((accepted, rejected) => {
    if (rejected?.length) {
      toast.error(rejected[0].errors[0]?.message || 'File rejected')
      return
    }
    if (accepted?.length) onFile(accepted[0])
  }, [onFile])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: ACCEPT, multiple: false, maxSize: MAX_MB * 1024 * 1024,
  })

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">{label}</span>
        {file && (
          <button type="button" onClick={() => onFile(null)} className="text-[11px] text-slate-400 hover:text-rose-300">
            Remove
          </button>
        )}
      </div>
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-2xl p-4 text-center cursor-pointer transition-all ${
          isDragActive ? 'border-emerald-500 bg-emerald-500/10' : 'border-slate-700 bg-slate-800/40 hover:border-slate-500'
        }`}
      >
        <input {...getInputProps()} />
        {preview ? (
          <img src={preview} alt={label} className="max-h-44 mx-auto rounded-xl object-contain" />
        ) : (
          <div className="py-6 space-y-2">
            <div className="text-3xl">📸</div>
            <div className="text-xs font-medium text-slate-200">{hint}</div>
            <p className="text-[11px] text-slate-500">JPG, PNG or WebP, max {MAX_MB} MB</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default function Upload() {
  const { farm } = useFarm()
  const [leafFile, setLeafFile] = useState(null)
  const [pestFile, setPestFile] = useState(null)
  const [scanType, setScanType] = useState('dual')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const needsLeaf = scanType !== 'pest'
  const canSubmit = !loading && (needsLeaf ? !!leafFile : !!(pestFile || leafFile))

  const handleAnalyze = async () => {
    const formData = new FormData()
    formData.append('scan_type', scanType)
    if (scanType !== 'pest' && leafFile) formData.append('file', leafFile)
    if (scanType !== 'disease' && pestFile) formData.append('pest_file', pestFile)
    if (scanType === 'pest' && !pestFile && leafFile) formData.append('pest_file', leafFile)
    if (farm) formData.append('farm_id', farm.id)

    setLoading(true)
    try {
      const response = await analyzeImage(formData)
      setResult(response.data)
      toast.success('Analysis complete.')
    } catch (err) {
      toast.error(errorMessage(err, 'Analysis failed'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">Scan & Diagnose</h2>
        <p className="text-sm text-slate-400 mt-1">
          Upload a leaf image for disease classification and an optional canopy/trap image for pest detection.
          {farm && <> Weather is taken from <strong className="text-slate-300">{farm.name}</strong>.</>}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-5 space-y-6">
          <div className="glass-panel p-4 rounded-xl space-y-3">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider block">Diagnostic Mode</span>
            <div className="grid grid-cols-3 gap-2">
              {SCAN_MODES.map((m) => (
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

          {scanType !== 'pest' && (
            <ImageDrop label="Leaf image" hint="Drop a mango leaf image here" file={leafFile} onFile={setLeafFile} />
          )}
          {scanType !== 'disease' && (
            <ImageDrop
              label={scanType === 'pest' ? 'Canopy / trap image' : 'Canopy / trap image (optional)'}
              hint="Drop a canopy or pest-trap image here"
              file={pestFile}
              onFile={setPestFile}
            />
          )}

          <button
            type="button"
            onClick={handleAnalyze}
            disabled={!canSubmit}
            className={`w-full py-3 px-4 rounded-xl font-medium text-sm flex items-center justify-center gap-2 shadow-lg transition-all ${
              !canSubmit
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                : 'bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white shadow-emerald-500/20'
            }`}
          >
            {loading ? 'Analyzing...' : '🚀 Run Diagnostic'}
          </button>
          {loading && (
            <p className="text-[11px] text-slate-500 text-center">The first scan after startup can take a while on CPU.</p>
          )}
        </div>

        <div className="lg:col-span-7">
          {result ? (
            <ScanResult result={result} />
          ) : (
            <div className="glass-panel rounded-2xl p-12 border-slate-800 text-center flex flex-col items-center justify-center space-y-3 h-full min-h-[400px]">
              <div className="text-5xl opacity-40">🍃</div>
              <h3 className="text-sm font-semibold text-slate-300">No Active Diagnosis Loaded</h3>
              <p className="text-xs text-slate-500 max-w-sm">
                Upload images on the left to run the analysis and view the Orchard Health Index.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

