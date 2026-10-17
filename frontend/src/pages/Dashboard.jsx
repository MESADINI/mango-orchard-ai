import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { errorMessage, getWeatherForecast, listScans } from '../services/api'
import { useFarm } from '../services/FarmContext'
import { OhiScore, StatusBadge, diagnosisSummary, formatDate, pestSummary } from '../components/ScanWidgets'

const RISK_STYLES = {
  High: 'bg-rose-500/20 text-rose-300 border border-rose-500/30',
  Moderate: 'bg-amber-500/20 text-amber-300 border border-amber-500/30',
  Low: 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30',
  Unknown: 'bg-slate-700/40 text-slate-300 border border-slate-600',
}

function Card({ title, children, footer }) {
  return (
    <div className="glass-panel rounded-2xl p-6 border-slate-700/60 flex flex-col justify-between">
      <div>
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{title}</span>
        <div className="mt-2">{children}</div>
      </div>
      {footer && <div className="text-xs text-slate-500 border-t border-slate-800 pt-3 mt-4">{footer}</div>}
    </div>
  )
}

/** Rounds a possibly-missing number for display ("–" when the weather service returned null). */
const fmt = (v, digits = 0) => (v == null ? '–' : Number(v).toFixed(digits))

export default function Dashboard() {
  const { farm, loading: farmLoading } = useFarm()
  const [weather, setWeather] = useState(null)
  const [weatherError, setWeatherError] = useState(null)
  const [loadingWeather, setLoadingWeather] = useState(true)
  const [scans, setScans] = useState(null)

  useEffect(() => {
    listScans(20).then((res) => setScans(res.data)).catch(() => setScans([]))
  }, [])

  useEffect(() => {
    if (farmLoading) return
    setLoadingWeather(true)
    setWeatherError(null)
    getWeatherForecast(farm?.latitude, farm?.longitude)
      .then((res) => setWeather(res.data))
      .catch((err) => {
        setWeather(null)
        setWeatherError(errorMessage(err, 'Weather unavailable'))
      })
      .finally(() => setLoadingWeather(false))
  }, [farm?.latitude, farm?.longitude, farmLoading])

  const farmScans = (scans || []).filter((s) => !farm || !s.farm_id || s.farm_id === farm.id)
  const latest = farmScans[0]

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Orchard Health Overview</h2>
          <p className="text-sm text-slate-400 mt-1">
            Latest scan results, deterministic OHI analytics and 7-day disease-spread risk.
          </p>
        </div>
        <Link
          to="/upload"
          className="inline-flex items-center gap-2 bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white font-medium text-sm px-4 py-2.5 rounded-lg shadow-lg shadow-emerald-500/20 transition-all"
        >
          <span>📸</span><span>New Tree Scan</span>
        </Link>
      </div>

      <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex items-start gap-3">
        <span className="text-xl">⚠️</span>
        <div className="text-xs text-amber-200/90 leading-relaxed">
          <strong className="font-semibold text-amber-200">Policy Disclosure:</strong> Soil/leaf nutrient sensors are not
          integrated. Per the OHI methodology the nutrient input is defaulted to 0 and its weight is reallocated to the
          disease and pest penalties.
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
        <Card title="Latest Orchard Health" footer={latest ? formatDate(latest.created_at) : 'No scans yet'}>
          {latest ? (
            <div className="space-y-2">
              <OhiScore scan={latest} size="text-5xl" />
              <StatusBadge status={latest.health_status} />
            </div>
          ) : (
            <div className="text-sm text-slate-400">
              {scans === null ? 'Loading...' : <Link to="/upload" className="text-emerald-400">Run your first scan →</Link>}
            </div>
          )}
        </Card>

        <Card title="Orchard" footer={farm ? `${farm.acreage} acres · ${farm.cultivar}` : null}>
          <div className="text-4xl font-extrabold text-white">{farm ? farm.tree_count : '-'}</div>
          <p className="text-xs text-slate-400 mt-1">trees in inventory</p>
          <p className="text-xs text-emerald-400 mt-2 font-medium">{farmScans.length} recent scan(s)</p>
        </Card>

        <Card title="Latest Foliar Diagnosis" footer={latest?.disease_results?.[0] ? 'EfficientNet-B3' : null}>
          <div className={`text-2xl font-bold ${latest?.disease_results?.[0]?.is_healthy ? 'text-emerald-400' : 'text-rose-400'}`}>
            {latest ? diagnosisSummary(latest) : '-'}
          </div>
          {latest?.disease_results?.[0] && (
            <p className="text-xs text-slate-400 mt-1">
              {Math.round(latest.disease_results[0].confidence * 100)}% confidence ·{' '}
              {latest.disease_results[0].severity_percentage}% lesion area
            </p>
          )}
        </Card>

        <Card title="Latest Pest Findings" footer={latest?.unreliable_pest_warning ? 'Includes unverified classes' : null}>
          <div className="text-lg font-bold text-mango-400">{latest ? pestSummary(latest) : '-'}</div>
        </Card>
      </div>

      <div className="glass-panel rounded-2xl p-6 border-slate-700/60 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <span>🌦️</span><span>7-Day Disease Spread Forecast</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Open-Meteo forecast for {farm ? farm.name : 'the default Krishnagiri location'}
            </p>
          </div>
          {weather?.current_temperature != null && (
            <div className="text-right text-xs text-slate-300">
              <span className="font-semibold text-emerald-400">{weather.current_temperature}°C</span>
              <span className="text-slate-500 mx-1.5">•</span>
              <span>RH: {weather.current_humidity}%</span>
            </div>
          )}
        </div>

        {loadingWeather ? (
          <div className="py-8 text-center text-xs text-slate-500 animate-pulse">Fetching forecast...</div>
        ) : weatherError ? (
          <div className="py-6 text-center text-xs text-rose-300">{weatherError}</div>
        ) : (
          <>
            {weather.pathogen_alerts.map((a) => (
              <div key={a} className="text-xs text-rose-200 bg-rose-500/10 border border-rose-500/20 rounded-lg p-2.5">{a}</div>
            ))}
            <div className="grid grid-cols-1 md:grid-cols-7 gap-3 pt-2">
              {weather.forecast_days.map((day) => (
                <div
                  key={day.date}
                  title={day.dominant_threat}
                  className="bg-slate-800/50 rounded-xl p-3 border border-slate-700/50 flex flex-col justify-between text-center space-y-2"
                >
                  <div className="text-[11px] font-semibold text-slate-400">
                    {new Date(`${day.date}T00:00:00`).toLocaleDateString(undefined, { weekday: 'short', month: 'numeric', day: 'numeric' })}
                  </div>
                  <div className="text-sm font-bold text-white">
                    {fmt(day.temperature_max)}° / <span className="text-slate-400 font-normal">{fmt(day.temperature_min)}°</span>
                  </div>
                  <div className="text-[10px] text-slate-400">💧 {fmt(day.relative_humidity_mean)}% · {fmt(day.precipitation_sum, 1)} mm</div>
                  <div className="text-[10px] text-slate-400 leading-tight">
                    Anthr. {day.anthracnose_risk} · Mildew {day.powdery_mildew_risk} · Canker {day.bacterial_canker_risk}
                  </div>
                  <div className={`text-[10px] font-bold py-0.5 px-1.5 rounded uppercase tracking-wider ${RISK_STYLES[day.disease_spread_risk]}`}>
                    {day.disease_spread_risk} Risk
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="glass-panel rounded-2xl p-6 border-slate-700/60 space-y-4">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
          <span>🛡️</span><span>Interventions From Latest Scan</span>
        </h3>
        {latest?.recommendations?.length ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {latest.recommendations.map((r, i) => (
              <div key={i} className="p-3.5 rounded-xl bg-slate-800/60 border border-slate-700/50 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-emerald-400">{r.product_name}</span>
                  <span className="text-[10px] font-semibold text-slate-400 bg-slate-700/60 px-2 py-0.5 rounded">{r.treatment_type}</span>
                </div>
                <p className="text-xs text-slate-300">{r.condition}: {r.dosage_rate}</p>
                <p className="text-[10px] text-slate-500 italic">Source: {r.citation_source}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-slate-500 italic">
            {latest ? 'The latest scan produced no treatment recommendations.' : 'Run a scan to receive recommendations.'}
          </p>
        )}
      </div>
    </div>
  )
}
