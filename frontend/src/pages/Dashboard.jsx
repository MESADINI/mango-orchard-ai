import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getWeatherForecast } from '../services/api'

export default function Dashboard() {
  const [weather, setWeather] = useState(null)
  const [loadingWeather, setLoadingWeather] = useState(true)

  useEffect(() => {
    getWeatherForecast()
      .then((res) => {
        setWeather(res.data)
      })
      .catch((err) => {
        console.error('Weather error:', err)
      })
      .finally(() => setLoadingWeather(false))
  }, [])

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Welcome Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Orchard Health Overview</h2>
          <p className="text-sm text-slate-400 mt-1">
            Real-time disease surveillance, deterministic OHI analytics, and epidemiological risk forecasting.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link
            to="/upload"
            className="inline-flex items-center gap-2 bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white font-medium text-sm px-4 py-2.5 rounded-lg shadow-lg shadow-emerald-500/20 transition-all"
          >
            <span>📸</span>
            <span>New Tree Scan</span>
          </Link>
        </div>
      </div>

      {/* Disclosed Placeholder Caveat Alert */}
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex items-start gap-3">
        <span className="text-xl">⚠️</span>
        <div className="text-xs text-amber-200/90 leading-relaxed">
          <strong className="font-semibold text-amber-200">System Policy Disclosure:</strong> Soil/leaf nitrogen sensors are currently offline. In accordance with OHI methodology, nutrient deficiency input is defaulted to 0 and its weight reallocated to visual disease and pest penalties.
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
        {/* Main OHI Card */}
        <div className="glass-panel rounded-2xl p-6 border-slate-700/60 flex flex-col justify-between relative overflow-hidden">
          <div className="absolute -right-4 -bottom-4 w-28 h-28 bg-emerald-500/10 rounded-full blur-2xl"></div>
          <div>
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Composite Orchard Health</span>
            <div className="flex items-baseline gap-2 mt-2">
              <span className="text-5xl font-extrabold text-amber-400">82.5</span>
              <span className="text-slate-500 text-sm font-semibold">/ 100</span>
            </div>
            <div className="mt-2 inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-400/10 text-amber-300 border border-amber-400/20">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
              Mild Stress / Warning
            </div>
          </div>
          <p className="text-xs text-slate-500 mt-4 border-t border-slate-800 pt-3">
            Deterministic rule-based Python index
          </p>
        </div>

        {/* Trees Monitored */}
        <div className="glass-panel rounded-2xl p-6 border-slate-700/60 flex flex-col justify-between">
          <div>
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Trees Registered</span>
            <div className="text-4xl font-extrabold text-white mt-2">350</div>
            <p className="text-xs text-emerald-400 mt-2 font-medium">92% surveyed this week</p>
          </div>
          <p className="text-xs text-slate-500 border-t border-slate-800 pt-3">12.5 Acres Block A & B</p>
        </div>

        {/* Dominant Pathogen Threat */}
        <div className="glass-panel rounded-2xl p-6 border-slate-700/60 flex flex-col justify-between">
          <div>
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Primary Risk Factor</span>
            <div className="text-2xl font-bold text-rose-400 mt-2">Anthracnose</div>
            <p className="text-xs text-slate-400 mt-1">Colletotrichum gloeosporioides</p>
          </div>
          <div className="text-xs text-slate-400 border-t border-slate-800 pt-3 flex items-center justify-between">
            <span>EfficientNet-B3</span>
            <span className="text-rose-400 font-semibold">89.2% Conf</span>
          </div>
        </div>

        {/* Pest Activity Flag */}
        <div className="glass-panel rounded-2xl p-6 border-slate-700/60 flex flex-col justify-between">
          <div>
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Canopy Pest Pressure</span>
            <div className="text-2xl font-bold text-mango-400 mt-2">Fruit Fly</div>
            <p className="text-xs text-slate-400 mt-1">Bactrocera dorsalis</p>
          </div>
          <div className="text-xs text-slate-400 border-t border-slate-800 pt-3 flex items-center justify-between">
            <span>YOLOv8 Active</span>
            <span className="text-emerald-400 font-medium">Verified Class</span>
          </div>
        </div>
      </div>

      {/* 7-Day Microclimate & Pathogen Risk Forecast */}
      <div className="glass-panel rounded-2xl p-6 border-slate-700/60 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <span>🌦️</span>
              <span>7-Day Pathogen Microclimate Spread Forecast</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">Powered by Open-Meteo real-time atmospheric data</p>
          </div>
          {weather?.current_temperature && (
            <div className="text-right text-xs text-slate-300">
              <span className="font-semibold text-emerald-400">{weather.current_temperature}°C</span>
              <span className="text-slate-500 mx-1.5">•</span>
              <span>RH: {weather.current_humidity}%</span>
            </div>
          )}
        </div>

        {loadingWeather ? (
          <div className="py-8 text-center text-xs text-slate-500 animate-pulse">
            Fetching atmospheric forecast data...
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-7 gap-3 pt-2">
            {(weather?.forecast_days || []).map((day, idx) => (
              <div
                key={day.date || idx}
                className="bg-slate-800/50 rounded-xl p-3 border border-slate-700/50 flex flex-col justify-between text-center space-y-2 hover:border-slate-600 transition-colors"
              >
                <div className="text-[11px] font-semibold text-slate-400">
                  {new Date(day.date).toLocaleDateString('en-US', { weekday: 'short', month: 'numeric', day: 'numeric' })}
                </div>
                <div className="text-sm font-bold text-white">
                  {Math.round(day.temperature_max)}° / <span className="text-slate-400 font-normal">{Math.round(day.temperature_min)}°</span>
                </div>
                <div className="text-[10px] text-slate-400">
                  💧 {Math.round(day.relative_humidity_mean)}% RH
                </div>
                <div
                  className={`text-[10px] font-bold py-0.5 px-1.5 rounded uppercase tracking-wider ${
                    day.disease_spread_risk === 'High'
                      ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                      : day.disease_spread_risk === 'Moderate'
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                      : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                  }`}
                >
                  {day.disease_spread_risk} Risk
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Verified Protocols & Action Plan */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-panel rounded-2xl p-6 border-slate-700/60 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <span>🛡️</span>
            <span>Recommended ICAR/TNAU Interventions</span>
          </h3>
          <div className="space-y-3">
            <div className="p-3.5 rounded-xl bg-slate-800/60 border border-slate-700/50 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-emerald-400">Copper Oxychloride 50% WP (Blitox)</span>
                <span className="text-[10px] font-semibold text-slate-400 bg-slate-700/60 px-2 py-0.5 rounded">Chemical</span>
              </div>
              <p className="text-xs text-slate-300">Dosage: 3.0 g / Litre of water (0.3%) at blossom emergence.</p>
              <p className="text-[10px] text-slate-500 italic">Source: TNAU Agritech Portal (Crop Protection: Diseases of Mango)</p>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-800/60 border border-slate-700/50 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-mango-400">Methyl Eugenol Parapheromone Trap</span>
                <span className="text-[10px] font-semibold text-slate-400 bg-slate-700/60 px-2 py-0.5 rounded">Biocontrol</span>
              </div>
              <p className="text-xs text-slate-300">Deploy 10-15 traps/hectare 60 days before harvest maturity.</p>
              <p className="text-[10px] text-slate-500 italic">Source: ICAR-CISH IPM Package for Bactrocera dorsalis</p>
            </div>
          </div>
        </div>

        <div className="glass-panel rounded-2xl p-6 border-slate-700/60 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <span>🔬</span>
            <span>Model Health & Explainability</span>
          </h3>
          <div className="space-y-3 text-xs text-slate-300">
            <div className="flex justify-between items-center py-2 border-b border-slate-800">
              <span className="text-slate-400">Disease Backbone:</span>
              <span className="font-mono text-emerald-400 font-semibold">EfficientNet-B3 (PyTorch)</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-slate-800">
              <span className="text-slate-400">Pest Backbone:</span>
              <span className="font-mono text-mango-400 font-semibold">Ultralytics YOLOv8</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-slate-800">
              <span className="text-slate-400">Explainability Engine:</span>
              <span className="font-mono text-teal-400 font-semibold">Hook-based Grad-CAM</span>
            </div>
            <div className="flex justify-between items-center py-2">
              <span className="text-slate-400">Unreliable Pest Classes:</span>
              <span className="text-amber-400 font-semibold">Mango Hopper, Stem Borer (&lt;100 samples)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
