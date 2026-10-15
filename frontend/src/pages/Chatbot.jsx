import React, { useState } from 'react'
import { errorMessage, sendChatMessage } from '../services/api'
import toast from 'react-hot-toast'

export default function Chatbot() {
  const [language, setLanguage] = useState('en') // 'en' or 'ta'
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([
    {
      sender: 'assistant',
      text: 'Hello! I am your Mango Orchard Advisor. Ask about diseases (Anthracnose, Powdery Mildew, Bacterial Canker, Sooty Mold) or pests (Fruit Fly, Mealy Bug, Scale Insect, Mango Hopper, Stem Borer). Answers are keyword-matched to ICAR and TNAU-cited guidance.',
      sources: ['ICAR-CISH Lucknow', 'TNAU Agritech Portal']
    }
  ])
  const [loading, setLoading] = useState(false)

  const handleLanguageChange = (lang) => {
    setLanguage(lang)
    if (lang === 'ta') {
      setMessages((prev) => [
        ...prev,
        {
          sender: 'assistant',
          text: 'வணக்கம்! நான் உங்கள் மாந்தோட்ட AI ஆலோசகர். இலைக்கருகல், சாம்பல் நோய் அல்லது பூச்சி தாக்குதல் மேலாண்மை பற்றி நீங்கள் தமிழில் கேட்கலாம்.',
          sources: ['TNAU Agritech Portal']
        }
      ])
    } else {
      setMessages((prev) => [
        ...prev,
        {
          sender: 'assistant',
          text: 'Switched to English. Ask any question regarding disease diagnostics, weather risks, or ICAR treatment guidelines.',
          sources: ['ICAR-CISH Lucknow']
        }
      ])
    }
  }

  const handleSend = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userText = input
    setInput('')
    setMessages((prev) => [...prev, { sender: 'user', text: userText }])
    setLoading(true)

    try {
      const res = await sendChatMessage(userText, language)
      setMessages((prev) => [
        ...prev,
        {
          sender: 'assistant',
          text: res.data.reply,
          sources: res.data.sources
        }
      ])
    } catch (err) {
      toast.error(errorMessage(err, 'Failed to receive response from advisor bot.'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto h-[calc(100vh-10rem)] flex flex-col space-y-4">
      {/* Header with Bilingual Toggle */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4 shrink-0">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span>💬</span>
            <span>Bilingual Orchard Advisor</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Verified agricultural advice in English & Tamil (தமிழ்)
          </p>
        </div>

        {/* Language Pill Selector */}
        <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800">
          <button
            type="button"
            onClick={() => handleLanguageChange('en')}
            className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
              language === 'en'
                ? 'bg-emerald-600/30 text-emerald-300 border border-emerald-500/40'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            English
          </button>
          <button
            type="button"
            onClick={() => handleLanguageChange('ta')}
            className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
              language === 'ta'
                ? 'bg-emerald-600/30 text-emerald-300 border border-emerald-500/40'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            தமிழ் (Tamil)
          </button>
        </div>
      </div>

      {/* Chat Messages Log */}
      <div className="flex-1 glass-panel rounded-2xl p-6 overflow-y-auto space-y-4 border-slate-700/60">
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`flex ${m.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xl rounded-2xl p-4 text-xs leading-relaxed ${
                m.sender === 'user'
                  ? 'bg-emerald-600 text-white rounded-br-none'
                  : 'bg-slate-800/90 text-slate-200 border border-slate-700 rounded-bl-none shadow-lg'
              }`}
            >
              <p className="whitespace-pre-line">{m.text}</p>
              {m.sources && m.sources.length > 0 && (
                <div className="mt-2.5 pt-2 border-t border-slate-700/60 flex items-center gap-1.5 text-[10px] text-emerald-400">
                  <span>📚 Source:</span>
                  <span className="italic">{m.sources.join(', ')}</span>
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-800/90 text-slate-400 border border-slate-700 rounded-2xl p-4 text-xs flex items-center gap-2">
              <span className="animate-bounce">●</span>
              <span className="animate-bounce [animation-delay:0.2s]">●</span>
              <span className="animate-bounce [animation-delay:0.4s]">●</span>
              <span className="text-[11px] ml-1">Consulting ICAR/TNAU knowledge repository...</span>
            </div>
          </div>
        )}
      </div>

      {/* Query Input Bar */}
      <form onSubmit={handleSend} className="flex gap-3 shrink-0">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            language === 'ta'
              ? 'மா நோய் அல்லது பூச்சி தாக்குதல் பற்றி கேளுங்கள் (எ.கா. ஆந்த்ராக்னோஸ் மருந்தளவு என்ன?)...'
              : 'Ask about mango diseases, dosages, or pest management (e.g. How to treat Anthracnose?)...'
          }
          className="flex-1 bg-slate-950 border border-slate-700 rounded-xl px-4 py-3 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 transition-colors"
        />
        <button
          type="submit"
          disabled={!input.trim() || loading}
          className="px-5 py-3 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white font-medium text-xs flex items-center gap-2 shadow-lg shadow-emerald-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          <span>Send</span>
          <span>✈️</span>
        </button>
      </form>
    </div>
  )
}
