import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
})

// Request interceptor to attach JWT auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('mango_auth_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export const checkHealth = () => api.get('/health')

export const analyzeImage = (formData) => api.post('/api/analyze/', formData, {
  headers: { 'Content-Type': 'multipart/form-data' }
})

export const getWeatherForecast = (lat = 12.9716, lon = 77.5946) => 
  api.get(`/api/weather/forecast?latitude=${lat}&longitude=${lon}`)

export const sendChatMessage = (message, language = 'en') => 
  api.post('/api/chatbot/', { message, language })

export const getFarms = () => api.get('/api/farm/')

export const getReportUrl = (scanId) => `${API_BASE_URL}/api/report/generate/${scanId}`

export default api
