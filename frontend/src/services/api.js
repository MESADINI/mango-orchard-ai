import axios from 'axios'

// Empty base URL = same origin. In development Vite proxies /api, /media and /health
// to the backend (see vite.config.js); in Docker, nginx does the same.
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
export const TOKEN_KEY = 'mango_auth_token'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // CPU inference on first request can be slow
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem(TOKEN_KEY)
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

let unauthorizedHandler = null
export const onUnauthorized = (handler) => {
  unauthorizedHandler = handler
}

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && unauthorizedHandler) {
      unauthorizedHandler()
    }
    return Promise.reject(error)
  }
)

/** Extracts a human-readable message from an axios error. */
export const errorMessage = (err, fallback = 'Request failed') => {
  const detail = err?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg
  if (err?.code === 'ECONNABORTED') return 'The server took too long to respond.'
  if (!err?.response) return 'Cannot reach the backend server.'
  return fallback
}

/** Turns a backend-relative media path (e.g. /media/results/x.jpg) into a usable URL. */
export const mediaUrl = (path) => (path ? `${API_BASE_URL}${path}` : null)

// Health
export const checkHealth = () => api.get('/health')

// Auth
export const login = (email, password) => api.post('/api/auth/login', { email, password })
export const register = (email, password, fullName) =>
  api.post('/api/auth/register', { email, password, full_name: fullName || null })
export const getMe = () => api.get('/api/auth/me')

// Analysis
export const analyzeImage = (formData) => api.post('/api/analyze/', formData)
export const listScans = (limit = 50) => api.get('/api/analyze/', { params: { limit } })
export const getScan = (scanId) => api.get(`/api/analyze/${scanId}`)

// Weather
export const getWeatherForecast = (latitude, longitude) =>
  api.get('/api/weather/forecast', { params: { latitude, longitude } })

// Chatbot
export const sendChatMessage = (message, language = 'en') =>
  api.post('/api/chatbot/', { message, language })

// Farms
export const getFarms = () => api.get('/api/farm/')
export const createFarm = (data) => api.post('/api/farm/', data)
export const updateFarm = (farmId, data) => api.put(`/api/farm/${farmId}`, data)
export const getFarmTrees = (farmId) => api.get(`/api/farm/${farmId}/trees`)
export const addTree = (farmId, data) => api.post(`/api/farm/${farmId}/trees`, data)

/**
 * Downloads the scan report (PDF, or HTML when the server lacks WeasyPrint's native
 * libraries) with the auth header and opens it in a new tab.
 */
export const openReport = async (scanId) => {
  const tab = window.open('', '_blank')
  try {
    const res = await api.get(`/api/report/generate/${scanId}`, { responseType: 'blob' })
    const url = URL.createObjectURL(res.data)
    if (tab) {
      tab.location.href = url
    } else {
      window.location.href = url
    }
    setTimeout(() => URL.revokeObjectURL(url), 60000)
  } catch (err) {
    if (tab) tab.close()
    throw err
  }
}

export default api
