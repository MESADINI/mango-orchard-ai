import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backend = process.env.VITE_PROXY_TARGET || 'http://localhost:8000'
const proxied = { target: backend, changeOrigin: true }

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': proxied,
      '/media': proxied,
      '/health': proxied,
    }
  }
})
