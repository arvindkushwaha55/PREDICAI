import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// The Rust edge API does not send CORS headers, so the browser cannot call
// http://localhost:18080 directly from this origin. During `npm run dev`
// the dashboard calls the same-origin path /api/* instead, and Vite
// proxies it to the API. A statically served production build has no
// proxy — the Live API panel then honestly reports the API as unreachable
// (see README.md).
export default defineConfig({
  // GitHub Pages serves the dashboard at /PREDICAI/ (project site); the
  // Pages workflow sets GITHUB_PAGES=true so asset URLs get that prefix.
  // Local dev and other builds stay at the root path.
  base: process.env.GITHUB_PAGES === 'true' ? '/PREDICAI/' : '/',
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:18080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
