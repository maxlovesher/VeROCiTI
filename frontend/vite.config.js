import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    watch: {
      // Crash dumps and logs are written (and locked) by other tools; watching them made
      // the dev server die with EBUSY on Windows, taking the whole app down with it.
      ignored: ['**/*.stackdump', '**/*.log'],
    },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      '/anpr': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      '/live': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      '/dossier': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
    },
  },
})
