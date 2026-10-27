import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base './': the site is served from /phoenix-museum-qa-evals/ on GitHub Pages.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true, sourcemap: false },
  server: { port: 5173, strictPort: true },
});
