import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Backend on 8000. Keeps the frontend origin-clean so no CORS in dev.
      "/api": "http://localhost:8000",
      "/stream": { target: "ws://localhost:8000", ws: true },
    },
  },
});
