import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// WHY a dev proxy: the React dev server runs on :5173 and FastAPI on
// :8000. Proxying /api/* to the backend lets the frontend call relative
// paths ("/api/predict") in both dev and production, instead of hardcoding
// localhost:8000 (which would break once deployed behind a real domain).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
