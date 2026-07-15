# Phase 11 — Deployment

Two pieces: the **API** (Dockerized, on a CPU host) and the **frontend** (static, on Netlify).
Everything is wired; this is gated only on your hosting accounts.

## Backend API (Docker → CPU host)

The lean, torch-free image (`Dockerfile`) serves the ONNX model. It expects
`outputs/serving/model.onnx` (from `scripts/export.py`) and `configs/taxonomy/birds.yaml`,
both COPYed in.

**Hugging Face Spaces (Docker)** — free tier, simplest:
1. Create a Docker Space, push this repo (or the built image).
2. Set Space secrets/vars: `WILDLIFE_CORS_ORIGINS=https://<your-site>.netlify.app`.
3. It serves on port 8000 over HTTPS at `https://<user>-<space>.hf.space`.

**Render (free tier, no credit card)** — the current target. `render.yaml` is a Blueprint:
in Render, **New → Blueprint → connect this repo**; it provisions a Docker web service from
the `Dockerfile`, on the **Free** plan, with `/health` as the health check.
- The image bakes in the **85 MB int8 ONNX** model (committed to the repo, under GitHub's
  100 MB limit). The 353 MB fp32 model would OOM Render's 512 MB free RAM, so int8 is used
  (slower — ~500 ms/img — but fits comfortably).
- The container binds to Render's injected `$PORT` (see `Dockerfile` CMD).
- After the frontend is live, set **`WILDLIFE_CORS_ORIGINS`** to its origin in the Render
  dashboard (Environment tab) and redeploy.
- Free-tier caveat: the service **sleeps after ~15 min idle**; the first request then takes
  ~30–60 s to wake + load the model. Warm requests are ~500 ms. Note this in the README.

**Railway / Fly.io** — same image; set the same env vars. Ensure HTTPS (required for
mobile camera access and to avoid mixed-content from the HTTPS frontend).

Key env vars (see `docs/API.md` for all): `WILDLIFE_MODEL_PATH`, `WILDLIFE_TAXONOMY`,
`WILDLIFE_CORS_ORIGINS`, `WILDLIFE_RATE_LIMIT`.

```bash
# Build & run the API image locally to sanity-check before deploying:
docker build -t bird-api .
docker run -p 8000:8000 -e WILDLIFE_CORS_ORIGINS=http://localhost:5173 bird-api
```

## Frontend (Netlify)

`netlify.toml` sets base `frontend/`, build `npm run build`, publish `dist`, SPA fallback.

1. Connect the GitHub repo in Netlify (auto-deploy on push).
2. Set environment variable **`VITE_API_BASE_URL`** = your deployed API URL
   (e.g. `https://<user>-<space>.hf.space`).
3. Deploy. To repoint at a new API later, change that one variable and redeploy.

## Cross-device smoke test (the acceptance bar)

From a phone on **cellular** (not just Wi-Fi): open the Netlify URL → upload/shoot a bird →
get top-k + Grad-CAM. Note the free-tier cold-start latency honestly in the README.

## What I need from you

- A host account for the API (HF Spaces / Render / Railway / Fly) — tell me which and I'll
  finalize the config.
- A Netlify account connected to this GitHub repo.
- A trained checkpoint exported to `outputs/serving/model.onnx` (Phase 8).
