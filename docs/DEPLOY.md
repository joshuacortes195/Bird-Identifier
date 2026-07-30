# Deployment

Two pieces, both live:

| Piece | Host | URL |
|-------|------|-----|
| Frontend (static React/Vite build) | Vercel | https://bird-identifier-zeta.vercel.app |
| Inference API (Dockerized FastAPI) | Render | https://bird-identifier-api.onrender.com |

Both auto-deploy from `main` on push.

> **The two URLs are coupled.** The frontend's CSP (`frontend/vercel.json`) whitelists the
> API origin in `connect-src`, and the API's `WILDLIFE_CORS_ORIGINS` whitelists the frontend
> origin. Changing either host means updating both sides — a CSP mismatch shows up in the
> browser as requests that fail before they're ever sent, which looks like the API is down.

## Backend API (Render)

`render.yaml` is a Blueprint: in Render, **New → Blueprint → connect this repo**. It
provisions a Docker web service from the `Dockerfile` on the **Free** plan with `/health`
as the health check and auto-deploy on.

- The image is **torch-free by design** — ONNX Runtime + Pillow + NumPy only — which keeps
  it small enough for free-tier CPU hosts.
- It bakes in the **85 MB int8 ONNX** model (`outputs/serving/model.int8.onnx`, committed,
  under GitHub's 100 MB limit). The 353 MB fp32 model would OOM Render's 512 MB free RAM,
  so int8 is used: slower (~500 ms/img warm) but it fits.
- The container binds Render's injected `$PORT`, falling back to 8000 locally (see the
  `Dockerfile` CMD).
- **Free-tier cold start:** the service sleeps after ~15 min idle, and the first request
  then takes ~30–60 s to wake and load the model. The frontend has a waiting screen built
  for exactly this.

Environment variables to set in the Render dashboard (Environment tab):

| Variable | Value | Notes |
|----------|-------|-------|
| `WILDLIFE_CORS_ORIGINS` | the deployed frontend origin | `sync: false` in the Blueprint, so it is deliberately not in git |
| `WILDLIFE_TOP_K` | `5` | set by the Blueprint; number of predictions returned |

`WILDLIFE_MODEL_PATH` and `WILDLIFE_TAXONOMY` are baked into the image and need no override.
See `docs/API.md` for the full env var table and defaults.

```bash
# Sanity-check the API image locally before deploying:
docker build -t bird-api .
docker run -p 8000:8000 -e WILDLIFE_CORS_ORIGINS=http://localhost:5173 bird-api
curl localhost:8000/health
```

## Frontend (Vercel)

| Setting | Value |
|---------|-------|
| Root directory | `frontend` |
| Build command | `npm run build` (`tsc -b && vite build`) |
| Output directory | `dist` |

Set **`VITE_API_BASE_URL`** to the deployed API URL under the project's Environment
Variables. To repoint at a different API later, change that variable *and* the `connect-src`
entry in `frontend/vercel.json`, then redeploy.

`frontend/vercel.json` also supplies the security headers — CSP, `X-Frame-Options`,
`X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`.

## Cross-device smoke test (the acceptance bar)

From a phone on **cellular** (not just Wi-Fi): open the frontend URL → upload or shoot a
bird → get top-k + Grad-CAM. Expect the cold-start delay on the first request.

## Alternative API hosts

The same image runs anywhere that accepts a Dockerfile and injects `$PORT` — Railway,
Fly.io, Hugging Face Spaces (Docker). Set the same env vars. HTTPS is required: mobile
camera access needs a secure context, and an HTTP API would be blocked as mixed content
from the HTTPS frontend.
