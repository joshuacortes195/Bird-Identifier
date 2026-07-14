# Bird Identifier — web app (Phase 10)

Cross-device React app: upload a bird photo (file, drag-and-drop, or phone camera) and get
top-k species with confidence, scientific names, and a Grad-CAM "what did it see?" overlay.
Talks to the FastAPI inference service (see `../docs/API.md`).

**Stack:** React 19 · TypeScript (strict) · Vite 6 · Tailwind CSS 4. Fully typed API client,
no `any`. Dark/light themes (system default + manual toggle, no flash on load).

## Develop

```bash
npm install
cp .env.example .env          # set VITE_API_BASE_URL (default http://localhost:8000)
npm run dev                   # http://localhost:5173, also exposed on your LAN for phone testing
```

Run the backend alongside it. With no trained model yet, use the deterministic stub:

```bash
cd .. && make serve-dev       # API on :8000 with a stub predictor + CORS for :5173
```

To try it from your phone on the same Wi-Fi: `npm run dev` prints a `Network:` URL; open it
on the phone. Point the app at your machine's LAN IP via `VITE_API_BASE_URL`, and add that
origin to the API's `WILDLIFE_CORS_ORIGINS`.

## Build / verify

```bash
npm run build       # tsc -b (strict typecheck) + vite production build -> dist/
npm run preview     # serve the production build locally
```

## Structure

| Path | Purpose |
|------|---------|
| `src/api/` | Typed client (`client.ts`) + contract types (`types.ts`) mirroring the API. |
| `src/lib/image.ts` | Client-side downscale/compress before upload; HEIC passthrough. |
| `src/hooks/usePredict.ts` | Upload → predict state machine (abortable) + on-demand Grad-CAM. |
| `src/components/` | Dropzone, ImagePreview (+ attention overlay), ResultsPanel, states, theme toggle. |
| `src/icons/` | Inline SVG icons (no emoji, no raster). |
| `src/index.css` | Tailwind v4 theme tokens (light/dark), reduced-motion, focus styles. |

## Accessibility notes

- Visible keyboard focus rings; `prefers-reduced-motion` respected (animations disabled).
- Confidence conveyed by number + bar (never color alone); low-confidence has an explicit banner.
- Touch targets ≥44px; camera capture on mobile; screen-reader labels on icon-only controls.

## Deploy (Phase 11)

Netlify (GitHub auto-deploy). Set `VITE_API_BASE_URL` per environment to the deployed HTTPS
API, and lock the API's `WILDLIFE_CORS_ORIGINS` to the Netlify origin.
