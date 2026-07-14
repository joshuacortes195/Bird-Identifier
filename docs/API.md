# Inference API (Phase 9)

FastAPI service that loads the model **once at startup** and classifies uploaded bird
photos. Torch-free at runtime: it runs on ONNX Runtime + Pillow + NumPy, so the container
stays small enough for free-tier CPU hosting.

## Run locally

```bash
# With the real exported model (after Phase 8 export):
WILDLIFE_MODEL_PATH=outputs/serving/model.onnx make serve

# Without a model — deterministic stub predictor (for frontend / contract dev):
make serve-dev          # sets WILDLIFE_ALLOW_STUB=1, hot-reload on :8000
```

`make demo` launches the optional internal Gradio UI (requires the `demo` extra).

## Endpoints

### `GET /health`

Liveness/readiness for the host's health checks.

```json
{ "status": "ok", "ready": true,
  "model": { "name": "convnextv2_base", "backend": "onnx", "num_classes": 555, "input_size": 224 } }
```

### `POST /predict`

Multipart form upload.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `file` | file (required) | — | JPEG / PNG / WEBP / HEIC. EXIF orientation + HEIC handled. |
| `top_k` | int | server `top_k` (5) | Number of ranked species to return. |
| `include_gradcam` | bool | `false` | Returns a base64 PNG overlay when the backend supports it (ONNX does not; the torch backend will). |

**200** response:

```json
{
  "predictions": [
    { "rank": 1, "class_id": "817", "common_name": "Wood Duck",
      "scientific_name": "Aix sponsa", "confidence": 0.93 }
  ],
  "top_prediction": { "rank": 1, "class_id": "817", "common_name": "Wood Duck",
                      "scientific_name": "Aix sponsa", "confidence": 0.93 },
  "low_confidence": false,
  "inference_ms": 42.1,
  "model": { "name": "convnextv2_base", "backend": "onnx", "num_classes": 555, "input_size": 224 },
  "gradcam_png_base64": null
}
```

**Errors** — JSON body `{ "error": { "code": "...", "message": "..." } }`:

| Status | `code` | When |
|--------|--------|------|
| 413 | `payload_too_large` | Upload exceeds `WILDLIFE_MAX_UPLOAD_BYTES` (default 10 MB). |
| 415 | `unsupported_media_type` | Content type not an accepted image type. |
| 422 | `invalid_image` / `empty_file` | Bytes could not be decoded as an image. |
| 429 | `rate_limited` | More than `WILDLIFE_RATE_LIMIT` requests/min from one client. |
| 503 | `model_unavailable` | Model not loaded yet. |

### Example

```bash
curl -F "file=@bird.jpg;type=image/jpeg" -F "top_k=3" http://localhost:8000/predict
curl -F "file=@iphone.heic;type=image/heic" http://localhost:8000/predict   # HEIC works
```

## Configuration (environment variables)

| Var | Default | Purpose |
|-----|---------|---------|
| `WILDLIFE_MODEL_PATH` | — | Path to the `.onnx` model. |
| `WILDLIFE_MODEL_DIR` | — | Dir containing `model.onnx` (used if `WILDLIFE_MODEL_PATH` unset). |
| `WILDLIFE_TAXONOMY` | `configs/taxonomy/birds.yaml` | Label space (class names + scientific names). |
| `WILDLIFE_ALLOW_STUB` | `0` | Allow the stub predictor when no model is present (dev only). |
| `WILDLIFE_TOP_K` | `5` | Default number of predictions. |
| `WILDLIFE_LOW_CONF` | `0.35` | Top-1 below this sets `low_confidence: true`. |
| `WILDLIFE_MAX_UPLOAD_BYTES` | `10485760` | Upload size cap. |
| `WILDLIFE_MAX_RESOLUTION` | `4096` | Longest side; larger images are downscaled server-side. |
| `WILDLIFE_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed browser origins (lock to the deployed frontend in prod). |
| `WILDLIFE_RATE_LIMIT` | `60` | Requests/min per client IP (0 disables). |
| `WILDLIFE_IMAGE_SIZE` / `WILDLIFE_RESIZE_RATIO` | `224` / `1.14` | Preprocess — must match the trained model's eval transform. |

The stub predictor is refused unless `WILDLIFE_ALLOW_STUB=1`, so production never silently
serves fake predictions.
