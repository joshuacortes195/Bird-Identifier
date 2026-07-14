"""FastAPI inference service (Phase 9).

Loads the model **once at startup** and serves:

* ``POST /predict`` — multipart image upload -> top-k species JSON (common + scientific
  name, confidence, inference time). HEIC/HEIF and EXIF orientation handled; oversized
  images are downscaled server-side; non-images are rejected with clean errors.
* ``GET /health`` — liveness/readiness for the host's health checks.

CORS is locked to configured origins; a lightweight in-memory rate limiter throttles per
client. The model sits behind the :class:`~wildlife.serve.predictor.Predictor` interface,
so the ONNX backend and the dev stub are interchangeable.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from wildlife.data.imageio import load_image_from_bytes
from wildlife.serve.config import ServeConfig
from wildlife.serve.predictor import Predictor, build_predictor

logger = logging.getLogger("wildlife.serve")


def _configure_logging() -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('{"level":"%(levelname)s","msg":"%(message)s"}'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- schemas


class PredictionOut(BaseModel):
    rank: int
    class_id: str
    common_name: str
    scientific_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class ModelInfoOut(BaseModel):
    name: str
    backend: str
    num_classes: int
    input_size: int


class PredictResponse(BaseModel):
    predictions: list[PredictionOut]
    top_prediction: PredictionOut
    low_confidence: bool
    inference_ms: float
    model: ModelInfoOut
    gradcam_png_base64: str | None = None


class HealthResponse(BaseModel):
    status: str
    ready: bool
    model: ModelInfoOut | None = None


class ErrorResponse(BaseModel):
    error: dict


# --------------------------------------------------------------------------- errors


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- rate limit


class RateLimiter:
    """Fixed-window-ish sliding limiter, per client IP, in memory (single process)."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        window = self._hits[client]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= self.per_minute:
            raise APIError(429, "rate_limited", "Too many requests. Try again shortly.")
        window.append(now)


# --------------------------------------------------------------------------- app


def create_app(config: ServeConfig | None = None, predictor: Predictor | None = None) -> FastAPI:
    """Build the app. ``predictor`` can be injected (tests); otherwise built at startup."""
    _configure_logging()
    cfg = config or ServeConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if predictor is not None:
            app.state.predictor = predictor
        else:
            app.state.predictor = build_predictor(
                model_path=cfg.model_path,
                taxonomy_path=cfg.taxonomy_path,
                preprocess_cfg=cfg.preprocess,
                allow_stub=cfg.allow_stub,
            )
        logger.info(f"model loaded: {app.state.predictor.info}")
        yield
        app.state.predictor = None

    app = FastAPI(title="Wildlife Classifier API", version="0.1.0", lifespan=lifespan)
    app.state.config = cfg
    app.state.limiter = RateLimiter(cfg.rate_limit_per_minute)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(APIError)
    async def _api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    def get_predictor(request: Request) -> Predictor:
        pred = getattr(request.app.state, "predictor", None)
        if pred is None:
            raise APIError(503, "model_unavailable", "Model is not loaded yet.")
        return pred

    def rate_limit(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        request.app.state.limiter.check(client)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        pred = getattr(request.app.state, "predictor", None)
        if pred is None:
            return HealthResponse(status="starting", ready=False, model=None)
        info = pred.info
        return HealthResponse(
            status="ok",
            ready=True,
            model=ModelInfoOut(
                name=info.name,
                backend=info.backend,
                num_classes=info.num_classes,
                input_size=info.input_size,
            ),
        )

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
        },
    )
    async def predict(
        request: Request,
        file: UploadFile = File(...),
        top_k: int = Form(default=None),
        include_gradcam: bool = Form(default=False),
        _: None = Depends(rate_limit),
        pred: Predictor = Depends(get_predictor),
    ) -> PredictResponse:
        # Content-type gate (bytes are re-sniffed by the decoder below).
        if file.content_type and file.content_type not in cfg.allowed_content_types:
            raise APIError(
                415,
                "unsupported_media_type",
                f"Unsupported content type '{file.content_type}'. Upload a JPEG/PNG/WEBP/HEIC image.",
            )

        # Read with a hard size cap (don't trust Content-Length).
        data = await _read_capped(file, cfg.max_upload_bytes)

        try:
            img = load_image_from_bytes(data)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise APIError(422, "invalid_image", "Could not decode the file as an image.") from exc

        img = _downscale(img, cfg.max_resolution)

        k = top_k if top_k and top_k > 0 else cfg.top_k
        t0 = time.perf_counter()
        preds = pred.predict(img, top_k=k)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        if not preds:
            raise APIError(500, "empty_prediction", "Model returned no predictions.")

        out = [
            PredictionOut(
                rank=p.rank,
                class_id=p.class_id,
                common_name=p.common_name,
                scientific_name=p.scientific_name,
                confidence=round(min(max(p.confidence, 0.0), 1.0), 6),
            )
            for p in preds
        ]
        top = out[0]
        gradcam_b64: str | None = None
        if include_gradcam:
            if pred.supports_gradcam:
                try:
                    gradcam_b64 = pred.gradcam_png(img)
                except Exception:  # noqa: BLE001 - overlay is best-effort, never fail the prediction
                    logger.exception("gradcam generation failed; returning null overlay")
            else:
                logger.info("gradcam requested but backend does not support it; returning null")

        return PredictResponse(
            predictions=out,
            top_prediction=top,
            low_confidence=top.confidence < cfg.low_confidence_threshold,
            inference_ms=round(inference_ms, 2),
            model=ModelInfoOut(
                name=pred.info.name,
                backend=pred.info.backend,
                num_classes=pred.info.num_classes,
                input_size=pred.info.input_size,
            ),
            gradcam_png_base64=gradcam_b64,
        )

    return app


async def _read_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Read the upload in chunks, rejecting once it exceeds ``max_bytes``."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 256)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise APIError(
                413,
                "payload_too_large",
                f"Image exceeds the {max_bytes // (1024 * 1024)} MB upload limit.",
            )
        chunks.append(chunk)
    if total == 0:
        raise APIError(422, "empty_file", "The uploaded file is empty.")
    return b"".join(chunks)


def _downscale(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longest side <= ``max_side`` (protects CPU + memory)."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)


# Module-level ASGI app for `uvicorn wildlife.serve.app:app`.
app = create_app()
