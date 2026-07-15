"""Serving configuration, 12-factor style (environment variables with safe defaults).

Deploy hosts (HF Spaces / Render / Fly) inject config via env, so the API reads from the
environment rather than a Hydra tree. CORS defaults to local dev origins and must be
locked to the deployed frontend origin in production (``WILDLIFE_CORS_ORIGINS``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from wildlife.serve.preprocess import PreprocessConfig


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val and val.strip() else default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val and val.strip() else default


def _env_list(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if not val or not val.strip():
        return default
    return [o.strip() for o in val.split(",") if o.strip()]


@dataclass
class ServeConfig:
    model_path: str | None = None
    taxonomy_path: str = "configs/taxonomy/birds.yaml"
    allow_stub: bool = False
    # Interactive API docs (/docs, /redoc, /openapi.json) leak the full schema; off in prod.
    enable_docs: bool = False

    top_k: int = 5
    low_confidence_threshold: float = 0.35

    # Input safety
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_resolution: int = 4096  # longest side; larger is downscaled server-side
    allowed_content_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
        "application/octet-stream",  # some clients send this for HEIC; we sniff bytes
    )

    # CORS + rate limiting
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    rate_limit_per_minute: int = 60

    # Preprocessing (must match the trained model's eval transform)
    image_size: int = 224
    resize_ratio: float = 1.14

    @property
    def preprocess(self) -> PreprocessConfig:
        return PreprocessConfig(image_size=self.image_size, resize_ratio=self.resize_ratio)

    @classmethod
    def from_env(cls) -> ServeConfig:
        model_dir = os.getenv("WILDLIFE_MODEL_DIR")
        default_model = os.getenv("WILDLIFE_MODEL_PATH")
        if default_model is None and model_dir:
            candidate = Path(model_dir) / "model.onnx"
            default_model = str(candidate) if candidate.exists() else None

        return cls(
            model_path=default_model,
            taxonomy_path=os.getenv("WILDLIFE_TAXONOMY", "configs/taxonomy/birds.yaml"),
            allow_stub=_env_bool("WILDLIFE_ALLOW_STUB", False),
            enable_docs=_env_bool("WILDLIFE_ENABLE_DOCS", False),
            top_k=_env_int("WILDLIFE_TOP_K", 5),
            low_confidence_threshold=_env_float("WILDLIFE_LOW_CONF", 0.35),
            max_upload_bytes=_env_int("WILDLIFE_MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
            max_resolution=_env_int("WILDLIFE_MAX_RESOLUTION", 4096),
            cors_origins=_env_list(
                "WILDLIFE_CORS_ORIGINS",
                ["http://localhost:5173", "http://127.0.0.1:5173"],
            ),
            rate_limit_per_minute=_env_int("WILDLIFE_RATE_LIMIT", 60),
            image_size=_env_int("WILDLIFE_IMAGE_SIZE", 224),
            resize_ratio=_env_float("WILDLIFE_RESIZE_RATIO", 1.14),
        )
