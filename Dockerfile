# Serving image for the FastAPI inference API (Phase 9).
# Torch-FREE by design: the app runs on ONNX Runtime + Pillow + NumPy, which keeps the
# image small enough for free-tier CPU hosts (HF Spaces / Render / Fly). The quantized
# ONNX model (Phase 8) is what makes this viable.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WILDLIFE_MODEL_PATH=/app/models/model.onnx \
    WILDLIFE_TAXONOMY=/app/configs/taxonomy/birds.yaml

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Serve runtime deps only — no torch, timm, hydra, or grad-cam in the serving image.
RUN pip install -U pip \
    && pip install \
        "fastapi>=0.112" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9" \
        "onnxruntime>=1.19" "pillow>=10.4" "pillow-heif>=0.18" "numpy>=1.26,<3" "pyyaml>=6"

COPY pyproject.toml README.md ./
COPY src/ ./src/
# Install the package without pulling its heavy ML core deps (already have serve runtime).
RUN pip install --no-deps .

# Taxonomy (label space) + baked-in model artifacts (populated by scripts/export.py / CI).
COPY configs/taxonomy/ ./configs/taxonomy/
COPY outputs/serving/ ./models/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "wildlife.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
