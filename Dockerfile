# Serving image for the FastAPI inference API (built out in Phase 9).
# CPU-only base — the quantized ONNX model is what makes free-tier hosting viable.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WILDLIFE_MODEL_DIR=/app/models

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

# CPU torch + serving extras only (no CUDA in the serving image).
RUN pip install -U pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e ".[serve]"

# Baked-in model artifacts (populated by the export step / CI).
COPY outputs/serving/ ./models/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "wildlife.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
