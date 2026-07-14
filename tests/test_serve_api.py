"""API contract tests (Phase 9). Run torch-free with an injected StubPredictor."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wildlife.data.taxonomy import TaxonEntry, Taxonomy
from wildlife.serve.app import create_app
from wildlife.serve.config import ServeConfig
from wildlife.serve.predictor import StubPredictor


def _taxonomy(n: int = 12) -> Taxonomy:
    entries = [
        TaxonEntry(
            idx=i,
            class_id=str(100 + i),
            common_name=f"Species {i}",
            scientific_name=f"Genus specius{i}",
            supercategory="bird",
        )
        for i in range(n)
    ]
    return Taxonomy(name="test", entries=entries)


def _png_bytes(size=(64, 48), color=(120, 160, 90)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _client(cfg: ServeConfig | None = None) -> TestClient:
    cfg = cfg or ServeConfig(allow_stub=True, rate_limit_per_minute=0)
    app = create_app(config=cfg, predictor=StubPredictor(_taxonomy()))
    return TestClient(app)


def test_health_reports_ready_model():
    with _client() as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["ready"] is True
        assert body["model"]["backend"] == "stub"
        assert body["model"]["num_classes"] == 12


def test_predict_returns_ranked_topk():
    with _client() as client:
        r = client.post(
            "/predict",
            files={"file": ("bird.png", _png_bytes(), "image/png")},
            data={"top_k": "5"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["predictions"]) == 5
        ranks = [p["rank"] for p in body["predictions"]]
        assert ranks == [1, 2, 3, 4, 5]
        confs = [p["confidence"] for p in body["predictions"]]
        assert confs == sorted(confs, reverse=True)
        assert all(0.0 <= c <= 1.0 for c in confs)
        assert body["top_prediction"] == body["predictions"][0]
        assert "scientific_name" in body["predictions"][0]
        assert body["inference_ms"] >= 0
        assert isinstance(body["low_confidence"], bool)


def test_predict_is_deterministic_for_same_image():
    with _client() as client:
        img = _png_bytes()
        a = client.post("/predict", files={"file": ("a.png", img, "image/png")}).json()
        b = client.post("/predict", files={"file": ("a.png", img, "image/png")}).json()
        assert a["predictions"] == b["predictions"]


def test_rejects_non_image_content_type():
    with _client() as client:
        r = client.post(
            "/predict",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        assert r.status_code == 415
        assert r.json()["error"]["code"] == "unsupported_media_type"


def test_rejects_undecodable_image():
    with _client() as client:
        r = client.post(
            "/predict",
            files={"file": ("broken.png", b"not really a png", "image/png")},
        )
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "invalid_image"


def test_rejects_oversized_upload():
    cfg = ServeConfig(allow_stub=True, rate_limit_per_minute=0, max_upload_bytes=1024)
    app = create_app(config=cfg, predictor=StubPredictor(_taxonomy()))
    with TestClient(app) as client:
        big = _png_bytes(size=(512, 512))
        assert len(big) > 1024
        r = client.post("/predict", files={"file": ("big.png", big, "image/png")})
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "payload_too_large"


def test_rate_limiter_throttles():
    cfg = ServeConfig(allow_stub=True, rate_limit_per_minute=2)
    app = create_app(config=cfg, predictor=StubPredictor(_taxonomy()))
    with TestClient(app) as client:
        img = _png_bytes()
        codes = [
            client.post("/predict", files={"file": ("a.png", img, "image/png")}).status_code
            for _ in range(3)
        ]
        assert codes[:2] == [200, 200]
        assert codes[2] == 429


def test_gradcam_null_when_unsupported():
    with _client() as client:
        r = client.post(
            "/predict",
            files={"file": ("a.png", _png_bytes(), "image/png")},
            data={"include_gradcam": "true"},
        )
        assert r.status_code == 200
        assert r.json()["gradcam_png_base64"] is None


def test_cors_header_for_allowed_origin():
    with _client() as client:
        r = client.get("/health", headers={"origin": "http://localhost:5173"})
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


class _GradcamStub(StubPredictor):
    """A predictor that claims Grad-CAM support and returns a fixed overlay."""

    OVERLAY = "aGVsbG8="  # base64("hello")

    @property
    def supports_gradcam(self) -> bool:
        return True

    def gradcam_png(self, img, target_category=None):
        return self.OVERLAY


def test_gradcam_returned_when_backend_supports_and_requested():
    cfg = ServeConfig(allow_stub=True, rate_limit_per_minute=0)
    app = create_app(config=cfg, predictor=_GradcamStub(_taxonomy()))
    with TestClient(app) as client:
        img = _png_bytes()
        with_cam = client.post(
            "/predict",
            files={"file": ("a.png", img, "image/png")},
            data={"include_gradcam": "true"},
        ).json()
        assert with_cam["gradcam_png_base64"] == _GradcamStub.OVERLAY

        # Not requested -> null even though the backend supports it.
        without = client.post("/predict", files={"file": ("a.png", img, "image/png")}).json()
        assert without["gradcam_png_base64"] is None


def test_production_refuses_stub_without_flag():
    # No model file + allow_stub False -> build_predictor raises at startup.
    cfg = ServeConfig(allow_stub=False, model_path=None)
    app = create_app(config=cfg)  # real build path, no injected predictor
    with pytest.raises(FileNotFoundError), TestClient(app):
        pass
