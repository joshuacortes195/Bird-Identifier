"""Inference behind a small interface so the API never hardcodes a model backend.

``Predictor`` is the contract the FastAPI app depends on. Two implementations ship:

* :class:`OnnxPredictor` — production path. Loads an exported ONNX model with ONNX
  Runtime (CPU) and the run's taxonomy; torch is **not** imported. This is what the
  deployed container serves. ``onnxruntime`` is imported lazily so environments without a
  wheel (e.g. Python 3.14 / macOS-x86 dev boxes) can still import this module and run the
  stub-backed tests.
* :class:`StubPredictor` — deterministic fake predictions derived from the image bytes,
  with no model file. Lets the frontend and the API contract be developed and tested
  end-to-end before a checkpoint exists. Never used in production (guarded by config).

Both return a ranked list of :class:`Prediction`; the app serializes them identically.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

from wildlife.data.taxonomy import Taxonomy, load_taxonomy
from wildlife.serve.preprocess import PreprocessConfig, preprocess


@dataclass(frozen=True)
class Prediction:
    rank: int
    class_id: str
    common_name: str
    scientific_name: str | None
    confidence: float


@dataclass(frozen=True)
class ModelInfo:
    name: str
    backend: str
    num_classes: int
    input_size: int


@runtime_checkable
class Predictor(Protocol):
    """What the API needs from any inference backend."""

    info: ModelInfo

    def predict(self, img: Image.Image, top_k: int = 5) -> list[Prediction]: ...

    @property
    def supports_gradcam(self) -> bool: ...

    def gradcam_png(self, img: Image.Image, target_category: int | None = None) -> str | None:
        """Base64 PNG attention overlay, or None if unsupported."""
        ...


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - logits.max()
    exp = np.exp(logits)
    return (exp / exp.sum()).astype(np.float64)


def _rank(probs: np.ndarray, taxonomy: Taxonomy, top_k: int) -> list[Prediction]:
    k = int(min(top_k, len(taxonomy)))
    top_idx = np.argsort(probs)[::-1][:k]
    out: list[Prediction] = []
    for rank, idx in enumerate(top_idx, start=1):
        entry = taxonomy[int(idx)]
        out.append(
            Prediction(
                rank=rank,
                class_id=entry.class_id,
                common_name=entry.common_name,
                scientific_name=entry.scientific_name,
                confidence=float(probs[idx]),
            )
        )
    return out


class OnnxPredictor:
    """CPU ONNX Runtime inference. Torch-free; ``onnxruntime`` imported lazily."""

    def __init__(
        self,
        model_path: str | Path,
        taxonomy: Taxonomy,
        *,
        preprocess_cfg: PreprocessConfig | None = None,
        name: str = "onnx-model",
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on platform wheels
            raise RuntimeError(
                "onnxruntime is required for OnnxPredictor. Install serve extras: "
                "pip install -e '.[serve]'"
            ) from exc

        self.taxonomy = taxonomy
        self.pre = preprocess_cfg or PreprocessConfig()
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self.info = ModelInfo(
            name=name,
            backend="onnx",
            num_classes=len(taxonomy),
            input_size=self.pre.image_size,
        )

    def predict(self, img: Image.Image, top_k: int = 5) -> list[Prediction]:
        x = preprocess(img, self.pre)
        logits = self._session.run(None, {self._input_name: x})[0][0]
        return _rank(_softmax(logits), self.taxonomy, top_k)

    @property
    def supports_gradcam(self) -> bool:
        # Grad-CAM needs backbone activations/gradients (the torch model, Phase 6),
        # not the frozen ONNX graph. The API reports this so the UI can hide the toggle.
        return False

    def gradcam_png(self, img: Image.Image, target_category: int | None = None) -> str | None:
        return None


class StubPredictor:
    """Deterministic, model-free predictions for local dev / API contract tests.

    Confidence is a fixed decreasing curve; the *which* classes are chosen deterministically
    from a hash of the image bytes, so the same image always yields the same ranking and
    tests are stable. Clearly not a real model — the app refuses to start with this unless
    ``allow_stub`` is set.
    """

    def __init__(self, taxonomy: Taxonomy, *, name: str = "stub") -> None:
        self.taxonomy = taxonomy
        self.info = ModelInfo(name=name, backend="stub", num_classes=len(taxonomy), input_size=224)

    def predict(self, img: Image.Image, top_k: int = 5) -> list[Prediction]:
        seed = int.from_bytes(hashlib.sha256(img.tobytes()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        k = int(min(top_k, len(self.taxonomy)))
        chosen = rng.choice(len(self.taxonomy), size=k, replace=False)
        # A plausible-looking decreasing confidence curve that sums to < 1.
        raw = np.sort(rng.random(k))[::-1]
        conf = raw / raw.sum() * float(rng.uniform(0.6, 0.98))
        out: list[Prediction] = []
        for rank, (idx, c) in enumerate(zip(chosen, conf, strict=True), start=1):
            entry = self.taxonomy[int(idx)]
            out.append(
                Prediction(
                    rank=rank,
                    class_id=entry.class_id,
                    common_name=entry.common_name,
                    scientific_name=entry.scientific_name,
                    confidence=float(c),
                )
            )
        return out

    @property
    def supports_gradcam(self) -> bool:
        return False

    def gradcam_png(self, img: Image.Image, target_category: int | None = None) -> str | None:
        return None


class TorchPredictor:
    """PyTorch backend — the only one that can produce Grad-CAM overlays.

    Heavier than ONNX (pulls torch/timm), so it's for a torch-enabled deployment or local
    use, not the lean serving image. Rebuilds the model from the checkpoint's embedded config
    and class list. NOTE: this integration must be validated against the first real Phase 5
    checkpoint — it is written against the checkpoint schema in ``utils/checkpoint.py`` but
    has not been exercised end-to-end on this dev box (no torch wheel here).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        taxonomy: Taxonomy,
        *,
        preprocess_cfg: PreprocessConfig | None = None,
        device: str = "cpu",
        name: str = "torch-model",
    ) -> None:
        import torch

        from wildlife.data.transforms import TransformConfig, build_eval_transform
        from wildlife.models.factory import ModelConfig, build_model
        from wildlife.utils.checkpoint import load_checkpoint

        self.torch = torch
        self.taxonomy = taxonomy
        self.pre = preprocess_cfg or PreprocessConfig()
        self.device = torch.device(device)

        ckpt = load_checkpoint(checkpoint_path, map_location=device)
        model_cfg_dict = (ckpt.get("config") or {}).get("model", {})
        model_cfg = ModelConfig(
            backbone=model_cfg_dict.get("backbone", ModelConfig.backbone),
            pretrained=False,  # weights come from the checkpoint
            head=model_cfg_dict.get("head", "linear"),
            head_kwargs=model_cfg_dict.get("head_kwargs", {}) or {},
            drop_rate=model_cfg_dict.get("drop_rate", 0.0),
            drop_path_rate=model_cfg_dict.get("drop_path_rate", 0.0),
        )
        self.model = build_model(model_cfg, num_classes=len(taxonomy))
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(self.device)

        self._transform = build_eval_transform(
            TransformConfig(
                image_size=self.pre.image_size,
                resize_ratio=self.pre.resize_ratio,
                mean=self.pre.mean,
                std=self.pre.std,
            )
        )
        self.info = ModelInfo(
            name=name,
            backend="torch",
            num_classes=len(taxonomy),
            input_size=self.pre.image_size,
        )

    def _tensor(self, img: Image.Image):
        return self._transform(img.convert("RGB")).unsqueeze(0).to(self.device)

    def predict(self, img: Image.Image, top_k: int = 5) -> list[Prediction]:
        x = self._tensor(img)
        with self.torch.no_grad():
            logits = self.model(x)[0].cpu().numpy()
        return _rank(_softmax(logits), self.taxonomy, top_k)

    @property
    def supports_gradcam(self) -> bool:
        return True

    def gradcam_png(self, img: Image.Image, target_category: int | None = None) -> str | None:
        from wildlife.eval.gradcam import gradcam_png_base64

        base = img.convert("RGB")
        x = self._tensor(base)
        if target_category is None:
            with self.torch.no_grad():
                target_category = int(self.model(x)[0].argmax().item())
        return gradcam_png_base64(self.model, x, base, target_category=target_category)


def build_predictor(
    *,
    model_path: str | Path | None,
    taxonomy_path: str | Path,
    preprocess_cfg: PreprocessConfig | None = None,
    allow_stub: bool = False,
) -> Predictor:
    """Select a backend from what's on disk.

    ``.onnx`` -> :class:`OnnxPredictor`. No model path (or a missing file) -> the stub,
    but only when ``allow_stub`` is true; otherwise this raises so production never
    silently serves fake predictions.
    """
    taxonomy = load_taxonomy(taxonomy_path)
    path = Path(model_path) if model_path else None

    if path and path.exists():
        if path.suffix == ".onnx":
            return OnnxPredictor(path, taxonomy, preprocess_cfg=preprocess_cfg, name=path.stem)
        if path.suffix in {".pt", ".pth"}:
            return TorchPredictor(path, taxonomy, preprocess_cfg=preprocess_cfg, name=path.stem)

    if allow_stub:
        return StubPredictor(taxonomy)

    raise FileNotFoundError(
        f"No servable model at {model_path!r}. Export one (scripts/export.py) or set "
        "WILDLIFE_ALLOW_STUB=1 for local/frontend development."
    )
