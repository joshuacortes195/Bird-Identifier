"""Torch-free eval preprocessing for the serving path.

The deployed CPU container runs ONNX Runtime + NumPy + Pillow only (no torch /
torchvision), which keeps the image small enough for free-tier hosting. So this module
reimplements :func:`wildlife.data.transforms.build_eval_transform` — deterministic
resize (shorter side) -> center-crop -> ImageNet normalize -> NCHW float32 — using pure
Pillow + NumPy. It must stay numerically equivalent to the training-time eval transform;
``tests/test_serve_preprocess.py`` pins that parity against torchvision when torch is present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessConfig:
    """Must match the training run's eval TransformConfig (image_size, resize_ratio, stats)."""

    image_size: int = 224
    resize_ratio: float = 1.14
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


def _resize_shorter_side(img: Image.Image, size: int) -> Image.Image:
    """Resize so the shorter side == ``size``, preserving aspect ratio (bilinear).

    Mirrors ``torchvision.transforms.Resize(size)`` given a single int.
    """
    w, h = img.size
    if w <= h:
        new_w, new_h = size, int(round(size * h / w))
    else:
        new_w, new_h = int(round(size * w / h)), size
    if (new_w, new_h) == (w, h):
        return img
    return img.resize((new_w, new_h), Image.BILINEAR)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    # If the image is smaller than the crop (shouldn't happen after resize), pad-safe clamp.
    left = max(0, left)
    top = max(0, top)
    return img.crop((left, top, left + size, top + size))


def preprocess(img: Image.Image, cfg: PreprocessConfig | None = None) -> np.ndarray:
    """PIL RGB image -> normalized ``(1, 3, H, W)`` float32 array ready for the model."""
    cfg = cfg or PreprocessConfig()
    if img.mode != "RGB":
        img = img.convert("RGB")
    resize = int(round(cfg.image_size * cfg.resize_ratio))
    img = _resize_shorter_side(img, resize)
    img = _center_crop(img, cfg.image_size)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    mean = np.asarray(cfg.mean, dtype=np.float32)
    std = np.asarray(cfg.std, dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # (3, H, W)
    return arr[np.newaxis, ...].astype(np.float32)  # (1, 3, H, W)
