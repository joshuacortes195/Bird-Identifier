"""Grad-CAM attention overlays (Phase 6).

Renders where the classifier looked — used both in the eval report (correct vs. incorrect
predictions, bird vs. background) and, via :func:`gradcam_png_base64`, by the torch serving
backend to return an overlay to the web app.

This module imports torch and ``pytorch_grad_cam`` and therefore runs on the training/CI
box, not the torch-free serving image. ConvNeXt-V2 backbones are convolutional, so the
target activations are already spatial (N, C, H, W) — no ViT-style reshape needed.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from wildlife.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def find_target_layer(model):
    """Pick the last spatial conv stage of a timm backbone for Grad-CAM.

    Tries the common ConvNeXt layout first, then falls back to the last ``nn.Conv2d``.
    """
    from torch import nn

    backbone = getattr(model, "backbone", model)
    stages = getattr(backbone, "stages", None)
    if stages is not None and len(stages) > 0:
        return stages[-1]
    last_conv = None
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("Could not locate a convolutional target layer for Grad-CAM.")
    return last_conv


def compute_gradcam(model, input_tensor, target_category: int | None = None) -> np.ndarray:
    """Return a (H, W) heatmap in [0, 1] for one image (``input_tensor``: (1, 3, H, W))."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    layer = find_target_layer(model)
    targets = None if target_category is None else [ClassifierOutputTarget(int(target_category))]
    with GradCAM(model=model, target_layers=[layer]) as cam:
        grayscale = cam(input_tensor=input_tensor, targets=targets)  # (1, H, W)
    return grayscale[0]


def _heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    """Map [0,1] -> RGB uint8 using matplotlib's 'jet' colormap."""
    from matplotlib import colormaps

    cmap = colormaps["jet"]
    rgba = cmap(np.clip(heatmap, 0.0, 1.0))  # (H, W, 4) float
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay_heatmap(base_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Composite a heatmap over the original RGB image (heatmap resized to match)."""
    base = base_img.convert("RGB")
    hm = Image.fromarray(_heatmap_to_rgb(heatmap)).resize(base.size, Image.BILINEAR)
    return Image.blend(base, hm, alpha)


def denormalize_tensor(input_tensor) -> Image.Image:
    """Recover a viewable PIL image from a normalized (1,3,H,W) tensor."""
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    arr = input_tensor.detach().cpu().numpy()[0]
    arr = np.clip(arr * std + mean, 0, 1)
    arr = (np.transpose(arr, (1, 2, 0)) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def gradcam_png_base64(
    model,
    input_tensor,
    base_img: Image.Image,
    target_category: int | None = None,
    alpha: float = 0.5,
) -> str:
    """Full path: heatmap -> overlay on ``base_img`` -> base64 PNG string (no data: prefix)."""
    heatmap = compute_gradcam(model, input_tensor, target_category)
    overlay = overlay_heatmap(base_img, heatmap, alpha)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
