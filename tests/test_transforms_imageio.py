"""Transform determinism/shape + bbox-crop correctness."""

from __future__ import annotations

import torch
from PIL import Image

from wildlife.data.imageio import crop_to_bbox
from wildlife.data.transforms import (
    TransformConfig,
    build_eval_transform,
    build_train_transform,
    denormalize,
)


def _img():
    return Image.new("RGB", (200, 150), color=(120, 60, 30))


def test_eval_transform_is_deterministic():
    cfg = TransformConfig(image_size=64)
    tf = build_eval_transform(cfg)
    a = tf(_img())
    b = tf(_img())
    assert torch.equal(a, b)
    assert a.shape == (3, 64, 64)


def test_train_transform_shape_and_randomness():
    cfg = TransformConfig(image_size=64, random_erasing=0.0)
    tf = build_train_transform(cfg)
    out = tf(_img())
    assert out.shape == (3, 64, 64)


def test_denormalize_range():
    cfg = TransformConfig(image_size=32)
    t = build_eval_transform(cfg)(_img())
    d = denormalize(t, cfg)
    assert float(d.min()) >= 0.0 and float(d.max()) <= 1.0


def test_crop_to_bbox_basic():
    img = _img()  # 200x150
    cropped = crop_to_bbox(img, (50, 40, 60, 40), pad_frac=0.0)
    assert cropped.size == (60, 40)


def test_crop_to_bbox_clamps_to_bounds():
    img = _img()  # 200x150
    # bbox near the edge with big padding must clamp, not overflow.
    cropped = crop_to_bbox(img, (180, 130, 30, 30), pad_frac=1.0)
    w, h = cropped.size
    assert 0 < w <= 200 and 0 < h <= 150


def test_crop_to_bbox_degenerate_returns_original():
    img = _img()
    out = crop_to_bbox(img, (0, 0, 0, 0), pad_frac=0.0)
    assert out.size == img.size
