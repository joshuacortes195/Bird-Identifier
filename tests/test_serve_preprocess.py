"""Serving preprocess: output contract + numerical parity with the training eval transform."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from wildlife.serve.preprocess import PreprocessConfig, preprocess


def test_output_shape_dtype_and_determinism():
    img = Image.new("RGB", (300, 200), (100, 150, 200))
    a = preprocess(img)
    b = preprocess(img)
    assert a.shape == (1, 3, 224, 224)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, b)


def test_converts_non_rgb():
    img = Image.new("L", (256, 256), 128)  # grayscale
    out = preprocess(img)
    assert out.shape == (1, 3, 224, 224)


def test_parity_with_torchvision_eval_transform():
    """The torch-free path must match build_eval_transform within tight tolerance."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from wildlife.data.transforms import TransformConfig, build_eval_transform

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(220, 340, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")

    cfg = PreprocessConfig(image_size=224, resize_ratio=1.14)
    ours = preprocess(img, cfg)

    tv = build_eval_transform(TransformConfig(image_size=224, resize_ratio=1.14))
    ref = tv(img).unsqueeze(0).numpy()

    # PIL resample vs torchvision may differ by a hair; require close agreement.
    assert ours.shape == ref.shape
    assert np.abs(ours - ref).mean() < 2e-2
    assert np.abs(ours - ref).max() < 0.2
    _ = torch  # silence unused
