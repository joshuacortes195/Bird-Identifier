"""Shared fixtures: a synthetic prepared dataset so data tests run without downloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wildlife.data.schema import Sample, write_manifest
from wildlife.data.taxonomy import TaxonEntry, Taxonomy, write_taxonomy


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> dict:
    """Create a tiny 3-class dataset (images + manifests + taxonomy) on disk.

    Returns paths so tests can build a SpeciesDataset / registry dataset against it.
    """
    num_classes = 3
    per_class = 4
    image_root = tmp_path / "images"
    processed = tmp_path / "processed"
    image_root.mkdir()
    processed.mkdir()

    entries = [
        TaxonEntry(idx=i, class_id=str(100 + i), common_name=f"Species {i}", supercategory="bird")
        for i in range(num_classes)
    ]
    write_taxonomy(Taxonomy(name="synthetic", entries=entries), processed / "taxonomy.yaml")

    # Spatial gradients (not solid colors) so cropping visibly changes the image.
    h, w = 80, 100
    gx = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    gy = np.linspace(0, 255, h, dtype=np.uint8)[:, None]

    samples: list[Sample] = []
    for c in range(num_classes):
        for k in range(per_class):
            arr = np.zeros((h, w, 3), dtype=np.uint8)
            arr[:, :, 0] = gx  # horizontal gradient
            arr[:, :, 1] = gy  # vertical gradient
            arr[:, :, 2] = (c * 80 + k * 10) % 256  # class/instance tint
            img = Image.fromarray(arr, "RGB")
            rel = f"cls{c}/img{k}.png"
            (image_root / f"cls{c}").mkdir(exist_ok=True)
            img.save(image_root / rel)
            # A centered bbox for crop tests.
            samples.append(
                Sample(
                    image_path=rel,
                    class_idx=c,
                    class_id=str(100 + c),
                    supercategory="bird",
                    bbox=(20, 15, 50, 40),
                )
            )

    write_manifest(samples, processed / "train.csv")
    write_manifest(samples[:3], processed / "val.csv")
    write_manifest(samples[:3], processed / "test.csv")

    return {
        "processed_dir": processed,
        "image_root": image_root,
        "num_classes": num_classes,
        "num_train": len(samples),
    }
