"""Concrete datasets, self-registered by name (registry seam #3).

NABirds and CUB share one implementation — they differ only in where their files
live, which comes from config. A future iNaturalist dataset registers here too; the
training/eval code only ever asks the registry for a dataset by its config name.

Import this module (done in ``wildlife.data.__init__``) to trigger registration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wildlife.data.base import SpeciesDataset
from wildlife.data.registry import register_dataset
from wildlife.data.taxonomy import load_taxonomy


def _resolve_image_root(image_root: str | Path) -> Path:
    """Find the directory containing images.txt (archives nest under a top folder)."""
    image_root = Path(image_root)
    if (image_root / "images.txt").exists():
        return image_root
    matches = list(image_root.rglob("images.txt"))
    return matches[0].parent if matches else image_root


class _ManifestDataset(SpeciesDataset):
    """Shared base: build a split from a processed dir + raw image root."""

    def __init__(
        self,
        split: str,
        processed_dir: str | Path,
        image_root: str | Path,
        *,
        transform=None,
        **kwargs: Any,
    ) -> None:
        processed_dir = Path(processed_dir)
        taxonomy = load_taxonomy(processed_dir / "taxonomy.yaml")
        super().__init__(
            manifest_path=processed_dir / f"{split}.csv",
            data_root=_resolve_image_root(image_root),
            taxonomy=taxonomy,
            transform=transform,
            **kwargs,
        )


@register_dataset("nabirds")
class NABirdsDataset(_ManifestDataset):
    """NABirds — 555 categories, ~48.5k images, with bounding boxes."""


@register_dataset("cub")
class CUBDataset(_ManifestDataset):
    """CUB-200-2011 — 200 categories, ~11.8k images (smoke/fast iteration)."""


# NOTE (seam): iNaturalist would register here later, e.g.
#   @register_dataset("inat")
#   class INatDataset(_ManifestDataset): ...
# It is intentionally NOT implemented in this bird build.
