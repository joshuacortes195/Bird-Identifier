"""The torch ``Dataset`` that every dataset shares, reading from split manifests.

Returns a dict per item so optional fields (supercategory, bbox) are plumbed through
without positional churn. The coarse ``supercategory`` label is constant ("bird") now
but always returned, so multi-taxon grouping works later with no change here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from wildlife.data.imageio import crop_to_bbox, load_image
from wildlife.data.schema import Sample, read_manifest
from wildlife.data.taxonomy import Taxonomy, load_taxonomy


class SpeciesDataset(Dataset):
    """Species-classification dataset backed by a CSV manifest.

    Args:
        manifest_path: CSV produced by ``prepare_splits`` (train/val/test).
        data_root: directory the manifest's ``image_path`` entries are relative to.
        taxonomy: label space (num_classes / names / supercategory map).
        transform: callable(PIL.Image) -> Tensor. If None, returns the PIL image.
        bbox_crop: crop to the subject bounding box before transform (when available).
        bbox_pad_frac: padding fraction around the bbox when cropping.
        return_supercategory: include the coarse taxon index in each item.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        taxonomy: Taxonomy,
        *,
        transform=None,
        bbox_crop: bool = False,
        bbox_pad_frac: float = 0.1,
        return_supercategory: bool = True,
    ) -> None:
        self.samples: list[Sample] = read_manifest(manifest_path)
        self.data_root = Path(data_root)
        self.taxonomy = taxonomy
        self.transform = transform
        self.bbox_crop = bbox_crop
        self.bbox_pad_frac = bbox_pad_frac
        self.return_supercategory = return_supercategory
        self._super_of_idx = taxonomy.idx_to_supercategory_idx()

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def num_classes(self) -> int:
        return self.taxonomy.num_classes

    def labels(self) -> list[int]:
        """All class indices, in dataset order (for samplers / class weighting)."""
        return [s.class_idx for s in self.samples]

    def __getitem__(self, index: int) -> dict[str, Any]:
        s = self.samples[index]
        img = load_image(self.data_root / s.image_path)
        if self.bbox_crop and s.has_bbox:
            img = crop_to_bbox(img, s.bbox, self.bbox_pad_frac)
        image = self.transform(img) if self.transform is not None else img

        item: dict[str, Any] = {
            "image": image,
            "label": s.class_idx,
        }
        if self.return_supercategory:
            item["supercategory"] = self._super_of_idx[s.class_idx]
        return item


def build_species_dataset(
    manifest_path: str | Path,
    data_root: str | Path,
    taxonomy_path: str | Path,
    **kwargs: Any,
) -> SpeciesDataset:
    """Convenience builder that loads the taxonomy from a YAML path."""
    taxonomy = load_taxonomy(taxonomy_path)
    return SpeciesDataset(manifest_path, data_root, taxonomy, **kwargs)


def default_collate_dict(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of item-dicts into a batched dict of tensors."""
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    out: dict[str, Any] = {"image": images, "label": labels}
    if "supercategory" in batch[0]:
        out["supercategory"] = torch.tensor([b["supercategory"] for b in batch], dtype=torch.long)
    return out
