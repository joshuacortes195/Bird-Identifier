"""Dataloader construction + class-imbalance handling.

Both levers are config options so their effect can be measured:
  * a WeightedRandomSampler (balances classes at the batch level), and
  * class-weighted loss weights (returned for the criterion).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from wildlife.data.base import SpeciesDataset, default_collate_dict
from wildlife.data.registry import build_dataset
from wildlife.data.transforms import (
    TransformConfig,
    build_eval_transform,
    build_train_transform,
)


@dataclass
class LoaderConfig:
    dataset: str = "cub"
    processed_dir: str = "data/processed/cub"
    image_root: str = "data/raw/cub200"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    bbox_crop: bool = False
    bbox_pad_frac: float = 0.1
    weighted_sampler: bool = False


def make_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Inverse-frequency sampler so rare classes appear as often as common ones."""
    counts = Counter(labels)
    weights = [1.0 / counts[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Normalized inverse-frequency weights for a class-weighted loss."""
    counts = Counter(labels)
    w = torch.tensor(
        [len(labels) / (num_classes * max(1, counts.get(c, 0))) for c in range(num_classes)],
        dtype=torch.float32,
    )
    return w * num_classes / w.sum()


def build_dataset_for_split(cfg: LoaderConfig, split: str, tcfg: TransformConfig) -> SpeciesDataset:
    is_train = split == "train"
    transform = build_train_transform(tcfg) if is_train else build_eval_transform(tcfg)
    return build_dataset(
        cfg.dataset,
        split=split,
        processed_dir=cfg.processed_dir,
        image_root=cfg.image_root,
        transform=transform,
        bbox_crop=cfg.bbox_crop,
        bbox_pad_frac=cfg.bbox_pad_frac,
    )


def build_dataloader(cfg: LoaderConfig, split: str, tcfg: TransformConfig) -> DataLoader:
    dataset = build_dataset_for_split(cfg, split, tcfg)
    is_train = split == "train"

    sampler = None
    shuffle = is_train
    if is_train and cfg.weighted_sampler:
        sampler = make_sampler(dataset.labels())
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        drop_last=is_train,
        collate_fn=default_collate_dict,
    )
