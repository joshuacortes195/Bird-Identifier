"""Registry resolution + SpeciesDataset item contract, on the synthetic dataset."""

from __future__ import annotations

import pytest
import torch

from wildlife.data import available_datasets, build_dataset
from wildlife.data.base import default_collate_dict
from wildlife.data.transforms import TransformConfig, build_eval_transform


def test_expected_datasets_registered():
    names = available_datasets()
    assert "nabirds" in names
    assert "cub" in names


def test_unknown_dataset_raises():
    with pytest.raises(KeyError, match="Unknown dataset"):
        build_dataset("does-not-exist", split="train", processed_dir=".", image_root=".")


def test_dataset_item_contract(synthetic_dataset):
    tf = build_eval_transform(TransformConfig(image_size=32))
    ds = build_dataset(
        "cub",
        split="train",
        processed_dir=synthetic_dataset["processed_dir"],
        image_root=synthetic_dataset["image_root"],
        transform=tf,
    )
    assert len(ds) == synthetic_dataset["num_train"]
    assert ds.num_classes == synthetic_dataset["num_classes"]

    item = ds[0]
    assert item["image"].shape == (3, 32, 32)
    assert 0 <= item["label"] < ds.num_classes
    assert item["supercategory"] == 0  # single "bird" group

    # Labels span the full class range.
    labels = set(ds.labels())
    assert labels == set(range(ds.num_classes))


def test_collate_produces_batched_tensors(synthetic_dataset):
    tf = build_eval_transform(TransformConfig(image_size=32))
    ds = build_dataset(
        "cub",
        split="train",
        processed_dir=synthetic_dataset["processed_dir"],
        image_root=synthetic_dataset["image_root"],
        transform=tf,
    )
    batch = default_collate_dict([ds[0], ds[1], ds[2], ds[3]])
    assert batch["image"].shape == (4, 3, 32, 32)
    assert batch["label"].dtype == torch.long
    assert batch["label"].shape == (4,)
    assert batch["supercategory"].shape == (4,)


def test_bbox_crop_toggle_changes_output(synthetic_dataset):
    tf = build_eval_transform(TransformConfig(image_size=32))
    common = {
        "split": "train",
        "processed_dir": synthetic_dataset["processed_dir"],
        "image_root": synthetic_dataset["image_root"],
        "transform": tf,
    }
    plain = build_dataset("cub", bbox_crop=False, **common)[0]["image"]
    cropped = build_dataset("cub", bbox_crop=True, **common)[0]["image"]
    # Cropping to the (smaller) bbox then resizing yields a different tensor.
    assert not torch.equal(plain, cropped)
