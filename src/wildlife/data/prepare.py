"""Turn a ParsedDataset into reproducible train/val/test manifests + taxonomy + summary.

The official train/test split is respected; a validation set is carved out of train,
stratified by class and seeded so it's identical on every machine.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from wildlife.data.schema import Sample, read_manifest, write_manifest
from wildlife.data.sources.cub_format import ParsedDataset
from wildlife.data.taxonomy import Taxonomy, load_taxonomy, write_taxonomy
from wildlife.utils.logging import get_logger

log = get_logger(__name__)


def stratified_val_split(
    samples: list[Sample],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    """Split ``samples`` into (train, val), holding out ``val_fraction`` per class."""
    by_class: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.class_idx].append(s)

    rng = random.Random(seed)
    train_out: list[Sample] = []
    val_out: list[Sample] = []
    for _, items in sorted(by_class.items()):
        items = sorted(items, key=lambda s: s.image_path)  # deterministic order
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_fraction)) if len(items) > 1 else 0
        val_out.extend(items[:n_val])
        train_out.extend(items[n_val:])
    return train_out, val_out


def _class_balance_stats(samples: list[Sample], num_classes: int) -> dict:
    counts = Counter(s.class_idx for s in samples)
    per_class = [counts.get(i, 0) for i in range(num_classes)]
    nonzero = [c for c in per_class if c > 0]
    return {
        "num_images": len(samples),
        "num_classes_present": len(nonzero),
        "min_per_class": min(per_class) if per_class else 0,
        "max_per_class": max(per_class) if per_class else 0,
        "mean_per_class": round(sum(per_class) / max(1, num_classes), 2),
        "empty_classes": sum(1 for c in per_class if c == 0),
        "imbalance_ratio": round(max(nonzero) / min(nonzero), 2) if nonzero else 0.0,
    }


def prepare_splits(
    parsed: ParsedDataset,
    out_dir: str | Path,
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """Write train/val/test manifests, taxonomy.yaml, and summary.json under ``out_dir``.

    Returns the summary dict (also written to disk).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_by_index = parsed.image_id_by_sample_index
    train_pool = [s for s, iid in zip(parsed.samples, id_by_index, strict=True) if iid in parsed.train_ids]
    test = [s for s, iid in zip(parsed.samples, id_by_index, strict=True) if iid in parsed.test_ids]

    train, val = stratified_val_split(train_pool, val_fraction=val_fraction, seed=seed)

    write_manifest(train, out_dir / "train.csv")
    write_manifest(val, out_dir / "val.csv")
    write_manifest(test, out_dir / "test.csv")
    write_taxonomy(parsed.taxonomy, out_dir / "taxonomy.yaml")

    n = parsed.taxonomy.num_classes
    summary = {
        "taxonomy": parsed.taxonomy.name,
        "num_classes": n,
        "seed": seed,
        "val_fraction": val_fraction,
        "with_bbox": sum(1 for s in parsed.samples if s.has_bbox),
        "splits": {
            "train": _class_balance_stats(train, n),
            "val": _class_balance_stats(val, n),
            "test": _class_balance_stats(test, n),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(
        "Prepared %s: %d classes | train=%d val=%d test=%d | bboxes=%d",
        parsed.taxonomy.name,
        n,
        len(train),
        len(val),
        len(test),
        summary["with_bbox"],
    )
    return summary


def make_subset(
    src_processed_dir: str | Path,
    out_dir: str | Path,
    *,
    num_classes: int = 10,
    seed: int = 42,
) -> dict:
    """Build a small K-class subset from a prepared dataset for fast iteration.

    Selects the first ``num_classes`` classes (deterministic), remaps class indices
    to a contiguous 0..K-1 space, and rewrites train/val/test manifests + taxonomy.
    """
    src = Path(src_processed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy(src / "taxonomy.yaml")
    keep_idx = list(range(min(num_classes, taxonomy.num_classes)))
    keep_set = set(keep_idx)
    remap = {old: new for new, old in enumerate(keep_idx)}

    from dataclasses import replace as dc_replace

    new_entries = [
        dc_replace(taxonomy.entries[old], idx=new) for old, new in remap.items()
    ]
    sub_tax = Taxonomy(name=f"{taxonomy.name}_subset{len(keep_idx)}", entries=new_entries)
    write_taxonomy(sub_tax, out_dir / "taxonomy.yaml")

    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        src_manifest = src / f"{split}.csv"
        if not src_manifest.exists():
            continue
        kept = [
            Sample(
                image_path=s.image_path,
                class_idx=remap[s.class_idx],
                class_id=s.class_id,
                supercategory=s.supercategory,
                bbox=s.bbox,
            )
            for s in read_manifest(src_manifest)
            if s.class_idx in keep_set
        ]
        write_manifest(kept, out_dir / f"{split}.csv")
        counts[split] = len(kept)

    summary = {
        "taxonomy": sub_tax.name,
        "num_classes": sub_tax.num_classes,
        "derived_from": str(src),
        "seed": seed,
        "counts": counts,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Subset (%d classes) written to %s: %s", sub_tax.num_classes, out_dir, counts)
    return summary
