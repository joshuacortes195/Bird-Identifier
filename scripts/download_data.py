#!/usr/bin/env python
"""Download a dataset, parse it, and write reproducible splits + taxonomy config.

Examples:
    python scripts/download_data.py --dataset cub        # small, fast (smoke)
    python scripts/download_data.py --dataset nabirds     # primary (~48.5k imgs)

Outputs:
    data/processed/<dataset>/{train,val,test}.csv   split manifests
    data/processed/<dataset>/taxonomy.yaml          label space (copy)
    data/processed/<dataset>/summary.json           counts + class balance
    configs/taxonomy/<birds|cub>.yaml               canonical taxonomy config
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Ensure src/ is importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wildlife.data.prepare import prepare_splits  # noqa: E402
from wildlife.data.sources.preparers import (  # noqa: E402
    SOURCE_PREPARERS,
    TAXONOMY_CONFIG_NAME,
    available_sources,
)
from wildlife.utils.logging import get_logger  # noqa: E402

log = get_logger("download_data")
REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=available_sources())
    p.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "processed"))
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    preparer = SOURCE_PREPARERS[args.dataset]
    log.info("Acquiring dataset '%s' ...", args.dataset)
    parsed = preparer(args.raw_dir)

    out_dir = Path(args.out_dir) / args.dataset
    summary = prepare_splits(parsed, out_dir, val_fraction=args.val_fraction, seed=args.seed)

    # Publish the taxonomy as a canonical config (birds.yaml / cub.yaml).
    cfg_name = TAXONOMY_CONFIG_NAME.get(args.dataset, f"{parsed.taxonomy.name}.yaml")
    cfg_dst = REPO_ROOT / "configs" / "taxonomy" / cfg_name
    cfg_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_dir / "taxonomy.yaml", cfg_dst)
    log.info("Taxonomy config -> %s", cfg_dst.relative_to(REPO_ROOT))

    print("\n=== DATA SUMMARY ===")
    print(f"dataset          : {args.dataset}")
    print(f"num_classes      : {summary['num_classes']}")
    print(f"images w/ bbox   : {summary['with_bbox']}")
    for split, stats in summary["splits"].items():
        print(
            f"{split:<5}: {stats['num_images']:>6} imgs | "
            f"{stats['num_classes_present']:>3} classes | "
            f"min/max per class {stats['min_per_class']}/{stats['max_per_class']} | "
            f"imbalance {stats['imbalance_ratio']}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
