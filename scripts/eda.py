#!/usr/bin/env python
"""Generate EDA figures + report for a prepared dataset.

    python scripts/eda.py --dataset cub
Outputs to outputs/eda/<dataset>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wildlife.data.eda import run_eda  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# The image root differs per dataset (archives nest under a top folder).
DATA_ROOT_HINT = {
    "cub": REPO_ROOT / "data" / "raw" / "cub200",
    "nabirds": REPO_ROOT / "data" / "raw" / "nabirds",
}


def _find_data_root(dataset: str) -> Path:
    base = DATA_ROOT_HINT.get(dataset, REPO_ROOT / "data" / "raw" / dataset)
    matches = list(base.rglob("images.txt"))
    return matches[0].parent if matches else base


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--size-sample", type=int, default=800)
    args = p.parse_args()

    processed = REPO_ROOT / "data" / "processed" / args.dataset
    if not (processed / "train.csv").exists():
        raise SystemExit(
            f"No manifests at {processed}. Run scripts/download_data.py --dataset {args.dataset} first."
        )

    out_dir = REPO_ROOT / "outputs" / "eda" / args.dataset
    run_eda(processed, _find_data_root(args.dataset), out_dir, size_sample=args.size_sample)
    print(f"EDA report: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
