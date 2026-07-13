#!/usr/bin/env python
"""Create a small K-class subset from a prepared dataset for fast iteration.

    python scripts/make_subset.py --source cub --num-classes 10
Writes data/processed/subset/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wildlife.data.prepare import make_subset  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="cub", help="prepared dataset to subset (cub|nabirds)")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "processed" / "subset"))
    args = p.parse_args()

    src = REPO_ROOT / "data" / "processed" / args.source
    if not (src / "train.csv").exists():
        raise SystemExit(
            f"No prepared dataset at {src}. Run download_data.py --dataset {args.source} first."
        )
    make_subset(src, args.out, num_classes=args.num_classes)
    print(f"Subset written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
