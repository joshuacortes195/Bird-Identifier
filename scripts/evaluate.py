#!/usr/bin/env python
"""Evaluate a trained checkpoint on a split (default: NABirds test set).

    python scripts/evaluate.py --checkpoint outputs/checkpoints/<run>/best.pt
Writes metrics JSON + confusion matrix + reliability diagram to outputs/eval/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wildlife.eval.runner import evaluate_checkpoint  # noqa: E402


def _latest_best() -> Path | None:
    ckpts = sorted(
        (REPO_ROOT / "outputs" / "checkpoints").glob("*/best.pt"), key=lambda p: p.stat().st_mtime
    )
    return ckpts[-1] if ckpts else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None, help="path to best.pt (default: latest run)")
    p.add_argument("--dataset", default="nabirds")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--bbox-crop", action="store_true")
    args = p.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else _latest_best()
    if ckpt is None or not ckpt.exists():
        raise SystemExit("No checkpoint found. Pass --checkpoint or train a model first.")

    results = evaluate_checkpoint(
        ckpt,
        dataset=args.dataset,
        processed_dir=str(REPO_ROOT / "data" / "processed" / args.dataset),
        image_root=str(REPO_ROOT / "data" / "raw" / args.dataset),
        split=args.split,
        batch_size=args.batch_size,
        image_size=args.image_size,
        bbox_crop=args.bbox_crop,
        out_dir=str(REPO_ROOT / "outputs" / "eval"),
    )
    print("\n=== EVAL RESULTS ===")
    for k in ("top1", "top5", "macro_f1", "mean_per_class_acc"):
        print(f"{k:20s}: {results[k]:.4f}")
    print(f"{'ECE':20s}: {results['calibration']['ece']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
