#!/usr/bin/env python
"""Benchmark a dataloader (imgs/sec) and dump a post-augmentation batch grid.

    python scripts/benchmark_loader.py --data cub --split train --batches 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from wildlife.data.loaders import LoaderConfig, build_dataloader  # noqa: E402
from wildlife.data.transforms import TransformConfig  # noqa: E402
from wildlife.data.visualize import save_batch_grid  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_data_cfg(name: str) -> dict:
    with (REPO_ROOT / "configs" / "data" / f"{name}.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="cub", help="data config name (cub|nabirds|subset)")
    p.add_argument("--split", default="train")
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=None)
    args = p.parse_args()

    raw = _load_data_cfg(args.data)
    tcfg = TransformConfig(**raw["transform"])
    lcfg = LoaderConfig(
        dataset=raw["name"] if raw["name"] in {"cub", "nabirds"} else "cub",
        processed_dir=str(REPO_ROOT / raw["processed_dir"]),
        image_root=str(REPO_ROOT / raw["image_root"]),
        batch_size=raw["batch_size"],
        num_workers=args.num_workers if args.num_workers is not None else raw["num_workers"],
        bbox_crop=raw["bbox_crop"],
        weighted_sampler=raw.get("weighted_sampler", False),
    )
    loader = build_dataloader(lcfg, args.split, tcfg)

    it = iter(loader)
    first = next(it)  # warm up workers
    class_names = loader.dataset.taxonomy.class_names
    out_dir = REPO_ROOT / "outputs" / "eda" / args.data
    grid = save_batch_grid(first["image"], first["label"], class_names, out_dir / "batch_grid.png", tcfg=tcfg)

    n_imgs = first["image"].size(0)
    t0 = time.perf_counter()
    for i, batch in enumerate(it):
        n_imgs += batch["image"].size(0)
        if i + 1 >= args.batches:
            break
    dt = time.perf_counter() - t0
    rate = n_imgs / dt if dt > 0 else float("nan")

    print(f"\n=== LOADER BENCHMARK ({args.data}/{args.split}) ===")
    print(f"batch_size      : {lcfg.batch_size}")
    print(f"num_workers     : {lcfg.num_workers}")
    print(f"images          : {n_imgs}")
    print(f"elapsed         : {dt:.2f}s")
    print(f"throughput      : {rate:.1f} imgs/sec")
    print(f"batch grid      : {grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
