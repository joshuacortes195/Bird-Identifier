#!/usr/bin/env python
"""Fit a temperature scalar on val, report test ECE before/after (Phase 6 calibration).

    python scripts/calibrate.py +checkpoint=outputs/checkpoints/<run>/best.pt \\
        data=nabirds model=convnextv2_base

Temperature is fit on the validation split (no test leakage) and applied to test logits.
Writes results/calibration.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wildlife.data.loaders import LoaderConfig, build_dataloader  # noqa: E402
from wildlife.data.transforms import TransformConfig  # noqa: E402
from wildlife.eval.calibration import apply_temperature, calibration, fit_temperature  # noqa: E402
from wildlife.models import ModelConfig, build_model  # noqa: E402
from wildlife.utils.checkpoint import load_checkpoint  # noqa: E402
from wildlife.utils.logging import get_logger  # noqa: E402

log = get_logger("calibrate")


def _abs(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def _collect(model, loader, device):
    import torch

    logits_all, targets_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device)).float().cpu().numpy()
            logits_all.append(logits)
            targets_all.append(batch["label"].numpy())
    return np.concatenate(logits_all), np.concatenate(targets_all)


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        raise SystemExit("Pass +checkpoint=<path/to/best.pt>")
    checkpoint = _abs(str(checkpoint))

    tcfg = TransformConfig(**OmegaConf.to_container(cfg.data.transform, resolve=True))
    lcfg = LoaderConfig(
        dataset="nabirds" if cfg.data.name == "nabirds" else "cub",
        processed_dir=_abs(cfg.data.processed_dir),
        image_root=_abs(cfg.data.image_root),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    val_loader = build_dataloader(lcfg, "val", tcfg)
    test_loader = build_dataloader(lcfg, "test", tcfg)
    num_classes = val_loader.dataset.taxonomy.num_classes

    mcfg = ModelConfig(
        backbone=cfg.model.backbone,
        pretrained=False,
        head=cfg.head.name,
        head_kwargs=OmegaConf.to_container(cfg.head.head_kwargs, resolve=True)
        if cfg.head.get("head_kwargs")
        else {},
        drop_rate=cfg.model.drop_rate,
        drop_path_rate=cfg.model.drop_path_rate,
    )
    model = build_model(mcfg, num_classes)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint, map_location=str(device))
    ema_state = ckpt.get("ema")
    if ema_state:
        state = {
            k.removeprefix("module."): v for k, v in ema_state.items() if k.startswith("module.")
        }
    else:
        state = ckpt["model"]
    model.load_state_dict(state)
    model.to(device)

    log.info("Collecting val logits (fit T)...")
    val_logits, val_targets = _collect(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_targets)
    log.info("Fitted temperature T=%.3f", temperature)

    log.info("Collecting test logits (report)...")
    test_logits, test_targets = _collect(model, test_loader, device)
    before = calibration(test_logits, test_targets)
    after = calibration(apply_temperature(test_logits, temperature), test_targets)

    out = REPO_ROOT / "results" / "calibration.md"
    out.write_text(
        "# Confidence calibration (temperature scaling)\n\n"
        f"- Temperature fit on val (no test leakage): **T = {temperature:.3f}**\n"
        f"- Test ECE: **{before.ece:.4f} → {after.ece:.4f}** "
        f"({'improved' if after.ece < before.ece else 'no improvement'})\n"
        f"- Test MCE: {before.mce:.4f} → {after.mce:.4f}\n"
        f"- Over-confident before: {before.overconfident}; after: {after.overconfident}\n\n"
        "Temperature scaling is a post-hoc, accuracy-preserving fix (it only rescales "
        "logits, so top-1/top-5 are unchanged). Apply T at serving time before softmax.\n",
        encoding="utf-8",
    )
    log.info("Test ECE %.4f -> %.4f (T=%.3f). Wrote %s", before.ece, after.ece, temperature, out)
    print(f"\nT={temperature:.3f} | test ECE {before.ece:.4f} -> {after.ece:.4f}")


if __name__ == "__main__":
    main()
