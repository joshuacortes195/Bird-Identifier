#!/usr/bin/env python
"""Train a classifier. Hydra-driven; every hyperparameter comes from configs/.

Examples:
    python scripts/train.py                              # smoke: subset + atto
    python scripts/train.py train=smoke data=subset model=convnextv2_atto
    python scripts/train.py data=nabirds model=convnextv2_base train=baseline
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wildlife.data.loaders import LoaderConfig, build_dataloader, class_weights  # noqa: E402
from wildlife.data.transforms import TransformConfig  # noqa: E402
from wildlife.models import ModelConfig, build_model, count_parameters  # noqa: E402
from wildlife.train.loop import TrainConfig, Trainer  # noqa: E402
from wildlife.train.optim import OptimConfig  # noqa: E402
from wildlife.utils.logging import get_logger  # noqa: E402
from wildlife.utils.seed import seed_everything  # noqa: E402

log = get_logger("train_script")


def _abs(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def _registry_name(data_name: str) -> str:
    # subset reuses the CUB images + the generic manifest-backed dataset class.
    return "nabirds" if data_name == "nabirds" else "cub"


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> float:
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    run_name = (
        cfg.run_name
        or f"{cfg.data.name}-{cfg.model.backbone.split('.')[0]}-{datetime.now():%Y%m%d_%H%M%S}"
    )
    log.info("Run: %s", run_name)

    tcfg = TransformConfig(**OmegaConf.to_container(cfg.data.transform, resolve=True))
    lcfg = LoaderConfig(
        dataset=_registry_name(cfg.data.name),
        processed_dir=_abs(cfg.data.processed_dir),
        image_root=_abs(cfg.data.image_root),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        bbox_crop=cfg.data.bbox_crop,
        bbox_pad_frac=cfg.data.bbox_pad_frac,
        weighted_sampler=cfg.data.weighted_sampler,
    )
    train_loader = build_dataloader(lcfg, "train", tcfg)
    val_loader = build_dataloader(lcfg, "val", tcfg)
    taxonomy = train_loader.dataset.taxonomy
    num_classes = taxonomy.num_classes
    log.info(
        "Dataset '%s': %d classes | train=%d val=%d",
        cfg.data.name,
        num_classes,
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    mcfg = ModelConfig(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        head=cfg.head.name,
        head_kwargs=OmegaConf.to_container(cfg.head.head_kwargs, resolve=True)
        if cfg.head.get("head_kwargs")
        else {},
        drop_rate=cfg.model.drop_rate,
        drop_path_rate=cfg.model.drop_path_rate,
    )
    model = build_model(mcfg, num_classes)
    total, trainable = count_parameters(model)
    log.info(
        "Model: %s + %s head | params: %.1fM total, %.1fM trainable",
        mcfg.backbone,
        mcfg.head,
        total / 1e6,
        trainable / 1e6,
    )

    cw = None
    if cfg.data.get("class_weighted_loss"):
        cw = class_weights(train_loader.dataset.labels(), num_classes)

    ocfg = OptimConfig(**OmegaConf.to_container(cfg.train.optim, resolve=True))
    train_cfg = TrainConfig(
        epochs=cfg.train.epochs,
        grad_accum=cfg.train.grad_accum,
        amp=cfg.train.amp,
        label_smoothing=cfg.train.label_smoothing,
        clip_grad=cfg.train.clip_grad,
        ema=cfg.train.ema,
        ema_decay=cfg.train.get("ema_decay", 0.9998),
        early_stop_patience=cfg.train.early_stop_patience,
        ckpt_dir=_abs(f"outputs/checkpoints/{run_name}"),
        log_interval=cfg.train.log_interval,
        mixup_alpha=cfg.train.mixup_alpha,
        cutmix_alpha=cfg.train.cutmix_alpha,
        mixup_prob=cfg.train.mixup_prob,
        optim=ocfg,
    )

    from wildlife.train.tracking import build_tracker

    tracker = build_tracker(cfg, run_name)
    run_config = OmegaConf.to_container(cfg, resolve=True)

    trainer = Trainer(
        model,
        train_cfg,
        num_classes,
        device=cfg.device,
        class_weights=cw,
        tracker=tracker,
        class_names=taxonomy.class_names,
        run_config=run_config,
    )
    history = trainer.fit(train_loader, val_loader)

    # Persist the run history as a CSV for quick comparison.
    hist_path = Path(train_cfg.ckpt_dir) / "history.csv"
    if history:
        with hist_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    if tracker is not None and hasattr(tracker, "finish"):
        tracker.finish()

    best = trainer.best_metric
    log.info("Done. Best val_top1=%.4f | checkpoints: %s", best, train_cfg.ckpt_dir)
    return best


if __name__ == "__main__":
    main()
