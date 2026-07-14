"""Training loop: AMP, gradient accumulation, cosine+warmup, label smoothing, Mixup,
optional EMA, checkpoint (best+last), early stopping, and resume-from-checkpoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from wildlife.data.transforms import MixupCutmix
from wildlife.train.ema import ModelEma
from wildlife.train.optim import CosineWarmupScheduler, OptimConfig, build_optimizer
from wildlife.utils.checkpoint import load_checkpoint, save_checkpoint
from wildlife.utils.logging import get_logger

log = get_logger("train")


@dataclass
class TrainConfig:
    epochs: int = 30
    grad_accum: int = 1
    amp: bool = True
    label_smoothing: float = 0.1
    clip_grad: float | None = 1.0
    ema: bool = False
    ema_decay: float = 0.9998
    early_stop_patience: int | None = None  # epochs without val improvement
    ckpt_dir: str = "outputs/checkpoints"
    log_interval: int = 20
    # mixup/cutmix (off by default; enabled for full runs)
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    mixup_prob: float = 0.0
    optim: OptimConfig = field(default_factory=OptimConfig)


def _resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        num_classes: int,
        *,
        device: str = "auto",
        class_weights: torch.Tensor | None = None,
        tracker: Any = None,
        class_names: list[str] | None = None,
        run_config: dict | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.cfg = cfg
        self.num_classes = num_classes
        self.tracker = tracker
        self.class_names = class_names
        self.run_config = run_config or {}

        self.optimizer = build_optimizer(self.model, cfg.optim)
        # On Ampere+ prefer bfloat16: it has fp32's dynamic range, so it needs no loss
        # scaling and avoids the fp16 overflow that silently skips optimizer steps.
        # Fall back to fp16 (+ GradScaler) only where bf16 is unavailable.
        cuda = self.device.type == "cuda"
        use_amp = cfg.amp and cuda
        self.amp_dtype = (
            torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
        )
        self.use_scaler = use_amp and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        self._amp_enabled = use_amp
        self.scheduler: CosineWarmupScheduler | None = None  # built in fit() once steps known

        self.mixup = MixupCutmix(
            num_classes,
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            prob=cfg.mixup_prob,
            label_smoothing=cfg.label_smoothing,
        )
        self.hard_criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None,
            label_smoothing=cfg.label_smoothing,
        )
        self.ema = ModelEma(self.model, cfg.ema_decay) if cfg.ema else None

        self.top1 = MulticlassAccuracy(num_classes=num_classes, average="micro").to(self.device)
        self.top5 = MulticlassAccuracy(
            num_classes=num_classes, average="micro", top_k=min(5, num_classes)
        ).to(self.device)

        self.start_epoch = 0
        self.best_metric = -1.0
        self.history: list[dict] = []

    # --- loss handles both hard (indices) and soft (mixup) targets ---
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 2:  # soft targets from mixup
            return torch.sum(-targets * torch.log_softmax(logits, dim=-1), dim=-1).mean()
        return self.hard_criterion(logits, targets)

    def _autocast(self):
        return torch.amp.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self._amp_enabled
        )

    def train_one_epoch(self, loader, epoch: int) -> dict:
        self.model.train()
        self.top1.reset()
        running = 0.0
        n_batches = len(loader)
        accum = max(1, self.cfg.grad_accum)
        self.optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()

        for i, batch in enumerate(loader):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            images, targets = self.mixup(images, labels)

            with self._autocast():
                logits = self.model(images)
                loss = self._loss(logits, targets) / accum

            self.scaler.scale(loss).backward()
            running += loss.item() * accum

            if (i + 1) % accum == 0 or (i + 1) == n_batches:
                if self.cfg.clip_grad is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)

            with torch.no_grad():
                self.top1.update(logits.detach(), labels)

            if (i + 1) % self.cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                log.info(
                    "epoch %d [%d/%d] loss=%.4f acc=%.3f lr=%.2e",
                    epoch,
                    i + 1,
                    n_batches,
                    running / (i + 1),
                    self.top1.compute().item(),
                    lr,
                )

        return {
            "train_loss": running / max(1, n_batches),
            "train_acc": self.top1.compute().item(),
            "epoch_time_s": round(time.perf_counter() - t0, 1),
        }

    @torch.no_grad()
    def evaluate(self, loader, *, use_ema: bool = False) -> dict:
        model = self.ema.module if (use_ema and self.ema is not None) else self.model
        model.eval()
        self.top1.reset()
        self.top5.reset()
        loss_sum, n = 0.0, 0
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            with self._autocast():
                logits = model(images)
                loss = self.hard_criterion(logits, labels)
            loss_sum += loss.item() * images.size(0)
            n += images.size(0)
            self.top1.update(logits, labels)
            self.top5.update(logits, labels)
        return {
            "val_loss": loss_sum / max(1, n),
            "val_top1": self.top1.compute().item(),
            "val_top5": self.top5.compute().item(),
        }

    def _build_scheduler(self, steps_per_epoch: int) -> None:
        updates_per_epoch = max(1, steps_per_epoch // max(1, self.cfg.grad_accum))
        total = updates_per_epoch * self.cfg.epochs
        warmup = int(updates_per_epoch * self.cfg.optim.warmup_epochs)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=warmup,
            total_steps=total,
            min_lr_ratio=self.cfg.optim.min_lr_ratio,
        )

    def fit(self, train_loader, val_loader) -> list[dict]:
        self._build_scheduler(len(train_loader))
        ckpt_dir = Path(self.cfg.ckpt_dir)
        epochs_no_improve = 0

        if self.tracker is not None and hasattr(self.tracker, "log_config"):
            self.tracker.log_config(self.run_config)

        for epoch in range(self.start_epoch, self.cfg.epochs):
            tr = self.train_one_epoch(train_loader, epoch)
            ev = self.evaluate(val_loader)
            if self.ema is not None:
                ema_ev = self.evaluate(val_loader, use_ema=True)
                ev.update({f"ema_{k}": v for k, v in ema_ev.items()})

            row = {"epoch": epoch, **tr, **ev, "lr": self.optimizer.param_groups[0]["lr"]}
            self.history.append(row)
            log.info(
                "epoch %d done | train_loss=%.4f val_top1=%.4f val_top5=%.4f (%.0fs)",
                epoch,
                tr["train_loss"],
                ev["val_top1"],
                ev["val_top5"],
                tr["epoch_time_s"],
            )
            if self.tracker is not None:
                self.tracker.log(row, step=epoch)

            metric = ev.get("ema_val_top1", ev["val_top1"])
            save_checkpoint(
                ckpt_dir / "last.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                ema=self.ema,
                epoch=epoch,
                best_metric=self.best_metric,
                metrics=row,
                config=self.run_config,
                class_names=self.class_names,
            )
            if metric > self.best_metric:
                self.best_metric = metric
                epochs_no_improve = 0
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model=self.model,
                    ema=self.ema,
                    epoch=epoch,
                    best_metric=self.best_metric,
                    metrics=row,
                    config=self.run_config,
                    class_names=self.class_names,
                )
                log.info("  new best val_top1=%.4f -> saved best.pt", metric)
            else:
                epochs_no_improve += 1
                if (
                    self.cfg.early_stop_patience
                    and epochs_no_improve >= self.cfg.early_stop_patience
                ):
                    log.info(
                        "Early stopping at epoch %d (no improvement for %d)",
                        epoch,
                        epochs_no_improve,
                    )
                    break

        return self.history

    def resume(self, path: str | Path) -> None:
        ckpt = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.ema is not None and ckpt.get("ema") is not None:
            self.ema.load_state_dict(ckpt["ema"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_metric = ckpt.get("best_metric", -1.0) or -1.0
        log.info(
            "Resumed from %s at epoch %d (best=%.4f)", path, self.start_epoch, self.best_metric
        )
