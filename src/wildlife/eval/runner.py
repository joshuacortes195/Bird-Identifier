"""Load a checkpoint and evaluate it on a split, producing a metrics report + figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wildlife.data.loaders import LoaderConfig, build_dataloader
from wildlife.data.transforms import TransformConfig
from wildlife.eval.metrics import (
    expected_calibration_error,
    macro_f1,
    most_confused_pairs,
    per_class_accuracy,
    topk_accuracy,
)
from wildlife.models import ModelConfig, build_model
from wildlife.utils.checkpoint import load_checkpoint
from wildlife.utils.logging import get_logger

log = get_logger("eval")


def _load_model_from_ckpt(
    ckpt: dict, num_classes: int, device: torch.device, *, prefer_ema: bool = True
):
    cfg = ckpt.get("config", {})
    mcfg = ModelConfig(
        backbone=cfg["model"]["backbone"],
        pretrained=False,  # weights come from the checkpoint
        head=cfg["head"]["name"],
        head_kwargs=cfg["head"].get("head_kwargs") or {},
        drop_rate=cfg["model"].get("drop_rate", 0.0),
        drop_path_rate=cfg["model"].get("drop_path_rate", 0.0),
    )
    model = build_model(mcfg, num_classes)
    if prefer_ema and ckpt.get("ema") is not None:
        # EMA weights (reported "best"); strip the ModelEma "module." prefix.
        state = {k[len("module.") :]: v for k, v in ckpt["ema"].items() if k.startswith("module.")}
        model.load_state_dict(state)
        log.info("Loaded EMA weights.")
    else:
        model.load_state_dict(ckpt["model"])
        log.info("Loaded raw model weights.")
    return model.to(device).eval()


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    all_logits, all_labels = [], []
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(images)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(batch["label"].numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    dataset: str,
    processed_dir: str,
    image_root: str,
    split: str = "test",
    batch_size: int = 64,
    num_workers: int = 6,
    image_size: int = 224,
    bbox_crop: bool = False,
    device: str = "auto",
    out_dir: str | Path = "outputs/eval",
) -> dict:
    dev = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else (device if device != "auto" else "cpu")
    )
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    class_names = ckpt.get("class_names") or []
    num_classes = len(class_names)

    tcfg = TransformConfig(image_size=image_size)
    lcfg = LoaderConfig(
        dataset=dataset,
        processed_dir=processed_dir,
        image_root=image_root,
        batch_size=batch_size,
        num_workers=num_workers,
        bbox_crop=bbox_crop,
    )
    loader = build_dataloader(lcfg, split, tcfg)
    if not num_classes:
        num_classes = loader.dataset.num_classes
        class_names = loader.dataset.taxonomy.class_names

    model = _load_model_from_ckpt(ckpt, num_classes, dev)
    log.info(
        "Evaluating %s split=%s (%d images, %d classes) on %s",
        dataset,
        split,
        len(loader.dataset),
        num_classes,
        dev,
    )
    logits, labels = collect_logits(model, loader, dev)
    preds = logits.argmax(axis=1)

    pca = per_class_accuracy(preds, labels, num_classes)
    results = {
        "checkpoint": str(checkpoint),
        "dataset": dataset,
        "split": split,
        "num_images": int(len(labels)),
        "num_classes": num_classes,
        "top1": topk_accuracy(logits, labels, k=1),
        "top5": topk_accuracy(logits, labels, k=5),
        "macro_f1": macro_f1(preds, labels),
        "mean_per_class_acc": float(np.nanmean(pca)),
        "worst_classes": [
            {"class": class_names[i], "acc": float(pca[i])} for i in np.argsort(pca)[:10]
        ],
        "most_confused_pairs": most_confused_pairs(preds, labels, class_names, top=15),
        "calibration": expected_calibration_error(logits, labels),
        "git_commit": ckpt.get("git_commit"),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{dataset}_{split}_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    _save_figures(logits, labels, preds, num_classes, results, out_dir, split)
    log.info(
        "RESULTS %s/%s: top1=%.4f top5=%.4f macroF1=%.4f ECE=%.4f",
        dataset,
        split,
        results["top1"],
        results["top5"],
        results["macro_f1"],
        results["calibration"]["ece"],
    )
    return results


def _save_figures(logits, labels, preds, num_classes, results, out_dir: Path, split: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reliability diagram.
    bins = results["calibration"]["bins"]
    if bins:
        conf = [b["conf"] for b in bins]
        acc = [b["acc"] for b in bins]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.plot(conf, acc, "o-", label="model")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(f"Reliability (ECE={results['calibration']['ece']:.3f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{split}_reliability.png", dpi=120)
        plt.close(fig)

    # Confusion matrix (downsampled if large) as an image.
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"Confusion matrix ({num_classes} classes, row-normalized)")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / f"{split}_confusion_matrix.png", dpi=130)
    plt.close(fig)
