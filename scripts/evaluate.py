#!/usr/bin/env python
"""Evaluate a trained checkpoint and write the Phase 6 report.

Produces top-1/top-5, macro-F1, per-class accuracy, a confusion matrix (+ most-confused
species pairs), a calibration reliability diagram + ECE, and Grad-CAM overlays on a few
correct and incorrect predictions — all from a real evaluation run.

Usage:
    python scripts/evaluate.py +checkpoint=outputs/checkpoints/<run>/best.pt data=nabirds \\
        model=convnextv2_base

Runs on the training/CI box (needs torch). The metric math itself is torch-free and unit
tested in tests/test_eval_metrics.py.
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
from wildlife.eval.calibration import calibration  # noqa: E402
from wildlife.eval.metrics import most_confused_pairs, per_class_accuracy, summarize  # noqa: E402
from wildlife.models import ModelConfig, build_model  # noqa: E402
from wildlife.utils.checkpoint import load_checkpoint  # noqa: E402
from wildlife.utils.logging import get_logger  # noqa: E402

log = get_logger("evaluate")


def _abs(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def _registry_name(data_name: str) -> str:
    return "nabirds" if data_name == "nabirds" else "cub"


def _collect_logits(model, loader, device):
    import torch

    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            logits = model(images).float().cpu().numpy()
            all_logits.append(logits)
            all_targets.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


def _plot_reliability(cal, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [(b.lower + b.upper) / 2 for b in cal.bins]
    accs = [b.accuracy for b in cal.bins]
    confs = [b.avg_confidence for b in cal.bins]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.bar(xs, accs, width=1 / len(cal.bins) * 0.9, alpha=0.7, label="accuracy")
    ax.plot(xs, confs, "o-", color="crimson", label="confidence", markersize=3)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability (ECE={cal.ece:.3f}, MCE={cal.mce:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_confusion(cm: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.where(row == 0, 1, row))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(norm, cmap="magma", vmin=0, vmax=1)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Row-normalized confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        raise SystemExit("Pass +checkpoint=<path/to/best.pt>")
    checkpoint = _abs(str(checkpoint))

    tcfg = TransformConfig(**OmegaConf.to_container(cfg.data.transform, resolve=True))
    lcfg = LoaderConfig(
        dataset=_registry_name(cfg.data.name),
        processed_dir=_abs(cfg.data.processed_dir),
        image_root=_abs(cfg.data.image_root),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        bbox_crop=cfg.data.bbox_crop,
        bbox_pad_frac=cfg.data.bbox_pad_frac,
        weighted_sampler=False,
    )
    test_loader = build_dataloader(lcfg, "test", tcfg)
    taxonomy = test_loader.dataset.taxonomy
    num_classes = taxonomy.num_classes

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

    device = torch.device(
        "cuda" if (cfg.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    ckpt = load_checkpoint(checkpoint, map_location=str(device))
    # Prefer EMA weights when present — they usually evaluate better.
    state = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state)
    model.to(device)
    log.info("Loaded %s (git %s)", checkpoint, ckpt.get("git_commit", "?"))

    logits, targets = _collect_logits(model, test_loader, device)
    report, cm = summarize(logits, targets, num_classes)
    cal = calibration(logits, targets)
    pca = per_class_accuracy(cm)
    pairs = most_confused_pairs(cm, taxonomy.class_names, top_n=20)

    out_dir = Path(_abs(f"outputs/eval/{Path(checkpoint).parent.name}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "confusion_matrix.npy", cm)
    _plot_reliability(cal, out_dir / "reliability.png")
    _plot_confusion(cm, out_dir / "confusion_matrix.png")

    # Grad-CAM samples (best-effort; skip if the CAM lib/layer isn't available).
    try:
        _gradcam_samples(model, test_loader, logits, targets, taxonomy, device, out_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("Grad-CAM sampling skipped: %s", exc)

    _write_report(out_dir, report, cal, pca, pairs, checkpoint, ckpt)
    log.info("Report written to %s", out_dir)
    log.info(
        "top1=%.4f top5=%.4f macroF1=%.4f ECE=%.4f",
        report.top1,
        report.top5,
        report.macro_f1,
        cal.ece,
    )


def _gradcam_samples(model, loader, logits, targets, taxonomy, device, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wildlife.eval.gradcam import compute_gradcam, denormalize_tensor, overlay_heatmap

    preds = logits.argmax(axis=1)
    correct_idx = np.where(preds == targets)[0][:3]
    wrong_idx = np.where(preds != targets)[0][:3]
    picks = list(correct_idx) + list(wrong_idx)
    if not picks:
        return

    # Re-fetch specific samples by walking the dataset (batch_size may not align to indices).
    ds = loader.dataset
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes = axes.ravel()
    for ax, i in zip(axes, picks, strict=False):
        item = ds[int(i)]
        x = item["image"].unsqueeze(0).to(device)
        cam = compute_gradcam(model, x, target_category=int(preds[i]))
        base = denormalize_tensor(x)
        ax.imshow(overlay_heatmap(base, cam))
        ok = preds[i] == targets[i]
        ax.set_title(
            f"{'✓' if ok else '✗'} {taxonomy.class_names[int(preds[i])][:18]}",
            fontsize=8,
            color="green" if ok else "crimson",
        )
        ax.axis("off")
    fig.suptitle("Grad-CAM — top row correct, bottom row incorrect")
    fig.tight_layout()
    fig.savefig(out_dir / "gradcam_samples.png", dpi=120)
    plt.close(fig)


def _write_report(out_dir, report, cal, pca, pairs, checkpoint, ckpt) -> None:
    valid_pca = pca[~np.isnan(pca)]
    worst = np.argsort(np.where(np.isnan(pca), np.inf, pca))[:10]
    lines = [
        "# Evaluation report",
        "",
        f"- Checkpoint: `{checkpoint}` (git `{ckpt.get('git_commit', '?')}`)",
        f"- Test samples: **{report.num_samples}** across **{report.num_classes}** classes",
        "",
        "## Headline metrics",
        "",
        "| metric | value |",
        "|--------|------:|",
        f"| Top-1 accuracy | {report.top1:.4f} |",
        f"| Top-5 accuracy | {report.top5:.4f} |",
        f"| Macro-F1 | {report.macro_f1:.4f} |",
        f"| Mean per-class acc | {report.mean_per_class_acc:.4f} |",
        f"| Median per-class acc | {float(np.median(valid_pca)):.4f} |",
        f"| ECE / MCE | {cal.ece:.4f} / {cal.mce:.4f} |",
        f"| Calibration | {'over-confident' if cal.overconfident else 'under-confident'} |",
        "",
        "## Hardest 10 classes (lowest recall)",
        "",
        "| class idx | per-class accuracy |",
        "|----------:|-------------------:|",
    ]
    lines += [f"| {int(i)} | {pca[i]:.3f} |" for i in worst]
    lines += [
        "",
        "## Most-confused species pairs",
        "",
        "| true | predicted | count |",
        "|------|-----------|------:|",
    ]
    lines += [f"| {p.true_name} | {p.pred_name} | {p.count} |" for p in pairs]
    lines += [
        "",
        "## Figures",
        "",
        "- `reliability.png` — calibration reliability diagram",
        "- `confusion_matrix.png` — row-normalized confusion matrix",
        "- `gradcam_samples.png` — attention on correct vs. incorrect predictions",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
