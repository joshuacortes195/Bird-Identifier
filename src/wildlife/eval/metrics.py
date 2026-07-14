"""Classification metrics for the evaluation report (Phase 6).

Implemented on NumPy arrays (logits + integer targets) rather than torch, so they run in
CI without a GPU and are unit-testable on any box. ``scripts/evaluate.py`` runs the model
to produce logits, then calls these. Nothing here hardcodes the class count — it flows
from the taxonomy / logits width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def top_k_accuracy(
    logits: np.ndarray, targets: np.ndarray, ks: tuple[int, ...] = (1, 5)
) -> dict[int, float]:
    """Top-k accuracy for each k. ``logits``: (N, C); ``targets``: (N,)."""
    n, num_classes = logits.shape
    order = np.argsort(-logits, axis=1)  # descending
    out: dict[int, float] = {}
    for k in ks:
        kk = min(k, num_classes)
        topk = order[:, :kk]
        hits = np.any(topk == targets[:, None], axis=1)
        out[k] = float(hits.mean()) if n else 0.0
    return out


def confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows = true class, cols = predicted class."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (targets, preds), 1)
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """Recall per class = diagonal / row sum. Classes with no samples -> NaN."""
    row_sums = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.diag(cm) / row_sums
    acc[row_sums == 0] = np.nan
    return acc


def macro_f1(cm: np.ndarray) -> float:
    """Unweighted mean F1 across classes (classes absent from truth are skipped)."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    present = cm.sum(axis=1) > 0  # class appears in ground truth
    f1 = np.where(np.isnan(f1), 0.0, f1)
    return float(f1[present].mean()) if present.any() else 0.0


@dataclass(frozen=True)
class ConfusedPair:
    true_idx: int
    pred_idx: int
    true_name: str
    pred_name: str
    count: int


def most_confused_pairs(
    cm: np.ndarray, class_names: list[str], top_n: int = 20
) -> list[ConfusedPair]:
    """Off-diagonal cells with the largest counts — the model's worst confusions."""
    cm = cm.copy()
    np.fill_diagonal(cm, 0)
    flat = cm.ravel()
    n = cm.shape[0]
    order = np.argsort(-flat)
    pairs: list[ConfusedPair] = []
    for idx in order[: top_n * 2]:
        count = int(flat[idx])
        if count == 0:
            break
        t, p = divmod(int(idx), n)
        pairs.append(
            ConfusedPair(
                true_idx=t,
                pred_idx=p,
                true_name=class_names[t] if t < len(class_names) else str(t),
                pred_name=class_names[p] if p < len(class_names) else str(p),
                count=count,
            )
        )
        if len(pairs) >= top_n:
            break
    return pairs


@dataclass
class MetricsReport:
    top1: float
    top5: float
    macro_f1: float
    num_samples: int
    num_classes: int
    mean_per_class_acc: float


def summarize(
    logits: np.ndarray, targets: np.ndarray, num_classes: int
) -> tuple[MetricsReport, np.ndarray]:
    """Compute the headline metrics and the confusion matrix in one pass."""
    preds = logits.argmax(axis=1)
    cm = confusion_matrix(preds, targets, num_classes)
    topk = top_k_accuracy(logits, targets, ks=(1, 5))
    pca = per_class_accuracy(cm)
    report = MetricsReport(
        top1=topk[1],
        top5=topk[5],
        macro_f1=macro_f1(cm),
        num_samples=int(len(targets)),
        num_classes=num_classes,
        mean_per_class_acc=float(np.nanmean(pca)) if num_classes else 0.0,
    )
    return report, cm
