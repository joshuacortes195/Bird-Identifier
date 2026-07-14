"""Classification metrics computed from a single pass of test logits.

Kept dependency-light (numpy + sklearn) so the whole eval report comes from one loop:
top-1/top-5, macro-F1, per-class accuracy, most-confused pairs, and calibration (ECE).
"""

from __future__ import annotations

import numpy as np


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))


def per_class_accuracy(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    acc = np.full(num_classes, np.nan)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            acc[c] = float((preds[mask] == c).mean())
    return acc


def most_confused_pairs(
    preds: np.ndarray, labels: np.ndarray, class_names: list[str], top: int = 15
) -> list[dict]:
    from collections import Counter

    wrong = Counter()
    for t, p in zip(labels, preds, strict=True):
        if t != p:
            wrong[(int(t), int(p))] += 1
    out = []
    for (t, p), n in wrong.most_common(top):
        out.append({"true": class_names[t], "pred": class_names[p], "count": int(n)})
    return out


def expected_calibration_error(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """ECE + reliability-diagram data using softmax max-prob confidence."""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            acc_bin = float(correct[mask].mean())
            conf_bin = float(conf[mask].mean())
            w = mask.sum() / n
            ece += w * abs(acc_bin - conf_bin)
            rows.append(
                {
                    "bin": [float(lo), float(hi)],
                    "acc": acc_bin,
                    "conf": conf_bin,
                    "count": int(mask.sum()),
                }
            )
    return {"ece": float(ece), "bins": rows, "mean_confidence": float(conf.mean())}


def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(labels, preds, average="macro", zero_division=0))
