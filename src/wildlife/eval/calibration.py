"""Confidence calibration (Phase 6): Expected Calibration Error + reliability curve.

NumPy-only so it's testable without a GPU. A well-calibrated classifier's confidence
matches its accuracy; ECE summarizes the gap, and the reliability bins feed the diagram.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    avg_confidence: float
    accuracy: float


@dataclass
class CalibrationReport:
    ece: float
    mce: float  # maximum calibration error
    bins: list[ReliabilityBin]
    overconfident: bool  # avg confidence > avg accuracy overall


def _nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    """Mean negative log-likelihood of the true class under temperature-scaled softmax."""
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=1))
    logp_true = z[np.arange(len(targets)), targets] - logsumexp
    return float(-logp_true.mean())


def fit_temperature(
    logits: np.ndarray, targets: np.ndarray, bounds: tuple[float, float] = (0.5, 5.0)
) -> float:
    """Fit a single softmax temperature T that minimizes NLL (Guo et al. 2017).

    Must be fit on a held-out split (val), then applied to test logits. T>1 softens
    over-confident predictions; T<1 sharpens under-confident ones.
    """
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(lambda t: _nll(logits, targets, t), bounds=bounds, method="bounded")
    return float(res.x)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / temperature


def calibration(logits: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> CalibrationReport:
    probs = softmax(logits)
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == targets).astype(np.float64)
    n = len(targets)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Last bin is inclusive of 1.0.
        in_bin = (
            (confidences > lo) & (confidences <= hi)
            if i > 0
            else (confidences >= lo) & (confidences <= hi)
        )
        count = int(in_bin.sum())
        if count == 0:
            bins.append(ReliabilityBin(lo, hi, 0, 0.0, 0.0))
            continue
        avg_conf = float(confidences[in_bin].mean())
        acc = float(correct[in_bin].mean())
        gap = abs(avg_conf - acc)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bins.append(ReliabilityBin(lo, hi, count, avg_conf, acc))

    return CalibrationReport(
        ece=float(ece),
        mce=float(mce),
        bins=bins,
        overconfident=bool(confidences.mean() > correct.mean()),
    )
