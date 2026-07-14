"""Eval metrics + calibration (Phase 6). NumPy-only, so they run without torch/GPU."""

from __future__ import annotations

import numpy as np

from wildlife.eval.calibration import calibration
from wildlife.eval.metrics import (
    macro_f1,
    most_confused_pairs,
    per_class_accuracy,
    summarize,
)


def _toy():
    # 3 samples, 4 classes. preds argmax = [0, 1, 0]; targets = [0, 1, 2].
    logits = np.array(
        [
            [5.0, 1.0, 0.0, 0.0],  # -> 0 (correct)
            [0.0, 3.0, 1.0, 0.0],  # -> 1 (correct)
            [2.0, 0.0, 1.0, 0.0],  # -> 0 (wrong; target 2)
        ]
    )
    targets = np.array([0, 1, 2])
    return logits, targets


def test_topk_and_confusion():
    logits, targets = _toy()
    report, cm = summarize(logits, targets, num_classes=4)
    assert report.top1 == 2 / 3
    assert report.top5 == 1.0  # k=5 >= 4 classes -> everything is in top-k
    assert report.num_samples == 3
    assert report.num_classes == 4
    assert cm[0, 0] == 1 and cm[1, 1] == 1 and cm[2, 0] == 1
    assert cm.sum() == 3


def test_per_class_accuracy_handles_absent_class():
    _, _ = _toy()
    logits, targets = _toy()
    _, cm = summarize(logits, targets, num_classes=4)
    pca = per_class_accuracy(cm)
    assert pca[0] == 1.0
    assert pca[1] == 1.0
    assert pca[2] == 0.0
    assert np.isnan(pca[3])  # class 3 never appears


def test_macro_f1_known_value():
    logits, targets = _toy()
    _, cm = summarize(logits, targets, num_classes=4)
    # class0 F1=0.6667, class1 F1=1, class2 F1=0; class3 absent -> mean over 3 present.
    assert abs(macro_f1(cm) - (2 / 3 + 1 + 0) / 3) < 1e-9


def test_most_confused_pairs():
    logits, targets = _toy()
    _, cm = summarize(logits, targets, num_classes=4)
    pairs = most_confused_pairs(cm, class_names=["A", "B", "C", "D"], top_n=5)
    assert len(pairs) == 1
    p = pairs[0]
    assert (p.true_idx, p.pred_idx, p.count) == (2, 0, 1)
    assert p.true_name == "C" and p.pred_name == "A"


def test_calibration_perfect_when_confident_and_correct():
    # Every prediction correct with ~prob 1 -> ECE ~ 0, not overconfident.
    logits = np.eye(4) * 12.0  # near one-hot
    targets = np.arange(4)
    rep = calibration(logits, targets, n_bins=10)
    assert rep.ece < 1e-3
    assert rep.overconfident is False


def test_calibration_flags_overconfidence():
    # High confidence but half wrong -> overconfident, sizeable ECE.
    logits = np.array([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
    targets = np.array([0, 0, 1, 1])  # 50% accuracy, ~100% confidence
    rep = calibration(logits, targets, n_bins=10)
    assert rep.overconfident is True
    assert rep.ece > 0.4
