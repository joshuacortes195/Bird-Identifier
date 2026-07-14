"""Temperature scaling: preserves predictions (accuracy), improves over-confidence."""

from __future__ import annotations

import numpy as np

from wildlife.eval.calibration import (
    _nll,
    apply_temperature,
    fit_temperature,
)


def _overconfident_logits(n=400, c=10, seed=0):
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, c, size=n)
    logits = rng.normal(0, 1, size=(n, c))
    # Make the true class win by a large, over-confident margin ~85% of the time.
    for i, t in enumerate(targets):
        if rng.random() < 0.85:
            logits[i, t] += 8.0
    return logits, targets


def test_apply_temperature_preserves_argmax():
    logits, _ = _overconfident_logits()
    for temp in (0.5, 2.0, 4.0):
        assert np.array_equal(
            logits.argmax(1), apply_temperature(logits, temp).argmax(1)
        )


def test_fit_temperature_minimizes_nll():
    # fit_temperature optimizes NLL, so the fitted T must not increase NLL vs T=1.
    logits, targets = _overconfident_logits()
    t = fit_temperature(logits, targets)
    assert t > 1.0  # over-confident model needs softening
    assert _nll(logits, targets, t) <= _nll(logits, targets, 1.0) + 1e-9
