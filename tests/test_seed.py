"""Seeding determinism — cheap but real coverage so the suite isn't empty."""

from __future__ import annotations

import random

import numpy as np

from wildlife.utils import seed_everything


def test_seed_returns_value():
    assert seed_everything(123) == 123


def test_seed_makes_python_and_numpy_deterministic():
    seed_everything(7)
    a_py = [random.random() for _ in range(5)]
    a_np = np.random.rand(5)

    seed_everything(7)
    b_py = [random.random() for _ in range(5)]
    b_np = np.random.rand(5)

    assert a_py == b_py
    assert np.array_equal(a_np, b_np)


def test_different_seeds_differ():
    seed_everything(1)
    first = np.random.rand(3)
    seed_everything(2)
    second = np.random.rand(3)
    assert not np.array_equal(first, second)
