"""Reconcile my-photo labels to NABirds taxonomy classes (Phase 7 ingest).

My field labels won't match NABirds' exact strings (which include age/morph qualifiers like
"Reddish Egret (Dark morph)"). This maps a free-form label to a taxonomy index by exact,
then normalized, then base-name (parenthetical-stripped) matching — and reports what didn't
match so I can fix labels rather than silently dropping photos.

Pure Python + unit-tested; ``scripts/ood_eval.py`` uses it before running inference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from wildlife.data.taxonomy import Taxonomy


def normalize_name(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def base_name(s: str) -> str:
    """Strip a trailing parenthetical qualifier, e.g. 'Reddish Egret (Dark morph)' -> 'reddish egret'."""
    return normalize_name(re.sub(r"\(.*?\)", "", s))


@dataclass
class Reconciliation:
    mapping: dict[str, int]  # user label -> taxonomy idx
    unmatched: list[str]
    ambiguous: dict[str, list[int]]  # base name matched >1 class


def _indexes(taxonomy: Taxonomy):
    exact: dict[str, int] = {}
    base: dict[str, list[int]] = {}
    for e in taxonomy.entries:
        exact[normalize_name(e.common_name)] = e.idx
        base.setdefault(base_name(e.common_name), []).append(e.idx)
    return exact, base


def reconcile_labels(labels: list[str], taxonomy: Taxonomy) -> Reconciliation:
    exact, base = _indexes(taxonomy)
    mapping: dict[str, int] = {}
    unmatched: list[str] = []
    ambiguous: dict[str, list[int]] = {}

    for label in labels:
        n = normalize_name(label)
        if n in exact:
            mapping[label] = exact[n]
            continue
        b = base_name(label)
        candidates = base.get(b, [])
        if len(candidates) == 1:
            mapping[label] = candidates[0]
        elif len(candidates) > 1:
            # Base name maps to multiple morphs/ages — pick the first but flag it.
            mapping[label] = candidates[0]
            ambiguous[label] = candidates
        else:
            unmatched.append(label)

    return Reconciliation(mapping=mapping, unmatched=unmatched, ambiguous=ambiguous)
