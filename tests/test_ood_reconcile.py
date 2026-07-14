"""Label reconciliation for the Phase 7 my-photos OOD test."""

from __future__ import annotations

from wildlife.data.taxonomy import TaxonEntry, Taxonomy
from wildlife.eval.ood import base_name, normalize_name, reconcile_labels


def _tax() -> Taxonomy:
    names = [
        "Wood Duck",
        "Reddish Egret (Dark morph)",
        "Reddish Egret (White morph)",
        "American Robin",
    ]
    return Taxonomy(
        name="t",
        entries=[TaxonEntry(idx=i, class_id=str(i), common_name=n) for i, n in enumerate(names)],
    )


def test_normalize_and_base():
    assert normalize_name("  Reddish Egret!! ") == "reddish egret"
    assert base_name("Reddish Egret (Dark morph)") == "reddish egret"


def test_exact_and_case_insensitive_match():
    r = reconcile_labels(["wood duck", "American robin"], _tax())
    assert r.mapping["wood duck"] == 0
    assert r.mapping["American robin"] == 3
    assert r.unmatched == []


def test_base_name_ambiguous_is_flagged():
    r = reconcile_labels(["Reddish Egret"], _tax())
    # Base name matches two morphs -> mapped to first, flagged ambiguous.
    assert r.mapping["Reddish Egret"] == 1
    assert r.ambiguous["Reddish Egret"] == [1, 2]


def test_unmatched_reported():
    r = reconcile_labels(["Painted Bunting"], _tax())
    assert "Painted Bunting" in r.unmatched
    assert r.mapping == {}
