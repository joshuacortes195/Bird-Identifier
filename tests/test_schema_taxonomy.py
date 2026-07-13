"""Manifest IO round-trip + taxonomy label-space behavior."""

from __future__ import annotations

import pytest

from wildlife.data.schema import NO_BBOX, Sample, read_manifest, write_manifest
from wildlife.data.taxonomy import TaxonEntry, Taxonomy, load_taxonomy, write_taxonomy


def test_sample_row_roundtrip():
    s = Sample("a/b.jpg", 5, "817", "bird", (1, 2, 3, 4))
    assert Sample.from_row(s.to_row()) == s
    assert s.has_bbox
    assert not Sample("x.jpg", 0, "0", bbox=NO_BBOX).has_bbox


def test_manifest_write_read(tmp_path):
    samples = [Sample(f"i{i}.jpg", i % 3, str(i), "bird", (i, i, i + 1, i + 1)) for i in range(6)]
    path = write_manifest(samples, tmp_path / "m.csv")
    assert read_manifest(path) == samples


def test_taxonomy_label_space(tmp_path):
    entries = [
        TaxonEntry(0, "10", "Wood Duck", supercategory="bird"),
        TaxonEntry(1, "11", "Mallard", supercategory="bird"),
    ]
    tax = Taxonomy("birds", entries)
    assert tax.num_classes == 2
    assert tax.class_names == ["Wood Duck", "Mallard"]
    assert tax.supercategories == ["bird"]
    assert tax.idx_to_supercategory_idx() == [0, 0]

    path = write_taxonomy(tax, tmp_path / "t.yaml")
    reloaded = load_taxonomy(path)
    assert reloaded.num_classes == 2
    assert reloaded.class_id_to_idx() == {"10": 0, "11": 1}


def test_taxonomy_rejects_non_contiguous(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: x\nsupercategory_default: bird\nclasses:\n"
        "  - {idx: 0, class_id: '1', common_name: A}\n"
        "  - {idx: 2, class_id: '2', common_name: B}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not contiguous"):
        load_taxonomy(bad)
