"""Config-driven label space (extensibility seam #2).

The class count, human-readable names, and per-class taxonomic metadata all come from
a ``configs/taxonomy/*.yaml`` file — never a hardcoded 555. The model head's output
dimension, the loss, and the eval reports all read from a :class:`Taxonomy`. Adding
mammals later is a new YAML (or a merged ``animals.yaml``), with no code change.

YAML schema::

    name: birds
    supercategory_default: bird
    classes:
      - idx: 0
        class_id: "817"          # dataset-native id (string; may be sparse)
        common_name: "Wood Duck"
        scientific_name: "Aix sponsa"   # optional
        supercategory: bird             # optional; falls back to default
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TaxonEntry:
    idx: int
    class_id: str
    common_name: str
    scientific_name: str | None = None
    supercategory: str = "bird"


@dataclass
class Taxonomy:
    name: str
    entries: list[TaxonEntry] = field(default_factory=list)

    # --- label-space properties (what the head/loss/eval consume) ---
    @property
    def num_classes(self) -> int:
        return len(self.entries)

    @property
    def class_names(self) -> list[str]:
        return [e.common_name for e in self.entries]

    @property
    def supercategories(self) -> list[str]:
        """Unique coarse taxa in first-seen order (a single group for birds now)."""
        seen: dict[str, None] = {}
        for e in self.entries:
            seen.setdefault(e.supercategory, None)
        return list(seen)

    def supercategory_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.supercategories)}

    def idx_to_supercategory_idx(self) -> list[int]:
        smap = self.supercategory_index()
        return [smap[e.supercategory] for e in self.entries]

    def class_id_to_idx(self) -> dict[str, int]:
        return {e.class_id: e.idx for e in self.entries}

    def __getitem__(self, idx: int) -> TaxonEntry:
        return self.entries[idx]

    def __len__(self) -> int:
        return len(self.entries)


def load_taxonomy(path: str | Path) -> Taxonomy:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    default_super = data.get("supercategory_default", "bird")
    entries = [
        TaxonEntry(
            idx=int(c["idx"]),
            class_id=str(c["class_id"]),
            common_name=c["common_name"],
            scientific_name=c.get("scientific_name"),
            supercategory=c.get("supercategory", default_super),
        )
        for c in data["classes"]
    ]
    entries.sort(key=lambda e: e.idx)
    # Contiguity check — idx must be 0..N-1 so it indexes the head cleanly.
    for expected, e in enumerate(entries):
        if e.idx != expected:
            raise ValueError(f"Taxonomy '{path}' idx not contiguous: expected {expected}, got {e.idx}")
    return Taxonomy(name=data["name"], entries=entries)


def write_taxonomy(taxonomy: Taxonomy, path: str | Path, *, supercategory_default: str = "bird") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = []
    for e in taxonomy.entries:
        row: dict = {"idx": e.idx, "class_id": e.class_id, "common_name": e.common_name}
        if e.scientific_name:
            row["scientific_name"] = e.scientific_name
        if e.supercategory != supercategory_default:
            row["supercategory"] = e.supercategory
        classes.append(row)
    doc = {
        "name": taxonomy.name,
        "supercategory_default": supercategory_default,
        "classes": classes,
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    return path
