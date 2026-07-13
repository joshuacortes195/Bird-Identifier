"""Torch-free data schema: the ``Sample`` record and split-manifest IO.

Manifests are the single source of truth for what's in each split. They are plain CSV
so they're diffable, reproducible, and inspectable without loading any images. The
torch ``Dataset`` classes (Phase 2) read these manifests; nothing hardcodes paths.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

# Manifest columns, in order. `bbox_*` are -1 when a dataset has no boxes.
MANIFEST_FIELDS = [
    "image_path",  # relative to the dataset root
    "class_idx",  # contiguous 0..num_classes-1 index into the taxonomy label space
    "class_id",  # dataset-native class id (may be sparse / string)
    "supercategory",  # coarse taxon label (constant "bird" now; plumbed for later)
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
]

NO_BBOX = (-1, -1, -1, -1)


@dataclass(frozen=True)
class Sample:
    """One labeled image. ``bbox`` is (x, y, w, h) in pixels or ``NO_BBOX``."""

    image_path: str
    class_idx: int
    class_id: str
    supercategory: str = "bird"
    bbox: tuple[int, int, int, int] = NO_BBOX

    @property
    def has_bbox(self) -> bool:
        return self.bbox != NO_BBOX

    def to_row(self) -> dict[str, str]:
        d = asdict(self)
        bx, by, bw, bh = d.pop("bbox")
        d.update(bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh)
        return {k: str(v) for k, v in d.items()}

    @classmethod
    def from_row(cls, row: dict[str, str]) -> Sample:
        return cls(
            image_path=row["image_path"],
            class_idx=int(row["class_idx"]),
            class_id=row["class_id"],
            supercategory=row.get("supercategory", "bird"),
            bbox=(
                int(row["bbox_x"]),
                int(row["bbox_y"]),
                int(row["bbox_w"]),
                int(row["bbox_h"]),
            ),
        )


def write_manifest(samples: list[Sample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for s in samples:
            writer.writerow(s.to_row())
    return path


def read_manifest(path: str | Path) -> list[Sample]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        return [Sample.from_row(row) for row in csv.DictReader(f)]
