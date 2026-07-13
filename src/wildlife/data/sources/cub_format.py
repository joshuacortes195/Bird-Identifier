"""Parser for the CUB-200-2011 file layout, shared by CUB and NABirds.

Both datasets ship the same whitespace-delimited index files:

    images.txt              <image_id> <relative_image_path>
    image_class_labels.txt  <image_id> <class_id>
    train_test_split.txt    <image_id> <is_training(1|0)>
    classes.txt             <class_id> <class_name>
    bounding_boxes.txt      <image_id> <x> <y> <w> <h>   (optional)

The label space is derived from the class_ids that actually appear on images — so the
class count (200 for CUB, 555 for NABirds) is discovered, never hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wildlife.data.schema import NO_BBOX, Sample
from wildlife.data.taxonomy import TaxonEntry, Taxonomy


def _read_pairs(path: Path) -> dict[str, str]:
    """Read a two-column ``<id> <value>`` file into a dict (value = rest of line)."""
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, value = line.partition(" ")
            out[key] = value.strip()
    return out


def _read_bboxes(path: Path) -> dict[str, tuple[int, int, int, int]]:
    boxes: dict[str, tuple[int, int, int, int]] = {}
    if not path.exists():
        return boxes
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                image_id = parts[0]
                x, y, w, h = (int(round(float(v))) for v in parts[1:5])
                boxes[image_id] = (x, y, w, h)
    return boxes


@dataclass
class ParsedDataset:
    samples: list[Sample]  # all images, split marked separately
    train_ids: set[str]  # image_ids in the official train split
    test_ids: set[str]  # image_ids in the official test split
    taxonomy: Taxonomy
    # image_id -> its position in `samples`, for split lookup
    image_id_by_sample_index: list[str]


def clean_class_name(raw: str) -> str:
    """'001.Black_footed_Albatross' -> 'Black footed Albatross'; NABirds names pass through."""
    name = raw.split(".", 1)[1] if "." in raw and raw.split(".", 1)[0].isdigit() else raw
    return name.replace("_", " ").strip()


def parse_cub_format(
    root: str | Path,
    *,
    taxonomy_name: str,
    supercategory: str = "bird",
    images_dir: str = "images",
) -> ParsedDataset:
    root = Path(root)
    images = _read_pairs(root / "images.txt")
    labels = _read_pairs(root / "image_class_labels.txt")
    split = _read_pairs(root / "train_test_split.txt")
    class_names_raw = _read_pairs(root / "classes.txt")
    bboxes = _read_bboxes(root / "bounding_boxes.txt")

    if not images or not labels:
        raise FileNotFoundError(
            f"Missing images.txt/image_class_labels.txt under {root} — is the dataset extracted?"
        )

    # Label space = class_ids that actually appear on images, sorted for stability.
    present_ids = sorted({labels[i] for i in images}, key=lambda s: (len(s), s))
    class_id_to_idx = {cid: idx for idx, cid in enumerate(present_ids)}

    entries = [
        TaxonEntry(
            idx=idx,
            class_id=cid,
            common_name=clean_class_name(class_names_raw.get(cid, cid)),
            supercategory=supercategory,
        )
        for cid, idx in class_id_to_idx.items()
    ]
    taxonomy = Taxonomy(name=taxonomy_name, entries=entries)

    samples: list[Sample] = []
    image_id_by_index: list[str] = []
    train_ids: set[str] = set()
    test_ids: set[str] = set()

    for image_id, rel_path in images.items():
        cid = labels[image_id]
        idx = class_id_to_idx[cid]
        samples.append(
            Sample(
                image_path=f"{images_dir}/{rel_path}",
                class_idx=idx,
                class_id=cid,
                supercategory=supercategory,
                bbox=bboxes.get(image_id, NO_BBOX),
            )
        )
        image_id_by_index.append(image_id)
        if split.get(image_id, "1") == "1":
            train_ids.add(image_id)
        else:
            test_ids.add(image_id)

    return ParsedDataset(
        samples=samples,
        train_ids=train_ids,
        test_ids=test_ids,
        taxonomy=taxonomy,
        image_id_by_sample_index=image_id_by_index,
    )
