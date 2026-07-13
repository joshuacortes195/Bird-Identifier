"""iNaturalist source — STUB ONLY (extensibility seam; NOT implemented in this build).

This file marks the seam where mammals/reptiles/etc. plug in later. When implemented,
it will download an iNat taxon subset, parse it into the same ``ParsedDataset`` shape,
and register under ``@register_dataset("inat")`` — with a merged ``animals.yaml``
taxonomy (birds + mammals + ...). No training/eval code changes are needed to add it.

Do NOT build this now (see the prompt's scope guardrails).
"""

from __future__ import annotations


def prepare_inat(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
    raise NotImplementedError(
        "iNaturalist source is a documented future seam and is intentionally not "
        "implemented in the bird build. See docs/EXTENSIBILITY.md."
    )
