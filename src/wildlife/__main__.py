"""Tiny CLI entry point. Real work lives in scripts/ (Hydra-driven)."""

from __future__ import annotations

import sys

from wildlife import __version__


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in {"-v", "--version", "version"}:
        print(f"wildlife-classifier {__version__}")
        return 0
    print(
        "wildlife-classifier — use the Hydra scripts:\n"
        "  python scripts/train.py [overrides...]\n"
        "  python scripts/evaluate.py\n"
        "  python scripts/export.py\n"
        "  python scripts/download_data.py\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
