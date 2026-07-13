"""Exploratory data analysis: reproducible figures + a markdown report.

Logic lives here (not only in a notebook) so it's testable and re-runnable. The
notebook in notebooks/ just calls these functions.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from wildlife.data.schema import read_manifest  # noqa: E402
from wildlife.data.taxonomy import load_taxonomy  # noqa: E402
from wildlife.utils.logging import get_logger  # noqa: E402

log = get_logger("eda")


def plot_class_distribution(counts: list[int], out_path: Path, title: str) -> None:
    counts_sorted = sorted(counts, reverse=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(counts_sorted)), counts_sorted, width=1.0)
    ax.set_xlabel("class rank (most -> least frequent)")
    ax.set_ylabel("# training images")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_image_size_stats(sizes: list[tuple[int, int]], out_path: Path) -> dict:
    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]
    aspects = [w / h for w, h in sizes if h]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(widths, bins=40, alpha=0.7, label="width")
    axes[0].hist(heights, bins=40, alpha=0.7, label="height")
    axes[0].set_title("Image dimensions (px)")
    axes[0].legend()
    axes[1].hist(aspects, bins=40, color="teal")
    axes[1].set_title("Aspect ratio (w/h)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return {
        "width": {"min": min(widths), "max": max(widths), "median": sorted(widths)[len(widths) // 2]},
        "height": {"min": min(heights), "max": max(heights), "median": sorted(heights)[len(heights) // 2]},
        "aspect_median": round(sorted(aspects)[len(aspects) // 2], 3) if aspects else None,
    }


def plot_sample_grid(
    manifest, data_root: Path, class_names: list[str], out_path: Path, *, rows: int = 4, cols: int = 6, seed: int = 0
) -> None:
    rng = random.Random(seed)
    picks = rng.sample(manifest, min(rows * cols, len(manifest)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    for ax, sample in zip(axes.flat, picks, strict=False):
        try:
            img = Image.open(data_root / sample.image_path).convert("RGB")
            ax.imshow(img)
            ax.set_title(class_names[sample.class_idx][:18], fontsize=7)
        except Exception as e:  # noqa: BLE001
            ax.text(0.5, 0.5, "load error", ha="center")
            log.warning("sample grid: %s", e)
        ax.axis("off")
    for ax in list(axes.flat)[len(picks):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_eda(
    processed_dir: str | Path,
    data_root: str | Path,
    out_dir: str | Path,
    *,
    size_sample: int = 800,
    seed: int = 42,
) -> dict:
    """Generate EDA figures + report.md for a prepared dataset. Returns the stats dict."""
    processed_dir = Path(processed_dir)
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy(processed_dir / "taxonomy.yaml")
    train = read_manifest(processed_dir / "train.csv")
    summary = json.loads((processed_dir / "summary.json").read_text(encoding="utf-8"))

    counts = Counter(s.class_idx for s in train)
    per_class = [counts.get(i, 0) for i in range(taxonomy.num_classes)]
    plot_class_distribution(per_class, out_dir / "class_distribution.png", f"{taxonomy.name}: train long tail")

    rng = random.Random(seed)
    sample_for_sizes = rng.sample(train, min(size_sample, len(train)))
    sizes: list[tuple[int, int]] = []
    for s in sample_for_sizes:
        try:
            with Image.open(data_root / s.image_path) as im:
                sizes.append(im.size)
        except Exception:  # noqa: BLE001, S112
            continue
    size_stats = plot_image_size_stats(sizes, out_dir / "image_size_stats.png") if sizes else {}

    plot_sample_grid(train, data_root, taxonomy.class_names, out_dir / "sample_grid.png", seed=seed)

    # Long-tail extremes for the report.
    ranked = sorted(
        ((taxonomy.class_names[i], per_class[i]) for i in range(taxonomy.num_classes)),
        key=lambda kv: kv[1],
    )
    rarest = ranked[:8]
    commonest = ranked[-8:][::-1]

    report = [
        f"# EDA — {taxonomy.name}\n",
        f"- **Classes:** {taxonomy.num_classes}",
        f"- **Train images:** {len(train)}",
        f"- **Images with bbox:** {summary.get('with_bbox', 0)}",
        f"- **Imbalance ratio (train):** {summary['splits']['train']['imbalance_ratio']}x",
        f"- **Image size (median W×H):** {size_stats.get('width', {}).get('median')}×"
        f"{size_stats.get('height', {}).get('median')}"
        if size_stats
        else "",
        "\n## Rarest classes\n",
        *[f"- {name}: {n}" for name, n in rarest],
        "\n## Commonest classes\n",
        *[f"- {name}: {n}" for name, n in commonest],
        "\n## Figures\n",
        "![class distribution](class_distribution.png)",
        "![image sizes](image_size_stats.png)",
        "![samples](sample_grid.png)",
        "\n> Hardest-confusable species pairs are surfaced quantitatively in Phase 6",
        "> (confusion matrix), once a model exists.",
    ]
    (out_dir / "report.md").write_text("\n".join(r for r in report if r != ""), encoding="utf-8")
    stats = {"size_stats": size_stats, "rarest": rarest, "commonest": commonest}
    log.info("EDA written to %s", out_dir)
    return stats
