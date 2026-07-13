"""Render a post-augmentation training batch to a PNG for sanity-checking transforms."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from wildlife.data.transforms import TransformConfig, denormalize  # noqa: E402


def save_batch_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    out_path: str | Path,
    *,
    max_images: int = 32,
    tcfg: TransformConfig | None = None,
) -> Path:
    """Save a grid of a (already-normalized) image batch with class-name titles."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(max_images, images.size(0))
    imgs = denormalize(images[:n].detach().cpu(), tcfg)
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0))
    axes = axes.flat if hasattr(axes, "flat") else [axes]
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(imgs[i].permute(1, 2, 0).numpy())
            name = class_names[int(labels[i])] if labels is not None else ""
            ax.set_title(name[:16], fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
