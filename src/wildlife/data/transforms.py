"""Image transforms tuned for fine-grained classification.

Train: resize/random-resized-crop, horizontal flip, RandAugment, normalize, random
erasing. Eval: deterministic resize + center-crop + normalize. Mixup/CutMix are
batch-level and applied in the training loop (see :class:`MixupCutmix`).

Normalization defaults to ImageNet stats (backbones are ImageNet-pretrained).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TransformConfig:
    image_size: int = 224
    resize_ratio: float = 1.14  # eval resize = image_size * ratio, then center-crop
    randaugment: bool = True
    rand_aug_n: int = 2
    rand_aug_m: int = 9
    hflip: bool = True
    random_erasing: float = 0.25  # probability; 0 disables
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


def build_train_transform(cfg: TransformConfig) -> T.Compose:
    ops: list = [T.RandomResizedCrop(cfg.image_size, scale=(0.5, 1.0), ratio=(0.75, 1.333))]
    if cfg.hflip:
        ops.append(T.RandomHorizontalFlip())
    if cfg.randaugment:
        ops.append(T.RandAugment(num_ops=cfg.rand_aug_n, magnitude=cfg.rand_aug_m))
    ops.append(T.ToTensor())
    ops.append(T.Normalize(cfg.mean, cfg.std))
    if cfg.random_erasing > 0:
        ops.append(T.RandomErasing(p=cfg.random_erasing))
    return T.Compose(ops)


def build_eval_transform(cfg: TransformConfig) -> T.Compose:
    resize = int(round(cfg.image_size * cfg.resize_ratio))
    return T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(cfg.image_size),
            T.ToTensor(),
            T.Normalize(cfg.mean, cfg.std),
        ]
    )


def denormalize(t: torch.Tensor, cfg: TransformConfig | None = None) -> torch.Tensor:
    """Undo Normalize for visualization. Accepts (C,H,W) or (N,C,H,W)."""
    cfg = cfg or TransformConfig()
    mean = torch.tensor(cfg.mean, device=t.device).view(-1, 1, 1)
    std = torch.tensor(cfg.std, device=t.device).view(-1, 1, 1)
    return (t * std + mean).clamp(0, 1)


class MixupCutmix:
    """Batch-level Mixup/CutMix wrapper around timm's implementation.

    Applied inside the training loop *after* the dataloader. Returns mixed images and
    soft targets; a no-op passthrough when both alphas are 0 (returns hard targets).
    """

    def __init__(
        self,
        num_classes: int,
        *,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        switch_prob: float = 0.5,
        label_smoothing: float = 0.1,
    ) -> None:
        self.enabled = (mixup_alpha > 0 or cutmix_alpha > 0) and prob > 0
        self._mix = None
        if self.enabled:
            from timm.data import Mixup

            self._mix = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=prob,
                switch_prob=switch_prob,
                label_smoothing=label_smoothing,
                num_classes=num_classes,
            )

    def __call__(self, images: torch.Tensor, targets: torch.Tensor):
        if self._mix is not None and images.size(0) % 2 == 0:
            return self._mix(images, targets)
        return images, targets
