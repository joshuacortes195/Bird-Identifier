"""One train step reduces loss; checkpoint round-trips; scheduler warms up then decays."""

from __future__ import annotations

import torch
from torch import nn

from wildlife.train.optim import CosineWarmupScheduler, OptimConfig, build_optimizer
from wildlife.utils.checkpoint import load_checkpoint, save_checkpoint


class _Tiny(nn.Module):
    def __init__(self, n_in=16, n_out=4):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.fc(x)


def test_scheduler_warmup_then_decay():
    model = _Tiny()
    opt = build_optimizer(model, OptimConfig(lr=1.0))
    sched = CosineWarmupScheduler(opt, warmup_steps=5, total_steps=25, min_lr_ratio=0.0)

    lrs = []
    for _ in range(25):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()

    # Warmup ramps up; peak around end of warmup; then decays toward ~0.
    assert lrs[0] < lrs[4]
    assert max(lrs) <= 1.0 + 1e-6
    assert lrs[-1] < lrs[5]
    assert lrs[-1] < 0.1


def test_one_train_step_reduces_loss():
    torch.manual_seed(0)
    model = _Tiny()
    opt = build_optimizer(model, OptimConfig(lr=0.1, weight_decay=0.0))
    crit = nn.CrossEntropyLoss()
    x = torch.randn(32, 16)
    y = torch.randint(0, 4, (32,))

    with torch.no_grad():
        loss_before = crit(model(x), y).item()
    for _ in range(20):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        loss_after = crit(model(x), y).item()
    assert loss_after < loss_before


def test_checkpoint_roundtrip(tmp_path):
    model = _Tiny()
    opt = build_optimizer(model, OptimConfig(lr=0.01))
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        optimizer=opt,
        epoch=3,
        best_metric=0.42,
        metrics={"val_top1": 0.42},
        config={"seed": 42},
        class_names=["a", "b", "c", "d"],
    )
    ckpt = load_checkpoint(path)
    assert ckpt["epoch"] == 3
    assert ckpt["best_metric"] == 0.42
    assert ckpt["class_names"] == ["a", "b", "c", "d"]
    assert "git_commit" in ckpt

    model2 = _Tiny()
    model2.load_state_dict(ckpt["model"])
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.equal(p1, p2)
