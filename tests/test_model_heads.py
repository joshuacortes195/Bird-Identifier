"""Model factory + head interface: output dim derives from num_classes, not hardcoded."""

from __future__ import annotations

import pytest
import torch

from wildlife.models import build_model, count_parameters
from wildlife.models.factory import ModelConfig
from wildlife.models.heads import LinearHead, available_heads, build_head


def test_linear_head_output_dim_follows_num_classes():
    head = build_head("linear", in_features=64, num_classes=7)
    out = head(torch.randn(3, 64))
    assert out.shape == (3, 7)
    assert isinstance(head, LinearHead)


def test_head_registry_lists_and_rejects_unknown():
    assert "linear" in available_heads()
    assert "hierarchical" in available_heads()
    with pytest.raises(KeyError):
        build_head("nope", in_features=8, num_classes=2)


def test_hierarchical_head_is_stub():
    with pytest.raises(NotImplementedError):
        build_head("hierarchical", in_features=8, num_classes=2)


@pytest.mark.parametrize("num_classes", [3, 11])
def test_build_model_forward_shape(num_classes):
    # Tiny backbone, no pretrained download, for a fast CPU test.
    cfg = ModelConfig(backbone="convnextv2_atto", pretrained=False, head="linear")
    model = build_model(cfg, num_classes)
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
        feats = model.forward_features(x)
    assert out.shape == (2, num_classes)
    assert feats.shape[0] == 2
    assert model.num_classes == num_classes
    total, trainable = count_parameters(model)
    assert total > 0 and trainable == total
