import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import SimpleCNN, get_model  # noqa: E402


@pytest.mark.parametrize("architecture", ["cnn", "resnet18"])
def test_get_model_output_shape(architecture):
    model = get_model(architecture=architecture, num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_simple_cnn_default_classes():
    model = SimpleCNN()
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 10)


def test_get_model_invalid_architecture():
    with pytest.raises(ValueError):
        get_model(architecture="not-a-real-model", num_classes=10)
