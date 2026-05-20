import pytest
import torch
from PIL import Image
import numpy as np

from src.models.backbone import FrozenVLMBackbone


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def dummy_input():
    imgs = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))]
    instrs = ["pick up the red block"]
    return imgs, instrs


@pytest.fixture(scope="module")
def backbone():
    return FrozenVLMBackbone()


@pytest.fixture(scope="module")
def embedding(backbone, dummy_input):
    imgs, instrs = dummy_input
    return backbone.encode(imgs, instrs)


def test_real_backbone_output_is_tensor(embedding):
    assert isinstance(embedding, torch.Tensor)


def test_real_backbone_output_shape(embedding):
    assert embedding.shape == (1, 4096)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available on this machine",
)
def test_real_backbone_runs_on_mps(embedding):
    assert embedding.device.type == "mps"