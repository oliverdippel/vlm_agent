import numpy as np
import torch
from PIL import Image

from src.models.backbone import FrozenVLMBackbone


class FakeInputs(dict):
    """Mimics Hugging Face BatchEncoding enough for this unit test."""

    def to(self, device):
        return self


class FakeProcessor:
    def __call__(self, images, text, return_tensors, padding):
        self.images = images
        self.text = text
        self.return_tensors = return_tensors
        self.padding = padding
        return FakeInputs(input_ids=torch.tensor([[1, 2, 3]]))


class FakeOutputs:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class FakeModel:
    def __call__(self, **kwargs):
        # Shape: (B=1, seq_len=3, D=2)
        hidden = torch.tensor(
            [
                [
                    [1.0, 2.0],   # CLS token
                    [3.0, 4.0],   # instruction token 1
                    [5.0, 6.0],   # instruction token 2
                ]
            ]
        )
        return FakeOutputs(hidden_states=[hidden])


def make_dummy_input():
    images = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))]
    instructions = ["pick up the red block"]
    return images, instructions


def test_encode_concatenates_cls_and_mean_pooled_language_embedding():
    backbone = FrozenVLMBackbone.__new__(FrozenVLMBackbone)
    backbone.device = torch.device("cpu")
    backbone.processor = FakeProcessor()
    backbone.model = FakeModel()

    images, instructions = make_dummy_input()

    emb = backbone.encode(images, instructions)

    expected = torch.tensor([[1.0, 2.0, 4.0, 5.0]])

    assert torch.equal(emb, expected)


def test_encode_returns_expected_shape():
    backbone = FrozenVLMBackbone.__new__(FrozenVLMBackbone)
    backbone.device = torch.device("cpu")
    backbone.processor = FakeProcessor()
    backbone.model = FakeModel()

    images, instructions = make_dummy_input()

    emb = backbone.encode(images, instructions)

    assert emb.shape == (1, 4)


def test_processor_is_called_with_expected_arguments():
    backbone = FrozenVLMBackbone.__new__(FrozenVLMBackbone)
    backbone.device = torch.device("cpu")
    backbone.processor = FakeProcessor()
    backbone.model = FakeModel()

    images, instructions = make_dummy_input()

    _ = backbone.encode(images, instructions)

    assert backbone.processor.images == images
    assert backbone.processor.text == instructions
    assert backbone.processor.return_tensors == "pt"
    assert backbone.processor.padding is True