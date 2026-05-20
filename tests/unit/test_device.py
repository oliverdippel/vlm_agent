import torch

from src.utils.device import get_device


def test_get_device_returns_torch_device():
    device = get_device()

    assert isinstance(device, torch.device)


def test_get_device_returns_valid_device_type():
    device = get_device()

    assert device.type in {"cuda", "mps", "cpu"}