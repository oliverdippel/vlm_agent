import torch


class MockVLMBackbone:
    """
    Lightweight fake VLM backbone for debugging.

    It avoids downloading/loading PaliGemma weights and returns dummy
    conditioning embeddings with the expected shape.
    """

    def __init__(self, conditioning_dim: int = 4096, device=None):
        self.conditioning_dim = conditioning_dim
        self.device = device or torch.device("cpu")

    def encode(self, images, instructions) -> torch.Tensor:
        batch_size = len(images)
        return torch.zeros(
            batch_size,
            self.conditioning_dim,
            device=self.device,
        )