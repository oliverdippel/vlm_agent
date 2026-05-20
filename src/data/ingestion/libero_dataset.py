from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.transformation.normalizer import ActionNormalizer


class LIBERODataset(Dataset):
    """
    PyTorch Dataset for LIBERO data stored in HDF5 files.
        - Each HDF5 file contains multiple demos, 
          each with RGB images and 7-DoF actions.
        - The dataset index is built lazily to avoid loading all data into memory.
        - __getitem__ loads only the requested sample from disk.

    Each returned item:
        image: Tensor, shape (3, H, W), float32 in [0, 1]
        action: Tensor, shape (7,), float32
        instruction: str
    """

    def __init__(
            self,
            data_dir: str, 
            transform=None, 
            action_normalizer: ActionNormalizer = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.action_normalizer = action_normalizer

        # List of all training samples as lightweight references.
        self.samples = self._build_index()

    def _build_index(self) -> list[dict]:
        """
        Build an index over all HDF5 files, demos, and timesteps.

        This does not load images/actions into memory.
        It only records where each sample lives.
        """
        samples = []

        for h5_path in sorted(self.data_dir.glob("*.hdf5")):
            with h5py.File(h5_path, "r") as f:
                instruction = f.attrs.get("instruction", "")

                for demo_key in f["data"].keys():
                    demo = f["data"][demo_key]
                    num_steps = len(demo["actions"])

                    for t in range(num_steps):
                        samples.append(
                            {
                                "h5_path": h5_path,
                                "demo_key": demo_key,
                                "timestep": t,
                                "instruction": instruction,
                            }
                        )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        h5_path = sample["h5_path"]
        demo_key = sample["demo_key"]
        timestep = sample["timestep"]
        
        with h5py.File(h5_path, "r") as f:
            demo = f["data"][demo_key]

            img = demo["obs"]["agentview_rgb"][timestep]  # (H, W, 3), uint8
            action = demo["actions"][timestep]              # (7,), float32

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        action = torch.from_numpy(action).float()

        if self.action_normalizer is not None:
            action = self.action_normalizer.normalize(action)

        assert action.shape[-1] == 7

        return {
            "image": img,
            "action": action,
            "instruction": sample["instruction"],
        }


def get_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    action_normalizer: ActionNormalizer = None,
) -> DataLoader:
    """
    DataLoader for LIBERO data.
    """
    from src.data.transformation.transforms import get_train_transform

    dataset = LIBERODataset(
        data_dir=data_dir,
        transform=get_train_transform(),
        action_normalizer=action_normalizer,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # MPS does not benefit from CUDA pinned memory.
    )