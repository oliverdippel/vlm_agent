from pathlib import Path

import h5py
import numpy as np
import torch


class ActionNormalizer:
    """
    Fits mean/std over all demo actions in the dataset.
    Must be fitted once, then passed to both DataLoader and Evaluator.

    Usage:
        normalizer = ActionNormalizer.fit_from_dir("data/raw/")
        normalizer.save("data/processed/action_stats.npz")

        # later
        normalizer = ActionNormalizer.load("data/processed/action_stats.npz")
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean  # (7,)
        self.std = std    # (7,)

    @classmethod
    def fit_from_dir(cls, data_dir: str) -> "ActionNormalizer":
        all_actions = []
        for h5_path in sorted(Path(data_dir).glob("*.hdf5")):
            with h5py.File(h5_path, "r") as f:
                for demo_key in f["data"].keys():
                    actions = f["data"][demo_key]["actions"][:]  # (T, 7)
                    all_actions.append(actions)

        all_actions = np.concatenate(all_actions, axis=0)  # (N, 7)
        mean = all_actions.mean(axis=0)
        std = all_actions.std(axis=0) + 1e-8

        print(f"[ActionNormalizer] fitted on {len(all_actions)} transitions")
        print(f"  mean: {mean.round(4)}")
        print(f"  std:  {std.round(4)}")
        print(f"  min:  {all_actions.min(axis=0).round(4)}")
        print(f"  max:  {all_actions.max(axis=0).round(4)}")

        return cls(mean, std)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)
        print(f"[ActionNormalizer] saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ActionNormalizer":
        data = np.load(path)
        return cls(data["mean"], data["std"])

    def normalize(self, action: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Raw action → normalized. Used in DataLoader."""
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        mean = torch.from_numpy(self.mean).float().to(action.device)
        std = torch.from_numpy(self.std).float().to(action.device)
        return (action - mean) / std

    def denormalize(self, action: torch.Tensor) -> np.ndarray:
        """Normalized action → raw robot action. Used in Evaluator."""
        mean = torch.from_numpy(self.mean).float().to(action.device)
        std = torch.from_numpy(self.std).float().to(action.device)
        return ((action * std) + mean).cpu().numpy()