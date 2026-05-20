import h5py
import numpy as np
import torch

from src.data.ingestion.libero_dataset import LIBERODataset, get_dataloader
from src.data.transformation.transforms import get_train_transform


def _create_mock_libero_hdf5(
    tmp_path,
    image_key: str = "agentview_rgb",
    image_size: int = 144,
    num_steps: int = 10,
):

    h5_path = tmp_path / "mock_task_demo.hdf5"

    with h5py.File(h5_path, "w") as f:
        f.attrs["instruction"] = "mock_task"

        data = f.create_group("data")
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")

        obs.create_dataset(
            image_key,
            data=np.random.randint(
                0,
                255,
                size=(num_steps, image_size, image_size, 3),
                dtype=np.uint8,
            ),
        )

        demo.create_dataset(
            "actions",
            data=np.random.randn(num_steps, 7).astype(np.float32),
        )

        demo.create_dataset("dones", data=np.zeros(num_steps, dtype=bool))
        demo.create_dataset("rewards", data=np.zeros(num_steps, dtype=np.float32))

    return h5_path


def test_libero_dataloader_batch_shapes(tmp_path):
    _create_mock_libero_hdf5(tmp_path)

    dataloader = get_dataloader(
        data_dir=str(tmp_path),
        batch_size=4,
        num_workers=0,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["action"].shape == (4, 7)
    assert len(batch["instruction"]) == 4


def test_libero_dataloader_batch_types(tmp_path):
    _create_mock_libero_hdf5(tmp_path)

    dataloader = get_dataloader(
        data_dir=str(tmp_path),
        batch_size=4,
        num_workers=0,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert isinstance(batch["image"], torch.Tensor)
    assert isinstance(batch["action"], torch.Tensor)
    assert isinstance(batch["instruction"], list | tuple)


def test_libero_dataloader_image_values_are_normalized(tmp_path):
    _create_mock_libero_hdf5(tmp_path)

    dataloader = get_dataloader(
        data_dir=str(tmp_path),
        batch_size=4,
        num_workers=0,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert torch.isfinite(batch["image"]).all()


def test_libero_dataset_uses_agentview_rgb_key(tmp_path):
    _create_mock_libero_hdf5(
        tmp_path,
        image_key="agentview_rgb",
        image_size=144,
    )

    dataset = LIBERODataset(
        data_dir=tmp_path,
        transform=get_train_transform(img_size=224),
    )

    item = dataset[0]

    assert item["image"].shape == (3, 224, 224)


def test_libero_dataset_resizes_raw_144_images_to_model_size_224(tmp_path):
    _create_mock_libero_hdf5(
        tmp_path,
        image_key="agentview_rgb",
        image_size=144,
    )

    dataset = LIBERODataset(
        data_dir=tmp_path,
        transform=get_train_transform(img_size=224),
    )

    item = dataset[0]

    assert item["image"].shape[-2:] == (224, 224)