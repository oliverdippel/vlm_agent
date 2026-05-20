import h5py
import numpy as np
import torch

from src.data.transformation.normalizer import ActionNormalizer


def _create_mock_action_hdf5(tmp_path):
    h5_path = tmp_path / "mock_task_demo.hdf5"

    actions_demo_0 = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )

    actions_demo_1 = np.array(
        [
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ],
        dtype=np.float32,
    )

    with h5py.File(h5_path, "w") as f:
        data = f.create_group("data")

        demo_0 = data.create_group("demo_0")
        demo_0.create_dataset("actions", data=actions_demo_0)

        demo_1 = data.create_group("demo_1")
        demo_1.create_dataset("actions", data=actions_demo_1)

    all_actions = np.concatenate([actions_demo_0, actions_demo_1], axis=0)

    return h5_path, all_actions


def test_action_normalizer_fits_mean_and_std_from_hdf5(tmp_path):
    _, all_actions = _create_mock_action_hdf5(tmp_path)

    normalizer = ActionNormalizer.fit_from_dir(str(tmp_path))

    expected_mean = all_actions.mean(axis=0)
    expected_std = all_actions.std(axis=0) + 1e-8

    np.testing.assert_allclose(normalizer.mean, expected_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(normalizer.std, expected_std, rtol=1e-6, atol=1e-6)


def test_action_normalizer_normalize_matches_manual_formula(tmp_path):
    _, all_actions = _create_mock_action_hdf5(tmp_path)

    normalizer = ActionNormalizer.fit_from_dir(str(tmp_path))

    action = torch.tensor(all_actions[0], dtype=torch.float32)
    normalized = normalizer.normalize(action)

    expected = (action - torch.tensor(normalizer.mean)) / torch.tensor(normalizer.std)

    torch.testing.assert_close(normalized, expected)


def test_action_normalizer_denormalize_inverts_normalize(tmp_path):
    _, all_actions = _create_mock_action_hdf5(tmp_path)

    normalizer = ActionNormalizer.fit_from_dir(str(tmp_path))

    raw_action = torch.tensor(all_actions[0], dtype=torch.float32)
    normalized = normalizer.normalize(raw_action)
    recovered = normalizer.denormalize(normalized)

    np.testing.assert_allclose(recovered, raw_action.numpy(), rtol=1e-6, atol=1e-6)


def test_action_normalizer_save_and_load_roundtrip(tmp_path):
    _, _ = _create_mock_action_hdf5(tmp_path)

    normalizer = ActionNormalizer.fit_from_dir(str(tmp_path))

    save_path = tmp_path / "action_stats.npz"
    normalizer.save(str(save_path))

    loaded = ActionNormalizer.load(str(save_path))

    np.testing.assert_allclose(loaded.mean, normalizer.mean)
    np.testing.assert_allclose(loaded.std, normalizer.std)