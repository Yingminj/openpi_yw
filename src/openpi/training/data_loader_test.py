import dataclasses
import itertools
import pathlib

import jax
import numpy as np
import polars as pl

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import keyframe_sampler


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_keyframe_repeat_sampler(tmp_path: pathlib.Path):
    label_dir = tmp_path / "label"
    label_dir.mkdir()
    labels = np.zeros(50, dtype=np.int8)
    labels[[20, 35]] = 1
    pl.DataFrame({"label": labels}).write_parquet(label_dir / "episode_000000.parquet")

    config = keyframe_sampler.KeyframeSamplingConfig(label_dir=str(label_dir))
    sampler = keyframe_sampler.KeyframeRepeatSampler([50], config, seed=7, shuffle=False)

    expected = np.ones(50, dtype=np.int32)
    expected[0:10] = 10
    expected[10:20] = 15
    expected[20] = 20
    expected[21:25] = 10
    expected[25:35] = 15
    expected[35] = 20
    expected[36:46] = 10
    np.testing.assert_array_equal(sampler.repeat_counts, expected)

    sampled = np.bincount(list(sampler), minlength=50)
    np.testing.assert_array_equal(sampled, expected)
    assert sampler.stats.keyframe_samples == 40
    assert sampler.stats.keyframe_sample_ratio == 40 / expected.sum()


def test_keyframe_sampler_shuffles_each_epoch(tmp_path: pathlib.Path):
    label_dir = tmp_path / "label"
    label_dir.mkdir()
    pl.DataFrame({"label": [0, 1, 0]}).write_parquet(label_dir / "episode_000000.parquet")

    config = keyframe_sampler.KeyframeSamplingConfig(label_dir=str(label_dir))
    sampler = keyframe_sampler.KeyframeRepeatSampler([3], config, seed=3)
    epoch_one = list(sampler)
    epoch_two = list(sampler)

    assert epoch_one != epoch_two
    np.testing.assert_array_equal(np.bincount(epoch_one), sampler.repeat_counts)
    np.testing.assert_array_equal(np.bincount(epoch_two), sampler.repeat_counts)


def test_continuous_sampler_keeps_complete_epochs(tmp_path: pathlib.Path):
    label_dir = tmp_path / "label"
    label_dir.mkdir()
    pl.DataFrame({"label": [0, 1, 0]}).write_parquet(label_dir / "episode_000000.parquet")

    config = keyframe_sampler.KeyframeSamplingConfig(label_dir=str(label_dir))
    epoch_sampler = keyframe_sampler.KeyframeRepeatSampler([3], config, shuffle=False)
    continuous_sampler = keyframe_sampler.ContinuousEpochSampler(epoch_sampler)
    samples = list(itertools.islice(continuous_sampler, 2 * len(epoch_sampler)))

    first_epoch = np.bincount(samples[: len(epoch_sampler)], minlength=3)
    second_epoch = np.bincount(samples[len(epoch_sampler) :], minlength=3)
    np.testing.assert_array_equal(first_epoch, epoch_sampler.repeat_counts)
    np.testing.assert_array_equal(second_epoch, epoch_sampler.repeat_counts)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
