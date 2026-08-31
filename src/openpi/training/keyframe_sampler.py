from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
import logging
import pathlib
import re

import numpy as np
import polars as pl
import torch


@dataclasses.dataclass(frozen=True)
class KeyframeSamplingConfig:
    """Per-epoch repetition policy around keyframes."""

    label_dir: str
    outer_pre_frames: int = 20
    inner_pre_frames: int = 10
    post_frames: int = 10
    outer_pre_repeat: int = 10
    inner_pre_repeat: int = 15
    keyframe_repeat: int = 20
    post_repeat: int = 10


@dataclasses.dataclass(frozen=True)
class KeyframeSamplingStats:
    raw_frames: int
    raw_keyframes: int
    epoch_samples: int
    keyframe_samples: int
    focused_frames: int
    focused_samples: int

    @property
    def raw_keyframe_ratio(self) -> float:
        return self.raw_keyframes / self.raw_frames

    @property
    def keyframe_sample_ratio(self) -> float:
        return self.keyframe_samples / self.epoch_samples

    @property
    def focused_sample_ratio(self) -> float:
        return self.focused_samples / self.epoch_samples

    def summary(self) -> str:
        return (
            f"raw_frames={self.raw_frames:,}, raw_keyframes={self.raw_keyframes:,} "
            f"({self.raw_keyframe_ratio:.4%}), epoch_samples={self.epoch_samples:,}, "
            f"keyframe_samples={self.keyframe_samples:,} ({self.keyframe_sample_ratio:.4%}), "
            f"focused_frames={self.focused_frames:,}, focused_samples={self.focused_samples:,} "
            f"({self.focused_sample_ratio:.4%})"
        )


def _episode_index(path: pathlib.Path) -> int:
    match = re.fullmatch(r"episode_(\d+)\.parquet", path.name)
    if match is None:
        raise ValueError(f"Unexpected label filename: {path.name}")
    return int(match.group(1))


def load_frame_labels(label_dir: str | pathlib.Path, episode_lengths: Sequence[int]) -> np.ndarray:
    """Load one binary label per frame, using filename and row order as episode/frame identity."""

    label_dir = pathlib.Path(label_dir)
    label_files = sorted(label_dir.glob("episode_*.parquet"), key=_episode_index)
    if len(label_files) != len(episode_lengths):
        raise ValueError(
            f"Expected {len(episode_lengths)} label files in {label_dir}, found {len(label_files)}."
        )

    labels: list[np.ndarray] = []
    for expected_episode, (path, expected_length) in enumerate(zip(label_files, episode_lengths, strict=True)):
        episode = _episode_index(path)
        if episode != expected_episode:
            raise ValueError(f"Expected episode_{expected_episode:06d}.parquet, found {path.name}.")

        frame_labels = pl.read_parquet(path, columns=["label"]).get_column("label").to_numpy()
        if len(frame_labels) != expected_length:
            raise ValueError(
                f"{path.name} contains {len(frame_labels)} labels, but episode {episode} has "
                f"{expected_length} frames."
            )
        if not np.isin(frame_labels, (0, 1)).all():
            raise ValueError(f"{path.name} contains label values other than 0 and 1.")
        labels.append(frame_labels.astype(np.uint8, copy=False))

    return np.concatenate(labels)


def build_repeat_counts(
    labels: np.ndarray,
    episode_lengths: Sequence[int],
    config: KeyframeSamplingConfig,
) -> np.ndarray:
    """Create exact per-frame epoch repetition counts.

    Overlapping windows use the maximum requested repetition count. This keeps an
    exact keyframe at 20 samples even when another keyframe window overlaps it.
    """

    labels = np.asarray(labels, dtype=np.uint8)
    if labels.ndim != 1 or len(labels) != sum(episode_lengths):
        raise ValueError("Labels must be a flat array matching the total episode length.")

    repeat_counts = np.ones(len(labels), dtype=np.int32)
    episode_start = 0
    for episode_length in episode_lengths:
        episode_labels = labels[episode_start : episode_start + episode_length]
        episode_repeats = repeat_counts[episode_start : episode_start + episode_length]

        for keyframe in np.flatnonzero(episode_labels == 1):
            outer_start = max(0, keyframe - config.outer_pre_frames)
            inner_start = max(0, keyframe - config.inner_pre_frames)
            post_end = min(episode_length, keyframe + config.post_frames + 1)

            episode_repeats[outer_start:inner_start] = np.maximum(
                episode_repeats[outer_start:inner_start], config.outer_pre_repeat
            )
            episode_repeats[inner_start:keyframe] = np.maximum(
                episode_repeats[inner_start:keyframe], config.inner_pre_repeat
            )
            episode_repeats[keyframe] = max(episode_repeats[keyframe], config.keyframe_repeat)
            episode_repeats[keyframe + 1 : post_end] = np.maximum(
                episode_repeats[keyframe + 1 : post_end], config.post_repeat
            )

        episode_start += episode_length

    return repeat_counts


class KeyframeRepeatSampler(torch.utils.data.Sampler[int]):
    """Sample each frame an exact number of times per epoch."""

    def __init__(
        self,
        episode_lengths: Sequence[int],
        config: KeyframeSamplingConfig,
        *,
        seed: int = 0,
        shuffle: bool = True,
    ):
        self._episode_lengths = tuple(int(length) for length in episode_lengths)
        self._config = config
        self._seed = seed
        self._shuffle = shuffle
        self._epoch = 0

        self.labels = load_frame_labels(config.label_dir, self._episode_lengths)
        self.repeat_counts = build_repeat_counts(self.labels, self._episode_lengths, config)
        self._indices = np.repeat(np.arange(len(self.labels), dtype=np.int64), self.repeat_counts)

        keyframe_mask = self.labels == 1
        focused_mask = self.repeat_counts > 1
        self.stats = KeyframeSamplingStats(
            raw_frames=len(self.labels),
            raw_keyframes=int(keyframe_mask.sum()),
            epoch_samples=len(self._indices),
            keyframe_samples=int(self.repeat_counts[keyframe_mask].sum()),
            focused_frames=int(focused_mask.sum()),
            focused_samples=int(self.repeat_counts[focused_mask].sum()),
        )
        logging.info("Keyframe sampler: %s", self.stats.summary())

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[int]:
        if self._shuffle:
            generator = torch.Generator()
            generator.manual_seed(self._seed + self._epoch)
            order = torch.randperm(len(self._indices), generator=generator).numpy()
            indices = self._indices[order]
        else:
            indices = self._indices
        self._epoch += 1
        return iter(indices.tolist())


class ContinuousEpochSampler(torch.utils.data.Sampler[int]):
    """Concatenate complete sampler epochs so fixed-size batches never drop tail samples."""

    def __init__(self, epoch_sampler: KeyframeRepeatSampler):
        self.epoch_sampler = epoch_sampler

    def __len__(self) -> int:
        return len(self.epoch_sampler)

    def __iter__(self) -> Iterator[int]:
        while True:
            yield from self.epoch_sampler
