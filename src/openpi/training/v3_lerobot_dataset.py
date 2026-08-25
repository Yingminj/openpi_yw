from __future__ import annotations

import json
import pathlib
from typing import SupportsIndex

from lerobot.common.datasets.video_utils import decode_video_frames
import numpy as np
import polars as pl

DEFAULT_VIDEO_KEYS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_r",
)


def is_lerobot_v3_dataset(repo_id: str) -> bool:
    root = pathlib.Path(repo_id)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return False
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    return str(info.get("codebase_version", "")).lstrip("v").startswith("3.")


class LeRobotV3Dataset:
    """Minimal local LeRobot v3 reader compatible with the pinned LeRobot runtime."""

    def __init__(self, repo_id: str, action_horizon: int):
        self.root = pathlib.Path(repo_id)
        self.action_horizon = action_horizon
        self._frame_labels: np.ndarray | None = None

        with (self.root / "meta" / "info.json").open(encoding="utf-8") as file:
            self.info = json.load(file)
        self.fps = float(self.info["fps"])

        data_paths = sorted((self.root / "data").rglob("*.parquet"))
        if not data_paths:
            raise FileNotFoundError(f"No data parquet files found under {self.root / 'data'}.")
        data = pl.concat(
            [
                pl.read_parquet(
                    path,
                    columns=[
                        "observation.state",
                        "action",
                        "timestamp",
                        "frame_index",
                        "episode_index",
                        "index",
                        "task_index",
                    ],
                )
                for path in data_paths
            ]
        ).sort("index")

        self._states = np.asarray(data["observation.state"].to_list(), dtype=np.float32)
        self._actions = np.asarray(data["action"].to_list(), dtype=np.float32)
        self._timestamps = data["timestamp"].to_numpy().astype(np.float64, copy=False)
        self._frame_indices = data["frame_index"].to_numpy().astype(np.int64, copy=False)
        self._episode_indices = data["episode_index"].to_numpy().astype(np.int64, copy=False)
        self._task_indices = data["task_index"].to_numpy().astype(np.int64, copy=False)
        indices = data["index"].to_numpy()
        if not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError("Dataset frame indices must be contiguous and start at zero.")

        self.video_keys = tuple(
            key
            for key, feature in self.info["features"].items()
            if isinstance(feature, dict) and feature.get("dtype") == "video"
        )
        if not self.video_keys:
            self.video_keys = DEFAULT_VIDEO_KEYS

        task_path = self.root / "meta" / "tasks.parquet"
        self._tasks_by_index: dict[int, str] = {}
        if task_path.is_file():
            tasks = pl.read_parquet(task_path)
            if {"task_index", "task"}.issubset(tasks.columns):
                self._tasks_by_index = {
                    int(row["task_index"]): str(row["task"]) for row in tasks.to_dicts()
                }

        episode_paths = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        if not episode_paths:
            raise FileNotFoundError(f"No episode metadata found under {self.root / 'meta' / 'episodes'}.")
        episodes = pl.concat([pl.read_parquet(path) for path in episode_paths]).sort("episode_index")
        self._episodes = episodes.to_dicts()
        self.episode_lengths = tuple(int(row["length"]) for row in self._episodes)
        if sum(self.episode_lengths) != len(self):
            raise ValueError("Episode lengths do not match the number of dataset frames.")

        for expected_episode, row in enumerate(self._episodes):
            if int(row["episode_index"]) != expected_episode:
                raise ValueError("Episode indices must be contiguous and start at zero.")
            start = int(row["dataset_from_index"])
            end = int(row["dataset_to_index"])
            if end - start != int(row["length"]):
                raise ValueError(f"Invalid frame bounds for episode {expected_episode}.")
            if not np.all(self._episode_indices[start:end] == expected_episode):
                raise ValueError(f"Frame episode indices do not match episode {expected_episode}.")
            if not np.array_equal(self._frame_indices[start:end], np.arange(end - start)):
                raise ValueError(f"Frame indices are not contiguous in episode {expected_episode}.")

    def __len__(self) -> int:
        return len(self._states)

    def set_frame_labels(self, labels: np.ndarray) -> None:
        labels = np.asarray(labels, dtype=np.uint8)
        if labels.shape != (len(self),):
            raise ValueError(f"Expected {len(self)} frame labels, got shape {labels.shape}.")
        self._frame_labels = labels

    def _decode_image(self, episode: dict, video_key: str, local_timestamp: float) -> np.ndarray:
        prefix = f"videos/{video_key}"
        chunk_index = int(episode[f"{prefix}/chunk_index"])
        file_index = int(episode[f"{prefix}/file_index"])
        from_timestamp = float(episode[f"{prefix}/from_timestamp"])
        video_path = (
            self.root
            / "videos"
            / video_key
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        frames = decode_video_frames(
            video_path,
            [from_timestamp + local_timestamp],
            tolerance_s=0.5 / self.fps,
            backend="pyav",
        )
        return np.asarray(frames[0])

    def __getitem__(self, index: SupportsIndex) -> dict:
        index = index.__index__()
        episode_index = int(self._episode_indices[index])
        episode = self._episodes[episode_index]
        episode_end = int(episode["dataset_to_index"])
        action_indices = np.minimum(np.arange(index, index + self.action_horizon), episode_end - 1)
        local_timestamp = float(self._timestamps[index])

        sample = {
            "observation.state": self._states[index],
            "action": self._actions[action_indices],
        }
        for video_key in self.video_keys:
            sample[video_key] = self._decode_image(episode, video_key, local_timestamp)

        tasks = episode.get("tasks") or []
        if tasks:
            sample["prompt"] = tasks[0]
        else:
            sample["prompt"] = self._tasks_by_index.get(int(self._task_indices[index]), "")
        if self._frame_labels is not None:
            sample["keyframe"] = np.int64(self._frame_labels[index])
        return sample
