"""Compute normalization statistics for a config with low CPU memory usage.

This variant avoids decoding LeRobot image columns when computing state/action
normalization statistics. It falls back to the standard dataloader for datasets
that cannot use the direct local LeRobot reader.
"""

import json
import math
import pathlib

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_local_lerobot_root(repo_id: str | None) -> pathlib.Path | None:
    """Return the local LeRobot root if the dataset is already present on disk."""
    if repo_id is None:
        return None

    candidates = [pathlib.Path(repo_id)]
    try:
        from lerobot.common.constants import HF_LEROBOT_HOME

        candidates.append(HF_LEROBOT_HOME / repo_id)
    except Exception:
        pass

    for root in candidates:
        has_v2_metadata = (root / "meta" / "episodes.jsonl").is_file()
        has_v3_data = any((root / "data").glob("chunk-*/file-*.parquet"))
        if (root / "meta" / "info.json").is_file() and (has_v2_metadata or has_v3_data):
            return root
    return None


def _fake_image_for_stats(feature: dict) -> np.ndarray:
    """Create a tiny channel-first image placeholder for transforms that require image keys."""
    shape = tuple(feature.get("shape", ()))
    channels = 3
    if len(shape) == 3:
        if shape[0] in (1, 3, 4):
            channels = shape[0]
        elif shape[-1] in (1, 3, 4):
            channels = shape[-1]
    return np.zeros((channels, 1, 1), dtype=np.float32)


def _cast_column(values: list, dtype: str | None) -> np.ndarray:
    array = np.asarray(values)
    if dtype is None:
        return array
    if dtype.startswith("float"):
        return array.astype(np.float32 if dtype == "float32" else np.float64, copy=False)
    if dtype.startswith("int"):
        return array.astype(np.int64 if dtype == "int64" else np.int32, copy=False)
    return array


def _read_episode_columns(path: pathlib.Path, columns: list[str], features: dict) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=columns)
    return {
        column: _cast_column(table[column].to_pylist(), features.get(column, {}).get("dtype"))
        for column in table.column_names
    }


def _episode_path(root: pathlib.Path, info: dict, episode_index: int) -> pathlib.Path:
    episode_chunk = episode_index // info["chunks_size"]
    return root / info["data_path"].format(episode_chunk=episode_chunk, episode_index=episode_index)


def _chunk_actions(actions: np.ndarray, start: int, end: int, horizon: int) -> np.ndarray:
    indices = np.arange(start, end)[:, None] + np.arange(horizon)[None, :]
    indices = np.minimum(indices, len(actions) - 1)
    return actions[indices]


def _load_v3_tasks(root: pathlib.Path) -> dict[int, str]:
    import pyarrow.parquet as pq

    tasks: dict[int, str] = {}
    # LeRobot v3 writes either meta/tasks.parquet or meta/tasks/chunk-*/file-*.parquet.
    paths = sorted((root / "meta" / "tasks").glob("chunk-*/file-*.parquet"))
    if (root / "meta" / "tasks.parquet").is_file():
        paths.append(root / "meta" / "tasks.parquet")
    for path in paths:
        table = pq.read_table(path, columns=["task_index", "task"])
        for item in table.to_pylist():
            tasks[int(item["task_index"])] = item["task"]
    return tasks


def create_lerobot_v3_direct_dataloader(
    root: pathlib.Path,
    data_config: _config.DataConfig,
    action_horizon: int,
    direct_chunk_size: int,
    max_frames: int | None,
) -> tuple[object, int]:
    """Read only numeric Parquet columns from a LeRobot v3 dataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text())
    features = info["features"]
    vector_columns = [key for key, feature in features.items() if feature.get("dtype") not in ("image", "video")]
    missing_action_columns = [key for key in data_config.action_sequence_keys if key not in vector_columns]
    if missing_action_columns:
        raise ValueError(f"Action columns missing from LeRobot dataset: {missing_action_columns}")
    if "episode_index" not in vector_columns:
        raise ValueError("LeRobot v3 dataset is missing episode_index")

    paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No LeRobot v3 data parquet files under {root / 'data'}")
    table = pa.concat_tables([pq.read_table(path, columns=vector_columns) for path in paths])
    episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    if "index" in table.column_names:
        order = np.argsort(np.asarray(table["index"].to_pylist(), dtype=np.int64), kind="stable")
    else:
        frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        order = np.lexsort((frame_indices, episode_indices))
    episode_indices = episode_indices[order]
    episode_data = {
        column: _cast_column(table[column].to_pylist(), features.get(column, {}).get("dtype"))[order]
        for column in table.column_names
    }

    boundaries = np.flatnonzero(np.r_[True, episode_indices[1:] != episode_indices[:-1], True])
    episode_ranges = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    remaining = max_frames
    num_batches = 0
    limited_ranges: list[tuple[int, int]] = []
    for start, end in episode_ranges:
        if remaining is not None:
            end = min(end, start + remaining)
            remaining -= end - start
        if end > start:
            limited_ranges.append((start, end))
            num_batches += math.ceil((end - start) / direct_chunk_size)
        if remaining is not None and remaining <= 0:
            break

    fake_images = {
        key: _fake_image_for_stats(feature)
        for key, feature in features.items()
        if feature.get("dtype") in ("image", "video")
    }
    tasks = _load_v3_tasks(root)
    transform = transforms.compose(
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, RemoveStrings()]
    )

    def iterator():
        for episode_start, episode_end in limited_ranges:
            for start in range(episode_start, episode_end, direct_chunk_size):
                end = min(start + direct_chunk_size, episode_end)
                action_chunks = {
                    key: _chunk_actions(episode_data[key][episode_start:episode_end], start - episode_start, end - episode_start, action_horizon)
                    for key in data_config.action_sequence_keys
                }
                batch_values = {"state": [], "actions": []}
                for offset, row in enumerate(range(start, end)):
                    raw = {key: value[row] for key, value in episode_data.items()}
                    raw.update(fake_images)
                    raw.update({key: value[offset] for key, value in action_chunks.items()})
                    if "task_index" in raw and int(raw["task_index"]) in tasks:
                        raw["task"] = tasks[int(raw["task_index"])]
                        if data_config.prompt_from_task:
                            raw["prompt"] = raw["task"]
                    transformed = transform(raw)
                    for key in batch_values:
                        batch_values[key].append(np.asarray(transformed[key]))
                yield {key: np.stack(values, axis=0) for key, values in batch_values.items()}

    return iterator(), num_batches


def create_lerobot_direct_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    direct_chunk_size: int,
    max_frames: int | None = None,
) -> tuple[object, int] | None:
    """Create a stats-only LeRobot iterator that never reads image bytes.

    LeRobot samples include images during normal training, but normalization stats only
    need the transformed ``state`` and ``actions`` arrays. This iterator reads only
    non-visual parquet columns and supplies tiny image placeholders for transforms whose
    state/action logic lives in image-aware policy input transforms.
    """
    root = _resolve_local_lerobot_root(data_config.repo_id)
    if root is None:
        return None
    if direct_chunk_size <= 0:
        raise ValueError(f"direct_chunk_size must be positive, got {direct_chunk_size}.")

    if not (root / "meta" / "episodes.jsonl").is_file():
        return create_lerobot_v3_direct_dataloader(
            root, data_config, action_horizon, direct_chunk_size, max_frames
        )

    info = json.loads((root / "meta" / "info.json").read_text())
    features = info["features"]
    episodes = sorted(_load_jsonl(root / "meta" / "episodes.jsonl"), key=lambda x: x["episode_index"])
    tasks_path = root / "meta" / "tasks.jsonl"
    tasks = {item["task_index"]: item["task"] for item in _load_jsonl(tasks_path)} if tasks_path.is_file() else {}

    vector_columns = [key for key, feature in features.items() if feature.get("dtype") not in ("image", "video")]
    missing_action_columns = [key for key in data_config.action_sequence_keys if key not in vector_columns]
    if missing_action_columns:
        raise ValueError(f"Action columns missing from LeRobot dataset: {missing_action_columns}")

    fake_images = {
        key: _fake_image_for_stats(feature)
        for key, feature in features.items()
        if feature.get("dtype") in ("image", "video")
    }
    transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ]
    )

    remaining_frames = max_frames
    num_batches = 0
    for episode in episodes:
        episode_len = episode["length"]
        if remaining_frames is not None:
            episode_len = min(episode_len, remaining_frames)
            remaining_frames -= episode_len
        if episode_len > 0:
            num_batches += math.ceil(episode_len / direct_chunk_size)
        if remaining_frames is not None and remaining_frames <= 0:
            break

    def iterator():
        processed = 0
        for episode in episodes:
            if max_frames is not None and processed >= max_frames:
                return

            episode_index = episode["episode_index"]
            episode_data = _read_episode_columns(_episode_path(root, info, episode_index), vector_columns, features)
            episode_len = len(next(iter(episode_data.values())))

            for start in range(0, episode_len, direct_chunk_size):
                if max_frames is not None and processed >= max_frames:
                    return
                end = min(start + direct_chunk_size, episode_len)
                if max_frames is not None:
                    end = min(end, start + max_frames - processed)

                action_chunks = {
                    key: _chunk_actions(episode_data[key], start, end, action_horizon)
                    for key in data_config.action_sequence_keys
                }
                batch_values = {"state": [], "actions": []}

                for offset, row in enumerate(range(start, end)):
                    raw = {key: value[row] for key, value in episode_data.items()}
                    raw.update(fake_images)
                    raw.update({key: value[offset] for key, value in action_chunks.items()})

                    if "task_index" in raw:
                        task_index = int(raw["task_index"])
                        if task_index in tasks:
                            raw["task"] = tasks[task_index]
                            if data_config.prompt_from_task:
                                raw["prompt"] = tasks[task_index]

                    transformed = transform(raw)
                    for key in batch_values:
                        batch_values[key].append(np.asarray(transformed[key]))

                processed += end - start
                yield {key: np.stack(values, axis=0) for key, values in batch_values.items()}

    return iterator(), num_batches


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    direct_lerobot: bool = True,
    direct_chunk_size: int = 1024,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    data_loader = None
    num_batches = 0

    if direct_lerobot and data_config.rlds_data_dir is None:
        direct_loader = create_lerobot_direct_dataloader(
            data_config,
            config.model.action_horizon,
            direct_chunk_size,
            max_frames,
        )
        if direct_loader is not None:
            data_loader, num_batches = direct_loader
            print("Using direct LeRobot stats reader (image columns are skipped).")
        else:
            print(
                f"No local LeRobot dataset at {data_config.repo_id!r} (missing meta/info.json); "
                "falling back to the LeRobot loader, which will treat it as a Hugging Face repo id."
            )

    if data_loader is None and data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, batch_size or config.batch_size, max_frames
        )
    elif data_loader is None:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            batch_size or config.batch_size,
            config.model,
            config.num_workers if num_workers is None else num_workers,
            max_frames,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
