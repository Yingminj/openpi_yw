"""Patch LeRobot's dataset loader for newer Hugging Face datasets versions.

Some LeRobot releases call ``torch.stack`` directly on ``datasets.Column``
objects. Newer ``datasets`` versions return a Column there instead of a plain
list of tensors, which breaks OpenPI training before the first batch is loaded.
Run this script after creating or refreshing the virtual environment.
"""

from __future__ import annotations

from pathlib import Path

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset


HELPER = '''

def _stack_hf_column(values) -> torch.Tensor:
    """Convert a Hugging Face column/list selection to a stacked torch tensor.

    Recent `datasets` versions may return a `Column` object here instead of the
    list-of-tensors shape older LeRobot code expects.
    """
    values = list(values)
    if not values:
        return torch.empty(0)
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(values)
    return torch.stack([torch.as_tensor(v) for v in values])
'''


REPLACEMENTS = {
    'timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()': (
        'timestamps = np.asarray(self.hf_dataset["timestamp"]).squeeze()'
    ),
    'episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()': (
        'episode_indices = np.asarray(self.hf_dataset["episode_index"]).squeeze()'
    ),
    "key: torch.stack(self.hf_dataset.select(q_idx)[key])": (
        "key: _stack_hf_column(self.hf_dataset.select(q_idx)[key])"
    ),
    "key: torch.as_tensor(list(self.hf_dataset.select(q_idx)[key]))": (
        "key: _stack_hf_column(self.hf_dataset.select(q_idx)[key])"
    ),
    "key: torch.stack(list(self.hf_dataset.select(q_idx)[key]))": (
        "key: _stack_hf_column(self.hf_dataset.select(q_idx)[key])"
    ),
    "query_timestamps[key] = torch.stack(timestamps).tolist()": (
        "query_timestamps[key] = _stack_hf_column(timestamps).tolist()"
    ),
    "query_timestamps[key] = torch.stack(list(timestamps)).tolist()": (
        "query_timestamps[key] = _stack_hf_column(timestamps).tolist()"
    ),
}


def main() -> None:
    path = Path(lerobot_dataset.__file__)
    source = path.read_text()
    patched = source

    if "def _stack_hf_column(" not in patched:
        marker = 'CODEBASE_VERSION = "v2.1"\n'
        if marker not in patched:
            raise SystemExit(f"could not find insertion marker {marker!r} in {path}")
        patched = patched.replace(marker, marker + HELPER, 1)

    for old, new in REPLACEMENTS.items():
        patched = patched.replace(old, new)

    if patched == source:
        print(f"no patch needed: {path}")
    else:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_text(source)
        path.write_text(patched)
        print(f"patched: {path}")

    remaining_pattern = "torch.stack(self.hf_dataset"
    if remaining_pattern in path.read_text():
        raise SystemExit(f"still contains {remaining_pattern!r}; inspect {path}")


if __name__ == "__main__":
    main()
