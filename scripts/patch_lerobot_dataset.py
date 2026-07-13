"""Patch LeRobot's dataset loader for newer Hugging Face datasets versions.

Some LeRobot releases call ``torch.stack`` directly on ``datasets.Column``
objects. Newer ``datasets`` versions return a Column there instead of a plain
list of tensors, which breaks OpenPI training before the first batch is loaded.
Run this script after creating or refreshing the virtual environment.
"""

from __future__ import annotations

from pathlib import Path

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset


REPLACEMENTS = {
    'timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()': (
        'timestamps = np.asarray(list(self.hf_dataset["timestamp"])).squeeze()'
    ),
    'episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()': (
        'episode_indices = np.asarray(list(self.hf_dataset["episode_index"])).squeeze()'
    ),
    "key: torch.stack(self.hf_dataset.select(q_idx)[key])": (
        "key: torch.as_tensor(list(self.hf_dataset.select(q_idx)[key]))"
    ),
}


def main() -> None:
    path = Path(lerobot_dataset.__file__)
    source = path.read_text()
    patched = source

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
