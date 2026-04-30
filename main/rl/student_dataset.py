from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


TRANSITION_KEYS = ["depth", "proprio", "command", "action", "done", "episode_start"]


def save_transition_dataset(path: Path, arrays: dict[str, np.ndarray], metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "arrays": {key: np.asarray(value) for key, value in arrays.items()},
    }
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path, pickle_protocol=4)
    tmp_path.replace(path)


def load_transition_dataset(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    arrays = payload["arrays"]
    metadata = payload.get("metadata", {})
    return arrays, metadata


def append_transition_datasets(
    old_arrays: dict[str, np.ndarray] | None,
    new_arrays: dict[str, np.ndarray],
    max_transitions: int | None = None,
) -> dict[str, np.ndarray]:
    if old_arrays is None:
        combined = {key: np.asarray(value) for key, value in new_arrays.items()}
    else:
        combined = {key: np.concatenate([old_arrays[key], new_arrays[key]], axis=0) for key in TRANSITION_KEYS}
    if max_transitions is not None and int(max_transitions) > 0:
        max_transitions = int(max_transitions)
        length = int(combined["action"].shape[0])
        if length > max_transitions:
            start = length - max_transitions
            combined = {key: value[start:] for key, value in combined.items()}
            combined["episode_start"][0] = True
    return combined


class RecurrentTransitionDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], bptt_len: int):
        self.arrays = arrays
        self.bptt_len = int(bptt_len)
        self.length = int(arrays["action"].shape[0])
        if self.length < self.bptt_len:
            raise ValueError(f"Dataset has {self.length} transitions, less than bptt_len={self.bptt_len}")
        self.starts = np.arange(0, self.length - self.bptt_len + 1, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index])
        end = start + self.bptt_len
        return {
            "depth": torch.as_tensor(self.arrays["depth"][start:end], dtype=torch.float32),
            "proprio": torch.as_tensor(self.arrays["proprio"][start:end], dtype=torch.float32),
            "command": torch.as_tensor(self.arrays["command"][start:end], dtype=torch.float32),
            "action": torch.as_tensor(self.arrays["action"][start:end], dtype=torch.float32),
            "episode_start": torch.as_tensor(self.arrays["episode_start"][start:end], dtype=torch.float32),
        }


def split_train_val(arrays: dict[str, np.ndarray], val_fraction: float = 0.1) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    length = int(arrays["action"].shape[0])
    val_count = max(1, int(round(length * float(val_fraction))))
    train_count = max(1, length - val_count)
    train_arrays = {key: value[:train_count] for key, value in arrays.items()}
    val_arrays = {key: value[train_count:] for key, value in arrays.items()}
    if int(val_arrays["action"].shape[0]) == 0:
        val_arrays = train_arrays
    train_arrays["episode_start"][0] = True
    val_arrays["episode_start"][0] = True
    return train_arrays, val_arrays
