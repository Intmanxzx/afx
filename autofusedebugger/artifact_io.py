from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "autofusedebugger-manifest/1"


def tensor_to_numpy(tensor: Any) -> tuple[np.ndarray, str]:
    value = tensor.detach().cpu()
    dtype = str(value.dtype).removeprefix("torch.")
    if dtype == "bfloat16":
        return value.float().numpy(), dtype
    return value.numpy(), dtype


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"json root must be an object: {path}")
    return data


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_array(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".npz":
        data = np.load(p)
        keys = list(data.keys())
        if len(keys) != 1:
            raise ValueError(f"npz must contain exactly one array: {p}")
        return data[keys[0]]
    raise ValueError(f"unsupported tensor file suffix: {p}")


def resolve_artifact(root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(root) / path
