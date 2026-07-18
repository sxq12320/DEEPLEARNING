"""Shared helpers for the citrus baseline scripts."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml


SUITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = SUITE_ROOT / "configs" / "baselines.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def load_registry(path: Path = DEFAULT_REGISTRY) -> Dict[str, Any]:
    """Load the baseline registry."""
    registry = load_yaml(path)
    if "baselines" not in registry:
        raise ValueError(f"Missing 'baselines' in {path}")
    return registry


def get_baseline(name: str, family: Optional[str] = None, registry_path: Path = DEFAULT_REGISTRY) -> Dict[str, Any]:
    """Return one validated baseline entry."""
    registry = load_registry(registry_path)
    try:
        baseline = dict(registry["baselines"][name])
    except KeyError as exc:
        choices = ", ".join(sorted(registry["baselines"]))
        raise ValueError(f"Unknown baseline '{name}'. Choices: {choices}") from exc
    if family and baseline.get("family") != family:
        raise ValueError(f"Baseline '{name}' belongs to {baseline.get('family')}, not {family}")
    baseline["id"] = name
    return baseline


def resolve_path(path: str | Path, base: Path = SUITE_ROOT) -> Path:
    """Resolve a possibly relative path against the suite root."""
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def require_new_directory(path: Path) -> None:
    """Refuse to reuse a non-empty experiment directory."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Any) -> None:
    """Write JSON using a stable, human-readable format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def command_text(argv: Optional[Iterable[str]] = None) -> str:
    """Return a shell-readable representation of the current command."""
    values = list(argv if argv is not None else sys.argv)
    return subprocess.list2cmdline(values)


def environment_snapshot() -> Dict[str, Any]:
    """Collect lightweight reproducibility metadata without importing frameworks."""
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command_text(),
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    snapshot["git"] = git_snapshot(SUITE_ROOT)
    return snapshot


def git_snapshot(path: Path) -> Dict[str, Any]:
    """Return the current Git revision and dirty state when available."""
    try:
        root = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        revision = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", root, "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"root": root, "revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"root": None, "revision": None, "dirty": None}


def framework_snapshot() -> Dict[str, Any]:
    """Collect PyTorch and accelerator details when PyTorch is installed."""
    try:
        import torch
    except ImportError:
        return {"torch": None}

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_mb": round(properties.total_memory / (1024**2), 1),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "devices": devices,
    }


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_runtime_yolo_yaml(dataset_root: Path, output_path: Path, class_names: Iterable[str]) -> Path:
    """Create an absolute-path Ultralytics dataset YAML for the current machine."""
    yolo_root = dataset_root.resolve() / "yolo"
    data = {
        "path": yolo_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    return output_path


def flatten_mapping(data: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested mappings for CSV output."""
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_mapping(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[name] = value
    return flat
