"""Small, framework-independent utilities; no training imports at module load."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def resolve_path(value, base=None):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(base or Path.cwd()) / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # FP32 here excludes TF32. This setting is identical within each AMP pair.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def snapshot():
    import importlib.metadata
    import platform
    import subprocess
    import torch

    versions = {}
    for package in ("torch", "torchvision", "ultralytics", "rfdetr", "mmdet", "mmcv", "mmengine", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    info = dict(
        python=sys.executable,
        versions=versions,
        platform=platform.platform(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    )
    try:
        info["git"] = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        info["git"] = "unavailable"
    return info
