"""Check the legacy MMDetection runtime API, not just package metadata."""

import importlib
import subprocess
import sys


def check_mmdet_environment():
    """Exercise Runner's environment-logging path before creating a training run.

    A missing CUDA_HOME is valid with prebuilt wheels. Import/collector failures
    are not hidden: they would otherwise terminate Runner.from_cfg before epoch 1.
    No packages are installed by this check.
    """
    repair = subprocess.list2cmdline(
        [sys.executable, "-m", "pip", "install", "setuptools==69.5.1"]
    )
    try:
        resources = importlib.import_module("pkg_resources")
        packaging = getattr(resources, "packaging")
        packaging.version.parse("2.1.0")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "MMDetection legacy runtime needs pkg_resources.packaging for Torch 2.1. "
            "This is an environment error, not a model or VRAM error. "
            f"In the dedicated citrus_mmdet environment run: {repair}. "
            "Do not install a package named pkg_resources or upgrade setuptools to latest."
        ) from exc
    importlib.import_module("torch.utils.cpp_extension")
    collector = importlib.import_module("mmengine.utils.dl_utils")
    return collector.collect_env()


if __name__ == "__main__":
    print(check_mmdet_environment())
