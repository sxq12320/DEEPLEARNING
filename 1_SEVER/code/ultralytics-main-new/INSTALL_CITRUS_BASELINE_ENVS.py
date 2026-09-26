"""Linux x86_64/NVIDIA: create two isolated environments with prebuilt wheels. Does not touch your training env."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def commands(environment, conda):
    modern = environment == "modern"
    prefix = Path.home() / ".conda/envs" / ("citrus_baseline" if modern else "citrus_mmdet")
    python = prefix / "bin/python"
    torch_version, vision_version = ("2.5.1", "0.20.1") if modern else ("2.1.0", "0.16.0")
    steps = [
        [conda, "create", "-y", "-p", str(prefix), "python=3.10", "pip"],
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        [
            str(python),
            "-m",
            "pip",
            "install",
            f"torch=={torch_version}",
            f"torchvision=={vision_version}",
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
        ],
        [str(python), "-m", "pip", "install", "-r", str(ROOT / "baseline_amp" / f"requirements-{environment}.txt")],
    ]
    if not modern:
        steps.append(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "mmcv==2.1.0",
                "--only-binary=mmcv",
                "-f",
                "https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html",
            ]
        )
    steps.append([str(python), "-m", "pip", "check"])
    if not modern:
        steps.append([str(python), "-I", str(ROOT / "baseline_amp/environment_check.py")])
    else:
        steps.append(
            [str(python), "-I", "-u", str(ROOT / "baseline_amp/worker.py"),
             "--check", "yolo", "rfdetr", "--cpu-check"]
        )
    return prefix, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("all", "modern", "mmdet"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--repair", action="store_true", help="Explicitly finish/reinstall this script's named baseline env"
    )
    args = parser.parse_args()
    if not args.dry_run and (platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64")):
        raise RuntimeError("This wheel-only installer targets the Linux NVIDIA server, not Windows/ARM.")
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda and not args.dry_run:
        raise RuntimeError("conda not found. Open a conda-initialized terminal first; no packages were changed.")
    for environment in ["modern", "mmdet"] if args.env == "all" else [args.env]:
        prefix, steps = commands(environment, conda or "conda")
        if prefix.exists():
            if not args.repair and not args.dry_run:
                raise FileExistsError(
                    f"{prefix} already exists. Use --repair only for this baseline env; old sxq is untouched."
                )
            steps = steps[1:]
        for command in steps:
            print(subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
        print(f"PYTHONS['{environment}'] = {str(prefix / 'bin/python')!r}")
    print("Install only; no models were trained. Next edit RUN_CITRUS_BASELINES_AMP.py and run --preflight-only.")


if __name__ == "__main__":
    main()
