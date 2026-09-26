"""Foreground, minimal repair for bug1; never alters the current/sxq/modern environment."""

import argparse
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MMDET_PYTHON = Path.home() / ".conda/envs/citrus_mmdet/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=MMDET_PYTHON)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    target = str(args.python.expanduser().absolute())
    checks = (
        "import importlib.metadata as m; "
        "assert m.version('torch').split('+')[0] == '2.1.0', 'Expected isolated Torch 2.1.0 env'; "
        "assert m.version('mmdet') == '3.3.0', 'Expected MMDetection 3.3.0 env'"
    )
    commands = [
        [target, "-I", "-c", checks],
        [target, "-m", "pip", "install", "setuptools==69.5.1"],
        [target, "-m", "pip", "check"],
        [target, "-I", str(ROOT / "baseline_amp/environment_check.py")],
    ]
    if not args.dry_run and platform.system() != "Linux":
        raise RuntimeError("Run this repair on the Linux server; no local packages were changed.")
    print("Target interpreter:", target, flush=True)
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    print("Dry run only." if args.dry_run else "Runtime repair verified. No training/results were changed.")


if __name__ == "__main__":
    main()
