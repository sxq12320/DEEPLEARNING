"""Repair the isolated modern baseline env from yanked RF-DETR 1.4.0 to 1.4.0.post0.

Run this file with any Python interpreter. It modifies only the explicit
``--python`` target (default: ``~/.conda/envs/citrus_baseline/bin/python``),
never the current sxq or MMDetection environments, and never starts training.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODERN_PYTHON = Path.home() / ".conda/envs/citrus_baseline/bin/python"


def repair_commands(target):
    """Build an auditable minimal repair followed by the real worker API probe."""
    target = str(Path(target).expanduser().absolute())
    identity = (
        "import sys; import importlib.metadata as m; "
        "print('Repair target:', sys.executable, 'prefix:', sys.prefix, flush=True); "
        "print('Installed RF-DETR:', m.version('rfdetr'), flush=True); "
        "assert m.version('torch').split('+')[0] == '2.5.1', 'Expected isolated Torch 2.5.1 env'; "
        "assert m.version('ultralytics') == '8.4.60', 'Expected official Ultralytics 8.4.60 env'; "
        "assert m.version('rfdetr') in {'1.4.0','1.4.0.post0'}, "
        "'Refusing to mutate an unexpected RF-DETR environment'"
    )
    return [
        [target, "-I", "-c", identity],
        [
            target,
            "-I",
            "-m",
            "pip",
            "--isolated",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            "--no-cache-dir",
            "--index-url",
            "https://pypi.org/simple",
            "rfdetr==1.4.0.post0",
        ],
        [target, "-I", "-m", "pip", "check"],
        [target, "-I", "-u", str(ROOT / "baseline_amp/worker.py"), "--check", "yolo", "rfdetr", "--cpu-check"],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=MODERN_PYTHON)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    target = args.python.expanduser().absolute()
    if not args.dry_run and platform.system() != "Linux":
        raise RuntimeError("Run this repair on the Linux server; no local packages were changed.")
    if not args.dry_run and not target.is_file():
        raise FileNotFoundError(f"Modern baseline interpreter not found: {target}")
    print("Target interpreter:", target, flush=True)
    phases = ("Verify target environment", "Install fixed RF-DETR only", "Check dependencies", "Verify worker API")
    for phase, command in zip(phases, repair_commands(target)):
        print(f"\n[{phase}]", flush=True)
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                raise SystemExit(
                    f"Repair failed at: {phase} (exit {exc.returncode}). "
                    "Do not start the batch yet; save the full output above for diagnosis."
                ) from exc
    print("Dry run only." if args.dry_run else "Modern baseline repair and CPU API preflight verified.")
    print("No training was started; no datasets, weights or results were changed. Only the specified env is targeted.")


if __name__ == "__main__":
    main()
