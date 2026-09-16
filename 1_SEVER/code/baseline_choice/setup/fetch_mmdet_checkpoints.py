"""Download official MMDetection COCO checkpoints declared in configs/baselines.yaml.

Run inside the mmdet conda environment (needs openmim + network access to
download.openmmlab.com). Files land under <mmdet-root>/checkpoints/ so the
registry `checkpoint_glob` patterns resolve and run_comparison_batch.py passes
them to train_mmdet.py --checkpoint.

Examples:
    python setup/fetch_mmdet_checkpoints.py --mmdet-root /data/sxq/code/mmdetection
    python setup/fetch_mmdet_checkpoints.py --mmdet-root /data/sxq/code/mmdetection --only rtmdet_ins_tiny --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SUITE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def main() -> int:
    """Download every missing mmdet baseline checkpoint via `mim download`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmdet-root", type=Path, required=True, help="Cloned MMDetection repo root.")
    parser.add_argument("--only", default="", help="Comma-separated mmdet baseline IDs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mmdet_root = args.mmdet_root.resolve()
    if not mmdet_root.is_dir():
        print(f"MMDetection repo not found: {mmdet_root}")
        print("Clone it first: git clone -b v3.3.0 --depth 1 https://github.com/open-mmlab/mmdetection.git")
        return 1

    try:
        from baseline_common import load_registry
    except ImportError as exc:
        print(f"Cannot import baseline_common (needs PyYAML): {exc}")
        return 1
    registry = load_registry()

    try:
        import mim  # noqa: F401
    except ImportError:
        print("openmim is not installed in this environment: pip install -U openmim")
        return 1

    requested = {item.strip() for item in args.only.split(",") if item.strip()}
    failures = 0
    for name, entry in registry["baselines"].items():
        if entry.get("family") != "mmdetection" or "checkpoint_glob" not in entry:
            continue
        if requested and name not in requested:
            continue
        pattern = str(entry["checkpoint_glob"])
        existing = list(mmdet_root.glob(pattern))
        if existing:
            print(f"SKIP {name}: checkpoint already present ({existing[0]})")
            continue
        config_stem = Path(str(entry["config"])).stem
        command = [
            sys.executable, "-m", "mim", "download", "mmdet",
            "--config", config_stem,
            "--dest", str(mmdet_root / "checkpoints"),
        ]
        print(f"GET  {name}: {subprocess.list2cmdline(command)}")
        if args.dry_run:
            continue
        result = subprocess.run(command)
        if result.returncode != 0 or not list(mmdet_root.glob(pattern)):
            print(
                f"FAIL {name}: mim download did not produce '{pattern}'.\n"
                "     Download the weight manually from the MMDetection model zoo\n"
                "     (https://mmdetection.readthedocs.io/en/latest/model_zoo.html)\n"
                f"     and place it under {mmdet_root / 'checkpoints'} so the glob resolves."
            )
            failures += 1
        else:
            print(f"OK   {name}: {pattern} resolved")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
