# -*- coding: utf-8 -*-
"""Root entrypoint for citrus experiment summarizer and visualization suite.

Automatically detects the best Python environment (e.g. conda yolo env)
and runs the full-featured summary suite.
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "ultralytics-main-new" / "citrus_experiment_summary.py"

PREFERRED_PYTHONS = [
    Path(r"E:\AppInstallion\0_4_annaconda\envs\yolo\python.exe"),
    Path(r"E:\AppInstallion\0_4_annaconda\python.exe"),
]

def find_best_python():
    # If currently running under a python with matplotlib, use it
    try:
        import matplotlib
        return sys.executable
    except Exception:
        pass

    for p in PREFERRED_PYTHONS:
        if p.is_file():
            return str(p)
    return sys.executable

if __name__ == "__main__":
    if not SCRIPT_PATH.exists():
        print(f"Error: Could not find {SCRIPT_PATH}")
        sys.exit(1)

    py_exe = find_best_python()
    cmd = [py_exe, str(SCRIPT_PATH)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
