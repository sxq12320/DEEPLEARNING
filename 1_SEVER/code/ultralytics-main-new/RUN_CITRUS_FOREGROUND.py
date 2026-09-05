"""Click the VS Code Run triangle to train one citrus series visibly and sequentially.

Only edit the clearly marked configuration block below. This file does not use
nohup, ``&``, a shell wrapper, multiprocessing, or concurrent model training.
The selected dated batch runner executes every experiment in its suite one at a
time and prints Ultralytics progress directly in the VS Code terminal.
"""

from __future__ import annotations

from citrus_foreground import run_foreground

# =============================================================================
# USER CONFIGURATION: edit this block on the server, then click the Run triangle.
# =============================================================================
SERIES = "SAGE_V4R"  # Reconstructed V4; SAGE_V4 still selects the unchanged SAGE30--35 runner.
DATA = "/data/sxq/datasets/orange_yolo/data.yaml"
SUITE = "screen"  # Valid suites depend on SERIES; see FOREGROUND_TRAINING_README.md.
EPOCHS = 50  # 先做同预算筛选；长训改 300，并同时改 PROJECT 为新目录。
DEVICE = "0"  # One GPU only. Use "cpu" only for a build/debug run.
PROJECT = "/data/sxq/results/SAGE/CITRUS_SAGE_V4R_SCREEN_50EP"

BATCH = 16
IMGSZ = 640
WORKERS = 4
SEEDS = "42"  # Formal final repeats: "42,43,44" (except legacy TOPO, one seed per launch).
ONLY = ""  # Optional exact experiment name(s), comma-separated.
PRETRAINED = ""  # Empty means <code root>/yolo11n-seg.pt.
CACHE = "false"  # false/disk/ram; fixed paper protocol uses false.
AMP = None  # None keeps the dated runner's fixed protocol. Do not change for formal comparisons.

DRY_RUN = True  # 第一次只检查构建；通过后改为 False 正式顺序训练。
SKIP_COMPLETED = True
FAIL_FAST = True

# Safety: prevent a second launcher on the same GPU and refuse an already occupied GPU.
DEVICE_LOCK = True
REFUSE_BUSY_GPU = True
SINGLE_GPU_ONLY = True
# =============================================================================


def main() -> None:
    """Run the configured series in the foreground."""
    run_foreground(
        series=SERIES,
        data=DATA,
        suite=SUITE,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        pretrained=PRETRAINED,
        seeds=SEEDS,
        only=ONLY,
        cache=CACHE,
        amp=AMP,
        dry_run=DRY_RUN,
        skip_completed=SKIP_COMPLETED,
        fail_fast=FAIL_FAST,
        refuse_busy_gpu=REFUSE_BUSY_GPU,
        device_lock=DEVICE_LOCK,
        single_gpu_only=SINGLE_GPU_ONLY,
    )


if __name__ == "__main__":
    main()
