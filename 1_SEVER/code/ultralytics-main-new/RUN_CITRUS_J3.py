"""J V3 privileged high-resolution teacher under incomplete labels. Edit DATA/DEVICE, then click "Run Python File".

Foreground and sequential. Order: J50 anchor -> frozen J50 teacher mines TRAIN images from original pixels ->
student arms. Every arm deploys the same YOLO11n-seg (same params/FLOPs). Validation is the I-series EV6Validator.
Val/test MUST be the hand-cleaned version, otherwise recovered fruit are scored as false positives.
"""

import os

DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"  # edit: dataset with CLEANED val/test
DEVICE = "0"  # physical GPU index, or "cpu"
SUITE = "main"  # main = J50..J55 (6 arms); ladder = J50..J53; final = J50+J53; oracle = J59; all = 8
EPOCHS = 50  # screen first, then 300 with a NEW PROJECT for selected arms
SEEDS = [42]  # final: [42, 43, 44]; the teacher stays J50 seed 42 for every student seed
INIT = "scratch"  # or "yolo11n-seg.pt" (pretrained); must be identical for every arm in one table
TEACHER = "auto"  # auto = this project's J50_noisy_anchor_seed42/weights/best.pt; or an explicit TRAIN-only .pt
ORACLE_DATA = ""  # J59 only: data.yaml with hand-cleaned TRAIN labels and the SAME val/test as DATA
PROJECT = "/data/sxq/results/J/v3/CITRUS_J3_{}_{}EP_{}_20261007".format(SUITE.upper(), EPOCHS, INIT.split("/")[-1].split(".")[0].upper())
ONLY = ""  # e.g. "J50_noisy_anchor,J53_privileged_soft"; overrides SUITE
DRY_RUN = False
CPU_THREADS = 4
BATCH = None  # None = locked recipe (16)


def main():
    if DEVICE != "cpu":
        if not DEVICE.isdigit():
            raise ValueError("DEVICE must name one physical GPU or 'cpu'")
        os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
    from citrus_j3_batch import run

    run(DATA, "cpu" if DEVICE == "cpu" else "0", PROJECT, SUITE, EPOCHS, SEEDS, ONLY, DRY_RUN, INIT, TEACHER,
        ORACLE_DATA, CPU_THREADS, batch=BATCH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("J V3 queue stopped by Ctrl+C. No next model will start.", flush=True)
        raise SystemExit(130)
