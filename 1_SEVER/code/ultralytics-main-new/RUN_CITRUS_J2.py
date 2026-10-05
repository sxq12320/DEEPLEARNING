"""J V2 structural screening. Edit DATA/DEVICE, then click VS Code "Run Python File".

Foreground and sequential. Mechanism = J20/J29/J36/J41/J42/J43; includes decoder repairs.
Training-time validation remains I-series EV6Validator. Additional gaze/PR passes are off by default.
Legacy gaze and resolution controls remain available explicitly; 960/1280 controls disable RAM cache.
"""

import os

DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"  # edit to YOUR cleaned dataset path
DEVICE = "0"  # physical GPU index, or "cpu"
SUITE = "mechanism"  # six targeted controls; priority=old five; decoder_repairs=4; all=24 (costly)
EPOCHS = 50  # screen first, then 300 with a NEW PROJECT for selected models
SEEDS = [42]  # final shortlisted arms: [42, 43, 44]
PROJECT = "/data/sxq/results/J/v2/CITRUS_J2_DEEP_{}_{}EP_NATIVEVAL_20261003".format(SUITE.upper(), EPOCHS)
ONLY = ""  # e.g. "J41_subpixel_p3,J42_subpixel_p2,J43_compact_subpixel_p2"; overrides SUITE
DRY_RUN = False
GAZE_EVAL = False  # optional, SLOW paired inference; never replaces native validation
DIAGNOSE_PR = False  # optional second pass; native PR and confusion matrices are always saved
CPU_THREADS = 4
BATCH = None  # None = locked recipe (16). Set only if a 1280 arm runs out of memory, and report it.


def main():
    if DEVICE != "cpu":
        if not DEVICE.isdigit():
            raise ValueError("DEVICE must name one physical GPU or 'cpu'")
        os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
    from citrus_j2_batch import run

    run(DATA, "cpu" if DEVICE == "cpu" else "0", PROJECT, SUITE, EPOCHS, SEEDS, ONLY, DRY_RUN, GAZE_EVAL, DIAGNOSE_PR,
        CPU_THREADS, batch=BATCH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("J V2 queue stopped by Ctrl+C. No next model will start.", flush=True)
        raise SystemExit(130)
