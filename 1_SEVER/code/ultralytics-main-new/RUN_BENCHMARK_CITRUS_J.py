"""Edit physical DEVICE; click VS Code's triangle to compare J computation costs."""

import os
from datetime import datetime

DEVICE = "1"
MODELS = ["J00_yolo11n_control", "J11_synthesis", "J17_budget"]
IMAGE_SIZE = 640
TRAIN_BATCH = 16
WARMUP = 5
REPEATS = 20
CPU_THREADS = 4
OUTPUT = "/data/sxq/results/J/J_SPEED_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    if DEVICE != "cpu":
        if not DEVICE.isdigit():
            raise ValueError("Choose one physical GPU or 'cpu'")
        os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
    from citrus_j_benchmark import run

    run(OUTPUT, "cpu" if DEVICE == "cpu" else "0", MODELS, IMAGE_SIZE, TRAIN_BATCH, WARMUP, REPEATS, CPU_THREADS)


if __name__ == "__main__":
    main()
