"""Four whole-image ablations; run this file directly in VS Code."""
import csv
import gc
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

from citrus_control78_batch import (
    ROOT, assert_resumable, csv_epoch, digest, project_lock, restore_best,
    save_best_and_terminal, save_json, training_args, training_done,
)

DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
PROJECT = "/data/sxq/results/E12_SIMPLE_20261106"
DEVICE = "1"
EPOCHS = 300
BATCH = 16
WORKERS = 4
CACHE = True
SEEDS = [42]
MODELS = ["ES00_baseline", "ES01_detail", "ES02_semantic", "ES03_detail_semantic"]
EVALUATE_TEST = False  # Enable only after the method is selected using val.
DRY_RUN = False
SMOKE = False  # One epoch in a separate project. Use a small data.yaml locally.


def main():
    budget = 1 if SMOKE else EPOCHS
    project = Path(PROJECT + ("_SMOKE" if SMOKE else ""))
    data = Path(DATA).resolve()
    if not SEEDS or len(set(SEEDS)) != len(SEEDS) or len(set(MODELS)) != len(MODELS):
        raise ValueError("Use distinct seeds and models")
    choices = {"ES00_baseline", "ES01_detail", "ES02_semantic", "ES03_detail_semantic"}
    if not MODELS or not set(MODELS) <= choices:
        raise ValueError("Unknown/empty model selection")
    queue = [(name, seed) for seed in SEEDS for name in MODELS]
    print("E12 SIMPLE | foreground sequential | TRAIN=WHOLE | VAL=WHOLE | TEST=%s" %
          ("WHOLE" if EVALUATE_TEST else "NOT RUN"), flush=True)
    print("No tiles, no quality reweighting, standard segmentation loss. Queue:", queue, flush=True)
    if DRY_RUN:
        return queue
    if not data.is_file():
        raise FileNotFoundError(data)
    if DEVICE != "cpu":
        import sys
        if not str(DEVICE).isdigit():
            raise ValueError("Use a single physical GPU index or cpu")
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            raise RuntimeError("Start this file in a fresh Python process")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import numpy as np
    import torch
    from ultralytics import YOLO
    from citrus_control78_training import Control78Trainer
    from citrus_e_v6_training import EV6Validator

    project.mkdir(parents=True, exist_ok=True)
    with project_lock(project):
        rows = []
        for name, seed in queue:
            directory = project / (name + "_seed%d" % seed)
            yaml_path = ROOT / "0_orange_yaml/E12_SIMPLE" / (name + ".yaml")
            recipe = training_args(SimpleNamespace(epochs=budget, batch=BATCH, workers=WORKERS), seed)
            recipe["cache"] = CACHE
            protocol = dict(recipe=recipe, model_sha256=digest(yaml_path), data=str(data),
                            data_yaml=data.read_text(encoding="utf-8"), train="whole", val="whole_EV6",
                            module_sha256=digest(ROOT / "ultralytics/nn/modules/citrus_e12_simple.py"),
                            semantic_sha256=digest(ROOT / "ultralytics/nn/modules/citrus_e_v12.py"))
            frozen = directory / "protocol.json"
            if directory.exists() and (not frozen.exists() or
                    json.loads(frozen.read_text(encoding="utf-8")) != protocol):
                raise FileExistsError("Changed/unrecorded protocol; use a new PROJECT: " + str(directory))
            save_json(frozen, protocol)
            model = None
            try:
                if not training_done(directory, budget):
                    last = directory / "weights/last.pt"
                    if csv_epoch(directory) and not last.is_file():
                        raise RuntimeError("Missing last.pt; refusing to overwrite a partial run")
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    model = YOLO(str(last if last.is_file() else yaml_path), task="segment")
                    if last.is_file():
                        assert_resumable(model, directory)
                    model.add_callback("on_pretrain_routine_end", restore_best)
                    model.add_callback("on_model_save", save_best_and_terminal)
                    model.train(trainer=Control78Trainer, **dict(
                        recipe, data=str(data), project=str(project), name=directory.name,
                        device=DEVICE, exist_ok=True, resume=str(last) if last.is_file() else False))
                    if not training_done(directory, budget):
                        raise RuntimeError("Training stopped without a completion marker")
                model = YOLO(str(directory / "weights/best_mask.pt"), task="segment")
                for split in (["val", "test"] if EVALUATE_TEST else ["val"]):
                    metrics = model.val(validator=EV6Validator, data=str(data), split=split, device=DEVICE,
                                        imgsz=640, batch=BATCH, workers=WORKERS, conf=.001, iou=.7,
                                        max_det=300, mask_ratio=2, augment=False, half=False,
                                        project=str(directory), name="eval_whole_" + split, exist_ok=True,
                                        plots=True, save_json=False)
                    row = dict(model=name, seed=seed, split=split, inference="whole_single_view",
                               epochs_completed=csv_epoch(directory), **metrics.results_dict)
                    rows.append(row)
                    save_json(directory / ("metrics_whole_" + split + ".json"), row)
                    target = project / "WHOLE_RESULTS.csv"
                    temporary = target.with_suffix(".csv.tmp")
                    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    temporary.replace(target)
            finally:
                model = None
                gc.collect()
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
    return queue


if __name__ == "__main__":
    main()
