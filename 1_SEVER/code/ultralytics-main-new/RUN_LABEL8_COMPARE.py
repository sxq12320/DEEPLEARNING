"""Compare full vs min-8px labels with two YOLO11n-seg fits and paired whole-image evaluation."""
import csv
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import yaml

from citrus_control78_batch import (
    ROOT, assert_resumable, csv_epoch, digest, project_lock, restore_best,
    save_best_and_terminal, save_json, training_args, training_done,
)

OLD_DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
NEW_DATA = "/data/sxq/datasets/orange_yolo_grouped_min8px640_20261107/data.yaml"
PROJECT = "/data/sxq/results/Y11_LABEL8_20261107"
DEVICE = "1"
SEEDS = [42]
EPOCHS = 300
BATCH = 16
WORKERS = 4
CACHE = True
EVALUATE_TEST = False
SMOKE = False
MODEL = ROOT / "0_orange_yaml/E12_SIMPLE/ES00_baseline.yaml"
METRICS = {
    "mask_ap50": "metrics/mAP50(M)", "mask_ap5095": "metrics/mAP50-95(M)",
    "mask_recall": "metrics/recall(M)", "rmax": "rmax", "r_at_p90": "r_at_p90",
}


def write_table(path, rows):
    if rows:
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def dataset_configs(old_data=None, new_data=None, evaluate_test=None):
    """Resolve the two uploaded folder trees locally, including the old Windows YAML."""
    evaluate_test = EVALUATE_TEST if evaluate_test is None else evaluate_test
    specs = {}
    for name, path in (("all", old_data or OLD_DATA), ("min8", new_data or NEW_DATA)):
        setting = "OLD_DATA" if name == "all" else "NEW_DATA"
        path = Path(path).expanduser()
        if path.is_dir():
            path = path / "data.yaml"
        path = path.resolve()
        print(f"{setting} YAML: {path}", flush=True)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        spec = {"names": cfg["names"]}
        for split in ["train", "val"] + (["test"] if evaluate_test else []):
            # These two packaged datasets each contain their own train/val/test folders.
            # A copied absolute split entry can otherwise point both arms to the original labels.
            local_images = path.parent / split / "images"
            configured = path.parent / (cfg.get("path") or ".") / cfg[split]
            images = (local_images if local_images.is_dir() else configured).resolve()
            print(f"  {split}: {images}  (labels: {images.parent / 'labels'})", flush=True)
            if not images.is_dir():
                raise FileNotFoundError(f"{setting}: {split} image directory missing: {images}")
            spec[split] = str(images)
        specs[name] = spec
    if specs["all"]["names"] != specs["min8"]["names"] or len(specs["all"]["names"]) != 1:
        raise ValueError("Both datasets must describe the same single class")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for split in ("train", "val", "test"):
        if split not in specs["all"]:
            continue
        if specs["all"][split] == specs["min8"][split]:
            raise ValueError(f"OLD_DATA and NEW_DATA both resolve {split} to {specs['all'][split]}. "
                             "Point each setting to data.yaml inside its own uploaded dataset folder. "
                             "The two runs would otherwise use the same labels.")
        members = [{p.name for p in Path(spec[split]).iterdir() if p.suffix.lower() in extensions}
                   for spec in specs.values()]
        if not members[0] or members[0] != members[1]:
            raise ValueError(f"Image membership differs in {split}. Select the grouped before/after pair.")
    return specs


def observed_validator():
    import numpy as np
    from citrus_e_v6_training import EV6Validator
    from eval_citrus_e_v5 import empirical_mask_pr

    class ObservedEV6Validator(EV6Validator):
        def get_stats(self):
            stats = {key: np.concatenate(self.metrics.stats[key], axis=0)
                     for key in ("target_cls", "pred_cls", "conf", "tp_m")}
            curve = empirical_mask_pr(stats, max_points=None).get("0")
            if curve is None:
                raise ValueError("No labelled targets in this evaluation split")
            qualified = [r for p, r in zip(curve["precision"], curve["recall"]) if p >= .90]
            save_json(self.save_dir / "observed_mask_pr.json",
                      dict(**curve, r_at_p90=max(qualified, default=0.0)))
            return super().get_stats()

    return ObservedEV6Validator


def fit(directory, recipe, train_yaml, protocol, model_yaml, device):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from citrus_control78_training import Control78Trainer

    frozen = directory / "protocol.json"
    if directory.exists() and (not frozen.exists() or json.loads(frozen.read_text(encoding="utf-8")) != protocol):
        raise FileExistsError(f"Different run settings in {directory}. Select a new PROJECT.")
    save_json(frozen, protocol)
    if training_done(directory, recipe["epochs"]):
        return
    last = directory / "weights/last.pt"
    if not last.is_file() and csv_epoch(directory):
        raise RuntimeError(f"Missing resumable checkpoint: {last}")
    seed = recipe["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = YOLO(str(last if last.is_file() else model_yaml), task="segment")
    if last.is_file():
        assert_resumable(model, directory)
    model.add_callback("on_pretrain_routine_end", restore_best)
    if "expected_targets" in protocol:
        def verify_loaded_labels(trainer):
            for split, loader in (("train", trainer.train_loader), ("val", trainer.test_loader)):
                actual = sum(len(row["cls"]) for row in loader.dataset.labels)
                expected = protocol["expected_targets"][split]
                if actual != expected:
                    raise RuntimeError(f"{split} loader read {actual} targets; expected {expected}. "
                                       "Check the dataset path and labels.cache before training.")
        model.add_callback("on_pretrain_routine_end", verify_loaded_labels)
    model.add_callback("on_model_save", save_best_and_terminal)
    model.train(trainer=Control78Trainer, **dict(
        recipe, data=str(train_yaml), project=str(directory.parent), name=directory.name, device=device,
        exist_ok=True, resume=str(last) if last.is_file() else False))
    if not training_done(directory, recipe["epochs"]):
        raise RuntimeError(f"Training incomplete: {directory}")


def main(*, model_yaml=None, project_dir=None, old_data=None, new_data=None, device=None,
         seeds=None, epochs=None, evaluate_test=None, smoke=None, check_labels=None):
    import sys
    model_yaml = Path(model_yaml or MODEL)
    device = DEVICE if device is None else device
    seeds = SEEDS if seeds is None else seeds
    epochs = EPOCHS if epochs is None else epochs
    smoke = SMOKE if smoke is None else smoke
    evaluate_test = EVALUATE_TEST if evaluate_test is None else evaluate_test
    if str(device) != "cpu":
        if not str(device).isdigit():
            raise ValueError("DEVICE must specify one GPU index or cpu")
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            raise RuntimeError("Run this file in a fresh Python process")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Set distinct seeds")
    specs = dataset_configs(old_data, new_data, evaluate_test)
    audit = check_labels(specs) if check_labels else None
    from ultralytics import YOLO

    project = Path(str(project_dir or PROJECT) + ("_SMOKE" if smoke else "")).resolve()
    project.mkdir(parents=True, exist_ok=True)
    budget = 1 if smoke else epochs
    splits = ["val", "test"] if evaluate_test else ["val"]
    print(f"{model_yaml.stem} label comparison | {2*len(seeds)} sequential fits | {budget} epochs maximum")
    print("Train views: whole | checkpoint selection: min8 val | final evaluation:", splits, "both label policies")
    rows = []
    deltas = []
    with project_lock(project):
        if audit is not None:
            save_json(project / "label_audit.json", audit)
        for seed in seeds:
            recipe = training_args(SimpleNamespace(epochs=budget, batch=BATCH, workers=WORKERS), seed)
            recipe["cache"] = CACHE
            for policy in ("all", "min8"):
                directory = project / f"train_{policy}_seed{seed}"
                # Same validation population and labels for checkpoint selection / early stopping.
                train_spec = dict(names=specs[policy]["names"], train=specs[policy]["train"], val=specs["min8"]["val"])
                train_yaml = project / f"train_{policy}.yaml"
                freeze = dict(train_spec=train_spec, recipe=recipe, model_sha256=digest(model_yaml))
                if audit is not None:
                    freeze["expected_targets"] = dict(train=audit["train"][policy], val=audit["val"]["min8"])
                # Ensure numeric class keys round-trip through the JSON protocol.
                freeze = json.loads(json.dumps(freeze))
                train_yaml.write_text(yaml.safe_dump(train_spec), encoding="utf-8")
                fit(directory, recipe, train_yaml, freeze, model_yaml, device)
                weights = directory / "weights/best_mask.pt"
                for eval_policy, spec in specs.items():
                    eval_yaml = project / f"eval_{eval_policy}.yaml"
                    eval_yaml.write_text(yaml.safe_dump(spec), encoding="utf-8")
                    for split in splits:
                        name = f"eval_{split}_{eval_policy}"
                        model = YOLO(str(weights), task="segment")
                        metric = model.val(validator=observed_validator(), data=str(eval_yaml), split=split,
                                           project=str(directory), name=name, exist_ok=True, device=device,
                                           imgsz=640, batch=BATCH, workers=WORKERS, conf=.001, iou=.7, max_det=300,
                                           augment=False, half=False, plots=True, save_json=False, mask_ratio=2)
                        curve = json.loads((directory / name / "observed_mask_pr.json").read_text(encoding="utf-8"))
                        if audit is not None and curve["targets"] != audit[split][eval_policy]:
                            raise RuntimeError(f"{eval_yaml} {split}: validator read {curve['targets']} targets, "
                                               f"expected {audit[split][eval_policy]}. Check dataset/labels.cache.")
                        scores = dict(metric.results_dict, rmax=curve["rmax"], r_at_p90=curve["r_at_p90"])
                        row = dict(train_labels=policy, eval_labels=eval_policy, split=split, seed=seed,
                                   targets=curve["targets"], epochs_completed=csv_epoch(directory))
                        row.update({key + "_percent": 100*float(scores[value]) for key, value in METRICS.items()})
                        rows.append(row)
                        write_table(project / "LABEL8_RESULTS.csv", rows)
                        if policy == "min8":
                            ref = next(r for r in rows if r["train_labels"] == "all" and r["eval_labels"] == eval_policy
                                       and r["split"] == split and r["seed"] == seed)
                            delta = dict(eval_labels=eval_policy, split=split, seed=seed)
                            delta.update({key + "_delta_pp": row[key+"_percent"] - ref[key+"_percent"]
                                          for key in METRICS})
                            deltas.append(delta)
                            write_table(project / "LABEL8_DELTA.csv", deltas)
                        del model
    print("Finished:", project / "LABEL8_DELTA.csv")
    return rows


if __name__ == "__main__":
    main()
