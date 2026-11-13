"""Full-label, whole-image pretrained comparison; run this file from the uploaded code folder.

Default: YOLO11n-seg, SAGE42, V12_06; one seed, 300 epochs, no tiles or min8.
Optional --include-candidate adds C01: SAGE42 with a zero-start contrast stem.
C01 is an unvalidated hypothesis, NOT a claimed improvement or a new proven method.

Examples:
    python RUN_WHOLE_PRETRAIN_COMPARE.py
    python RUN_WHOLE_PRETRAIN_COMPARE.py --check-only --device cpu
    python RUN_WHOLE_PRETRAIN_COMPARE.py --smoke --device cpu --workers 0 --batch 2
    python RUN_WHOLE_PRETRAIN_COMPARE.py --include-candidate

Edit the settings below for your server, or supply equivalent CLI arguments.
This runner does not clean annotations automatically or certify their completeness.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parent
DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
PROJECT = "/data/sxq/results/WHOLE_PRETRAIN_COMPARE_20261111"
DEVICE = "1"
EPOCHS = 300
BATCH = 16
WORKERS = 4
SEEDS = [42]
AMP = False  # Same setting for every arm; not a claim that disabling AMP improves AP.
CACHE = False  # Entry points may explicitly freeze a common RAM-cache preference.
INCLUDE_CANDIDATE = False
MODELS = {
    "B00_yolo11n": "ultralytics/cfg/models/11/yolo11-seg.yaml",
    "S42_sage42": "0_orange_yaml/SAGE_series/SAGE42_asym_semantic_detail.yaml",
    "V06_v12": "0_orange_yaml/E_V12_series/V12_06_recognition_contrast.yaml",
    "C01_sage42_contrast": "0_orange_yaml/SAGE_series/SAGE42_asym_semantic_detail.yaml",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def completed_run(run, epochs):
    from citrus_control78_batch import csv_epoch, training_done
    return training_done(run, epochs) and csv_epoch(run) >= epochs


def verify_eval_population(run, curve):
    """Require final validation to use the same targets as checkpoint selection."""
    expected = read_json(Path(run) / "effective_runtime.json")["actual"]["val"]["targets"]
    if curve["targets"] != expected:
        raise RuntimeError(f"Final validation targets {curve['targets']} != training validation {expected}")


def arguments(argv=None, *, model_catalog=None, default_models=None, project_default=None):
    parser = argparse.ArgumentParser(description=__doc__)
    catalog = MODELS if model_catalog is None else model_catalog
    local_data = ROOT.parents[2] / "data/orange_yolo_grouped_dedup_20260820/data.yaml"
    parser.add_argument("--data", type=Path, default=local_data if os.name == "nt" else DATA)
    default_project = ROOT / "1_results/WHOLE_PRETRAIN_COMPARE_20261111" if os.name == "nt" else PROJECT
    parser.add_argument("--project", type=Path, default=project_default or default_project)
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolo11n-seg.pt")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--only", nargs="+", choices=list(catalog))
    parser.add_argument("--include-candidate", action="store_true", default=INCLUDE_CANDIDATE)
    parser.add_argument("--check-only", action="store_true", help="Audit data, build/profile models and check transfer; no fit")
    parser.add_argument("--smoke", action="store_true", help="One epoch, four images per split, separate SMOKE project")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.model_catalog = catalog
    args.default_models = default_models
    args.source_paths = []
    if args.device != "cpu" and not args.device.isdigit():
        parser.error("Use one GPU index or cpu; this runner intentionally uses sequential single-device fits.")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error("epochs/batch must be positive and workers nonnegative")
    try:
        args.seed_list = [int(s) for s in args.seeds.split(",")]
        if not args.seed_list or len(set(args.seed_list)) != len(args.seed_list) or min(args.seed_list) < 0:
            raise ValueError
    except ValueError:
        parser.error("seeds must be distinct nonnegative integers")
    if args.smoke:
        args.project = args.project.with_name(args.project.name + "_SMOKE")
        args.epochs, args.workers, args.batch = 1, 0, min(args.batch, 2)
    return args


def audit_dataset(data_path):
    """Audit actual packaged images/labels and group manifest; never fall back to legacy data."""
    data_path = data_path.resolve()
    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    names = cfg.get("names", {})
    if len(names) != 1 or "min8" in str(data_path).lower():
        raise ValueError("Expected a single-class full-label grouped dataset, not min8.")
    manifest = data_path.parent / "audit/split_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing group membership evidence: {manifest}")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    membership, group_splits = {}, {}
    for row in manifest_rows:
        key = (row["new_split"], row["image"])
        if key in membership:
            raise ValueError(f"Duplicate manifest member: {key}")
        membership[key] = row
        group_splits.setdefault(row["group_id"], set()).add(row["new_split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("Manifest contains a group crossing splits")
    spec, records, seen_hashes, counts = {"names": names}, [], {}, {}
    for split in ("train", "val", "test"):
        # The packaged split tree takes precedence over stale Windows paths in an uploaded YAML.
        images = data_path.parent / split / "images"
        labels = data_path.parent / split / "labels"
        files = sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"Empty split: {images}")
        expected = {name for member_split, name in membership if member_split == split}
        if {p.name for p in files} != expected:
            raise ValueError(f"{split}: actual images differ from the grouped manifest")
        spec[split] = str(images.resolve())
        targets = 0
        for image in files:
            label = labels / (image.stem + ".txt")
            lines = [line.split() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            for line in lines:
                values = list(map(float, line))
                if (len(values) < 7 or len(values) % 2 != 1 or values[0] != 0
                        or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values[1:])):
                    raise ValueError(f"Invalid single-class polygon in {label}")
            image_hash = sha(image)
            row = membership[split, image.name]
            if image_hash != row["sha256"]:
                raise ValueError(f"Image no longer matches manifest: {image}")
            if image_hash in seen_hashes and seen_hashes[image_hash] != split:
                raise ValueError(f"Image content crosses splits: {image}")
            seen_hashes[image_hash] = split
            targets += len(lines)
            records.append(dict(split=split, image=str(image), label=str(label), group=row["group_id"],
                                image_sha256=image_hash, label_sha256=sha(label), targets=len(lines)))
        counts[split] = dict(images=len(files), targets=targets)
    return spec, dict(source=str(data_path), manifest_sha256=sha(manifest), counts=counts, records=records,
                      label_review="Current labels frozen by hash; completeness not certified by this script")


def model_config(name, catalog=None):
    cfg = yaml.safe_load((ROOT / (MODELS if catalog is None else catalog)[name]).read_text(encoding="utf-8"))
    cfg.update(nc=1, scale="n", scales={"n": cfg["scales"]["n"]})
    if name == "C01_sage42_contrast":
        # Keep SAGE42's asymmetric neck and 16-channel detail decoder; add ONLY the tested contrast operator.
        # Avoid expanding G10's expensive high-resolution mask-detail path. Accuracy is unknown until trained.
        cfg["backbone"][0][2] = "EV10ContrastStem"
    return cfg


def recipe(args):
    from citrus_control78_batch import RECIPE78
    result = dict(RECIPE78)
    result.update(epochs=args.epochs, patience=0, batch=args.batch, workers=args.workers, imgsz=640,
                  pretrained=str(args.pretrained.resolve()), amp=AMP, mask_ratio=2, dropout=0.0,
                  copy_paste=0.3, cache=CACHE, plots=not args.smoke, save=True, save_period=-1,
                  close_mosaic=min(10, args.epochs), rect=False)
    for key in ("citrus_boundary", "citrus_concavity", "citrus_query", "citrus_contrast", "citrus_exclusive",
                "citrus_quality", "citrus_topology", "citrus_vfl", "nwd_ratio"):
        result[key] = 0.0
    return result


def source_fingerprint(extra_paths=()):
    paths = [Path(__file__), ROOT / "citrus_e_v6_training.py", ROOT / "citrus_control78_batch.py",
             ROOT / "RUN_LABEL8_COMPARE.py", ROOT / "eval_citrus_e_v5.py"]
    paths.extend((ROOT / "ultralytics").rglob("*.py"))
    paths.extend(map(Path, extra_paths))
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted(set(paths))}


def prepare(args):
    project = args.project.resolve()
    old = project / "suite_protocol.json"
    if not old.exists() and any(p.name != ".writer.lock" for p in project.iterdir()):
        raise FileExistsError(f"Nonempty project has no matching suite protocol: {project}")
    if not args.pretrained.is_file():
        raise FileNotFoundError(f"Supply the local official pretrained checkpoint: {args.pretrained}")
    names = args.only or args.default_models or list(args.model_catalog)[:3 + int(args.include_candidate)]
    spec, audit = audit_dataset(args.data)
    plan = dict(data_audit=audit, pretrained=str(args.pretrained.resolve()), pretrained_sha256=sha(args.pretrained),
                recipe=recipe(args), device=args.device, seeds=args.seed_list, models=names,
                configs={name: model_config(name, args.model_catalog) for name in names},
                source_hashes=source_fingerprint(args.source_paths),
                smoke=args.smoke, checkpoint_metric="all-label val metrics/mAP50-95(M)",
                views="whole image; no tiling or min8", test_policy="frozen membership only; no test evaluation")
    if old.exists() and read_json(old) != json.loads(json.dumps(plan)):
        raise FileExistsError("Existing project has different data/code/recipe. Use a new --project.")
    if args.smoke:
        spec = {"names": spec["names"]}
        for split in ("train", "val"):
            subset = [r for r in audit["records"] if r["split"] == split and r["targets"]][:4]
            for record in subset:
                for kind in ("image", "label"):
                    output = project / "smoke_data" / split / (kind + "s") / Path(record[kind]).name
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if not output.exists():
                        shutil.copy2(record[kind], output)
            spec[split] = str(project / "smoke_data" / split / "images")
    project.mkdir(parents=True, exist_ok=True)
    save_json(old, plan)
    save_json(project / "data_audit.json", audit)
    (project / "data_all.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    (project / "configs").mkdir(exist_ok=True)
    for name in names:
        (project / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(plan["configs"][name]), encoding="utf-8")
    print(json.dumps(dict(queue=names, data=audit["counts"], epochs=args.epochs, seeds=args.seed_list,
                          project=str(project), pretrained=True, mask_ratio=2, tiles=False), indent=2), flush=True)
    return plan


def transfer_report(model, source):
    """Inspect semantic layer mapping and measure actual named parameter equality."""
    import torch
    source_state = source.state_dict()
    mapping = model.yaml.get("pretrained_layer_map", {})
    aligned, unmatched, layers = {}, [], []
    equal_numel = total_numel = 0
    for i, layer in enumerate(model.model):
        j = int(mapping.get(i, i))
        layers.append(dict(target=i, target_type=type(layer).__name__, source=j,
                           source_type=type(source.model[j]).__name__ if 0 <= j < len(source.model) else None))
    for name, param in model.named_parameters():
        total_numel += param.numel()
        pieces = name.split(".")
        pieces[1] = str(mapping.get(int(pieces[1]), int(pieces[1])))
        key = ".".join(pieces)
        original = source_state.get(key)
        if original is not None and original.shape == param.shape and torch.equal(
                original.detach().float().cpu(), param.detach().float().cpu()):
            aligned[name] = key
            equal_numel += param.numel()
        else:
            unmatched.append(name)
    if equal_numel / total_numel < 0.80 or "model.0.conv.weight" not in aligned:
        raise RuntimeError("Pretrained transfer unexpectedly low; refuse to silently train from scratch")
    return dict(equal_parameter_numel=equal_numel, total_parameter_numel=total_numel,
                equal_fraction=equal_numel / total_numel, aligned_keys=aligned, other_keys=unmatched,
                layer_mapping=layers, note="Equality is not proof of feature equivalence; new operators remain random")


def check_models(project, plan):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    torch.set_num_threads(1)
    source = YOLO(plan["pretrained"], task="segment", verbose=False).model.float()

    def finite_tree(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_tree(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return all(finite_tree(item) for item in value)
        return True

    reports = []
    for name in plan["models"]:
        torch.manual_seed(plan["seeds"][0])
        model = YOLO(str(project / "configs" / f"{name}.yaml"), task="segment", verbose=False)
        model.load(plan["pretrained"])
        report = transfer_report(model.model, source)
        model.model.eval()
        with torch.no_grad():
            output = model.model(torch.zeros(1, 3, 640, 640))
        if not finite_tree(output):
            raise RuntimeError(f"Nonfinite forward output: {name}")
        report.update(model=name, parameters=sum(p.numel() for p in model.model.parameters()),
                      gflops_640=get_flops(model.model, imgsz=640))
        save_json(project / "preflight" / f"{name}.json", report)
        reports.append({key: report[key] for key in ("model", "parameters", "gflops_640", "equal_fraction")})
        del model
        gc.collect()
    save_json(project / "MODEL_COSTS.json", reports)
    print(json.dumps(reports, indent=2), flush=True)


def benchmark_batch1(model, project, plan):
    """Time real full-image prediction including preprocessing, NMS and mask decoding; exclude disk I/O."""
    import cv2
    import numpy as np
    import torch
    spec = yaml.safe_load((project / "data_all.yaml").read_text(encoding="utf-8"))
    paths = sorted(p for p in Path(spec["val"]).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    selected = np.linspace(0, len(paths) - 1, min(8, len(paths)), dtype=int)
    images = [cv2.imread(str(paths[i])) for i in selected]
    if any(image is None for image in images):
        raise ValueError("Unreadable validation image during latency measurement")
    warmup, repeats = (1, 2) if plan["smoke"] else (5, 24)
    samples = []
    for i in range(warmup + repeats):
        if plan["device"] != "cpu":
            torch.cuda.synchronize()
        started = time.perf_counter()
        model.predict(images[i % len(images)], imgsz=640, device=plan["device"], batch=1, rect=False,
                      half=False, conf=.25, iou=.7, max_det=300, augment=False, verbose=False,
                      save=False, save_txt=False, save_crop=False, show=False, retina_masks=False)
        if plan["device"] != "cpu":
            torch.cuda.synchronize()
        if i >= warmup:
            samples.append(1000 * (time.perf_counter() - started))
    return dict(median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)),
                samples_ms=samples, warmup=warmup, repeats=repeats, batch=1, conf=.25,
                scope="FP32 whole-image predict, preprocess+network+NMS+mask decode; excludes disk I/O",
                device=plan["device"], smoke=plan["smoke"])


def fit_one(project, name, seed, plan):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.data import build_yolo_dataset
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.utils.torch_utils import unwrap_model
    from citrus_e_v6_training import EV6Validator
    from citrus_control78_batch import assert_resumable, save_best_and_terminal
    from RUN_LABEL8_COMPARE import observed_validator

    class WholeTrainer(SegmentationTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            stride = max(int(unwrap_model(self.model).stride.max()), 32)
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=stride)

        def get_validator(self):
            from copy import copy
            self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
            return EV6Validator(self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                                _callbacks=self.callbacks)

        def final_eval(self):
            # Final reporting below evaluates the selected mask checkpoint explicitly, never combined box/mask best.
            # Leave last.pt resumable if interruption occurs before the completion marker is written.
            return None

    run = project / f"{name}_seed{seed}"
    result_file = run / "final_val.json"
    if result_file.is_file() and completed_run(run, plan["recipe"]["epochs"]):
        print(f"Skip completed: {run.name}", flush=True)
        return
    if not completed_run(run, plan["recipe"]["epochs"]):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1 if plan["device"] == "cpu" else min(8, os.cpu_count() or 1))
        last = run / "weights/last.pt"
        if run.exists() and not last.exists():
            raise FileExistsError(f"Run directory exists without a resumable checkpoint: {run}; use a new --project")
        model = YOLO(str(last if last.exists() else project / "configs" / f"{name}.yaml"), task="segment")
        if last.exists():
            assert_resumable(model, run)
        else:
            model.load(plan["pretrained"])

        def before_training(trainer):
            best_file = trainer.save_dir / "best_mask_selection.json"
            trainer.control78_best = read_json(best_file)["mask_mAP50_95"] if best_file.is_file() else -math.inf
            actual = {}
            for split, loader in (("train", trainer.train_loader), ("val", trainer.test_loader)):
                dataset = loader.dataset
                actual[split] = dict(images=len(dataset), targets=sum(len(r["cls"]) for r in dataset.labels))
                mask_format = dataset.transforms.transforms[-1]
                if type(dataset).__name__ != "YOLODataset" or dataset.rect or mask_format.mask_ratio != 2:
                    raise RuntimeError(f"Unexpected dataset route/grid: {split}")
                if not plan["smoke"] and actual[split] != plan["data_audit"]["counts"][split]:
                    raise RuntimeError(f"Loaded labels differ from audit: {split}")
            if bool(trainer.amp) != bool(plan["recipe"]["amp"]):
                raise RuntimeError("AMP changed unexpectedly")
            save_json(trainer.save_dir / "effective_runtime.json", dict(
                actual=actual, dataset="YOLODataset", tiles=False, label_policy="all", best_selection="all",
                train_mask_ratio=2, val_mask_ratio=2, rect=False, amp=bool(trainer.amp),
                initialization="resume" if trainer.resume else "official pretrained", device=str(trainer.device),
                python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda,
                hardware=torch.cuda.get_device_name(trainer.device) if trainer.device.type == "cuda" else "CPU"))
            if not trainer.resume:
                source = YOLO(plan["pretrained"], task="segment", verbose=False).model.float()
                save_json(trainer.save_dir / "initialization_transfer.json", transfer_report(trainer.model, source))

        model.add_callback("on_pretrain_routine_end", before_training)
        model.add_callback("on_model_save", save_best_and_terminal)
        model.train(trainer=WholeTrainer, **dict(plan["recipe"], seed=seed, data=str(project / "data_all.yaml"),
                    project=str(project), name=run.name, device=plan["device"], exist_ok=True,
                    resume=str(last) if last.exists() else False))
        if not completed_run(run, plan["recipe"]["epochs"]):
            raise RuntimeError(f"Training stopped before the shared budget: {run}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model = YOLO(str(run / "weights/best_mask.pt"), task="segment")
    metrics = model.val(validator=observed_validator(), data=str(project / "data_all.yaml"), split="val",
                        device=plan["device"], imgsz=640, batch=plan["recipe"]["batch"],
                        workers=plan["recipe"]["workers"], project=str(run), name="eval_val_all", exist_ok=True,
                        mask_ratio=2, rect=False, augment=False, half=False, conf=.001, iou=.7, max_det=300,
                        plots=not plan["smoke"], save_json=False, cache=False)
    curve = read_json(run / "eval_val_all/observed_mask_pr.json")
    verify_eval_population(run, curve)
    selected = read_json(run / "best_mask_selection.json")
    latency = benchmark_batch1(model, project, plan)
    save_json(run / "latency_batch1.json", latency)
    save_json(result_file, dict(model=name, seed=seed, best_epoch=selected["epoch"],
              mask_ap50=100 * float(metrics.results_dict["metrics/mAP50(M)"]),
              mask_ap5095=100 * float(metrics.results_dict["metrics/mAP50-95(M)"]),
              recall=100 * float(metrics.results_dict["metrics/recall(M)"]),
              r_at_p90=100 * float(curve["r_at_p90"]), targets=curve["targets"],
              latency_batch1_median_ms=latency["median_ms"], latency_batch1_p95_ms=latency["p95_ms"],
              speed_ms={k: float(v) for k, v in metrics.speed.items()},
              timing_note="Validator batch timing; not a batch1 end-to-end deployment benchmark"))


def summarize(project):
    costs = {r["model"]: r for r in read_json(project / "MODEL_COSTS.json")}
    rows = []
    for file in sorted(project.glob("*_seed*/final_val.json")):
        row = read_json(file)
        row.pop("speed_ms")
        row.pop("timing_note")
        row.update(parameters=costs[row["model"]]["parameters"], gflops_640=costs[row["model"]]["gflops_640"])
        rows.append(row)
    if rows:
        path = project / "FULL_LABEL_RESULTS.csv"
        with path.with_suffix(".csv.tmp").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        path.with_suffix(".csv.tmp").replace(path)


def main(argv=None, *, model_catalog=None, default_models=None, project_default=None, entrypoint=None):
    args = arguments(argv, model_catalog=model_catalog, default_models=default_models, project_default=project_default)
    args.source_paths = [entrypoint] if entrypoint is not None else []
    project = args.project.resolve()
    if args.worker:
        name, seed = args.worker.rsplit(":", 1)
        fit_one(project, name, int(seed), read_json(project / "suite_protocol.json"))
        return
    from citrus_control78_batch import project_lock
    # Fail before writing a training run if the selected environment has incompatible core packages.
    import matplotlib
    import numpy
    import torch
    import torchvision
    project.mkdir(parents=True, exist_ok=True)
    with project_lock(project):
        plan = prepare(args)
        try:
            git_status = subprocess.run(["git", "status", "--short", "--", str(ROOT)], cwd=ROOT,
                                        capture_output=True, text=True, check=False).stdout
        except OSError:
            git_status = "git unavailable"
        launch = dict(time_utc=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
                      git_status=git_status, python=sys.version,
                      packages={"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                                "numpy": numpy.__version__, "matplotlib": matplotlib.__version__})
        with (project / "launches.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(launch, ensure_ascii=False) + "\n")
        check_models(project, plan)
        if args.check_only:
            print("Preflight complete. No training started.", flush=True)
            return
        for seed in plan["seeds"]:
            for name in plan["models"]:
                # Fresh process per model: no CUDA/RNG/criterion state leaks across arms.
                subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--project", str(project),
                                "--worker", f"{name}:{seed}"], check=True)
                summarize(project)
    print(f"Finished: {project / 'FULL_LABEL_RESULTS.csv'}", flush=True)


if __name__ == "__main__":
    main()
