"""J V3 privileged-teacher arms: foreground serial training with the I-series EV6 validator.

Order of work inside one call: (1) the J50 anchor trains on the released labels; (2) the frozen J50 teacher (seed
TEACHER_SEED) mines candidate missing fruit on TRAINING sources only, from original pixels; (3) student arms train
with those instances injected under their policy. Every arm deploys the identical YOLO11n-seg graph.
"""

import gc
import hashlib
import json
import os
import platform
import sys
from functools import partial
from pathlib import Path

from citrus_j3_suite import CACHE, FACTORS, TEACHER_ARM, TEACHER_SEED, YAML_DIR, needs_teacher, select_names

ROOT = Path(__file__).resolve().parent
PROTOCOL_ID = "citrus_j3_privileged_teacher_incomplete_labels_v1_20261007"
SOURCES = (
    "citrus_j3_batch.py",
    "citrus_j3_suite.py",
    "citrus_j3_data.py",
    "citrus_j3_teacher.py",
    "citrus_j2_batch.py",
    "citrus_j2_data.py",
    "citrus_j_batch.py",
    "citrus_protocol.py",
    "citrus_slicing.py",
    "citrus_e_v5_slicing.py",
    "citrus_e_v6_training.py",
    "eval_citrus_e_v5.py",
    "eval_citrus_sliced.py",
    "ultralytics/utils/citrus_j3_loss.py",
    "ultralytics/utils/loss.py",
    "ultralytics/utils/tal.py",
    "ultralytics/data/augment.py",
    "ultralytics/data/dataset.py",
    "ultralytics/models/yolo/segment/val.py",
    "ultralytics/utils/metrics.py",
    "protocols/citrus_paper1_formal_v2_ram.yaml",
)


def source_digest():
    digest = hashlib.sha256()
    for name in SOURCES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings(epochs, seed, imgsz=640, cache=True, init="scratch"):
    """J V2 locked recipe (AMP off, copy_paste .3, input/2 masks, EV6 validation); init is explicit and recorded."""
    from citrus_j2_batch import settings as j2_settings

    args = j2_settings(epochs, seed, imgsz, cache)
    if init != "scratch":
        path = Path(init).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"INIT checkpoint not found: {path} (use 'scratch' or a .pt path)")
        args["pretrained"] = str(path.resolve())
    return args


def record_start(trainer, init, cpu_threads=None):
    """Initial state digest, runtime, loaded membership and the actual teacher injection of this run."""
    import torch
    from ultralytics.utils.torch_utils import unwrap_model

    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    model = unwrap_model(trainer.model)
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    path = Path(trainer.save_dir)
    record = dict(state_sha256=digest.hexdigest(), initialization=init,
                  parameters=sum(p.numel() for p in model.parameters()), amp=bool(trainer.amp), data=str(trainer.args.data))
    (path / "initialization.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    runtime = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda, platform=platform.platform(),
                   device=str(trainer.device), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   cpu_threads=torch.get_num_threads(), argv=sys.argv,
                   gpu=torch.cuda.get_device_name(trainer.device) if trainer.device.type == "cuda" else None)
    (path / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    for split, loader in (("train", trainer.train_loader), ("val", trainer.test_loader)):
        files = getattr(loader.dataset, "im_files", [])
        (path / (split + "_loaded_files.txt")).write_text("\n".join(map(str, files)), encoding="utf-8")
    summary = getattr(trainer.train_loader.dataset, "pseudo_summary", None)
    (path / "j3_injection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def record_loss_stats(trainer):
    """Cumulative trust-code statistics seen by the J3 loss (positives, pseudo, ignore regions, ignored anchors)."""
    from ultralytics.utils.torch_utils import unwrap_model

    criterion = getattr(unwrap_model(trainer.model), "criterion", None)
    if criterion is not None and hasattr(criterion, "stats"):
        (Path(trainer.save_dir) / "j3_loss_stats.json").write_text(json.dumps(criterion.stats, indent=2),
                                                                     encoding="utf-8")


def teacher_weights(project, teacher):
    if teacher == "auto":
        directory = project / f"{TEACHER_ARM}_seed{TEACHER_SEED}"
        if not (directory / "j3_completed.json").is_file():
            raise RuntimeError(f"Teacher {directory.name} has not completed; include {TEACHER_ARM} (seed "
                               f"{TEACHER_SEED}) in this run or set TEACHER to an explicit checkpoint")
        return directory / "weights" / "best.pt"
    path = Path(teacher).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TEACHER checkpoint not found: {path}")
    print(f"WARNING explicit teacher {path}: it must have been trained on THIS split's train images only.", flush=True)
    return path


def run(data, device, project, suite="main", epochs=50, seeds=(42,), only="", dry_run=False, init="scratch",
        teacher="auto", oracle_data="", cpu_threads=4, workers=None, batch=None):
    """Train selected arms one by one; refuse to overwrite partial or conflicting outputs."""
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds

    torch.set_num_threads(cpu_threads)
    selected = select_names(suite, only)
    selected.sort(key=lambda n: n != TEACHER_ARM)  # the anchor (= teacher source) always trains first
    if epochs < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Positive epochs and distinct seeds required")
    if init != "scratch" and not Path(init).expanduser().is_file():
        raise FileNotFoundError(f"INIT must be 'scratch' or an existing checkpoint: {init}")
    data, project = Path(data).expanduser().resolve(), Path(project).expanduser().resolve()
    print(f"J V3 FOREGROUND / SEQUENTIAL | {selected} | {epochs} epochs | seeds={seeds} | init={init}", flush=True)
    print(f"GPU visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unchanged')} logical={device} | AMP=False | "
          f"teacher={teacher}", flush=True)
    if dry_run:
        for name in selected:
            model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment", verbose=False)
            print(f"BUILD {name}: {sum(p.numel() for p in model.model.parameters()):,} parameters | {FACTORS[name]}")
        return
    if not data.is_file():
        raise FileNotFoundError(f"Edit DATA in RUN_CITRUS_J3.py: {data}")
    if any(FACTORS[n]["oracle"] for n in selected) and not (oracle_data and Path(oracle_data).is_file()):
        raise FileNotFoundError("J59 needs ORACLE_DATA: a data.yaml whose TRAIN labels were cleaned by hand and whose "
                                "val/test are identical to DATA")
    pending = []
    for seed in seeds:
        for name in selected:
            factors = FACTORS[name]
            train = settings(epochs, seed, factors["imgsz"], CACHE[name], init)
            if workers is not None:
                train["workers"] = int(workers)
            if batch is not None:
                train["batch"] = int(batch)
            source = Path(oracle_data).resolve() if factors["oracle"] else data
            static = dict(protocol=PROTOCOL_ID, name=name, seed=seed, data=str(data), train_source=str(source),
                          epochs=epochs, init=init, teacher=teacher if needs_teacher(name) else None,
                          yaml_sha256=file_sha256(YAML_DIR / (name + ".yaml")), code_sha256=source_digest(),
                          train=train, factors=factors, validator="EV6Validator (I series)")
            directory = project / f"{name}_seed{seed}"
            completed = directory / "j3_completed.json"
            if completed.is_file() and json.loads(completed.read_text(encoding="utf-8")).get("static") == static:
                print(f"SKIP completed {directory}", flush=True)
                continue
            if directory.exists():
                raise RuntimeError(f"Partial/conflicting output exists: {directory}. Use a NEW PROJECT or remove it yourself.")
            pending.append((name, seed, directory, static))
    if not pending:
        return
    project.mkdir(parents=True, exist_ok=True)
    from citrus_j2_data import prepare_resampled_views
    from citrus_j3_data import make_trainer
    from citrus_j3_teacher import mine
    from citrus_j_batch import save_mask_checkpoints

    mined = {}
    for name, seed, directory, static in pending:
        factors = static["factors"]
        pseudo_path, provenance = None, {}
        if needs_teacher(name):
            mode = factors["teacher"]
            if mode not in mined:
                weights = teacher_weights(project, teacher)
                output = project / f"_teacher_{mode}_{file_sha256(weights)[:12]}"
                mined[mode] = (mine(weights, data, output, device=device, mode=mode), weights)
            pseudo_path, weights = mined[mode]
            provenance = dict(teacher_weights=str(weights), teacher_sha256=file_sha256(weights),
                              pseudo=str(pseudo_path), pseudo_sha256=file_sha256(pseudo_path))
        source = Path(static["train_source"])
        tag = "oracle_" if factors["oracle"] else ""
        train_data = prepare_resampled_views(source, project / f"_prepared_views_{tag}{factors['imgsz']}_linear",
                                             factors["imgsz"], "linear")
        init_seeds(seed, deterministic=True)
        model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment")
        model.add_callback("on_pretrain_routine_end", partial(record_start, init=static["init"],
                                                              cpu_threads=cpu_threads))
        model.add_callback("on_model_save", save_mask_checkpoints)
        model.add_callback("on_fit_epoch_end", record_loss_stats)
        print(f"TRAIN {name} -> {directory} | {factors} | {provenance or 'no teacher'}", flush=True)
        model.train(trainer=make_trainer(factors, pseudo_path), data=str(train_data), device=device,
                    project=str(project), name=directory.name, exist_ok=False, **static["train"])
        record = dict(static=static, provenance=provenance)
        (directory / "j3_protocol.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        (directory / "j3_completed.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
