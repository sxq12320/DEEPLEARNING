"""Foreground sequential J4 experiments with unchanged I-series EV6 validation."""

import gc
import hashlib
import json
from functools import partial
from pathlib import Path

from citrus_j4_suite import ROOT, YAML_DIR, select_names

PROTOCOL_ID = "j4_detail_preserving_compression_scratch_ev6_v1_20261009"
SOURCES = (
    "citrus_j4_batch.py", "citrus_j4_suite.py", "citrus_j2_batch.py", "citrus_j2_data.py",
    "citrus_protocol.py", "citrus_slicing.py", "citrus_e_v5_slicing.py", "citrus_e_v6_training.py",
    "ultralytics/nn/tasks.py", "ultralytics/nn/modules/citrus_j4.py",
    "ultralytics/nn/modules/citrus_j2_structure.py", "ultralytics/nn/modules/citrus_e_v11.py",
    "ultralytics/nn/modules/citrus_e_v12.py", "ultralytics/utils/citrus_e_v11_loss.py",
    "ultralytics/utils/citrus_e_v9_loss.py", "ultralytics/utils/citrus_e_v3_loss.py",
    "ultralytics/utils/sage_v4r_loss.py", "ultralytics/utils/loss.py", "ultralytics/utils/tal.py",
    "ultralytics/utils/metrics.py", "protocols/citrus_paper1_formal_v2_ram.yaml",
)


def implementation_hash():
    """Hash scoped local model/training/evaluation code; not a data-confirmation gate."""
    value = hashlib.sha256()
    for name in tracked_sources():
        value.update(name.encode())
        value.update((ROOT / name).read_bytes())
    return value.hexdigest()


def tracked_sources():
    directories = ("ultralytics/nn", "ultralytics/engine", "ultralytics/data", "ultralytics/utils",
                   "ultralytics/models/yolo/segment", "ultralytics/models/yolo/detect")
    files = set(SOURCES) | {"citrus_j_batch.py", "eval_citrus_j2_gaze.py", "eval_citrus_e_v5.py",
                            "eval_citrus_sliced.py", "ultralytics/cfg/default.yaml"}
    for directory in directories:
        files.update(p.relative_to(ROOT).as_posix() for p in (ROOT / directory).rglob("*.py"))
    return sorted(files)


def run(data, device, project, suite="priority", epochs=50, seeds=(42,), only="", dry_run=False,
        cpu_threads=4, workers=4, batch=16):
    """Scratch-only architecture ladder; no pseudo labels or changed val/test labels."""
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    from citrus_j2_batch import settings
    from citrus_j_batch import save_initial_state, save_mask_checkpoints

    if epochs < 1 or batch < 1 or workers < 0 or cpu_threads < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Positive epochs/batch/threads, nonnegative workers and distinct seeds required")
    torch.set_num_threads(cpu_threads)
    selected = select_names(suite, only)
    project = Path(project).expanduser().resolve()
    data = Path(data).expanduser().resolve()
    train_settings = settings(epochs, seeds[0], imgsz=640, cache=True)
    train_settings.update(workers=workers, batch=batch)
    print(f"J4 FOREGROUND / SEQUENTIAL / SCRATCH / AMP=False / EV6 | {selected}", flush=True)
    print(f"DATA={data}\nPROJECT={project}\nCACHE=True; no GPU occupancy guard or data-confirmation gate", flush=True)
    print("If labels/data changed at the same DATA path, choose a NEW PROJECT; completed outputs are not rerun.",
          flush=True)
    if dry_run:
        for name in selected:
            model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment", verbose=False)
            print(f"BUILD {name}: {sum(p.numel() for p in model.model.parameters()):,} parameters", flush=True)
            del model
        return
    if not data.is_file():
        raise FileNotFoundError(f"Edit DATA in RUN_CITRUS_J4.py: {data}")
    pending = []
    digest = implementation_hash()
    for seed in seeds:
        for name in selected:
            train = dict(train_settings, seed=int(seed))
            record = dict(protocol=PROTOCOL_ID, name=name, data=str(data), seed=int(seed), train=train,
                          yaml_sha256=hashlib.sha256((YAML_DIR / (name + ".yaml")).read_bytes()).hexdigest(),
                          code_sha256=digest, validator="EV6Validator (I series)", pseudo_labels=False)
            directory = project / f"{name}_seed{seed}"
            marker = directory / "j4_completed.json"
            if marker.is_file() and json.loads(marker.read_text("utf-8")) == record:
                print(f"SKIP completed {directory.name}", flush=True)
            elif directory.exists():
                raise FileExistsError(f"Partial/conflicting output: {directory}. Use a NEW PROJECT or ONLY other models.")
            else:
                pending.append((name, seed, directory, record))
    if not pending:
        return
    from citrus_j2_data import make_trainer, prepare_resampled_views

    project.mkdir(parents=True, exist_ok=True)
    train_data = prepare_resampled_views(data, project / "_prepared_views_640_linear", 640, "linear")
    factors = dict(gaze_train=False, antialias=False, density=False, detach=False, imgsz=640)
    for name, seed, directory, record in pending:
        init_seeds(seed, deterministic=True)
        model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment")
        model.add_callback("on_pretrain_routine_end", partial(save_initial_state, cpu_threads=cpu_threads))
        model.add_callback("on_model_save", save_mask_checkpoints)

        def record_protocol(trainer, protocol=record):
            (Path(trainer.save_dir) / "j4_protocol.json").write_text(
                json.dumps(dict(protocol, code_files=tracked_sources()), ensure_ascii=False, indent=2), encoding="utf-8")

        model.add_callback("on_pretrain_routine_end", record_protocol)
        print(f"TRAIN {name} -> {directory}", flush=True)
        model.train(trainer=make_trainer(factors), data=str(train_data), device=device, project=str(project),
                    name=directory.name, exist_ok=False, **record["train"])
        actual_dir = Path(model.trainer.save_dir).resolve()
        if actual_dir != directory:
            raise RuntimeError(f"Unexpected output redirect: {actual_dir} != {directory}")
        (directory / "j4_completed.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
