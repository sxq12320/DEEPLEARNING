"""J V2 structural and retained gaze controls: foreground serial training with the I-series EV6 validator."""

import gc
import hashlib
import json
import os
import platform
import sys
from functools import partial
from pathlib import Path

from citrus_j2_suite import CACHE, FACTORS, YAML_DIR, select_names
from citrus_protocol import fixed_train_args, load_protocol

ROOT = Path(__file__).resolve().parent
PROTOCOL_ID = "citrus_j2_structural_phase_repair_nativeval_v3_20261003"
SOURCES = (
    "citrus_j2_batch.py",
    "citrus_j2_suite.py",
    "citrus_j2_structure_suite.py",
    "citrus_j_batch.py",
    "citrus_j2_data.py",
    "eval_citrus_j2_gaze.py",
    "citrus_protocol.py",
    "citrus_slicing.py",
    "citrus_e_v5_slicing.py",
    "citrus_e_v6_training.py",
    "eval_citrus_e_v5.py",
    "eval_citrus_sliced.py",
    "ultralytics/nn/modules/citrus_j2.py",
    "ultralytics/nn/modules/citrus_j2_structure.py",
    "ultralytics/nn/modules/citrus_j2_refined.py",
    "ultralytics/nn/modules/head.py",
    "ultralytics/nn/tasks.py",
    "ultralytics/utils/citrus_j2_loss.py",
    "ultralytics/utils/citrus_j2_structure_loss.py",
    "ultralytics/utils/citrus_j_loss.py",
    "ultralytics/nn/modules/citrus_i_v2.py",
    "ultralytics/nn/modules/citrus_e_v11.py",
    "ultralytics/models/yolo/segment/val.py",
    "ultralytics/utils/metrics.py",
    "ultralytics/utils/loss.py",
    "protocols/citrus_paper1_formal_v2_ram.yaml",
)


def source_digest():
    digest = hashlib.sha256()
    for name in SOURCES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def settings(epochs, seed, imgsz=640, cache=True):
    """Same locked recipe as J V1 (scratch, AMP off, copy_paste .3, input/2 masks); imgsz is an explicit factor."""
    args = fixed_train_args()
    args.update(load_protocol()["fixed_validation"])
    args.update(
        epochs=int(epochs),
        seed=int(seed),
        imgsz=int(imgsz),
        pretrained=False,
        amp=False,
        cache=cache,
        mask_ratio=2,
        copy_paste=0.3,
        patience=int(epochs) + 1,
        val=True,
        plots=True,
        conf=0.001,
        save_json=False,
        save_txt=False,
    )
    return args


def save_initial_state(trainer, cpu_threads=None):
    """Record the actual initial state, loaded membership and runtime."""
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
    record = dict(state_sha256=digest.hexdigest(), initialization="scratch",
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


def run(data, device, project, suite="priority", epochs=300, seeds=(42,), only="", dry_run=False, gaze_eval=False,
        diagnose_pr=False, cpu_threads=4, workers=None, batch=None):
    """Train selected arms one by one; refuse to overwrite partial or conflicting outputs."""
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds

    torch.set_num_threads(cpu_threads)
    selected = select_names(suite, only)
    if epochs < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Positive epochs and distinct seeds required")
    data, project = Path(data).expanduser().resolve(), Path(project).expanduser().resolve()
    print(f"J V2 FOREGROUND / SEQUENTIAL | {selected} | {epochs} epochs | seeds={seeds}", flush=True)
    print(f"GPU visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unchanged')} logical={device} | AMP=False scratch",
          flush=True)
    if dry_run:
        for name in selected:
            model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment", verbose=False)
            print(f"BUILD {name}: {sum(p.numel() for p in model.model.parameters()):,} parameters | {FACTORS[name]}")
        return
    if not data.is_file():
        raise FileNotFoundError(f"Edit DATA in RUN_CITRUS_J2.py: {data}")
    pending = []
    for seed in seeds:
        for name in selected:
            factors = FACTORS[name]
            train = settings(epochs, seed, factors["imgsz"], CACHE[name])
            if workers is not None:
                train["workers"] = int(workers)
            if batch is not None:
                train["batch"] = int(batch)
            directory = project / f"{name}_seed{seed}"
            record = dict(protocol=PROTOCOL_ID, name=name, seed=seed, data=str(data), epochs=epochs,
                          yaml_sha256=hashlib.sha256((YAML_DIR / (name + ".yaml")).read_bytes()).hexdigest(),
                          code_sha256=source_digest(), train=train, factors=factors, validator="EV6Validator (I series)",
                          diagnose_pr=bool(diagnose_pr), gaze_eval=bool(gaze_eval))
            completed = directory / "j2_completed.json"
            if completed.is_file() and json.loads(completed.read_text(encoding="utf-8")) == record:
                print(f"SKIP completed {directory}", flush=True)
                continue
            if directory.exists():
                raise RuntimeError(f"Partial/conflicting output exists: {directory}. Use a NEW PROJECT or remove it yourself.")
            pending.append((name, seed, directory, record))
    if not pending:
        return
    project.mkdir(parents=True, exist_ok=True)
    from citrus_j2_data import make_trainer, prepare_resampled_views

    for name, seed, directory, record in pending:
        factors = record["factors"]
        resample = "area" if factors["antialias"] else "linear"
        train_data = prepare_resampled_views(data, project / f"_prepared_views_{factors['imgsz']}_{resample}",
                                             factors["imgsz"], resample)
        init_seeds(seed, deterministic=True)
        model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment")
        model.add_callback("on_pretrain_routine_end", partial(save_initial_state, cpu_threads=cpu_threads))
        from citrus_j_batch import save_mask_checkpoints

        model.add_callback("on_model_save", save_mask_checkpoints)
        print(f"TRAIN {name} -> {directory} | {factors}", flush=True)
        model.train(trainer=make_trainer(factors), data=str(train_data), device=device, project=str(project),
                    name=directory.name, exist_ok=False, **record["train"])
        (directory / "j2_protocol.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        best = directory / "weights/best.pt"
        if diagnose_pr:
            from analyze_citrus_pr import DiagnosticSegmentationValidator
            from citrus_j_diagnostics import observed_mask_pr
            from citrus_e_v6_training import EV6Validator

            class NativeDiagnosticValidator(DiagnosticSegmentationValidator, EV6Validator):
                """Observed matches using the unchanged I-series input/2 evaluation raster."""

            loaded = YOLO(str(best), task="segment")
            metrics = loaded.val(data=str(data), validator=NativeDiagnosticValidator, device=device,
                                 imgsz=factors["imgsz"], batch=16, workers=4, mask_ratio=2, conf=0.001, iou=0.7,
                                 max_det=300, half=False, plots=False, project=str(directory), name="observed_pr")
            diagnostics = observed_mask_pr(metrics.diagnostic_stats, directory / "observed_pr_points.csv")
            (directory / "observed_pr.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            del loaded
        if gaze_eval:
            from eval_citrus_j2_gaze import evaluate

            evaluate(best, data, directory / "gaze_eval", device=device, imgsz=factors["imgsz"],
                     antialias=factors["antialias"])
        (directory / "j2_completed.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
