"""Foreground J experiments using the historical I-series native segmentation validator."""

import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

from citrus_j_suite import FACTORS, RUN_OVERRIDES, YAML_DIR, select_names
from citrus_protocol import fixed_train_args, load_protocol

ROOT = Path(__file__).resolve().parent
PROTOCOL_ID = "citrus_j_rgb_scratch_sourcebalanced_native_val_v2_20261002"


def source_digest():
    """Record scoped implementation identity without inspecting unrelated projects."""
    files = (
        "citrus_j_batch.py",
        "citrus_j_suite.py",
        "citrus_j_diagnostics.py",
        "citrus_protocol.py",
        "citrus_e_v5_slicing.py",
        "citrus_e_v6_training.py",
        "citrus_slicing.py",
        "ultralytics/nn/modules/citrus_j.py",
        "ultralytics/nn/modules/head.py",
        "ultralytics/nn/tasks.py",
        "ultralytics/utils/citrus_j_loss.py",
        "ultralytics/utils/loss.py",
        "ultralytics/utils/torch_utils.py",
        "ultralytics/utils/metrics.py",
        "ultralytics/utils/ops.py",
        "ultralytics/models/yolo/segment/val.py",
        "ultralytics/engine/validator.py",
        "ultralytics/engine/trainer.py",
        "protocols/citrus_paper1_formal_v2_ram.yaml",
    )
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def settings(epochs, seed):
    """Lock common recipe; new backbone has no silently partially transferred weights."""
    args = fixed_train_args()
    args.update(load_protocol()["fixed_validation"])
    args.update(
        epochs=int(epochs),
        seed=int(seed),
        pretrained=False,
        amp=False,
        cache=True,
        mask_ratio=2,
        copy_paste=0.3,
        patience=int(epochs) + 1,
        val=True,
        conf=0.001,
        plots=True,
        save_json=False,
        save_txt=False,
    )
    return args


def make_native_trainer(source_balanced):
    """Use the exact I-series EV6 validator, including the unsliced debug route."""
    from citrus_e_v6_training import EV6TrainingTrainer
    from ultralytics.models.yolo.segment import SegmentationTrainer

    if source_balanced:
        return EV6TrainingTrainer

    class JPlainTrainer(EV6TrainingTrainer):
        """Disable training views only; retain I-series validation data and mask grids."""

        def build_dataset(self, img_path, mode="train", batch=None):
            if mode == "train":
                return SegmentationTrainer.build_dataset(self, img_path, mode, batch)
            return super().build_dataset(img_path, mode, batch)

    return JPlainTrainer


def save_mask_checkpoints(trainer):
    """Keep I-style mask-only snapshots without changing the native best.pt selection."""
    for metric, attribute, filename, record_name, score_name in (
        ("metrics/mAP50-95(M)", "j_best_mask", "best_mask.pt", "best_mask_selection.json", "mask_mAP50_95"),
        ("metrics/mAP50(M)", "j_best_mask50", "best_mask50.pt", "best_mask50_selection.json", "mask_mAP50"),
    ):
        score = float(trainer.metrics.get(metric, -1))
        if score > getattr(trainer, attribute, -1):
            setattr(trainer, attribute, score)
            directory = Path(trainer.save_dir)
            shutil.copy2(trainer.last, directory / "weights" / filename)
            (directory / record_name).write_text(
                json.dumps({"epoch": trainer.epoch + 1, score_name: score}, indent=2), encoding="utf-8"
            )


def save_initial_state(trainer, cpu_threads=None):
    """Record actual training initialization and loaded membership, not an unused preview model."""
    from ultralytics.utils.torch_utils import unwrap_model
    import torch

    if cpu_threads is not None:
        # Trainer setup may replace the caller's thread count with its own default.
        torch.set_num_threads(cpu_threads)
    model = unwrap_model(trainer.model)
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    path = Path(trainer.save_dir)
    record = dict(
        state_sha256=digest.hexdigest(),
        initialization="scratch",
        parameters=sum(p.numel() for p in model.parameters()),
        amp=bool(trainer.amp),
        data=str(trainer.args.data),
    )
    (path / "initialization.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    runtime = dict(
        python=sys.version,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        platform=platform.platform(),
        device=str(trainer.device),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cpu_threads=torch.get_num_threads(),
        argv=sys.argv,
        gpu=torch.cuda.get_device_name(trainer.device) if trainer.device.type == "cuda" else None,
    )
    try:
        runtime["git_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        runtime["git_status_scoped"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--short", "--", "."], text=True, stderr=subprocess.DEVNULL
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        runtime["git_status_scoped"] = "Unavailable (uploaded folder may have no .git); scoped source SHA256 recorded."
    (path / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    for split, loader in (("train", trainer.train_loader), ("val", trainer.test_loader)):
        files = getattr(loader.dataset, "im_files", [])
        (path / (split + "_loaded_files.txt")).write_text("\n".join(map(str, files)), encoding="utf-8")


def run(
    data,
    device,
    project,
    suite="priority",
    epochs=50,
    seeds=(42,),
    only="",
    dry_run=False,
    source_balanced=True,
    common_validation=False,  # Compatibility only: True is rejected, never rerouted to baselines.
    diagnose_pr=False,
    cpu_threads=4,
):
    """Train selected arms one by one; refuse to overwrite partial or conflicting outputs."""
    if common_validation:
        raise ValueError("J uses the I-series EV6Validator only. Keep COMMON_VALIDATION=False; baselines are separate.")
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds

    torch.set_num_threads(cpu_threads)
    selected = select_names(suite, only)
    if epochs < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Positive epochs and distinct seeds required")
    data, project = Path(data).expanduser().resolve(), Path(project).expanduser().resolve()
    print("J FOREGROUND / SEQUENTIAL | {} | {} epochs | seeds={}".format(selected, epochs, seeds), flush=True)
    print(
        "GPU visible={} logical device={} | AMP=False CACHE=True scratch".format(
            os.environ.get("CUDA_VISIBLE_DEVICES", "unchanged"), device
        ),
        flush=True,
    )
    print("VAL: I-series EV6Validator | every epoch | mask grid=input/2 | max_det=300 | native PR/CM", flush=True)
    if dry_run:
        for name in selected:
            model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment", verbose=False)
            print("BUILD {}: {:,} parameters".format(name, sum(p.numel() for p in model.model.parameters())))
        return
    if not data.is_file():
        raise FileNotFoundError("Edit DATA in RUN_CITRUS_J.py: {}".format(data))
    pending = []
    for seed in seeds:
        for name in selected:
            directory = project / (name + "_seed" + str(seed))
            record = dict(
                protocol=PROTOCOL_ID,
                name=name,
                seed=seed,
                data=str(data),
                epochs=epochs,
                source_balanced=source_balanced,
                common_validation=False,
                diagnose_pr=bool(diagnose_pr),
                validation=dict(
                    validator="citrus_e_v6_training.EV6Validator",
                    protocol="i_series_native_input2",
                    every_epoch=True,
                    mask_ratio=2,
                    conf=0.001,
                    iou=0.7,
                    max_det=300,
                    half=False,
                    plots=True,
                    checkpoint_selection="best.pt=native box+mask; best_mask.pt=mask AP50-95; best_mask50.pt=mask AP50",
                ),
                yaml_sha256=hashlib.sha256((YAML_DIR / (name + ".yaml")).read_bytes()).hexdigest(),
                code_sha256=source_digest(),
                train={**settings(epochs, seed), **RUN_OVERRIDES[name]},
                factors=FACTORS[name],
            )
            completed = directory / "j_completed.json"
            if completed.is_file() and json.loads(completed.read_text()) == record:
                print("SKIP completed {}".format(directory), flush=True)
                continue
            if directory.exists():
                raise RuntimeError(
                    "Partial/conflicting output exists: {}. Resume last.pt separately or use NEW PROJECT.".format(
                        directory
                    )
                )
            pending.append((name, seed, directory, record))
    if not pending:
        return
    project.mkdir(parents=True, exist_ok=True)
    from citrus_e_v5_slicing import prepare_multiscale_views

    train_data = prepare_multiscale_views(data, project / "_prepared_multiscale", 640) if source_balanced else data
    trainer = make_native_trainer(source_balanced)
    for name, seed, directory, record in pending:
        init_seeds(seed, deterministic=True)
        model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment")
        model.add_callback("on_pretrain_routine_end", partial(save_initial_state, cpu_threads=cpu_threads))
        model.add_callback("on_model_save", save_mask_checkpoints)
        print("TRAIN {} -> {}".format(name, directory), flush=True)
        model.train(
            trainer=trainer,
            data=str(train_data),
            device=device,
            project=str(project),
            name=directory.name,
            exist_ok=False,
            **record["train"],
        )
        (directory / "j_protocol.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        selected_checkpoint = directory / "weights/best.pt"
        print("SELECTED native I-series best.pt: {}".format(selected_checkpoint), flush=True)
        if diagnose_pr:
            from analyze_citrus_pr import DiagnosticSegmentationValidator
            from citrus_e_v6_training import EV6Validator
            from citrus_j_diagnostics import observed_mask_pr

            class JDiagnosticValidator(DiagnosticSegmentationValidator, EV6Validator):
                """Retain raw matches without changing the I-series validation grid."""

            # Explicit opt-in only: same native protocol, no extra default validation pass.
            # Observed points never overwrite the standard PR curve or checkpoint choice.
            loaded = YOLO(str(selected_checkpoint), task="segment")
            metrics = loaded.val(
                data=str(data),
                validator=JDiagnosticValidator,
                device=device,
                imgsz=640,
                batch=16,
                workers=4,
                mask_ratio=2,
                conf=0.001,
                iou=0.7,
                max_det=300,
                half=False,
                save_json=False,
                save_txt=False,
                plots=False,
                project=str(directory),
                name="observed_pr",
            )
            diagnostics = observed_mask_pr(metrics.diagnostic_stats, directory / "observed_pr_points.csv")
            (directory / "observed_pr.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            del loaded
        (directory / "j_completed.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
