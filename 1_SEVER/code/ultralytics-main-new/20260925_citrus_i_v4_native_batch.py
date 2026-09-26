"""Foreground, sequential training of the independent I_V4 architecture ablations.

Run from VS Code with RUN_CITRUS_I_V4_NATIVE.py. No nohup, GPU reservation, or dataset
identity gate. These arms train from scratch: YOLO11 pretrained weights are incompatible.
"""

import gc
import json
from pathlib import Path

import torch

from citrus_e_v5_slicing import prepare_multiscale_views
from citrus_e_v6_training import EV6TrainingTrainer
from citrus_protocol import fixed_train_args
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer


ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/I_V4_series"
NAMES = (
    "I48_native_set",
    "I49_native_evidence",
    "I50_native_scene",
    "I51_native_fallback",
    "I52_native_review",
    "I53_native_edge",
)


class NativeTrainingTrainer(EV6TrainingTrainer):
    """Source-balanced training with truthful names for our five non-YOLO loss terms."""

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = ("mask_loss", "obj_loss", "scene_loss", "evid_loss", "rev_loss")
        return validator


class NativePlainTrainer(SegmentationTrainer):
    """Unsliced debug route with the same native loss names; not the formal input protocol."""

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = ("mask_loss", "obj_loss", "scene_loss", "evid_loss", "rev_loss")
        return validator


def run(data, device, project, epochs=50, seed=42, only="", dry_run=False, source_balanced=True):
    """Run selected arms in order, reusing one uniformly prepared multiscale dataset."""
    data = Path(data).expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    project = Path(project).expanduser().resolve()
    chosen = [part.strip() for part in only.split(",") if part.strip()] if only else list(NAMES)
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(NAMES):
        raise ValueError("ONLY must be unique names from {}".format(NAMES))
    print("I_V4 NATIVE: {} | epochs={} | device={} | data={}".format(chosen, epochs, device, data), flush=True)
    print("Initialization: scratch; same augmentation/optimizer, NOT matched YOLO pretraining.", flush=True)
    if dry_run:
        for name in chosen:
            model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment")
            print("BUILD OK {}: {} parameters".format(name, sum(p.numel() for p in model.model.parameters())))
        return
    pending = []
    for name in chosen:
        target = project / (name + "_seed" + str(seed))
        if (target / "native_completed.json").is_file():
            print("SKIP completed {}".format(target), flush=True)
            continue
        if target.exists():
            raise RuntimeError(
                "Partial output exists at {}. Resume from its last.pt separately or use a new PROJECT; "
                "the batch runner will not overwrite it.".format(target)
            )
        pending.append((name, target))
    if not pending:
        return
    project.mkdir(parents=True, exist_ok=True)
    if source_balanced:
        train_data = prepare_multiscale_views(data, project / "_prepared_multiscale", 640)
    else:
        train_data = data
    for name, target in pending:
        marker = target / "native_completed.json"
        model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment")
        args = fixed_train_args()
        args.update(
            data=str(train_data), project=str(project), name=target.name, epochs=int(epochs),
            device=str(device), seed=int(seed), exist_ok=False, amp=False, cache=True,
            mask_ratio=2, copy_paste=0.3, cos_lr=False, plots=True, pretrained=False,
        )
        # These coefficients belong to YOLO's TAL/box/DFL losses; our criterion ignores them.
        # Retain the fixed optimizer, learning schedule, augmentations and input protocol.
        print("TRAIN {} -> {}".format(name, target), flush=True)
        model.train(trainer=NativeTrainingTrainer if source_balanced else NativePlainTrainer, **args)
        marker.write_text(json.dumps({"name": name, "seed": seed, "epochs": epochs,
                                      "data": str(data), "source_balanced": bool(source_balanced)}, indent=2),
                          encoding="utf-8")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit("Use RUN_CITRUS_I_V4_NATIVE.py for one-click foreground training.")
