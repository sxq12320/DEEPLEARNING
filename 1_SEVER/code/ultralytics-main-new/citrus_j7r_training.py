"""J7R foreground training and unchanged EV6 validation over explicit packed15 inputs."""

from __future__ import annotations

import hashlib
import json
from copy import copy
from pathlib import Path

import torch

from citrus_e_v6_training import EV6Validator
from citrus_j7_pr_audit import J7FinalAuditValidator
from citrus_j7_training import J7Trainer
from citrus_j7r_data import build_j7r_dataset, j7r_model_spec, pack_j7r_batch
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model


def rgb_plot_batch(batch):
    """Shallow plotting view; never replace the model's packed input in-place."""
    result = dict(batch)
    image = batch["img"]
    if image.ndim != 4 or image.shape[1] not in (3, 15):
        raise ValueError("J7R plots require RGB3 or packed15 BCHW")
    result["img"] = image[:, :3]
    return result


def _tensor_summary(value):
    if not isinstance(value, torch.Tensor):
        return None
    array = value.detach().cpu().contiguous().numpy()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "sum": float(value.detach().double().sum().cpu())}


def record_first_batch(owner, batch, spec, phase, source_high=None):
    """Save one observed pixel/GT summary, not a claim of identical random streams."""
    marker = "_j7r_{}_first_batch_written".format(phase)
    if getattr(owner, marker, False):
        return
    destination = Path(owner.save_dir) / "j7r_{}_first_batch.json".format(phase)
    if destination.exists():
        # Never overwrite a previous stream audit when resuming an existing run.
        setattr(owner, marker, True)
        return
    payload = {
        "schema_version": 1, "phase": phase, "input": dict(spec), "sensor_channels": 3,
        "model_channels": 15, "canonical_shape": list(batch["img"].shape[-2:]),
        "images": [str(value) for value in batch.get("im_file", ())],
        "global_rgb": _tensor_summary(batch["img"][:, :3]),
        "effective_high_packed": _tensor_summary(batch["img"][:, 3:]),
        "source_high": source_high,
        "ground_truth": {key: _tensor_summary(batch.get(key)) for key in ("masks", "bboxes", "cls", "batch_idx")},
        "limitation": "One first-batch observation only; same seed does not prove identical augmentations or streams.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    setattr(owner, marker, True)


def _validate_options(args, training=False):
    if args.augment or args.compile:
        raise ValueError("J7R first release requires augment=False and compile=False; packed phases cannot use RGB TTA")
    if args.save_json or args.save_txt:
        raise ValueError("J7R uses the unchanged EV6 raster; save_json/save_txt must be False")
    if int(args.imgsz) != 640:
        raise ValueError("J7R fixes the canonical image size to 640")
    if training and (args.amp or args.cache not in (True, "ram") or args.multi_scale or args.rect):
        raise ValueError("J7R controlled training fixes amp=False, cache=True, multi_scale=0, rect=False")


class _J7RValidationInput:
    """Input-only mixin; postprocessing, matching, PR and AP are never replaced."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _validate_options(self.args)
        self._j7r_spec = None

    def get_warmup_shape(self, model, batch_size, imgsz):
        self._j7r_spec = j7r_model_spec(model)
        if int(getattr(model, "channels", 15)) != 15:
            raise ValueError("J7R standalone warmup requires a packed15 model")
        return (batch_size, 15, imgsz, imgsz)

    def build_dataset(self, img_path, mode="val", batch=None):
        if self._j7r_spec is None:
            raise RuntimeError("Identify the J7R model before building standalone validation data")
        return build_j7r_dataset(self.args, img_path, batch, self.data, mode, self.stride, rect=True,
                                 data_policy=self._j7r_spec.get("data_policy", "paired1280"))

    def init_metrics(self, model):
        _validate_options(self.args)
        self._j7r_spec = j7r_model_spec(model)
        return super().init_metrics(model)

    def preprocess(self, batch):
        if self._j7r_spec is None:
            raise RuntimeError("J7R validation must identify the active raw/EMA model before preprocessing")
        batch = super().preprocess(batch)
        source = None
        if not getattr(self, "_j7r_val_first_batch_written", False):
            source = _tensor_summary(batch.get("high_img"))
        batch = pack_j7r_batch(batch, self._j7r_spec["observation"])
        record_first_batch(self, batch, self._j7r_spec, "val", source)
        return batch

    def update_metrics(self, preds, batch):
        # Includes confusion-matrix plot_matches when visualize=True.
        return super().update_metrics(preds, rgb_plot_batch(batch))

    def plot_val_samples(self, batch, ni):
        return super().plot_val_samples(rgb_plot_batch(batch), ni)

    def plot_predictions(self, batch, preds, ni, **kwargs):
        return super().plot_predictions(rgb_plot_batch(batch), preds, ni, **kwargs)


class J7RValidator(_J7RValidationInput, EV6Validator):
    """Train-epoch and standalone validation with paired pixels and original EV6 metrics."""


class J7RFinalAuditValidator(_J7RValidationInput, J7FinalAuditValidator):
    """Same input adapter plus the inherited selected-checkpoint raw PR audit."""


class J7RTrainer(J7Trainer):
    """Official YOLO.train adapter; fixed shared recipe, explicit input, mask-selected final evaluation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _validate_options(self.args, training=True)
        self.add_callback("on_train_start", self.seed_local_data_stream)

    @staticmethod
    def seed_local_data_stream(trainer):
        """Reset workers=0 CPU data RNG after lazy imports/model setup, never CUDA.

        Multiworker loaders retain the native worker_init_fn/generator. This
        local reset and a first-batch digest do not certify bit-identical full
        streams, especially with RAM caching and asynchronous worker prefetch.
        """
        if int(trainer.args.workers) != 0:
            return
        import random

        import numpy as np

        seed = int(trainer.args.seed)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.set_rng_state(generator.get_state())
        destination = Path(trainer.save_dir) / "j7r_local_data_rng.json"
        if not destination.exists():
            with destination.open("x", encoding="utf-8") as handle:
                json.dump({"seed": seed, "scope": "workers0 Python/NumPy/CPU Torch; CUDA untouched",
                           "full_stream_bitexact_claim": False}, handle, ensure_ascii=False, indent=2)

    def get_model(self, cfg=None, weights=None, verbose=True):
        source_spec = None
        source_yaml = None
        if weights is not None:
            source = weights
            if isinstance(weights, (str, Path)):
                from ultralytics.nn.tasks import load_checkpoint

                source, _ = load_checkpoint(weights)
            source_spec = j7r_model_spec(source)
            source_yaml = getattr(source, "yaml", source if isinstance(source, dict) else None)
            if cfg is not None and isinstance(cfg, dict) and source_spec != j7r_model_spec(cfg):
                raise ValueError("Refusing to reinterpret checkpoint observation semantics as another J7R arm")
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        target_spec = j7r_model_spec(model)
        if source_spec is not None and source_spec != target_spec:
            raise ValueError("Checkpoint and target model must use the same J7R observation semantics")
        if source_yaml is not None:
            architecture_keys = ("backbone", "head", "scale", "scales", "citrus_data_policy", "citrus_native_initialization",
                                 "citrus_native_layer_map")
            if any(source_yaml.get(key) != model.yaml.get(key) for key in architecture_keys):
                raise ValueError("J7R checkpoints require the same architecture, not merely the same input protocol")
        if weights is None and model.yaml.get("citrus_native_initialization"):
            from citrus_j7r_initialization import initialize_native_pair

            initialize_native_pair(model, int(self.args.seed))
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        spec = j7r_model_spec(unwrap_model(self.model))
        return build_j7r_dataset(self.args, img_path, batch, self.data, mode, stride,
                                 data_policy=spec.get("data_policy", "paired1280"))

    def preprocess_batch(self, batch):
        spec = j7r_model_spec(unwrap_model(self.model))
        batch = super().preprocess_batch(batch)
        source = None
        if not getattr(self, "_j7r_train_first_batch_written", False):
            source = _tensor_summary(batch.get("high_img"))
        batch = pack_j7r_batch(batch, spec["observation"])
        record_first_batch(self, batch, spec, "train", source)
        return batch

    def plot_training_samples(self, batch, ni):
        return super().plot_training_samples(rgb_plot_batch(batch), ni)

    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return J7RValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def final_eval(self):
        """Keep J7's mask-selected checkpoint contract, changing only the input adapter."""
        selected = Path(self.save_dir) / "weights" / "best_mask.pt"
        selection_file = Path(self.save_dir) / "best_mask_selection.json"
        if not selected.is_file() or not selection_file.is_file():
            raise RuntimeError("J7R requires best_mask.pt and best_mask_selection.json; best.pt is not substituted")
        selection = json.loads(selection_file.read_text(encoding="utf-8"))
        output = Path(self.save_dir) / "selected_mask_validation"
        if output.exists():
            raise FileExistsError("Refusing to overwrite selected validation output: {}".format(output))
        args = copy(self.args)
        args.half, args.compile = False, False
        args.plots, args.save_json, args.save_txt, args.split = bool(self.args.plots), False, False, "val"
        LOGGER.info("\nJ7R final validation: best_mask.pt, epoch {}, unchanged EV6".format(selection["epoch"]))
        validator = J7RFinalAuditValidator(self.test_loader, save_dir=output, args=args, _callbacks=self.callbacks)
        self.metrics = validator(model=str(selected))
        self.metrics.pop("fitness", None)
        self.validator = validator
        record = {
            "checkpoint": str(selected.resolve()), "native_fitness_checkpoint": str(Path(self.best).resolve()),
            "checkpoint_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(), "selection": selection,
            "validator": "J7RFinalAuditValidator / unchanged EV6", "mask_ratio": 2, "half": False,
            "metrics": self.metrics, "plots": bool(args.plots),
            "input_protocol": "packed_rgb_hr_2x_v1; sensor RGB3/model packed15/canonical640",
            "spatial_contract": "imgsz640 nominal; original EV6 rect padding retained; high canvas 2x; mask canvas /2",
            "note": "Native best.pt is retained; final metrics/plots use best_mask.pt with paired input.",
        }
        with (output / "selected_checkpoint.json").open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, default=str)
        self.j7_final_eval_record = self.j7r_final_eval_record = record
        self.j7_native_best = self.best
        self.best = selected
        self.epoch += 1
        try:
            self.run_callbacks("on_fit_epoch_end")
        finally:
            self.epoch -= 1
