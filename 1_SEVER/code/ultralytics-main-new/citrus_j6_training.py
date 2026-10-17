"""Native YOLO train/EV6-val/predict adapters for the J6 dual-view model.

Only model-specific data and mask decoding change. EV6's ground-truth raster,
matching, PR interpolation and AP integration remain inherited and untouched.
Exported engines, tensor-only prediction, TTA and compiled inference do not yet
carry the second raw observation and are rejected rather than silently degraded.
"""

from __future__ import annotations

import csv
import json
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from citrus_e_v6_training import EV6Validator
from citrus_j6_data import build_j6_dataset, prepare_j6_batch
from citrus_j6_inference import assert_j6_scout_geometry
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment import SegmentationPredictor, SegmentationTrainer
from ultralytics.utils import RANK, ops
from ultralytics.utils.torch_utils import unwrap_model


J6_LOSS_NAMES = (
    "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss",
    "scout_loss", "local_loss", "identity_loss", "gain_loss",
)


def find_j6_head(model, required=True):
    """Find the head in raw, EMA/DDP or AutoBackend.backend.model wrappers."""
    visited = set()
    queue = [model]
    while queue:
        current = queue.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if type(current).__name__ == "SegmentCitrusJ6" and callable(getattr(current, "bind_high_view", None)):
            return current
        if isinstance(current, torch.nn.Module):
            queue.extend(current.children())
        # New AutoBackend stores its Python backend outside nn.Module.children.
        for name in ("backend", "model", "module", "ema", "_orig_mod"):
            child = getattr(current, name, None)
            if child is not current:
                queue.append(child)
    if required:
        raise TypeError("J6 adapter requires a SegmentCitrusJ6 head in a native PyTorch model")
    return None


def _postprocess(head, preds, args):
    # Lazy import keeps plain YOLO startup independent of J6 inference code.
    from citrus_j6_inference import postprocess_j6

    predictions = postprocess_j6(
        head, preds, conf=args.conf, iou=args.iou, max_det=args.max_det, mask_ratio=2
    )
    if args.classes is not None:
        for pred in predictions:
            classes = torch.as_tensor(args.classes, device=pred["cls"].device)
            keep = (pred["cls"][:, None] == classes[None]).any(1)
            for key in ("bboxes", "conf", "cls", "masks"):
                pred[key] = pred[key][keep]
    return predictions


class J6Trainer(SegmentationTrainer):
    """Ordinary segmentation training with one augmented high/low observation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._j6_step_totals = {"successful_optimizer_steps": 0, "amp_skipped_optimizer_steps": 0}
        self.add_callback("on_train_start", self._start_j6_epoch_log)
        self.add_callback("on_train_epoch_end", self._write_j6_epoch_log)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Check checkpoint semantics before transferring equal-shaped legacy scout tensors."""
        if weights is not None:
            source = weights
            if isinstance(weights, (str, Path)):
                from ultralytics.nn.tasks import load_checkpoint

                source, _ = load_checkpoint(weights)
            source_head = find_j6_head(source, required=False)
            if source_head is not None:
                assert_j6_scout_geometry(source_head, "J6 training/checkpoint resume")
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        assert_j6_scout_geometry(find_j6_head(model), "J6 training model")
        return model

    def setup_model(self):
        """Also cover prebuilt models that the native trainer would otherwise reuse without get_model()."""
        if isinstance(self.model, torch.nn.Module):
            assert_j6_scout_geometry(find_j6_head(self.model), "J6 training/prebuilt model")
        return super().setup_model()

    def optimizer_step(self):
        previous_scale = self.scaler.get_scale()
        super().optimizer_step()
        success = self.scaler.get_scale() >= previous_scale
        criterion = getattr(unwrap_model(self.model), "criterion", None)
        if criterion is None or not callable(getattr(criterion, "commit_updates", None)):
            raise RuntimeError("J6 optimizer steps require J6Loss.commit_updates(success=...)")
        criterion.commit_updates(success=success)
        if not hasattr(self, "_j6_step_totals"):
            self._j6_step_totals = {"successful_optimizer_steps": 0, "amp_skipped_optimizer_steps": 0}
        key = "successful_optimizer_steps" if success else "amp_skipped_optimizer_steps"
        self._j6_step_totals[key] += 1
        # Generic ModelEMA averages only floating tensors. J6's integer update
        # counters and boolean trained flag are discrete control state, not
        # parameters to average; otherwise every EMA checkpoint remains in the
        # untrained base-only inference branch despite successful optimisation.
        self._sync_j6_ema_training_state(self)

    @staticmethod
    @torch.no_grad()
    def _sync_j6_ema_training_state(trainer):
        ema = getattr(trainer, "ema", None)
        if ema is None or getattr(ema, "ema", None) is None:
            return
        source = find_j6_head(unwrap_model(trainer.model))
        target = find_j6_head(ema.ema)
        for name in ("auxiliary_updates", "gain_updates", "identity_positive_updates", "identity_negative_updates"):
            getattr(target, name).copy_(getattr(source, name))
        if source.enabled:
            target.relation.relation_trained.copy_(source.relation.relation_trained)

    def validate(self):
        self._sync_j6_ema_training_state(self)
        self.validator.j6_epoch = self.epoch + 1
        return super().validate()

    @staticmethod
    def _j6_log_snapshot(trainer):
        model = unwrap_model(trainer.model)
        criterion = getattr(model, "criterion", None)
        totals = {f"forward/{key}": float(value) for key, value in getattr(criterion, "cumulative", {}).items()}
        totals.update({f"optimizer/{key}": float(value) for key, value in trainer._j6_step_totals.items()})
        head = find_j6_head(model)
        for name in ("auxiliary_updates", "gain_updates", "identity_positive_updates", "identity_negative_updates"):
            totals[f"state/{name}"] = float(getattr(head, name))
        return totals

    @staticmethod
    def _start_j6_epoch_log(trainer):
        trainer._j6_log_previous = J6Trainer._j6_log_snapshot(trainer)

    @staticmethod
    def _write_j6_epoch_log(trainer):
        """Append true epoch differences, not statistics from the last batch.

        Forward counters include attempted batches; successful/AMP-skipped step
        counters and committed head state are separate. In DDP these are rank-0
        local diagnostics, not an unannounced global sum.
        """
        if RANK not in {-1, 0}:
            return
        snapshot = J6Trainer._j6_log_snapshot(trainer)
        previous = getattr(trainer, "_j6_log_previous", {})
        path = Path(trainer.save_dir) / "j6_auxiliary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("epoch", "rank", "metric", "total", "delta"))
            if is_new:
                writer.writeheader()
            for key, value in sorted(snapshot.items()):
                writer.writerow(dict(epoch=trainer.epoch + 1, rank=RANK, metric=key, total=value,
                                     delta=value - previous.get(key, 0.0)))
        trainer._j6_log_previous = snapshot

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_j6_dataset(self.args, img_path, batch, self.data, mode=mode, stride=stride)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        return prepare_j6_batch(batch, self.device)

    def get_validator(self):
        self.loss_names = J6_LOSS_NAMES
        return J6Validator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)


class J6Validator(EV6Validator):
    """J6 decoded masks evaluated by the unmodified I/E/J EV6 protocol."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._j6_head = None
        self._check_options()

    def _check_options(self):
        if self.args.augment or self.args.compile:
            raise ValueError("J6 EV6 validation currently requires augment=False and compile=False")
        if self.args.save_json:
            raise ValueError("J6 uses EV6 metrics, not COCO JSON evaluation; save_json must remain False")

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_j6_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride, rect=True)

    def init_metrics(self, model):
        self._check_options()
        self._j6_head = find_j6_head(model)
        assert_j6_scout_geometry(self._j6_head, "J6 validation/checkpoint")
        super().init_metrics(model)
        # This controls inherited GT preparation, not our prediction decoding.
        # save_txt must not silently switch EV6 to a full-resolution GT grid.
        self.process = ops.process_mask
        self.evaluation_mask_ratio = 2
        self._j6_decode_totals = {}
        self._j6_decode_batches = 0
        self._j6_evaluation_number = getattr(self, "_j6_evaluation_number", 0) + 1

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        batch = prepare_j6_batch(batch, self.device, half=self.args.half)
        if self._j6_head is None:
            raise RuntimeError("J6Validator.init_metrics must bind the active raw or EMA model first")
        assert_j6_scout_geometry(self._j6_head, "J6 validation batch")
        self._j6_head.bind_high_view(batch["high_img"])
        return batch

    def postprocess(self, preds):
        predictions = _postprocess(self._j6_head, preds, self.args)
        expected = tuple(size // 2 for size in self.input_shape)
        for pred in predictions:
            if tuple(pred["masks"].shape[-2:]) != expected:
                raise RuntimeError(f"J6 decoder violated EV6 input/2 mask contract: expected {expected}")
        totals = getattr(self, "_j6_decode_totals", {})
        for key, value in getattr(self._j6_head, "last_inference", {}).items():
            totals[key] = totals.get(key, 0) + int(value)
        self._j6_decode_totals = totals
        self._j6_decode_batches = getattr(self, "_j6_decode_batches", 0) + 1
        return predictions

    def get_stats(self):
        metrics = super().get_stats()
        if RANK not in {-1, 0}:
            return metrics
        training = bool(getattr(self, "training", False))
        epoch = getattr(self, "j6_epoch", None) if training else None
        summary = {
            "protocol": "EV6 unchanged input/2 GT, matching, PR interpolation and AP integration",
            "epoch": epoch,
            "training_validation": training,
            "images": int(self.seen),
            "batches": getattr(self, "_j6_decode_batches", 0),
            "counts": getattr(self, "_j6_decode_totals", {}),
            "count_scope": "rank-0 local decode diagnostics; metrics remain the inherited distributed protocol",
        }
        head = self._j6_head
        summary["committed_head_state"] = {
            name: int(getattr(head, name, 0))
            for name in ("auxiliary_updates", "gain_updates", "identity_positive_updates", "identity_negative_updates")
        }
        summary["committed_head_state"]["enabled"] = bool(getattr(head, "enabled", False))
        summary["committed_head_state"]["relation_trained"] = bool(
            getattr(getattr(head, "relation", None), "relation_trained", False)
        )
        suffix = f"epoch{epoch:04d}" if epoch is not None else f"eval{getattr(self, '_j6_evaluation_number', 0):04d}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for name in ("j6_decode_summary.json", f"j6_decode_{suffix}.json"):
            (self.save_dir / name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    def _prepare_batch(self, si, batch):
        prepared = super()._prepare_batch(si, batch)
        if "j6_raw_to_global" in batch:
            prepared["j6_raw_to_global"] = batch["j6_raw_to_global"][si]
        return prepared

    def scale_preds(self, predn, pbatch):
        """Restore save_txt output without applying full-size padding to half masks."""
        if "j6_raw_to_global" not in pbatch:
            raise RuntimeError("J6 prediction export requires the paired dataset's recorded original-image affine")
        return restore_j6_prediction(
            predn, pbatch["ori_shape"], pbatch["imgsz"], pbatch["j6_raw_to_global"]
        )


def restore_j6_prediction(pred, original_shape, global_shape, raw_to_global, chunk_size=16, output_shape=None):
    """Map boxes and binary masks to original pixels using the recorded affine.

    Nearest sampling here is display/export restoration of already-thresholded
    masks, not validation. EV6 always evaluates the original input/2 raster.
    ``output_shape=None`` restores full original masks for explicit retina or
    validation text export. A smaller output_shape samples the same original
    content coordinate domain on a bounded display grid, without letterbox pads.
    Chunking bounds temporary float-mask memory; returned masks cost N*H_out*W_out
    uint8 bytes. It does not make an explicitly requested full retina mask cheap.
    """
    h0, w0 = map(int, original_shape[:2])
    gh, gw = map(int, global_shape)
    mh, mw = (h0, w0) if output_shape is None else tuple(map(int, output_shape))
    if chunk_size < 1 or min(h0, w0, gh, gw, mh, mw) < 1:
        raise ValueError("Image geometry and chunk_size must be positive")
    affine = torch.as_tensor(raw_to_global, device=pred["bboxes"].device, dtype=torch.float32)
    gx, gy = affine[0, 0], affine[1, 1]
    boxes = pred["bboxes"].clone()
    boxes[:, (0, 2)] = (boxes[:, (0, 2)] - affine[0, 2]) / gx
    boxes[:, (1, 3)] = (boxes[:, (1, 3)] - affine[1, 2]) / gy
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clamp(0, w0)
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clamp(0, h0)
    if not len(boxes):
        return {**pred, "bboxes": boxes, "masks": None}
    x = (torch.arange(mw, device=boxes.device, dtype=torch.float32) + 0.5) * (w0 / mw) * gx + affine[0, 2]
    y = (torch.arange(mh, device=boxes.device, dtype=torch.float32) + 0.5) * (h0 / mh) * gy + affine[1, 2]
    yy, xx = torch.meshgrid(y / gh * 2 - 1, x / gw * 2 - 1, indexing="ij")
    grid = torch.stack((xx, yy), -1)[None]
    masks = torch.empty((len(boxes), mh, mw), dtype=torch.uint8, device=boxes.device)
    for start in range(0, len(boxes), chunk_size):
        source = pred["masks"][start : start + chunk_size, None].float()
        restored = F.grid_sample(source, grid.expand(len(source), -1, -1, -1), mode="nearest", align_corners=False)
        masks[start : start + len(source)] = restored[:, 0].to(torch.uint8)
    return {**pred, "bboxes": boxes, "masks": masks}


class J6Predictor(SegmentationPredictor):
    """Native prediction with a true high observation and bounded display masks.

    Boxes always use original-image pixels. Unless retina_masks=True is explicit,
    masks cover the unpadded original content with long side <= global canvas's
    long side (normally 640/672), never N x original-H x original-W. Standard
    Results.save_txt then exports those bounded contours. Use explicit retina
    prediction or J6 validation save_txt when full original-raster contours are
    required; their memory cost is correspondingly larger.
    """

    def _check_options(self):
        if self.args.augment or self.args.compile or self.args.embed:
            raise ValueError("J6 prediction does not support TTA, compile, or embedding-only output")
        if getattr(self.model, "format", "pt") != "pt":
            raise ValueError("J6 currently requires native PyTorch weights, not an exported single-input engine")
        assert_j6_scout_geometry(find_j6_head(self.model), "J6 prediction/checkpoint")

    def preprocess(self, im):
        self._check_options()
        if isinstance(im, torch.Tensor):
            raise ValueError("J6 tensor-only prediction lacks raw high-resolution pixels; provide original images")
        low = super().preprocess(im)  # invokes paired pre_transform below
        self._j6_high = prepare_j6_batch(
            {"high_img": self._j6_high_cpu}, self.device, half=self.model.fp16
        )["high_img"]
        return low

    def pre_transform(self, im):
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz, auto=same_shapes and self.args.rect and self.model.format == "pt", stride=self.model.stride
        )
        lows, highs, geometry = [], [], []
        for raw in im:
            if raw.ndim != 3 or raw.shape[2] != 3 or raw.dtype != np.uint8:
                raise ValueError("J6 prediction requires original uint8 BGR images with three channels")
            params = letterbox.get_params({"img": raw})
            low = letterbox.apply_image({"img": raw}, params)["img"]
            h, w = low.shape[:2]
            cw, ch = params["new_unpad"]
            left, top = params["left"], params["top"]
            high = np.full((h * 2, w * 2, 3), 114, dtype=np.uint8)
            high[top * 2 : (top + ch) * 2, left * 2 : (left + cw) * 2] = cv2.resize(
                raw, (cw * 2, ch * 2), interpolation=cv2.INTER_LINEAR
            )
            affine = np.asarray(
                [[cw / raw.shape[1], 0.0, left], [0.0, ch / raw.shape[0], top], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            lows.append(low)
            highs.append(np.ascontiguousarray(high[..., ::-1].transpose(2, 0, 1)))
            geometry.append(affine)
        self._j6_high_cpu = torch.from_numpy(np.stack(highs))
        self._j6_original_geometry = geometry
        return lows

    def inference(self, im, *args, **kwargs):
        head = find_j6_head(self.model)
        assert_j6_scout_geometry(head, "J6 prediction forward")
        head.bind_high_view(self._j6_high)
        return super().inference(im, *args, **kwargs)

    def postprocess(self, preds, img, orig_imgs):
        predictions = _postprocess(find_j6_head(self.model), preds, self.args)
        results = []
        for pred, original, path, affine in zip(
            predictions, orig_imgs, self.batch[0], self._j6_original_geometry
        ):
            output_shape = None
            if not self.args.retina_masks:
                h0, w0 = original.shape[:2]
                scale = min(1.0, max(img.shape[-2:]) / max(h0, w0))
                output_shape = (max(1, round(h0 * scale)), max(1, round(w0 * scale)))
            restored = restore_j6_prediction(pred, original.shape, img.shape[-2:], affine,
                                             output_shape=output_shape)
            boxes = torch.cat(
                (restored["bboxes"], restored["conf"][:, None], restored["cls"][:, None]), dim=1
            )
            results.append(Results(original, path=path, names=self.model.names, boxes=boxes, masks=restored["masks"]))
        return results
