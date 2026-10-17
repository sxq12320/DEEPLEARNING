"""Process-local adapters for pinned OFFICIAL trainers; optimization/assignment/losses stay native."""

from __future__ import annotations

import contextlib
import copy
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image

from inference_input import OrderedInputPredictor
from validation import CommonValidation


def cpu_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def yolo_predictor(model, nc, protocol, prefetch_workers=2):
    """Native LetterBox/NMS/prototype decoder, without fusing or mutating the live training model."""
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import nms, ops
    from inference_runtime import decode_yolo_masks_bounded

    letterbox = LetterBox(new_shape=(640, 640), auto=False, stride=32)
    device = next(model.parameters()).device

    def prepare(path):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        resized = letterbox(image=bgr)
        tensor = torch.from_numpy(np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)))
        return tensor, rgb.shape

    def infer(prepared):
        tensor, shape = prepared
        tensor = tensor.to(device).float().unsqueeze(0) / 255.0
        outputs = model(tensor)
        # Ultralytics 8.4.60 segmentation exports and non-export models use these two layouts.
        protos = outputs[0][1] if isinstance(outputs[0], tuple) else outputs[1]
        detections = nms.non_max_suppression(
            outputs[0],
            conf_thres=protocol["score_min"],
            iou_thres=0.7,
            max_det=300,
            nc=nc,
            end2end=bool(getattr(model, "end2end", False)),
        )[0]
        if not len(detections):
            return [], [], []
        detections = detections[detections[:, 4].argsort(descending=True)[:protocol["max_dets"]]]
        ops.scale_boxes(tensor.shape[2:], detections[:, :4], shape)
        masks = decode_yolo_masks_bounded(ops.process_mask_native, protos[0], detections[:, 6:], detections[:, :4],
                                          shape[:2])
        masks = masks.cpu().numpy()
        keep = masks.any(axis=(-2, -1))
        # Boolean indexing copies the whole dense original-image mask tensor.
        # The common all-nonempty case can safely retain the existing CPU array.
        selected_masks = masks if keep.all() else masks[keep]
        return selected_masks, detections[:, 4].cpu().numpy()[keep], detections[:, 5].cpu().numpy()[keep].astype(int) + 1

    return OrderedInputPredictor(prepare, infer, prefetch_workers)


def mmdet_predictor(model, cfg, prefetch_workers=2):
    import torch
    from mmcv.ops import RoIPool
    from mmdet.utils import get_test_pipeline_cfg
    from mmengine.dataset import Compose, pseudo_collate
    from mmdet_common import prediction_pipeline

    # Official inference_detector requires model.cfg even with an explicit
    # pipeline, and rejects unsupported CPU RoIPool before reading any images.
    if not hasattr(model, "cfg"):
        raise AttributeError("MMDetection inference requires model.cfg")
    if model.data_preprocessor.device.type == "cpu":
        for module in model.modules():
            assert not isinstance(module, RoIPool), "CPU inference with RoIPool is not supported currently."
    pipeline = Compose(prediction_pipeline(get_test_pipeline_cfg(cfg)))

    def prepare(path):
        return pipeline(dict(img_path=str(path), img_id=0))

    def infer(data):
        # Equivalent to the official single-image inputs/data_samples lists;
        # a list passed to inference_detector itself still runs sequentially.
        with torch.no_grad():
            result = model.test_step(pseudo_collate([data]))[0].pred_instances
        return result.masks.cpu().numpy(), result.scores.cpu().numpy(), result.labels.cpu().numpy().astype(int) + 1

    return OrderedInputPredictor(prepare, infer, prefetch_workers)


def rfdetr_predictor(model, resolution, num_select, protocol, prefetch_workers=2):
    """Pinned Nano normalization/top-K/logit-mask threshold, with bounded full-resolution decoding."""
    import torch
    from torch.nn import functional as F
    from torchvision.transforms import functional as TF

    device = next(model.parameters()).device

    def prepare(path):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            tensor = TF.to_tensor(rgb)
        return tensor, width, height

    def infer(prepared):
        tensor, width, height = prepared
        tensor = tensor.to(device)
        tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        tensor = TF.resize(tensor, (resolution, resolution)).unsqueeze(0)
        outputs = model(tensor)
        if isinstance(outputs, tuple):
            outputs = dict(pred_boxes=outputs[0], pred_logits=outputs[1], pred_masks=outputs[2])
        probabilities = outputs["pred_logits"][0].sigmoid()
        scores, indexes = torch.topk(probabilities.flatten(), num_select)
        labels = indexes % probabilities.shape[1]
        queries = indexes // probabilities.shape[1]
        keep = (labels != 0) & (scores >= protocol["score_min"])
        # Preserve official foreground order; unused output zero must NOT consume the foreground cap.
        scores, labels, queries = scores[keep], labels[keep], queries[keep]
        scores, labels, queries = (value[:protocol["max_dets"]] for value in (scores, labels, queries))
        masks = []
        chunk_size = max(1, min(16, 4_000_000 // (height * width)))
        for start in range(0, len(queries), chunk_size):
            logits = outputs["pred_masks"][0][queries[start:start + chunk_size]]
            expanded = F.interpolate(logits.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)
            masks.extend((expanded[:, 0] > 0.0).cpu().numpy())
        return masks, scores.cpu().numpy(), labels.cpu().numpy().astype(int)

    return OrderedInputPredictor(prepare, infer, prefetch_workers, max_prefetch=2)


def make_yolo_trainer(job, prepared, run_dir, nc):
    import torch
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from ultralytics.models.yolo.segment.val import SegmentationValidator

    manager = CommonValidation(job, prepared, run_dir)

    class Validator(SegmentationValidator):

        def __call__(self, trainer=None, model=None):
            if trainer is None:
                raise RuntimeError("Common validator is training-only; use the separate final evaluation worker")
            candidate = trainer.ema.ema if trainer.ema else trainer.model

            def save_best(epoch):
                path = Path(run_dir) / "train/weights/best_unified.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                # FP32 snapshot of the exact evaluated weights; no mutation/half quantization of live EMA.
                torch.save(
                    dict(model=copy.deepcopy(candidate).cpu().float(),
                         ema=None,
                         epoch=epoch - 1,
                         train_args=vars(trainer.args),
                         validation_protocol=manager.protocol), path)
                return path

            report = manager.run(trainer.epoch + 1, candidate,
                                 yolo_predictor(candidate, nc, manager.protocol, job.get("workers", 2)),
                                 save_best, "ema" if trainer.ema else "regular")
            quick = report["quick"]
            # Official native CSV keys remain fixed. These are labeled quick-grid MASK metrics, not bbox AP.
            result = {key: float("nan") for key in self.metrics.keys}
            result.update({
                "metrics/precision(M)": quick["mask_precision"],
                "metrics/recall(M)": quick["mask_recall"],
                "metrics/mAP50(M)": quick["mask_ap_50"],
                "metrics/mAP50-95(M)": quick["mask_ap_50_95"],
                "fitness": report["original"]["mask_ap_50_95"] if report["original"] else -1.0,
            })
            return result

    class Trainer(SegmentationTrainer):
        common_validation = manager

        def get_validator(self):
            native = super().get_validator()
            return Validator(native.dataloader,
                             save_dir=native.save_dir,
                             args=copy.copy(native.args),
                             _callbacks=self.callbacks)

        def final_eval(self):
            # Parent waits for THIS CUDA process to exit, then launches common FP32 original-image eval.
            manager.checkpoint()

    return Trainer


def install_mmdet_loop(job, prepared, run_dir, cfg):
    """EMA before/after-val hooks stay intact; save the evaluated state BEFORE hooks swap weights back."""
    import torch
    from mmengine.registry import LOOPS
    from mmengine.runner.loops import ValLoop

    manager = CommonValidation(job, prepared, run_dir)

    class CitrusCommonValLoop(ValLoop):

        def run(self):
            runner = self.runner
            runner.call_hook("before_val")
            runner.call_hook("before_val_epoch")
            candidate = runner.model.module if hasattr(runner.model, "module") else runner.model
            has_ema = any(type(hook).__name__ == "EMAHook" for hook in runner.hooks)

            def save_best(epoch):
                path = Path(run_dir) / "train/best_unified.pth"
                # runner.save_checkpoint invokes EMAHook.before_save_checkpoint and would swap
                # the already-swapped validation weights a SECOND time. Save inference weights directly.
                torch.save(
                    dict(state_dict=cpu_state(candidate),
                         meta=dict(epoch=epoch,
                                   dataset_meta=runner.train_dataloader.dataset.metainfo,
                                   cfg=cfg.pretty_text,
                                   validation_protocol=manager.protocol)), path)
                return path

            existed = hasattr(candidate, "cfg")
            old_cfg = getattr(candidate, "cfg", None)
            candidate.cfg = cfg
            metrics = {}
            try:
                report = manager.run(runner.epoch, candidate, mmdet_predictor(candidate, cfg, job.get("workers", 2)),
                                     save_best,
                                     "ema" if has_ema else "regular")
                metrics = {
                    f"quick/{key}": value for key, value in report["quick"].items() if isinstance(value, (int, float))
                }
                if report["original"]:
                    metrics.update({
                        f"original/{key}": value
                        for key, value in report["original"].items()
                        if isinstance(value, (int, float))
                    })
                return metrics
            finally:
                if existed:
                    candidate.cfg = old_cfg
                else:
                    del candidate.cfg
                runner.call_hook("after_val_epoch", metrics=metrics)
                runner.call_hook("after_val")

    # Each worker process trains one job. Registration never edits installed MMEngine source.
    LOOPS.register_module(name="CitrusCommonValLoop", module=CitrusCommonValLoop, force=True)
    cfg.val_cfg = dict(type="CitrusCommonValLoop")
    cfg.train_cfg.val_interval = 1
    cfg.train_cfg.pop("dynamic_intervals", None)
    cfg.custom_hooks = [h for h in cfg.get("custom_hooks", []) if h["type"] != "EarlyStoppingHook"]
    cfg.default_hooks.checkpoint.save_best = None
    return manager


def rfdetr_stats(report):
    """Only mask statistics are computed. Native JSON must NOT claim any bbox AP was measured."""
    metrics = report["original"] or report["quick"]
    suffixes = ("ap_50_95", "ap_50", "ap_75", "ap_small", "ap_medium", "ap_large", "ar_1", "ar_10", "ar_100",
                "ar_small", "ar_medium", "ar_large")
    return dict(
        coco_eval_masks=[metrics[f"mask_{s}"] for s in suffixes],
        results_json=dict(
            class_map=[],
            scope="common_mask_only_original" if report["original"] else "common_mask_only_quick640",
            note="No bbox metrics/class-wise extended metrics computed; use validation_common and val_common.json."))


@contextlib.contextmanager
def rfdetr_validation(owner, job, prepared, run_dir):
    """Replace only pinned main-module validation entry; restore it even if training fails."""
    import torch
    import rfdetr.main as main_module

    manager = CommonValidation(job, prepared, run_dir)
    original_train, original_eval = main_module.train_one_epoch, main_module.evaluate
    if tuple(inspect.signature(original_eval).parameters) != (
            "model", "criterion", "postprocess", "data_loader", "base_ds", "device",
            "args") or "epoch" not in inspect.signature(original_train).parameters:
        raise RuntimeError("RF-DETR validation entry does not match pinned 1.4.0.post0; refusing unsafe patch")
    current_epoch = 0

    def train_one_epoch(*args, **kwargs):
        nonlocal current_epoch
        bound = inspect.signature(original_train).bind(*args, **kwargs)
        current_epoch = int(bound.arguments["epoch"]) + 1
        return original_train(*args, **kwargs)

    def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None):
        ema = getattr(owner.model, "ema_m", None)
        is_ema = ema is not None and model is ema.module
        eligible = is_ema if args.use_ema else True

        def save_best(epoch):
            path = Path(run_dir) / "train/checkpoint_best_unified.pth"
            torch.save(dict(model=cpu_state(model), epoch=epoch - 1, args=args, validation_protocol=manager.protocol),
                       path)
            return path

        report = manager.run(current_epoch,
                             model,
                             rfdetr_predictor(model, job["recipe"]["imgsz"], postprocess.num_select, manager.protocol,
                                              job.get("workers", 2)),
                             save_best,
                             "ema" if is_ema else "regular",
                             eligible=eligible)
        return rfdetr_stats(report), None

    main_module.train_one_epoch, main_module.evaluate = train_one_epoch, evaluate
    try:
        yield manager
    finally:
        main_module.train_one_epoch, main_module.evaluate = original_train, original_eval


def evaluate_unified(job, prepared, run_dir, checkpoint, output_dir, profile_images=0):
    """Final worker uses the SAME predictor and evaluator as original-image checkpoint selection."""
    from common import save_json, seed_everything
    from inference_runtime import install_inference_runtime, runtime_status
    from validation import checked_protocol, collect_metrics, validation_state

    seed_everything(job["seed"])
    install_inference_runtime()
    protocol = checked_protocol(job["validation"])
    prepared, run_dir, output_dir = Path(prepared), Path(run_dir), Path(output_dir)
    if profile_images and output_dir.resolve() == run_dir.resolve():
        raise ValueError("A speed profile requires a separate output directory to protect existing results")
    selection = None
    is_selected_checkpoint = False
    if not profile_images:
        selection_path = run_dir / "validation_common/best.json"
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            is_selected_checkpoint = Path(selection["checkpoint"]).resolve() == Path(checkpoint).resolve()
        if not is_selected_checkpoint and output_dir.resolve() == run_dir.resolve():
            raise ValueError("A custom checkpoint requires a separate output directory to protect existing results")
    nc = len(json.loads((prepared / "summary.json").read_text(encoding="utf-8"))["names"])
    family = job["recipe"]["family"]
    if family == "yolo":
        from ultralytics import YOLO

        owner = YOLO(str(checkpoint), task="segment")
        model = owner.model.float().eval()
        model.to("cuda:0")
        predictor = yolo_predictor(model, nc, protocol, job.get("workers", 2))
    elif family == "mmdet":
        from mmdet.apis import init_detector
        from mmdet_memory import install_memory_safe_rtmdet

        install_memory_safe_rtmdet()
        model = init_detector(str(run_dir / "effective_config.py"), str(checkpoint), device="cuda:0")
        model.float().eval()
        predictor = mmdet_predictor(model, model.cfg, job.get("workers", 2))
    else:
        from rfdetr import RFDETRSegNano

        owner = RFDETRSegNano(pretrain_weights=str(checkpoint),
                              num_classes=nc,
                              resolution=job["recipe"]["imgsz"],
                              positional_encoding_size=26,
                              device="cuda",
                              amp=False)
        model = owner.model.model.float().eval()
        predictor = rfdetr_predictor(model, job["recipe"]["imgsz"], owner.model.postprocess.num_select, protocol,
                                     job.get("workers", 2))
    with validation_state(model):
        metrics = collect_metrics(prepared / "coco/annotations/instances_val.json", prepared / "coco/images/val",
                                  predictor, protocol, profile_images=profile_images)
    if profile_images:
        save_json(output_dir / "evaluation_timing.json",
                  dict(metrics, runtime=runtime_status(), protocol=protocol, first_image_includes_warmup=True))
        return metrics
    if is_selected_checkpoint:
        # CPU FP32 checkpoint must reproduce the selected model; fail visibly rather than report a different best.
        if abs(metrics["mask_ap_50_95"] - selection["mask_ap_50_95"]) > 1e-5:
            raise RuntimeError("Saved best weights do not reproduce selection AP (tolerance 1e-5); job not completed")
    metrics.update(checkpoint=str(checkpoint),
                   params=sum(p.numel() for p in model.parameters()),
                   is_selected_checkpoint=is_selected_checkpoint,
                   checkpoint_role="selected" if is_selected_checkpoint else "custom",
                   selected_epoch=selection["epoch"] if is_selected_checkpoint else None,
                   weight_policy=selection["weight_policy"] if is_selected_checkpoint else None,
                   evaluation="original-val / common COCO mask AP maxDets100 / FP32 / same training predictor")
    save_json(
        output_dir / "evaluation_timing.json",
        dict(total_seconds=metrics["seconds"], images=metrics["images"], timing=metrics["timing"],
             per_image_timing=metrics["per_image_timing"], validation_engine=metrics["validation_engine"],
             runtime=runtime_status(), protocol=protocol, first_image_includes_warmup=True))
    save_json(output_dir / "val_common.json", metrics)
    return metrics
