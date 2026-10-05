"""Inference-only performance fixes shared by the official baseline workers.

PyTorch 2.1 F.interpolate dispatches CUDA bilinear resizing to a slow Python
decomposition when deterministic algorithms are enabled, even under no_grad.
The restriction concerns the native kernel's BACKWARD, not its forward. Route
only inference calls to the same native ATen forward used by ordinary PyTorch.
Do not change training determinism, precision, masks, or detection thresholds.

Source: https://github.com/pytorch/pytorch/blob/v2.1.0/torch/nn/functional.py
"""

from __future__ import annotations

import math
import time

_ORIGINAL_INTERPOLATE = None
_NATIVE_CALLS = 0


def _native_bilinear(input, size, scale_factor, align_corners, recompute_scale_factor):
    """Map supported F.interpolate arguments to ATen without changing sampling coordinates."""
    import torch

    if (size is None) == (scale_factor is None):
        return None  # Let the original function raise its standard argument error.
    scales = None
    if size is not None:
        if recompute_scale_factor:
            return None
        output = [size, size] if isinstance(size, int) else list(size)
        if len(output) != 2 or any(not isinstance(value, int) or value < 1 for value in output):
            return None
    else:
        scales = [scale_factor, scale_factor] if isinstance(scale_factor, (int, float)) else list(scale_factor)
        if len(scales) != 2 or any(not isinstance(value, (int, float)) or value <= 0 for value in scales):
            return None
        output = [math.floor(input.shape[i + 2] * scales[i]) for i in range(2)]
        if min(output) < 1:
            return None
        if recompute_scale_factor:
            scales = None
    return torch.ops.aten.upsample_bilinear2d.default(
        input,
        output,
        bool(align_corners),
        None if scales is None else float(scales[0]),
        None if scales is None else float(scales[1]),
    )


def inference_interpolate(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
    antialias=False,
):
    """Bypass only the inference CUDA bilinear decomposition; delegate everything else."""
    import torch

    global _NATIVE_CALLS
    if (
        not torch.is_grad_enabled()
        and input.is_cuda
        and input.ndim == 4
        and mode == "bilinear"
        and not antialias
        and torch.are_deterministic_algorithms_enabled()
        and not torch.jit.is_scripting()
        and not torch.jit.is_tracing()
        and not torch.overrides.has_torch_function_variadic(input)
    ):
        result = _native_bilinear(input, size, scale_factor, align_corners, recompute_scale_factor)
        if result is not None:
            _NATIVE_CALLS += 1
            return result
    return _ORIGINAL_INTERPOLATE(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
        antialias=antialias,
    )


def install_inference_runtime():
    """Install once, in this worker only; never edit torch/site-packages or global flags."""
    import torch.nn.functional as functional

    global _ORIGINAL_INTERPOLATE
    if functional.interpolate is not inference_interpolate:
        _ORIGINAL_INTERPOLATE = functional.interpolate
        functional.interpolate = inference_interpolate
    return runtime_status()


def runtime_status():
    """Small auditable record; this count is not a measured speedup."""
    return dict(
        adapter="native_bilinear_inference_v1",
        native_bilinear_calls=_NATIVE_CALLS,
        gradient_enabled_dispatch_changed=False,
        deterministic_flags_changed=False,
        threshold_or_resolution_changed=False,
    )


def decode_yolo_masks_bounded(decode, protos, coefficients, boxes, shape, pixel_budget=4_000_000):
    """Keep official native mask decoding but release each CUDA chunk after CPU transfer."""
    import torch

    step = max(1, min(16, pixel_budget // (shape[0] * shape[1])))
    output = torch.empty((len(coefficients), *shape), dtype=torch.uint8, device="cpu")
    for start in range(0, len(coefficients), step):
        output[start : start + step].copy_(
            decode(protos, coefficients[start : start + step], boxes[start : start + step], shape).cpu()
        )
    return output


def make_validation_timing_hook(run_dir):
    """Measure actual MMEngine validation wall time, including RLE/metrics, every epoch."""
    import torch
    from mmengine.hooks import Hook
    from common import save_json

    class ValidationTiming(Hook):
        def before_val_epoch(self, runner):
            torch.cuda.synchronize()
            self.started = self.previous = time.perf_counter()
            self.predict_seconds = self.input_wait_seconds = 0.0
            self.batches = 0
            self.native_before = _NATIVE_CALLS

        def before_val_iter(self, runner, batch_idx, data_batch=None):
            self.iter_started = time.perf_counter()
            self.input_wait_seconds += self.iter_started - self.previous

        def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            torch.cuda.synchronize()
            self.previous = time.perf_counter()
            self.predict_seconds += self.previous - self.iter_started
            self.batches += 1

        def after_val_epoch(self, runner, metrics=None):
            torch.cuda.synchronize()
            total = time.perf_counter() - self.started
            record = dict(
                epoch=runner.epoch,
                batches=self.batches,
                total_seconds=total,
                input_wait_seconds=self.input_wait_seconds,
                predict_transfer_rle_seconds=self.predict_seconds,
                metrics_and_epoch_hooks_seconds=max(0.0, total - self.input_wait_seconds - self.predict_seconds),
                native_bilinear_calls=_NATIVE_CALLS - self.native_before,
            )
            save_json(run_dir / "validation_timing.json", record)
            runner.logger.info("VALIDATION TIMING: " + str(record))

    return ValidationTiming()
