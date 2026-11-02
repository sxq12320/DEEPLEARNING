"""Explicit RGB/high packing over the tested J6 paired-image geometry.

The dataset remains sensor-RGB with the original canonical-640 EV6 labels.
Only the model tensor is packed15. ``upsampled`` is an explicit degraded
observation control, never a substitute for raw high pixels in the raw arm.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from citrus_j6_data import J6Dataset, aligned_high_view, build_j6_dataset


J7R_INPUT_PROTOCOL = "packed_rgb_hr_2x_v1"
J7R_OBSERVATIONS = frozenset(("raw", "upsampled", "global"))
J7R_DATA_POLICIES = frozenset(("paired1280", "native640"))


def j7r_model_spec(model):
    """Find and validate the packed-input YAML on a raw/EMA/DDP/backend model."""
    pending, visited = [model], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        yaml = current if isinstance(current, dict) else getattr(current, "yaml", None)
        if isinstance(yaml, dict) and yaml.get("citrus_input_protocol") == J7R_INPUT_PROTOCOL:
            observation = yaml.get("citrus_observation")
            if yaml.get("channels") != 15 or observation not in J7R_OBSERVATIONS:
                raise ValueError("J7R requires YAML channels=15 and an explicit raw/upsampled/global observation")
            policy = yaml.get("citrus_data_policy", "paired1280")
            if policy not in J7R_DATA_POLICIES or (policy == "native640" and observation != "global"):
                raise ValueError("native640 is the global-only loader control; raw/upsampled require paired1280")
            spec = {"protocol": J7R_INPUT_PROTOCOL, "channels": 15, "observation": observation}
            if "citrus_data_policy" in yaml:
                spec["data_policy"] = policy
            return spec
        for attribute in ("backend", "model", "module", "ema", "_orig_mod"):
            child = getattr(current, attribute, None)
            if child is not current:
                pending.append(child)
    raise TypeError("This adapter requires a J7R packed_rgb_hr_2x_v1 model, not an RGB-only historical model")


def pack_j7r_images(global_rgb, high_rgb, observation="raw"):
    """Pack normalized global RGB and explicit 2x high evidence into one tensor.

    Raw uint8 high pixels are normalized once. Floating high tensors are assumed
    already normalized, exactly as in prepare_j6_batch. Geometry and dtype/device
    are checked without per-batch CPU/GPU synchronization over all pixels.
    """
    if observation not in J7R_OBSERVATIONS:
        raise ValueError("Unknown J7R observation: {}".format(observation))
    if not isinstance(global_rgb, torch.Tensor) or global_rgb.ndim != 4 or global_rgb.shape[1] != 3:
        raise ValueError("J7R packing requires BCHW global RGB3, not an already-packed tensor")
    if not global_rgb.is_floating_point() or min(global_rgb.shape[-2:]) < 1:
        raise ValueError("Global RGB must already be normalized floating point")
    expected = (global_rgb.shape[0], 3, global_rgb.shape[-2] * 2, global_rgb.shape[-1] * 2)
    if observation == "raw":
        if not isinstance(high_rgb, torch.Tensor) or tuple(high_rgb.shape) != expected:
            raise ValueError("Raw J7R requires a paired high_img with shape {}".format(expected))
        if high_rgb.device != global_rgb.device:
            raise ValueError("High/global images must share a device before packing")
        if high_rgb.dtype == torch.uint8:
            high_rgb = high_rgb.to(global_rgb.dtype) / 255.0
        elif high_rgb.is_floating_point():
            high_rgb = high_rgb.to(global_rgb.dtype)
        else:
            raise ValueError("Raw high_img must be uint8 or normalized floating point")
    elif observation == "upsampled":
        # Intentionally weaker information; the YAML and audit identify this arm.
        high_rgb = F.interpolate(global_rgb, scale_factor=2, mode="bilinear", align_corners=False)
    else:
        # The global-only graph must not read this zero payload as real evidence.
        high_rgb = global_rgb.new_zeros(expected)
    return torch.cat((global_rgb, F.pixel_unshuffle(high_rgb, 2)), dim=1)


def pack_j7r_batch(batch, observation="raw"):
    """Replace only img after ordinary normalization; release the redundant high tensor."""
    batch["img"] = pack_j7r_images(batch["img"], batch.get("high_img"), observation)
    batch.pop("high_img", None)
    return batch


def build_j7r_dataset(cfg, img_path, batch, data, mode="train", stride=32, rect=None, high_cache_mb=128,
                      data_policy="paired1280"):
    """Reuse J6's one-augmentation/canonical-GT contract without changing sensor channels."""
    if int(data.get("channels", 3)) != 3:
        raise ValueError("J7R datasets are RGB3; model packing must not change data.yaml channels")
    if data_policy not in J7R_DATA_POLICIES:
        raise ValueError("Unknown J7R data policy: {}".format(data_policy))
    if mode not in {"train", "val", "test"}:
        raise ValueError("J7R mode must be train, val, or test")
    if data_policy == "native640":
        from ultralytics.data.build import build_yolo_dataset

        use_rect = mode != "train" if rect is None else bool(rect)
        return build_yolo_dataset(cfg, img_path, batch, data, mode, rect=use_rect, stride=stride)
    return build_j6_dataset(cfg, img_path, batch, data, mode, stride, rect, high_cache_mb)


__all__ = ("J6Dataset", "aligned_high_view", "J7R_INPUT_PROTOCOL", "J7R_OBSERVATIONS", "j7r_model_spec",
           "pack_j7r_images", "pack_j7r_batch", "build_j7r_dataset")
