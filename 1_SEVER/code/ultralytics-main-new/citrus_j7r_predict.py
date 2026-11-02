"""Native prediction for explicit J7R packed inputs; no model-side pending-view state."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from citrus_j7r_data import j7r_model_spec, pack_j7r_images
from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.segment import SegmentationPredictor


class J7RPredictor(SegmentationPredictor):
    """Read original RGB images or explicitly packed tensors, never synthesize missing raw evidence."""

    def _check_options(self):
        if self.args.augment or self.args.compile or self.args.embed:
            raise ValueError("J7R prediction does not support packed-input TTA, compile or embedding-only output")
        if getattr(self.model, "format", "pt") != "pt":
            raise ValueError("J7R export is not verified; prediction requires native PyTorch weights")
        return j7r_model_spec(self.model)

    def preprocess(self, images):
        spec = self._check_options()
        dtype = torch.float16 if self.model.fp16 else torch.float32
        if isinstance(images, torch.Tensor):
            if images.ndim != 4 or images.shape[1] != 15 or not images.is_floating_point():
                raise ValueError("J7R tensor prediction requires explicitly normalized BCHW packed15, not RGB3")
            if any(value % 32 for value in images.shape[-2:]):
                raise ValueError("J7R packed spatial dimensions must be multiples of32")
            if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                raise ValueError("J7R packed tensors must contain finite normalized pixels in [0,1]")
            return images.to(device=self.device, dtype=dtype)
        if not isinstance(images, (list, tuple)) or not images:
            raise ValueError("J7R image prediction requires a nonempty list of original uint8 BGR images")
        same_shapes = len({image.shape for image in images}) == 1
        letterbox = LetterBox(self.imgsz, auto=same_shapes and self.args.rect, stride=self.model.stride)
        lows, highs = [], []
        for raw in images:
            if not isinstance(raw, np.ndarray) or raw.ndim != 3 or raw.shape[2] != 3 or raw.dtype != np.uint8:
                raise ValueError("J7R source files remain original HWC uint8 BGR3 images")
            params = letterbox.get_params({"img": raw})
            low = letterbox.apply_image({"img": raw}, params)["img"]
            lows.append(np.ascontiguousarray(low[..., ::-1].transpose(2, 0, 1)))
            if spec["observation"] == "raw":
                height, width = low.shape[:2]
                content_width, content_height = params["new_unpad"]
                left, top = params["left"], params["top"]
                high = np.full((height * 2, width * 2, 3), 114, dtype=np.uint8)
                high[top * 2:(top + content_height) * 2, left * 2:(left + content_width) * 2] = cv2.resize(
                    raw, (content_width * 2, content_height * 2), interpolation=cv2.INTER_LINEAR)
                highs.append(np.ascontiguousarray(high[..., ::-1].transpose(2, 0, 1)))
        global_rgb = torch.from_numpy(np.stack(lows)).to(device=self.device, dtype=dtype) / 255.0
        high_rgb = None
        if highs:
            high_rgb = torch.from_numpy(np.stack(highs)).to(device=self.device)
        return pack_j7r_images(global_rgb, high_rgb, spec["observation"])

    def postprocess(self, preds, img, orig_imgs):
        # Tensor-source Results must display ordinary RGB, not the12 phase channels.
        if isinstance(orig_imgs, torch.Tensor):
            if orig_imgs.ndim != 4 or orig_imgs.shape[1] != 15:
                raise ValueError("J7R tensor-source results require a packed15 original input")
            orig_imgs = orig_imgs[:, :3]
        return super().postprocess(preds, img, orig_imgs)
