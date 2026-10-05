# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""J V2 loss: unchanged YOLO segmentation terms plus CenterNet-style small-fruit density supervision.

The density term is written into the fifth loss slot (``sem_loss`` in results.csv), which the standard
Segment head leaves at zero, so its trajectory is visible without changing any existing component.
"""

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8SegmentationLoss

SMALL_AREA = 32 * 32  # visible-mask area in CURRENT-view input pixels; same bucket boundary as the evaluator


def density_focal_loss(logits, target, alpha=2.0, beta=4.0):
    """Penalty-reduced pixel focal loss (CenterNet). Normalised by the number of exact peaks (>=1)."""
    logits, target = logits.float(), target.float()
    log_p, log_q = F.logsigmoid(logits), F.logsigmoid(-logits)
    p = log_p.exp()
    positive = target.eq(1).float()
    pos_loss = -(log_p * (1 - p).pow(alpha) * positive).sum()
    neg_loss = -(log_q * p.pow(alpha) * (1 - target).pow(beta) * (1 - positive)).sum()
    return (pos_loss + neg_loss) / positive.sum().clamp_min(1.0)


class J2SegmentationLoss(v8SegmentationLoss):
    """Standard assignment/box/mask/class terms; density added only when the head provides it."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.density_gain = float(getattr(model.model[-1], "density_gain", 0.0))

    def density_targets(self, masks, batch_idx, batch_size, grid, imgsz):
        """Gaussian centre heat of small visible instances on the density grid.

        Args:
            masks (torch.Tensor): Overlap index maps (B,h,w) or per-instance masks (N,h,w).
            batch_idx (torch.Tensor): Image index per instance (non-overlap mode).
            batch_size (int): Images in the batch.
            grid (tuple[int, int]): Density grid (gh, gw).
            imgsz (tuple[float, float]): Network input size (H, W) in pixels.

        Returns:
            (torch.Tensor): (B,1,gh,gw) heat in [0,1]; exactly 1 at each small instance's rounded centre.
        """
        gh, gw = grid
        heat = torch.zeros((batch_size, 1, gh, gw), device=masks.device)
        mh, mw = masks.shape[-2:]
        scale = (imgsz[0] / mh) * (imgsz[1] / mw)
        stride = imgsz[0] / gh
        ys = torch.arange(gh, device=masks.device, dtype=torch.float32)[None, :, None]
        xs = torch.arange(gw, device=masks.device, dtype=torch.float32)[None, None, :]
        for image in range(batch_size):
            instances = self.image_instance_masks(masks, batch_idx, image, (mh, mw))
            if not len(instances):
                continue
            area = instances.sum((1, 2))
            active = (area > 0) & (area * scale < SMALL_AREA)
            if not active.any():
                continue
            instances, area = instances[active], area[active]
            yy = torch.arange(mh, device=masks.device, dtype=torch.float32)
            xx = torch.arange(mw, device=masks.device, dtype=torch.float32)
            cy = (instances.sum(2) * yy).sum(1) / area
            cx = (instances.sum(1) * xx).sum(1) / area
            gy, gx = (cy + 0.5) * gh / mh - 0.5, (cx + 0.5) * gw / mw - 0.5
            sigma = ((area * scale).sqrt() / (4 * stride)).clamp(0.75, 1.5)[:, None, None]
            gaussian = torch.exp(-((ys - gy[:, None, None]) ** 2 + (xs - gx[:, None, None]) ** 2) / (2 * sigma**2))
            plane = gaussian.amax(0)
            plane[gy.round().clamp(0, gh - 1).long(), gx.round().clamp(0, gw - 1).long()] = 1.0
            heat[image, 0] = plane
        return heat

    def loss(self, preds, batch):
        total, components = super().loss(preds, batch)
        density = preds.get("j2_density") if isinstance(preds, dict) else None
        if density is None:
            return total, components
        if self.density_gain:
            imgsz = [float(v) * float(self.stride[0]) for v in preds["feats"][0].shape[2:]]
            target = self.density_targets(
                batch["masks"].to(density.device).float(),
                batch["batch_idx"].view(-1).to(density.device),
                density.shape[0],
                tuple(density.shape[-2:]),
                imgsz,
            )
            term = self.density_gain * density_focal_loss(density, target)
        else:
            term = density.sum() * 0.0
        zero = term * 0.0
        addition = torch.stack((zero, zero, zero, zero, term)).to(total.dtype)
        return total + addition * density.shape[0], components + addition.detach()
