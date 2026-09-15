# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Visible-mask foreground/background supervision, with bounded regional contrast.

Inspired by regional prototypes/hard queries in ReCo (ICLR 2022) and foreground
vs background distinction in ConDSeg (AAAI 2025). Independent two-class adaptation:
no queue, teacher, pairwise HW-by-HW affinity, second encoder or convex target.
"""

import torch
import torch.nn.functional as F

from .citrus_e_v3_loss import EV3SegmentationLoss


def visible_foreground(batch, size, overlap, device):
    """Union visible instance masks, without eroding tiny instances or filling holes."""
    masks = batch["masks"].to(device).float()
    if overlap:
        union = masks > 0
    else:
        b = batch["img"].shape[0]
        union = masks.new_zeros((b, *masks.shape[-2:]))
        if len(masks):
            union.index_add_(0, batch["batch_idx"].view(-1).long().to(device), (masks > 0).float())
        union = union > 0
    # Normally masks and detail both have stride 4. If downscaling, preserve any
    # foreground pixel for this AUXILIARY only; instance loss targets unchanged.
    union = union[:, None].float()
    if union.shape[-2:] != size:
        if size[0] <= union.shape[-2] and size[1] <= union.shape[-1]:
            union = F.adaptive_max_pool2d(union, size)
        else:
            union = F.interpolate(union, size, mode="nearest")
    return union


def regional_discrimination(logits, embedding, foreground, queries=128, temperature=0.2):
    """Balanced pixel BCE + two-prototype contrast on at most 256 hard queries.

    A five-pixel dilation defines local background, excluding ALL visible fruits.
    It supplies no leaf labels: background may include leaves, branches or sky.
    Empty/full-foreground images are supported without NaN or fake instances.
    """
    fg = foreground.bool()
    bg = ~fg
    ring = (F.max_pool2d(foreground, 5, 1, 2) > 0) & bg
    pixel = F.binary_cross_entropy_with_logits(logits, foreground, reduction="none")
    dims = (1, 2, 3)
    # Equal foreground/background contribution per image, rather than a huge
    # background majority. Missing regions contribute zero, not 0/0.
    fg_loss = (pixel * fg).sum(dims) / fg.sum(dims).clamp_min(1)
    bg_loss = (pixel * bg).sum(dims) / bg.sum(dims).clamp_min(1)
    ring_loss = (pixel * ring).sum(dims) / ring.sum(dims).clamp_min(1)
    balanced = (fg_loss + bg_loss + 0.5 * ring_loss).mean()
    rep = F.normalize(embedding.float(), dim=1).permute(0, 2, 3, 1).reshape(-1, embedding.shape[1])
    positive = fg.flatten().nonzero(as_tuple=False).flatten()
    negative = ring.flatten().nonzero(as_tuple=False).flatten()
    if not len(negative):
        negative = bg.flatten().nonzero(as_tuple=False).flatten()
    contrast = embedding.sum() * 0
    if len(positive) and len(negative):
        # Detached class means are targets, as in regional contrast. Deterministic
        # hard-query selection does not consume augmentation/initialization RNG.
        prototypes = F.normalize(torch.stack((rep[negative].mean(0), rep[positive].mean(0))).detach(), dim=1)
        score = logits.detach().flatten()
        p = positive[torch.topk(-score[positive], min(queries, len(positive))).indices]
        n = negative[torch.topk(score[negative], min(queries, len(negative))).indices]
        query = torch.cat((p, n))
        labels = torch.cat((torch.ones_like(p), torch.zeros_like(n)))
        contrast = F.cross_entropy(rep[query] @ prototypes.T / temperature, labels)
    return balanced + 0.1 * contrast, balanced.detach(), contrast.detach()


class EV5SegmentationLoss(EV3SegmentationLoss):
    """Keep TAL, box/mask losses and IoU-quality supervision unchanged."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.region_gain = model.model[-1].region_gain
        self.last_region = None

    def loss(self, preds, batch):
        total, components = super().loss(preds, batch)
        # Validation inference omits these TRAINING-ONLY outputs on purpose.
        if not self.region_gain or "ev5_region_logits" not in preds:
            return total, components
        logits = preds["ev5_region_logits"]
        target = visible_foreground(batch, logits.shape[-2:], self.overlap, logits.device)
        region, bce, contrast = regional_discrimination(logits, preds["ev5_region_embedding"], target)
        self.last_region = dict(bce=bce, contrast=contrast)
        q = self.region_gain * region
        addition = torch.stack((q * 0, q * 0, q * 0, q * 0, q))
        return total + addition * logits.shape[0], components + addition.detach()
