"""Independent scene--object citrus instance segmenter inside the Ultralytics I_V4 experiment series.

The only YOLO dependency is the outer trainer/validator interface. This model has its own
encoder, object queries, masks and set loss; it has no Detect/Segment, PAN, TAL or DFL.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Conv(nn.Sequential):
    def __init__(self, c1, c2, stride=1):
        super().__init__(nn.Conv2d(c1, c2, 3, stride, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU())


class _Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.pw(self.act(self.bn(self.dw(x))))


class CitrusNativeSegment(nn.Module):
    """Scene-first, set-prediction segmenter with optional evidence and bidirectional review.

    Args are deliberately fixed-width: no implicit YOLO width/depth scaling is applied.
    `evidence=False` uses learned object slots only; otherwise image-driven slots are
    gathered at the top evidence locations and optional fallback slots are appended.
    """

    def __init__(self, c1=3, width=40, evidence=True, scene=True, fallback=16, review=True, edge=False):
        super().__init__()
        if c1 != 3:
            raise ValueError("CitrusNativeSegment expects RGB images")
        self.nc = 1
        self.nm = 64 + int(fallback) if evidence else 64
        self.evidence_enabled = bool(evidence)
        self.scene_enabled = bool(scene)
        self.review_enabled = bool(review)
        self.edge_enabled = bool(edge)
        c2, c4, c8, c16 = 24, int(width), int(width) * 2, int(width) * 3
        self.stem = _Conv(3, c2, 2)
        self.stage2 = nn.Sequential(_Block(c2), _Block(c2))
        self.down4 = _Conv(c2, c4, 2)
        self.stage4 = nn.Sequential(_Block(c4), _Block(c4))
        self.down8 = _Conv(c4, c8, 2)
        self.stage8 = nn.Sequential(_Block(c8), _Block(c8))
        self.down16 = _Conv(c8, c16, 2)
        self.stage16 = _Block(c16)
        self.context = nn.Conv2d(c16, c8, 1)
        self.pixel = nn.Conv2d(c4, 64, 1)
        self.position = nn.Conv2d(10, 64, 1, bias=False)
        self.sem4 = nn.Conv2d(c8, c4, 1)
        self.memory = nn.Conv2d(c8, 64, 1)
        self.evidence = nn.Conv2d(c4, 1, 1)
        self.scene = nn.Conv2d(c4, 2, 1)
        self.query_seed = nn.Linear(c4, 64)
        learned = self.nm if not evidence else int(fallback)
        self.fallback = nn.Embedding(max(learned, 1), 64)
        self.query_read = nn.MultiheadAttention(64, 4, batch_first=True)
        self.query_norm = nn.LayerNorm(64)
        self.query_update = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 64))
        self.classifier = nn.Linear(64, 1)
        self.write = nn.Conv2d(64, c4, 1, bias=False)
        self.write_gain = nn.Parameter(torch.zeros(()))
        self.detail = nn.Conv2d(c2, 64, 3, 2, 1)
        self.detail_gain = nn.Parameter(torch.zeros(()))
        if self.edge_enabled:
            sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
            self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
            self.register_buffer("sobel_y", sobel_x.T.contiguous().view(1, 1, 3, 3))
            self.edge_project = nn.Conv2d(1, c2, 1, bias=False)
            self.edge_gain = nn.Parameter(torch.zeros(()))
        nn.init.constant_(self.classifier.bias, -3.0)

    @staticmethod
    def _coordinates(h, w, reference):
        y = torch.linspace(-1, 1, h, device=reference.device, dtype=reference.dtype)
        x = torch.linspace(-1, 1, w, device=reference.device, dtype=reference.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy, (math.pi * xx).sin(), (math.pi * xx).cos(),
                            (math.pi * yy).sin(), (math.pi * yy).cos(),
                            (2 * math.pi * xx).sin(), (2 * math.pi * xx).cos(),
                            (2 * math.pi * yy).sin(), (2 * math.pi * yy).cos()), 0).unsqueeze(0)

    def _seed_queries(self, features, evidence, position):
        b, c, h, w = features.shape
        if self.evidence_enabled:
            # Local NMS prevents all image-driven slots from collapsing onto one broad peak.
            peak = F.max_pool2d(evidence, 3, 1, 1)
            values = evidence.masked_fill(evidence < peak, -1e4).flatten(2)
            indices = values.topk(min(64, h * w), dim=-1).indices.expand(-1, c, -1)
            slots = features.flatten(2).gather(2, indices).transpose(1, 2)
            locations = position.expand(b, -1, -1, -1).flatten(2)
            locations = locations.gather(2, indices[:, :1].expand(-1, 64, -1)).transpose(1, 2)
            slots = self.query_seed(slots) + locations
            if self.nm > 64:
                learned = self.fallback.weight[: self.nm - 64].unsqueeze(0).expand(b, -1, -1)
                slots = torch.cat((slots, learned), 1)
            return slots
        return self.fallback.weight[:64].unsqueeze(0).expand(b, -1, -1)

    @staticmethod
    def _boxes_from_masks(logits, image_size):
        # Derived only at inference. No box head or box regression is used for learning.
        b, q, h, w = logits.shape
        occupied = logits.sigmoid() > 0.5
        yy = torch.arange(h, device=logits.device).view(1, 1, h, 1)
        xx = torch.arange(w, device=logits.device).view(1, 1, 1, w)
        x1 = torch.where(occupied, xx, w).flatten(2).amin(-1).float()
        x2 = torch.where(occupied, xx, -1).flatten(2).amax(-1).float()
        y1 = torch.where(occupied, yy, h).flatten(2).amin(-1).float()
        y2 = torch.where(occupied, yy, -1).flatten(2).amax(-1).float()
        # Empty masks remain valid but small; their low existence scores are filtered.
        x1 = torch.where(x1 == w, torch.full_like(x1, w / 2 - 1), x1)
        x2 = torch.where(x2 < 0, torch.full_like(x2, w / 2 + 1), x2)
        y1 = torch.where(y1 == h, torch.full_like(y1, h / 2 - 1), y1)
        y2 = torch.where(y2 < 0, torch.full_like(y2, h / 2 + 1), y2)
        x1, x2 = (x1 - 0.5).clamp(0, w), (x2 + 0.5).clamp(0, w)
        y1, y2 = (y1 - 0.5).clamp(0, h), (y2 + 0.5).clamp(0, h)
        scale_x, scale_y = image_size[1] / w, image_size[0] / h
        return torch.stack(((x1 + x2) * scale_x / 2, (y1 + y2) * scale_y / 2,
                            (x2 - x1) * scale_x, (y2 - y1) * scale_y), 1)

    def forward(self, image):
        f2 = self.stage2(self.stem(image))
        if self.edge_enabled:
            luminance = (image[:, :1] * 0.299 + image[:, 1:2] * 0.587 + image[:, 2:3] * 0.114)
            gx = F.conv2d(luminance, self.sobel_x, stride=2, padding=1)
            gy = F.conv2d(luminance, self.sobel_y, stride=2, padding=1)
            edge = (gx.square() + gy.square() + 1e-6).sqrt()
            edge = edge / (F.avg_pool2d(edge, 9, 1, 4) + 0.05)
            f2 = f2 + self.edge_gain.tanh() * self.edge_project(edge)
        f4 = self.stage4(self.down4(f2))
        f8 = self.stage8(self.down8(f4))
        f16 = self.stage16(self.down16(f8))
        f8 = f8 + F.interpolate(self.context(f16), size=f8.shape[-2:], mode="nearest")
        pixels = f4 + F.interpolate(self.sem4(f8), size=f4.shape[-2:], mode="nearest")
        evidence = self.evidence(pixels)
        scene = self.scene(pixels)
        position4 = self.position(self._coordinates(*pixels.shape[-2:], pixels))
        queries = self._seed_queries(pixels, evidence, position4)
        memory8 = self.memory(f8) + self.position(self._coordinates(*f8.shape[-2:], f8))
        memory = F.adaptive_avg_pool2d(memory8, (20, 20)).flatten(2).transpose(1, 2)
        read = self.query_read(queries, memory, memory, need_weights=False)[0]
        queries = self.query_norm(queries + read)
        if self.scene_enabled:
            # A soft prompt, not a hard foreground gate: false-negative pixels stay accessible.
            pixels = pixels * (1.0 + 0.2 * scene[:, :1].sigmoid())
        if self.review_enabled:
            coarse = torch.einsum("bqc,bchw->bqhw", queries, memory8) / 8.0
            attention = coarse.flatten(2).softmax(-1)
            pooled = torch.einsum("bqn,bcn->bqc", attention, memory8.flatten(2))
            queries = self.query_norm(queries + self.query_update(torch.cat((queries, pooled), -1)))
            # Bounded, zero-initialized object-to-scene feedback. Original pixels are always preserved.
            scene_write = torch.einsum("bqn,bqc->bcn", attention, queries).view(
                image.shape[0], 64, *f8.shape[-2:]
            )
            pixels = pixels + self.write_gain.tanh() * F.interpolate(
                self.write(scene_write), size=pixels.shape[-2:], mode="nearest"
            )
        pixel_embedding = self.pixel(pixels) + position4 + self.detail_gain.tanh() * self.detail(f2)
        logits = torch.einsum("bqc,bchw->bqhw", queries, pixel_embedding) / 8.0
        scores = self.classifier(queries).squeeze(-1)
        raw = {"masks": logits, "scores": scores, "scene": scene, "evidence": evidence}
        if self.training:
            return raw
        boxes = self._boxes_from_masks(logits, image.shape[-2:])
        identity = torch.eye(self.nm, device=image.device, dtype=logits.dtype).unsqueeze(0)
        identity = identity.expand(image.shape[0], -1, -1)
        # Adapter only: each query's actual mask is one prototype; YOLO's validator reconstructs it.
        prediction = torch.cat((boxes, scores.sigmoid().unsqueeze(1), identity), 1)
        return prediction, logits, raw
