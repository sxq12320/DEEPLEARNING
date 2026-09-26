"""Bound RTMDet-Ins v3.3.0 inference memory without dropping predictions.

Postprocessing follows OpenMMLab MMDetection v3.3.0 (Apache-2.0):
https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/rtmdet_ins_head.py
Only mask interpolation scheduling and final mask storage are changed.
"""

import math


def decode_masks_bounded(logits, stride, rescale, img_meta, threshold, chunk_limit=8, pixel_budget=4_000_000):
    """Preserve both official bilinear stages; store only final bool masks on CPU."""
    import torch
    import torch.nn.functional as functional

    if chunk_limit < 1 or pixel_budget < 1:
        raise ValueError("Mask chunk limits must be positive")
    height, width = logits.shape[-2:]
    intermediate = (height * stride, width * stride)
    expanded = intermediate
    if rescale:
        inverse = [1 / value for value in img_meta["scale_factor"]]
        # Preserve the pinned official implementation's exact size convention.
        expanded = (math.ceil(intermediate[0] * inverse[0]), math.ceil(intermediate[1] * inverse[1]))
        original = img_meta["ori_shape"][:2]
        output_shape = (min(expanded[0], original[0]), min(expanded[1], original[1]))
    else:
        output_shape = intermediate
    step = max(1, min(chunk_limit, pixel_budget // max(math.prod(intermediate), math.prod(expanded))))
    output = torch.empty((len(logits), *output_shape), dtype=torch.bool, device="cpu")
    for start in range(0, len(logits), step):
        chunk = functional.interpolate(logits[start:start + step].unsqueeze(0),
                                       scale_factor=stride, mode="bilinear")
        if rescale:
            chunk = functional.interpolate(chunk, size=list(expanded), mode="bilinear", align_corners=False)
            chunk = chunk[..., :original[0], :original[1]]
        output[start:start + step].copy_((chunk.sigmoid().squeeze(0) > threshold).cpu())
        del chunk
    return output


def _box_ops():
    from mmcv.ops import batched_nms
    from mmdet.structures.bbox import get_box_tensor, get_box_wh, scale_boxes

    return batched_nms, get_box_tensor, get_box_wh, scale_boxes


def decode_solo_masks_bounded(probabilities, stride, img_meta, threshold):
    """SOLOv2: resize probabilities, crop padding, then resize to original shape."""
    import torch
    import torch.nn.functional as functional

    original = img_meta["ori_shape"][:2]
    height, width = img_meta["img_shape"][:2]
    expanded = (probabilities.shape[-2] * stride, probabilities.shape[-1] * stride)
    step = max(1, min(8, 4_000_000 // max(math.prod(original), math.prod(expanded))))
    output = torch.empty((len(probabilities), *original), dtype=torch.bool, device="cpu")
    for start in range(0, len(probabilities), step):
        chunk = functional.interpolate(probabilities[start:start + step].unsqueeze(0), size=expanded,
                                       mode="bilinear", align_corners=False)[..., :height, :width]
        chunk = functional.interpolate(chunk, size=original, mode="bilinear", align_corners=False)
        output[start:start + step].copy_((chunk.squeeze(0) > threshold).cpu())
        del chunk
    return output


def _solo_ops():
    from mmengine.structures import InstanceData
    from mmdet.models.layers import mask_matrix_nms

    return InstanceData, mask_matrix_nms


def bounded_solo_predict(self, kernel_preds, cls_scores, mask_feats, img_meta, cfg=None):
    """Pinned SOLOv2 selection is unchanged; only the final two resizes are streamed."""
    import torch.nn.functional as functional

    InstanceData, matrix_nms = _solo_ops()

    def empty_results(scores):
        return InstanceData(scores=scores.new_ones(0), masks=scores.new_zeros(0, *img_meta["ori_shape"][:2]),
                            labels=scores.new_ones(0), bboxes=scores.new_zeros(0, 4))

    cfg = self.test_cfg if cfg is None else cfg
    assert len(kernel_preds) == len(cls_scores)
    selected = cls_scores > cfg.score_thr
    cls_scores = cls_scores[selected]
    if len(cls_scores) == 0:
        return empty_results(cls_scores)
    indices = selected.nonzero()
    labels = indices[:, 1]
    kernel_preds = kernel_preds[indices[:, 0]]
    intervals = labels.new_tensor(self.num_grids).pow(2).cumsum(0)
    strides = kernel_preds.new_ones(intervals[-1])
    strides[:intervals[0]] *= self.strides[0]
    for level in range(1, self.num_levels):
        strides[intervals[level - 1]:intervals[level]] *= self.strides[level]
    strides = strides[indices[:, 0]]
    kernels = kernel_preds.view(len(kernel_preds), -1, self.dynamic_conv_size, self.dynamic_conv_size)
    probabilities = functional.conv2d(mask_feats, kernels, stride=1).squeeze(0).sigmoid()
    masks = probabilities > cfg.mask_thr
    area = masks.sum((1, 2)).float()
    keep = area > strides
    if keep.sum() == 0:
        return empty_results(cls_scores)
    masks, probabilities, area = masks[keep], probabilities[keep], area[keep]
    cls_scores, labels = cls_scores[keep], labels[keep]
    cls_scores *= (probabilities * masks).sum((1, 2)) / area
    scores, labels, _, keep = matrix_nms(masks, labels, cls_scores, mask_area=area, nms_pre=cfg.nms_pre,
                                         max_num=cfg.max_per_img, kernel=cfg.kernel, sigma=cfg.sigma,
                                         filter_thr=cfg.filter_thr)
    if len(keep) == 0:
        return empty_results(cls_scores)
    masks = decode_solo_masks_bounded(probabilities[keep], self.mask_stride, img_meta, cfg.mask_thr)
    return InstanceData(masks=masks, labels=labels, scores=scores, bboxes=scores.new_zeros(len(scores), 4))


def bounded_rtmdet_postprocess(self, results, mask_feat, cfg, rescale=False, with_nms=True, img_meta=None):
    """Keep official box selection/order/scores; bound original-resolution mask temporaries."""
    import torch

    batched_nms, get_box_tensor, get_box_wh, scale_boxes = _box_ops()
    if rescale:
        assert img_meta.get("scale_factor") is not None
        results.bboxes = scale_boxes(results.bboxes, [1 / value for value in img_meta["scale_factor"]])
    if hasattr(results, "score_factors"):
        results.scores = results.scores * results.pop("score_factors")
    if cfg.get("min_bbox_size", -1) >= 0:
        width, height = get_box_wh(results.bboxes)
        valid = (width > cfg.min_bbox_size) & (height > cfg.min_bbox_size)
        if not valid.all():
            results = results[valid]
    assert with_nms, "with_nms must be True for RTMDet-Ins"
    if results.bboxes.numel() == 0:
        shape = img_meta["ori_shape"][:2] if rescale else img_meta["img_shape"][:2]
        results.masks = torch.zeros((len(results.bboxes), *shape), dtype=torch.bool, device="cpu")
        return results
    detections, keep = batched_nms(get_box_tensor(results.bboxes), results.scores, results.labels, cfg.nms)
    results = results[keep]
    results.scores = detections[:, -1]
    results = results[:cfg.max_per_img]
    logits = self._mask_predict_by_feat_single(mask_feat, results.kernels, results.priors)
    results.masks = decode_masks_bounded(logits, self.prior_generator.strides[0][0], rescale,
                                        img_meta, cfg.mask_thr_binary)
    return results


def install_memory_safe_rtmdet():
    """Process-local, pinned-version adapter; no site-packages files are edited."""
    import importlib.metadata
    from mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsHead
    from mmdet.models.dense_heads.solov2_head import SOLOV2Head

    if importlib.metadata.version("mmdet") != "3.3.0":
        raise RuntimeError("The verified bounded RTMDet adapter requires mmdet==3.3.0")
    RTMDetInsHead._bbox_mask_post_process = bounded_rtmdet_postprocess
    SOLOV2Head._predict_by_feat_single = bounded_solo_predict
    return dict(adapter="rtmdet_ins_solov2_bounded_masks_v1", final_mask_device="cpu",
                final_mask_dtype="bool", chunk_limit=8, expanded_pixel_budget=4_000_000,
                threshold_or_max_detections_changed=False)
