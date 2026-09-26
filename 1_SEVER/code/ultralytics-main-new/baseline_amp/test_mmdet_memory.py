"""Compare bounded decoding with pinned official postprocessing on CPU."""

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mmdet_memory


@pytest.mark.parametrize("rescale", [False, True])
@pytest.mark.parametrize("count", [1, 9, 17])
@pytest.mark.parametrize("threshold", [0.25, 0.5, 0.75])
def test_chunked_masks_equal_full_decode(rescale, count, threshold):
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(count, 7, 9, generator=generator)
    meta = dict(scale_factor=(0.61, 0.59), ori_shape=(87, 117))
    full = functional.interpolate(logits.unsqueeze(0), scale_factor=8, mode="bilinear")
    if rescale:
        full = functional.interpolate(full, size=[math.ceil(full.shape[-2] / 0.61),
                                                  math.ceil(full.shape[-1] / 0.59)],
                                      mode="bilinear", align_corners=False)[..., :87, :117]
    expected = full.sigmoid().squeeze(0) > threshold
    actual = mmdet_memory.decode_masks_bounded(logits, 8, rescale, meta, threshold,
                                               chunk_limit=3, pixel_budget=15_000)
    assert actual.device.type == "cpu" and actual.dtype == torch.bool
    assert torch.equal(actual, expected)


def test_every_interpolation_is_bounded(monkeypatch):
    calls = []
    original = functional.interpolate

    def recorded(tensor, *args, **kwargs):
        calls.append(tensor.shape[1])
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(functional, "interpolate", recorded)
    result = mmdet_memory.decode_masks_bounded(torch.zeros(23, 8, 8), 8, True,
                                              dict(scale_factor=(0.5, 0.5), ori_shape=(128, 128)),
                                              0.5, chunk_limit=8, pixel_budget=32768)
    assert max(calls) == 2 and len(result) == 23 and not result.any()


@pytest.mark.parametrize("rescale", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_official_postprocess_equivalence(monkeypatch, rescale, empty):
    pytest.importorskip("mmengine")
    from mmengine.config import ConfigDict
    from mmengine.structures import InstanceData

    spec = importlib.util.find_spec("mmdet")
    if spec is None:
        pytest.skip("Official mmdet source unavailable")
    path = Path(spec.origin).parent / "models/dense_heads/rtmdet_ins_head.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RTMDetInsHead")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_bbox_mask_post_process")
    method.decorator_list = []

    # Both paths use exactly the same controlled NMS oracle. This verifies
    # selection/order/score plumbing, not the already-compiled NMS implementation.
    def nms(boxes, scores, labels, cfg):
        keep = scores.argsort(descending=True)
        return torch.cat([boxes[keep], scores[keep, None] * 0.9], dim=1), keep

    def box_wh(boxes):
        return boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]

    def scale_boxes(boxes, factors):
        return boxes * boxes.new_tensor(factors * 2)

    ops = (nms, lambda boxes: boxes, box_wh, scale_boxes)
    monkeypatch.setattr(mmdet_memory, "_box_ops", lambda: ops)
    namespace = dict(torch=torch, math=math, F=functional, batched_nms=nms,
                     get_box_tensor=ops[1], get_box_wh=box_wh, scale_boxes=scale_boxes)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    count = 0 if empty else 5
    results = InstanceData(
        bboxes=torch.tensor([[0., 0., 8., 9.], [0., 0., 0., 0.], [1., 2., 7., 8.],
                             [2., 3., 7., 9.], [1., 1., 5., 6.]])[:count],
        scores=torch.tensor([.4, .9, .7, .6, .3])[:count], labels=torch.zeros(count, dtype=torch.long),
        kernels=torch.arange(count).reshape(-1, 1), priors=torch.zeros(count, 4),
        score_factors=torch.ones(count) * .8,
    )
    logits = torch.randn(5, 7, 9, generator=torch.Generator().manual_seed(1))
    head = SimpleNamespace(prior_generator=SimpleNamespace(strides=[(8, 8)]),
                           _mask_predict_by_feat_single=lambda feat, kernels, priors: logits[kernels[:, 0]])
    cfg = ConfigDict(min_bbox_size=0, nms=dict(type="nms"), max_per_img=3, mask_thr_binary=.5)
    meta = dict(scale_factor=(.61, .59), ori_shape=(87, 117), img_shape=(56, 72))
    expected = namespace["_bbox_mask_post_process"](head, results.clone(), None, cfg, rescale, True, meta)
    actual = mmdet_memory.bounded_rtmdet_postprocess(head, results.clone(), None, cfg, rescale, True, meta)
    assert set(actual.keys()) == set(expected.keys())
    for name in actual.keys():
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


@pytest.mark.parametrize("case", ["normal", "no_scores", "no_masks", "nms_empty"])
def test_solo_official_equivalence(monkeypatch, case):
    import runpy
    pytest.importorskip("mmengine")
    from mmengine.config import ConfigDict
    from mmengine.structures import InstanceData

    spec = importlib.util.find_spec("mmdet")
    if spec is None:
        pytest.skip("Official mmdet source unavailable")
    root = Path(spec.origin).parent
    path = root / "models/dense_heads/solov2_head.py"
    cls = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
               if isinstance(node, ast.ClassDef) and node.name == "SOLOV2Head")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_predict_by_feat_single")
    nms = runpy.run_path(str(root / "models/layers/matrix_nms.py"))["mask_matrix_nms"]
    namespace = dict(F=functional, InstanceData=InstanceData, mask_matrix_nms=nms)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    monkeypatch.setattr(mmdet_memory, "_solo_ops", lambda: (InstanceData, nms))
    head = SimpleNamespace(mask_stride=4, num_grids=[2], strides=[1], num_levels=1, dynamic_conv_size=1)
    scores = torch.tensor([[.4], [.8], [.7], [.6]])
    generator = torch.Generator().manual_seed(8)
    kernels = torch.randn(4, 2, generator=generator)
    features = torch.randn(1, 2, 7, 9, generator=generator)
    cfg = ConfigDict(score_thr=1 if case == "no_scores" else .001,
                     mask_thr=1 if case == "no_masks" else .5, nms_pre=10, max_per_img=3,
                     kernel="gaussian", sigma=2, filter_thr=1 if case == "nms_empty" else 0)
    meta = dict(img_shape=(25, 33), ori_shape=(101, 137))
    expected = namespace["_predict_by_feat_single"](head, kernels.clone(), scores.clone(), features, meta, cfg)
    actual = mmdet_memory.bounded_solo_predict(head, kernels.clone(), scores.clone(), features, meta, cfg)
    assert set(actual.keys()) == set(expected.keys())
    for name in actual.keys():
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name
