"""E V7 contracts: four factors on the phase-fine base, official API, backward, P2 graph."""

import importlib.util
from copy import deepcopy

import pytest
import torch
import yaml

from citrus_e_v7_suite import BOUNDARY_GAIN, FACTORS, NAMES, NEIGHBOR_GAIN, ROOT, RUN_OVERRIDES, YAML_DIR
from scripts.generate_citrus_e_v7_yaml import configs
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def batch(empty=False, mask_ratio=4):
    size = 128 // mask_ratio
    masks = torch.zeros(2, size, size)
    masks[0, 12 // mask_ratio:108 // mask_ratio, 12 // mask_ratio:108 // mask_ratio] = 1
    masks[1, 20 // mask_ratio:116 // mask_ratio, 20 // mask_ratio:116 // mask_ratio] = 1
    return dict(
        img=torch.rand(2, 3, 128, 128),
        batch_idx=torch.tensor([0.0, 1.0])[: 0 if empty else 2],
        cls=torch.zeros(0 if empty else 2, 1),
        bboxes=torch.tensor([[0.46875, 0.46875, 0.75, 0.75], [0.53125, 0.53125, 0.75, 0.75]])[: 0 if empty else 2],
        masks=masks * (not empty),
    )


@pytest.mark.parametrize("name,factors", zip(NAMES, FACTORS))
@pytest.mark.parametrize("empty", [False, True])
def test_official_api_backward_and_inference(name, factors, empty):
    assert YOLO(str(YAML_DIR / (name + ".yaml")), verbose=False).task == "segment"
    model = SegmentationModel(YAML_DIR / (name + ".yaml"), nc=1, verbose=False)
    args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    args.overlap_mask = True
    args.mask_ratio = RUN_OVERRIDES[name]["mask_ratio"]
    model.args = args
    loss, items = model.loss(batch(empty, args.mask_ratio))
    assert torch.isfinite(loss).all() and len(items) == 5
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    head = model.model[-1]
    if factors[1] and not empty:
        assert head.deform.weight.weight.grad.abs().sum() > 0
    if factors[2] and not empty:
        assert head.enhance[0].out.weight.grad.abs().sum() > 0
    assert get_flops(model, 640) < 15.0
    with torch.inference_mode():
        model.eval()
        x = torch.rand(1, 3, 128, 160)
        out = model(x)
        assert "ev5_region_logits" not in out[1]
        fused = deepcopy(model).fuse(verbose=False)(x)
        torch.testing.assert_close(out[0][0], fused[0][0], rtol=0.002, atol=0.02)
        torch.testing.assert_close(out[0][1], fused[0][1], rtol=0.002, atol=0.02)


def test_control_exact_parity_with_v6_phase():
    """V7_00 is the V6_08 phase-fine arm under the EV7 class, bit-exact."""
    old = SegmentationModel(
        ROOT / "0_orange_yaml/E_V6_series/V6_08_phase.yaml", nc=1, verbose=False
    ).eval()
    new = SegmentationModel(YAML_DIR / (NAMES[0] + ".yaml"), nc=1, verbose=False).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    with torch.inference_mode():
        x = torch.rand(1, 3, 128, 128)
        a, b = old(x), new(x)
        torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
        torch.testing.assert_close(a[0][1], b[0][1], rtol=0, atol=0)


@pytest.mark.parametrize("name,factors", zip(NAMES, FACTORS))
def test_scale_count_strides_and_geometry_flags(name, factors):
    model = SegmentationModel(YAML_DIR / (name + ".yaml"), nc=1, verbose=False).eval()
    head = model.model[-1]
    p2, _deform, _pmce, boundary = factors
    assert head.nl == (4 if p2 else 3)
    assert head.stride.tolist() == ([8.0, 16.0, 32.0, 4.0] if p2 else [8.0, 16.0, 32.0])
    assert len(head.cv2) == head.nl and len(head.cv3) == head.nl and len(head.cv4) == head.nl
    assert len(head.quality_predictor) == head.nl
    assert head.boundary_gain == (BOUNDARY_GAIN if boundary else 0.0)
    assert head.neighbor_gain == (NEIGHBOR_GAIN if boundary else 0.0)
    with torch.inference_mode():
        out = model(torch.rand(1, 3, 256, 256))
    assert out[0][0].shape[1] == 4 + head.nc + head.nm
    # 256-input anchor grids: P3+P4+P5 = 32^2+16^2+8^2; the P2 arm adds 64^2.
    assert out[0][0].shape[-1] == (5440 if p2 else 1344)
    assert out[0][1].shape[-2:] == (128, 128)  # phase-fine proto at stride 2 on every arm


def test_p2_towers_train_from_scratch_and_inherited_keep_indices():
    """P2 appends AFTER the inherited scales so cv2.0-2/cv3.0-2/cv4.0-2 stay aligned."""
    control = SegmentationModel(YAML_DIR / (NAMES[0] + ".yaml"), nc=1, verbose=False)
    p2 = SegmentationModel(YAML_DIR / (NAMES[1] + ".yaml"), nc=1, verbose=False)
    cs, ps = control.model[-1], p2.model[-1]
    assert len(ps.cv2) == 4 and ps.cv2[-1][0].conv.in_channels == 32
    csd, psd = cs.state_dict(), ps.state_dict()
    # Key+shape identity is what keeps pretrained P3/P4/P5 tensors aligned to
    # their semantic scale; freshly seeded weights need not be equal.
    for prefix in ("cv2.0", "cv2.1", "cv2.2", "cv3.0", "cv3.1", "cv3.2", "cv4.0", "cv4.1", "cv4.2",
                   "quality_predictor.0", "quality_predictor.1", "quality_predictor.2"):
        for key, value in csd.items():
            if key.startswith(prefix + "."):
                assert key in psd and value.shape == psd[key].shape, key
    for prefix in ("cv2.3", "cv3.3", "cv4.3", "quality_predictor.3"):
        assert any(k.startswith(prefix + ".") for k in psd)
        assert not any(k.startswith(prefix + ".") for k in csd)


def test_boundary_arm_runs_geometry_loss():
    model = SegmentationModel(YAML_DIR / (NAMES[4] + ".yaml"), nc=1, verbose=False)
    args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    args.overlap_mask = True
    args.mask_ratio = 2
    model.args = args
    loss, _ = model.loss(batch(mask_ratio=2))
    assert torch.isfinite(loss).all()
    geometry = model.criterion.last_geometry
    assert geometry is not None and geometry.shape == (2,)
    assert torch.isfinite(geometry).all()


@pytest.mark.parametrize("name", NAMES)
def test_pretraining_and_checkpoint(name, tmp_path):
    model = YOLO(str(YAML_DIR / (name + ".yaml")), verbose=False).load(str(ROOT / "yolo11n-seg.pt"))
    source = YOLO(str(ROOT / "yolo11n-seg.pt"), verbose=False).model.state_dict()
    head_index = len(model.model.model) - 1
    for key in ("model.2.cv1.conv.weight", "model.4.cv2.conv.weight"):
        torch.testing.assert_close(model.model.state_dict()[key], source[key], rtol=0, atol=0)
    # The head keeps inherited towers aligned even when the graph shifted.
    torch.testing.assert_close(
        model.model.state_dict()[f"model.{head_index}.cv4.0.0.conv.weight"],
        source["model.23.cv4.0.0.conv.weight"],
        rtol=0,
        atol=0,
    )
    path = tmp_path / "model.pt"
    model.save(str(path))
    restored = YOLO(str(path), verbose=False)
    assert type(restored.model.model[-1]) is type(model.model.model[-1])
    assert restored.model.model[-1].p2_head == model.model.model[-1].p2_head
    assert restored.model.model[-1].boundary_gain == model.model.model[-1].boundary_gain


def test_overrides_and_manifest_consistency():
    assert configs() == [yaml.safe_load((YAML_DIR / (n + ".yaml")).read_text()) for n in NAMES]
    for name, factors in zip(NAMES, FACTORS):
        ov = RUN_OVERRIDES[name]
        assert ov["mask_ratio"] == 2 and ov["nwd_ratio"] == 0.0
        assert ov["copy_paste"] == (0.3 if name == "V7_05_cp" else 0.0)
        cfg = yaml.safe_load((YAML_DIR / (name + ".yaml")).read_text())
        args = cfg["head"][-1][3]
        assert args[6] == "phase"  # adopted V6 fine base, not a V7 factor
        assert args[7] == bool(factors[0]) and args[8] == bool(factors[1]) and args[9] == bool(factors[2])
        assert (args[10] > 0) == bool(factors[3]) and (args[11] > 0) == bool(factors[3])
        if factors[0]:
            assert cfg["pretrained_layer_map"] == {19: -1, 20: 17, 22: 19, 26: 23}
            assert cfg["head"][-1][0] == [16, 22, 10, 19, 2, 0]
        else:
            assert "pretrained_layer_map" not in cfg
            assert cfg["head"][-1][0] == [16, 19, 10, 2, 0]
        assert cfg["pretrained_map_family"] == ("citrus_ev7_p2" if factors[0] else "citrus_ev7_control")
    spec = importlib.util.spec_from_file_location("ev7_runner_test", ROOT / "20260912_citrus_e_v7_batch.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.NAMES == NAMES and runner.RUN_OVERRIDES == RUN_OVERRIDES
    source = (ROOT / "RUN_CITRUS_E_V7.py").read_text()
    assert "cache=True" in source and "amp=False" in source and "device_lock=False" in source


def test_control_to_p2_transfer_and_reconstruction():
    source = SegmentationModel(YAML_DIR / (NAMES[0] + ".yaml"), nc=1, verbose=False)
    target = SegmentationModel(YAML_DIR / (NAMES[1] + ".yaml"), nc=1, verbose=False)
    fresh_p2 = deepcopy(target.model[19].state_dict())
    target.load(source, verbose=False)
    for dst, src in ((20, 17), (22, 19)):
        for key, value in source.model[src].state_dict().items():
            torch.testing.assert_close(target.model[dst].state_dict()[key], value, rtol=0, atol=0)
    for key, value in fresh_p2.items():
        torch.testing.assert_close(target.model[19].state_dict()[key], value, rtol=0, atol=0)
    restored = SegmentationModel(YAML_DIR / (NAMES[1] + ".yaml"), nc=1, verbose=False)
    restored.load(target, verbose=False)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("name", NAMES)
def test_phase_fine_base_is_active(name):
    """Every V7 arm inherits the V6 phase decoder: stride-2 proto, trained path."""
    model = SegmentationModel(YAML_DIR / (name + ".yaml"), nc=1, verbose=False)
    head = model.model[-1]
    assert head.fine_mode == "phase" and head.proto_stride == 2
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2})
    loss, _ = model.loss(batch(mask_ratio=2))
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert head.fine_to_proto.weight.grad.abs().sum() > 0
    with torch.no_grad():
        out = model.eval()(torch.rand(1, 3, 128, 160))
    assert out[0][1].shape[-2:] == (64, 80)  # stride-2 proto on a 128x160 input


def test_pmce_constant_input_has_no_padding_edge():
    from ultralytics.nn.modules.citrus_e_v7 import EV7PMCE

    module = EV7PMCE(16).eval()
    captured = []
    hook = module.local.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
    module(torch.ones(1, 16, 9, 11))
    hook.remove()
    assert captured[0].abs().max() < 1e-6


@pytest.mark.parametrize("fallback", [False, True])
def test_flops_uses_finite_input_under_determinism(monkeypatch, fallback):
    import thop

    from ultralytics.utils.torch_utils import get_flops

    seen = []

    def profile(model, inputs, verbose):
        assert torch.isfinite(inputs[0]).all()
        seen.append(inputs[0].shape[-1])
        if fallback and len(seen) == 1:
            raise ValueError("exercise full-resolution fallback")
        return 1e6, 0

    monkeypatch.setattr(thop, "profile", profile)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        assert get_flops(torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3)), 64) > 0
        assert seen == ([32, 64] if fallback else [32])
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)


def test_deform_preflight_preserves_rng():
    runner = importlib.import_module("20260912_citrus_e_v7_batch")
    rng = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    try:
        runner.check_deform_backend("cpu")
        assert torch.equal(rng, torch.random.get_rng_state())
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("name", NAMES)
def test_assigner_uses_spatial_order_not_tower_order(name):
    from ultralytics.utils.citrus_e_v7_loss import EV7SegmentationLoss
    from ultralytics.utils.tal import TaskAlignedAssigner

    model = SegmentationModel(YAML_DIR / (name + ".yaml"), nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2})
    loss = model.init_criterion()
    assert isinstance(loss, EV7SegmentationLoss)
    strides = model.model[-1].stride.tolist()
    assert loss.stride.tolist() == strides  # anchor/detection order must not change
    ordered = sorted(strides)
    assert loss.assigner.stride == ordered
    assert loss.assigner.stride_val == ordered[1]
    reference = TaskAlignedAssigner(stride=ordered)
    anchors = torch.tensor([[2., 2.], [6., 2.], [10., 2.], [14., 2.]])
    boxes = torch.tensor([[[3., 1., 9., 3.], [5., 1., 7., 3.]]])
    valid = torch.ones(1, 2, 1)
    expected = reference.select_candidates_in_gts(anchors, boxes, valid)
    actual = loss.assigner.select_candidates_in_gts(anchors, boxes, valid)
    assert torch.equal(actual, expected)
    if model.model[-1].p2_head:
        legacy = TaskAlignedAssigner(stride=strides).select_candidates_in_gts(anchors, boxes, valid)
        assert not torch.equal(actual, legacy)  # detects the original overly wide candidate support
