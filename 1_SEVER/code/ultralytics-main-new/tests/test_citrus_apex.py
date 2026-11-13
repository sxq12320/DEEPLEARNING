"""Unit, non-zero step-1 gradient, pure-loss, and model-fuse tests for APEX00-APEX08."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RUN_CITRUS_APEX import _ensure_optional_vision_stubs

_ensure_optional_vision_stubs()

from ultralytics.nn.modules.citrus_apex import C2PhaseBridge, SegmentCitrusBSD, _subpixel_phase_decompose
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import v8SegmentationLoss

APEX_YAMLS = sorted((ROOT / "0_orange_yaml" / "APEX_series").glob("APEX*.yaml"))


def _make_dummy_batch(bs: int = 2, imgsz: int = 128) -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(bs, 3, imgsz, imgsz),
        "batch_idx": torch.tensor([0, 1], dtype=torch.float32),
        "cls": torch.tensor([[0.0], [0.0]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25], [0.4, 0.6, 0.2, 0.2]], dtype=torch.float32),
        "masks": torch.ones(2, imgsz // 4, imgsz // 4, dtype=torch.float32),
    }


@pytest.mark.parametrize("yaml_path", APEX_YAMLS, ids=[p.stem for p in APEX_YAMLS])
def test_apex_build_forward_nonzero_step1_grads_and_fuse(yaml_path: Path):
    """Verify build, pure native v8SegmentationLoss, strictly non-zero step-1 internal grads, and fuse."""
    model = SegmentationModel(str(yaml_path), ch=3, nc=1, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, overlap_mask=True)

    # Verify default pure native loss (no positive-only q_head calibration interference)
    criterion = model.init_criterion()
    assert type(criterion) is v8SegmentationLoss, f"{yaml_path.stem} should default to pure v8SegmentationLoss"

    # 1. Train forward + backward through loss
    model.train()
    batch = _make_dummy_batch(bs=2, imgsz=128)
    loss_vec, _ = model.loss(batch)
    loss_scalar = loss_vec.sum()
    assert torch.isfinite(loss_scalar), f"{yaml_path.stem} produced non-finite loss"
    loss_scalar.backward()

    # 2. Verify BOTH residual gates AND internal branch convs receive strictly NON-ZERO gradients on step 1!
    if isinstance(model.model[-1], SegmentCitrusBSD):
        named = dict(model.named_parameters())
        # Head (M3) checks present in all APEX01-08 models:
        for suffix in (
            "c2_proj.conv.weight",
            "c2_dw3.conv.weight",
            "p2_to_proto.weight",
            "boundary_head.weight",
            "bp_down.conv.weight",
            "bp_scale",
            "proto_gain",
            "tdc_blocks.0.tdc_gain",
            "tdc_blocks.0.out_proj.conv.weight",
            "tdc_blocks.0.fine_proj.conv.weight",
            "tdc_blocks.0.fine_unshuffle.conv.weight",
        ):
            hits = [p for n, p in named.items() if n.endswith(suffix)]
            assert hits, f"Missing {suffix} in {yaml_path.stem}"
            for p in hits:
                assert p.grad is not None and torch.isfinite(p.grad).all()
                assert p.grad.abs().max().item() > 0.0, f"Dead zero grad on {suffix} in {yaml_path.stem}"

        # Neck (M2) checks when present (all except APEX06):
        if yaml_path.stem not in {"APEX00_baseline", "APEX06_m3_bsd_only"}:
            for suffix in ("bifpn_w", "sem_proj.conv.weight", "det_proj.conv.weight", "c2_proj.conv.weight", "p3_proj.conv.weight", "bridge_gain"):
                hits = [p for n, p in named.items() if n.endswith(suffix)]
                assert hits, f"Missing M2 {suffix} in {yaml_path.stem}"
                for p in hits:
                    assert p.grad is not None and torch.isfinite(p.grad).all()
                    assert p.grad.abs().max().item() > 0.0, f"Dead zero grad on {suffix} in {yaml_path.stem}"

        # Backbone (M1) checks when present (APEX01-05, APEX08):
        if yaml_path.stem not in {"APEX00_baseline", "APEX06_m3_bsd_only", "APEX07_m2_m3_bifreq_bsd"}:
            for suffix in ("contrast_proj.conv.weight", "stem_gain", "wave_in.conv.weight", "sppf_gain"):
                hits = [p for n, p in named.items() if n.endswith(suffix)]
                assert hits, f"Missing M1 {suffix} in {yaml_path.stem}"
                for p in hits:
                    assert p.grad is not None and torch.isfinite(p.grad).all()
                    assert p.grad.abs().max().item() > 0.0, f"Dead zero grad on {suffix} in {yaml_path.stem}"

    # 3. Eval forward + Conv+BN fuse consistency
    model.eval()
    with torch.no_grad():
        x = torch.rand(1, 3, 128, 128)
        out_pre_fuse = model(x)
        assert isinstance(out_pre_fuse, tuple) and len(out_pre_fuse) == 2

        model.fuse(verbose=False)
        out_post_fuse = model(x)
        assert torch.allclose(out_pre_fuse[0][0], out_post_fuse[0][0], atol=1e-4, rtol=1e-3)
        assert torch.allclose(out_pre_fuse[0][1], out_post_fuse[0][1], atol=1e-4, rtol=1e-3)


def test_subpixel_phase_decompose_and_joint_gating_rescue():
    """Verify sub-pixel phase differences survive DC smoothing and C2 impulses activate C2PhaseBridge even when P3==0."""
    torch.manual_seed(0)
    z = torch.zeros(1, 2, 16, 16)
    z[:, :, 4, 4] = 10.0  # Single sub-pixel impulse at top-left phase of (2, 2)
    decomposed = _subpixel_phase_decompose(z, (8, 8)).view(1, 2, 4, 8, 8)
    # The 4 intra-block phases at (2, 2) must remain distinct (not averaged out before unshuffle).
    phase_vals = decomposed[0, 0, :, 2, 2]
    assert phase_vals[0] > phase_vals[1] + 5.0
    assert torch.allclose(phase_vals[1], phase_vals[2]) and torch.allclose(phase_vals[2], phase_vals[3])

    bridge = C2PhaseBridge(c_p3=64, c_c2=64, width=16).eval()
    p3_zero = torch.zeros(1, 64, 8, 8)
    c2_impulse = torch.zeros(1, 64, 16, 16)
    c2_impulse[:, :, 4:6, 4:6] = 5.0
    with torch.no_grad():
        out = bridge([p3_zero, c2_impulse])
    assert out.abs().max().item() > 0.0, "C2PhaseBridge failed to rescue missing P3 response from C2 impulse"
