"""V7R route isolation, parameter-matched controls, gradients and foreground suites."""

from copy import deepcopy

import pytest
import torch

from citrus_sage_v7_suite import NAMES, SUITES, YAML_DIR
from ultralytics.nn.modules.citrus_sage_v7r import SAGEV7ContextPyramid, SegmentCitrusSAGEV7R
from ultralytics.nn.tasks import SegmentationModel


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_head(route="cls_only", multiscale=True):
    return SegmentCitrusSAGEV7R(nc=1, npr=64, route=route, multiscale=multiscale, ch=(64, 128, 256, 64))


def features():
    return [torch.rand(2, c, h, w) for c, h, w in [(64, 8, 10), (128, 4, 5), (256, 2, 3), (64, 16, 20)]]


def test_class_context_cannot_directly_change_geometry_or_prototypes():
    torch.manual_seed(42)
    head = make_head().eval()
    original = features()
    saved = [x.clone() for x in original]
    # Keep training output structure while BN is frozen for this causal path probe.
    head.training = True
    before = head(original)
    changed = deepcopy(head)
    with torch.no_grad():
        changed.semantic_route.project.conv.weight.add_(0.3)
    after = changed(original)
    for name in ("boxes", "mask_coefficient", "proto"):
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    assert not torch.equal(before["scores"][..., :320], after["scores"][..., :320])
    torch.testing.assert_close(before["scores"][..., 320:], after["scores"][..., 320:], rtol=0, atol=0)
    for x, copy in zip(original, saved):
        torch.testing.assert_close(x, copy, rtol=0, atol=0)


def test_shared_and_class_routes_differ_only_in_geometry_consumers():
    torch.manual_seed(42)
    shared = make_head("shared").eval()
    routed = make_head("cls_only").eval()
    routed.load_state_dict(shared.state_dict(), strict=True)
    shared.training = routed.training = True
    x = features()
    a, b = shared(x), routed(x)
    torch.testing.assert_close(a["scores"], b["scores"], rtol=0, atol=0)
    torch.testing.assert_close(a["proto"], b["proto"], rtol=0, atol=0)
    assert not torch.equal(a["boxes"][..., :320], b["boxes"][..., :320])
    assert not torch.equal(a["mask_coefficient"][..., :320], b["mask_coefficient"][..., :320])
    torch.testing.assert_close(a["boxes"][..., 320:], b["boxes"][..., 320:], rtol=0, atol=0)


def test_multiscale_control_same_weights_and_tiny_rectangular_shapes():
    multi, single = make_head(multiscale=True), make_head(multiscale=False)
    single.load_state_dict(multi.state_dict(), strict=True)
    assert sum(p.numel() for p in multi.parameters()) == sum(p.numel() for p in single.parameters())
    for size in ((1, 1), (3, 5), (9, 7)):
        x = torch.randn(2, 32, *size, requires_grad=True)
        module = SAGEV7ContextPyramid(32)
        y = module(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        y.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_zero_residual_gain_restores_local_tensor_and_all_context_parameters_learn():
    head = make_head()
    local = torch.randn(2, 32, 16, 20)
    semantic = torch.randn(2, 64, 8, 10)
    zero = deepcopy(head.semantic_route)
    with torch.no_grad():
        zero.gain.zero_()
    torch.testing.assert_close(zero(local, semantic), local, rtol=0, atol=0)
    head.semantic_route(local, semantic).square().mean().backward()
    for name, p in head.semantic_route.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


@pytest.mark.parametrize("name", NAMES[5:])
def test_multiclass_official_output(name):
    model = SegmentationModel(YAML_DIR / f"{name}.yaml", nc=3, verbose=False).eval()
    with torch.inference_mode():
        (decoded, proto), raw = model(torch.rand(1, 3, 128, 160))
    assert decoded.shape == (1, 39, 1700)
    assert proto.shape == (1, 32, 32, 40)
    assert raw["scores"].shape[1] == 3


def test_new_launcher_is_inert_and_old_suites_remain_unchanged():
    import RUN_SAGE_V7R as run
    from citrus_foreground import resolve_runner

    assert run.EPOCHS == 300 and not run.DRY_RUN
    assert run.SUITE == "refusion" and "V7R" in run.PROJECT
    assert SUITES["structure"] == tuple(NAMES[:5])
    assert SUITES["refusion"] == (NAMES[0], NAMES[2], *NAMES[5:])
    _, spec = resolve_runner("SAGE_V7")
    assert "refusion" in spec.suites
