"""Counterexamples and controlled repairs for J2 prototype spatial expressivity."""

import copy

import pytest
import torch
import torch.nn.functional as F

from tests.test_citrus_j2 import build
from ultralytics.nn.modules.citrus_j2_refined import J2PhaseProto
from ultralytics.nn.modules.citrus_j2_structure import SegmentCitrusJ2Structural

torch.set_num_threads(2)


def original_decoder():
    return SegmentCitrusJ2Structural(ch=(32, 64, 96), shared=True, proto_stride=4,
                                    compact_proto=True).proto.eval()


def test_original_compact_decoder_has_exact_two_by_two_copy_constraint():
    output = original_decoder()(torch.rand(2, 64, 12, 16))
    reference = output[..., 0::2, 0::2]
    for row, col in ((0, 1), (1, 0), (1, 1)):
        torch.testing.assert_close(reference, output[..., row::2, col::2], rtol=0, atol=0)


def test_phase_decoder_reproduces_old_function_at_initialization():
    original = original_decoder()
    repaired = J2PhaseProto(copy.deepcopy(original)).eval()
    x = torch.randn(2, 64, 12, 16)
    torch.testing.assert_close(original(x), repaired(x), rtol=1e-5, atol=2e-6)
    old_parameters = sum(p.numel() for p in original.parameters())
    new_parameters = sum(p.numel() for p in repaired.parameters())
    assert new_parameters - old_parameters == 3168


def test_phase_decoder_can_express_one_pixel_foreground_and_three_background():
    repaired = J2PhaseProto(original_decoder()).eval()
    with torch.no_grad():
        repaired.phase.weight.zero_()
        repaired.phase.bias.zero_()
        repaired.phase.bias[:4].copy_(torch.tensor([1.0, -1.0, -1.0, -1.0]))
    mask = repaired(torch.zeros(1, 64, 3, 4))[0, 0] > 0
    assert mask[0::2, 0::2].all()
    assert not mask[0::2, 1::2].any()
    assert not mask[1::2, :].any()


def test_phase_gradients_are_independent_and_finite():
    repaired = J2PhaseProto(original_decoder()).train()
    output = repaired(torch.randn(2, 64, 12, 16))
    target = torch.zeros_like(output)
    target[..., 0::2, 0::2] = 1
    F.binary_cross_entropy_with_logits(output, target).backward()
    gradient = repaired.phase.bias.grad.reshape(32, 4)
    assert torch.isfinite(gradient).all()
    assert (gradient[:, 0] - gradient[:, 1]).abs().max() > 0


def test_phase_conversion_preserves_cpu_rng_and_rejects_other_scaling():
    original = original_decoder()
    before = torch.random.get_rng_state().clone()
    J2PhaseProto(copy.deepcopy(original))
    assert torch.equal(before, torch.random.get_rng_state())
    original[2] = torch.nn.Upsample(scale_factor=3, mode="nearest")
    with pytest.raises(ValueError, match="nearest x2"):
        J2PhaseProto(original)


@pytest.mark.parametrize("old,new", [("J36_structure_complete", "J41_subpixel_p3"),
                                    ("J38_compact_complete", "J44_compact_subpixel_p3")])
def test_decoder_only_graph_has_same_initial_predictions(old, new):
    torch.manual_seed(42)
    previous = build(old).eval()
    torch.manual_seed(42)
    repaired = build(new).eval()
    x = torch.rand(1, 3, 96, 128)
    with torch.no_grad():
        expected = previous(x)[0]
        observed = repaired(x)[0]
    for old_tensor, new_tensor in zip(expected, observed):
        torch.testing.assert_close(old_tensor, new_tensor, rtol=2e-5, atol=2e-6)


def test_linear_interpolation_commutes_with_prototype_combination_and_gradients():
    proto = torch.randn(32, 13, 19, dtype=torch.float64, requires_grad=True)
    coefficients = torch.randn(7, 32, dtype=torch.float64, requires_grad=True)
    interpolated = F.interpolate(proto[None], size=(31, 43), mode="bilinear", align_corners=False)[0]
    first = (coefficients @ interpolated.flatten(1)).reshape(7, 31, 43)
    coarse = (coefficients @ proto.flatten(1)).reshape(7, 13, 19)
    second = F.interpolate(coarse[None], size=(31, 43), mode="bilinear", align_corners=False)[0]
    torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)
    first_grads = torch.autograd.grad(first.square().sum(), (proto, coefficients), retain_graph=True)
    second_grads = torch.autograd.grad(second.square().sum(), (proto, coefficients))
    for expected, observed in zip(first_grads, second_grads):
        torch.testing.assert_close(expected, observed, rtol=1e-12, atol=1e-10)
