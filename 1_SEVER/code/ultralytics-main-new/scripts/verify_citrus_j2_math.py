"""Exact symbolic checks of conditional J2 properties; never claims AP or generalization guarantees."""

import argparse
import json
from pathlib import Path

import sympy as sp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gamma = sp.symbols("gamma", real=True)
    var, eps = sp.symbols("var eps", positive=True)
    mean, beta, w, b, x = sp.symbols("mean beta w b x", real=True)
    direct = gamma * (w * x + b - mean) / sp.sqrt(var + eps) + beta
    folded = gamma * w * x / sp.sqrt(var + eps) + beta + (b - mean) * gamma / sp.sqrt(var + eps)
    bn_exact = sp.simplify(direct - folded) == 0
    alpha = sp.symbols("alpha", real=True)
    gain_gradient = sp.simplify(sp.diff(sp.tanh(alpha) / 2, alpha))
    detail, semantic = sp.symbols("detail semantic", real=True)
    residual_exact = sp.simplify((detail + sp.tanh(alpha) * semantic / 2) - detail) == semantic * sp.tanh(alpha) / 2
    d, delta = sp.symbols("d delta", positive=True)
    iou = (d - delta)**2 / (2 * d**2 - (d - delta)**2)
    sensitivity = sp.factor(sp.diff(iou, delta).subs(delta, 0))
    assert sensitivity == -4 / d
    points = {str(size): str(iou.subs({d: size, delta: 1})) for size in (4, 8, 16)}
    mac_high, mac_low = sp.symbols("mac_high mac_low", positive=True)
    # 2x resize in both spatial axes: moving a fixed-width convolution before it saves 4x MACs.
    ratio = sp.cancel((4 * mac_low) / mac_low)
    area_gt, area_pred = sp.symbols("area_gt area_pred", positive=True)
    error = sp.symbols("error", nonnegative=True)
    threshold = sp.symbols("threshold", nonnegative=True)
    overlap = (area_gt + area_pred - error) / 2
    union = (area_gt + area_pred + error) / 2
    mask_iou = sp.cancel(overlap / union)
    allowed_error = (1 - threshold) * (area_gt + area_pred) / (1 + threshold)
    assert sp.simplify(mask_iou.subs(error, allowed_error) - threshold) == 0
    height, width, hidden, prototypes = sp.symbols("height width hidden prototypes", positive=True)
    phase_mac_equal = sp.simplify(4 * height * width * hidden * prototypes
                                 - height * width * hidden * (4 * prototypes)) == 0
    additional_parameters = 3 * prototypes * (hidden + 1)
    assert additional_parameters.subs({hidden: 32, prototypes: 32}) == 3168
    noise_gain = sp.Rational(1, 5)
    noise_variance_ratio = 1 + sp.Rational(24, 25) * (2 * noise_gain + noise_gain**2)
    assert noise_variance_ratio == sp.Rational(889, 625)
    checks = dict(sympy=sp.__version__, bn_affine_identity=bn_exact,
                  bounded_residual_identity=bool(residual_exact), gain_derivative=str(gain_gradient),
                  gain_derivative_at_zero=str(gain_gradient.subs(alpha, 0)),
                  square_mask_iou_displacement_sensitivity=str(sensitivity),
                  square_mask_iou_one_pixel=points, pre_resize_conv_mac_ratio=str(ratio),
                  parameter_ratio_two_shared_kernels_vs_three_independent=str(sp.Rational(1, 3)),
                  symmetric_difference_mask_iou=str(mask_iou),
                  maximum_symmetric_difference_at_iou_threshold=str(allowed_error),
                  phase_projection_mac_equal=bool(phase_mac_equal),
                  phase_additional_parameters_h32_k32=3168,
                  centre_surround_noise_counterexample_variance_ratio=str(noise_variance_ratio),
                  candidate_grid_ratio_640=str(sp.Rational(160**2 + 80**2 + 40**2, 80**2 + 40**2 + 20**2)),
                  native_mask_assembly_mac_per_instance=32 * 320**2,
                  limitations=["BN fusion requires frozen statistics; no nonlinearities inside branch sums.",
                               "Bound applies before post-fusion learned refinement only.",
                               "Sharing kernels reduces kernel parameters, NOT spatial MACs by itself.",
                               "Square-mask example is not a model of irregular visible citrus masks.",
                               "None of these identities proves higher recall/AP or faster GPU training."])
    assert bn_exact and residual_exact and ratio == 4
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
