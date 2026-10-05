"""Exact symbolic verification of J fusion and size-aware Dice derivatives."""

import argparse
import json
from pathlib import Path

import sympy as sp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    s0, s1, s2, v0, v1, v2 = sp.symbols("s0 s1 s2 v0 v1 v2", real=True)
    scores = (s0, s1, s2)
    weights = [sp.exp(s) / sum(sp.exp(z) for z in scores) for s in scores]
    mixed = sum(a * v for a, v in zip(weights, (v0, v1, v2)))
    fusion_difference = sp.simplify(sp.diff(mixed, s0) - weights[0] * (v0 - mixed))
    p, t, remainder, intersection, epsilon = sp.symbols("p t remainder intersection epsilon", positive=True)
    numerator = 2 * (p * t + intersection) + epsilon
    denominator = p + remainder + epsilon
    dice_difference = sp.simplify(
        sp.diff(1 - numerator / denominator, p) - (numerator - 2 * t * denominator) / denominator**2
    )
    c = sp.symbols("c", real=True)
    shifted = sp.exp(s0 - c) / sum(sp.exp(z - c) for z in scores)
    softmax_difference = sp.simplify(shifted - weights[0])
    assert fusion_difference == dice_difference == softmax_difference == 0
    result = dict(
        sympy_version=sp.__version__,
        fusion_derivative_residual=str(fusion_difference),
        dice_derivative_residual=str(dice_difference),
        softmax_shift_residual=str(softmax_difference),
        fusion_derivative_latex=sp.latex(weights[0] * (v0 - mixed)),
        dice_derivative_latex=sp.latex((numerator - 2 * t * denominator) / denominator**2),
        note="Exact algebra only; does not prove recognition, generalization or a training gain.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
