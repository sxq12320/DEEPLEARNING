"""Structural J_V2 controls and explicit decoder repairs; old graphs stay available."""

STRUCTURE_NAMES = (
    "J29_i20_structure_anchor", "J30_rep_pan", "J31_rep_direct", "J32_p2_discovery",
    "J33_structure_residual", "J34_shared_prediction", "J35_visible_tiny_loss",
    "J36_structure_complete", "J37_density_complete", "J38_compact_complete", "J39_compact_decoder",
    "J40_shared_highres_proto",
    "J41_subpixel_p3", "J42_subpixel_p2", "J43_compact_subpixel_p2", "J44_compact_subpixel_p3",
)
STRUCTURE_FACTORS = {
    name: dict(density=False, detach=False, gaze_train=False, antialias=False, imgsz=640)
    for name in STRUCTURE_NAMES
}
STRUCTURE_FACTORS["J37_density_complete"]["density"] = True
STRUCTURE_DESCRIPTIONS = {
    "J29_i20_structure_anchor": "Exact current I20/J01 graph, scratch; retained historical structural anchor",
    "J30_rep_pan": "Non-CSP reparameterizable backbone with original PAN neck and P3/P4/P5 prediction",
    "J31_rep_direct": "Same rep backbone; single top-down detail-first neck; P3/P4/P5",
    "J32_p2_discovery": "Direct neck with P2/P3/P4 candidates, P5 retained as semantic context",
    "J33_structure_residual": "J32 plus bounded local-contrast residual; RGB is not replaced",
    "J34_shared_prediction": "J40 changes proto source P2 to P3 and stride 2 to 4; reconstruction grid unchanged",
    "J35_visible_tiny_loss": "J34 plus equal-instance bounded tiny-Dice auxiliary loss (gain 0.1)",
    "J36_structure_complete": "J35 plus local-contrast residual; structural full hypothesis",
    "J37_density_complete": "J36 plus optional density auxiliary branch for sparse gaze tests",
    "J38_compact_complete": "J36 narrowed backbone/neck channels; capacity/cost tradeoff arm",
    "J39_compact_decoder": "J32 plus narrow prototype decoder; high-resolution 1x1 output only",
    "J40_shared_highres_proto": "J39 plus joint narrow shared-task head; proto source/resolution comparison vs J34",
    "J41_subpixel_p3": "J36 decoder-only repair: same P3 source and stride 4, independent phases, repeated initialization",
    "J42_subpixel_p2": "J41 with P2 source and stride-2 phases; detail-source/resolution joint comparison",
    "J43_compact_subpixel_p2": "J42 narrower backbone/neck channels; head hyperparameters and losses retained",
    "J44_compact_subpixel_p3": "J38 decoder-only repair: P3 source and stride-4 independent phases",
}


def structural_graph(name):
    """Return conventional YAML layer rows, including every feature connection explicitly."""
    if name == "J29_i20_structure_anchor":
        return None
    repaired = name in {"J41_subpixel_p3", "J42_subpixel_p2", "J43_compact_subpixel_p2", "J44_compact_subpixel_p3"}
    compact = name in {"J38_compact_complete", "J43_compact_subpixel_p2", "J44_compact_subpixel_p3"}
    c1, c2, c3, c4, c5 = (24, 40, 80, 128, 192) if compact else (24, 48, 96, 160, 224)
    contrast = name in {"J33_structure_residual", "J36_structure_complete", "J37_density_complete",
                        "J38_compact_complete"} or repaired
    backbone = [
        [-1, 1, "Conv", [c1, 3, 2]],
        [-1, 1, "J2Contrast" if contrast else "nn.Identity", [5] if contrast else []],
        [-1, 1, "Conv", [c2, 3, 2]],
        [-1, 2, "J2RepStage", [c2, 2.0]],
        [-1, 1, "Conv", [c3, 3, 2]],
        [-1, 3, "J2RepStage", [c3, 2.0]],
        [-1, 1, "Conv", [c4, 3, 2]],
        [-1, 4, "J2RepStage", [c4, 2.0]],
        [-1, 1, "Conv", [c5, 3, 2]],
        [-1, 2, "J2RepStage", [c5, 2.0]],
        [-1, 1, "SPPF", [c5, 5]],
    ]
    if name == "J30_rep_pan":
        neck = [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 7], 1, "Concat", [1]],
            [-1, 1, "C3k2", [128, False]], [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 5], 1, "Concat", [1]], [-1, 1, "C3k2", [96, False]],
            [-1, 1, "Conv", [96, 3, 2]], [[-1, 13], 1, "Concat", [1]],
            [-1, 1, "C3k2", [128, False]], [-1, 1, "Conv", [128, 3, 2]],
            [[-1, 10], 1, "Concat", [1]], [-1, 1, "C3k2", [192, False]],
            [[16, 19, 22], 1, "Segment", [1, 32, 64]],
        ]
        return backbone, neck
    neck = [
        [[7, 10], 1, "J2ResidualFuse", [96 if not compact else 80, 1]],  # 11: P4
        [[5, 11], 1, "J2ResidualFuse", [64 if not compact else 48, 1]],  # 12: P3
    ]
    if name == "J31_rep_direct":
        neck += [[10, 1, "Conv", [192, 1]], [[12, 11, 13], 1, "Segment", [1, 32, 64]]]
        return backbone, neck
    neck += [[[3, 12], 1, "J2ResidualFuse", [32, 1]]]  # 13: narrow P2 discovery
    shared = name in {"J34_shared_prediction", "J35_visible_tiny_loss", "J36_structure_complete",
                      "J37_density_complete", "J38_compact_complete", "J40_shared_highres_proto"} or repaired
    gain = 0.1 if name in {"J35_visible_tiny_loss", "J36_structure_complete", "J37_density_complete",
                          "J38_compact_complete"} or repaired else 0.0
    compact_proto = shared or name == "J39_compact_decoder"
    proto_stride = 4 if shared and name != "J40_shared_highres_proto" else 2
    if name in {"J42_subpixel_p2", "J43_compact_subpixel_p2"}:
        proto_stride = 2
    head_args = [1, 32, 64, shared, gain, name == "J37_density_complete", proto_stride, 32, compact_proto]
    module = "SegmentCitrusJ2Expressive" if repaired else "SegmentCitrusJ2Structural"
    neck += [[[13, 12, 11], 1, module, head_args]]
    return backbone, neck
