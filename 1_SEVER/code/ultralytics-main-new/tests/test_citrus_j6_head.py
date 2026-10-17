"""Independent J6 head contracts; synthetic CPU checks, not AP/GPU evidence."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.citrus_j6 import SegmentCitrusJ6, expand_roi


ROOT = Path(__file__).resolve().parents[1]
YAMLS = ROOT / "0_orange_yaml" / "J_V6_series"


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17102026)
    yield
    torch.set_num_threads(old)


def head(**kwargs):
    model = SegmentCitrusJ6(nm=4, npr=8, ch=(8, 16, 32, 3), **kwargs)
    model.stride = torch.tensor([8., 16., 32.])
    return model


def features(batch=2):
    return [torch.randn(batch, 8, 8, 12), torch.randn(batch, 16, 4, 6),
            torch.randn(batch, 32, 2, 3), torch.rand(batch, 3, 64, 96)]


def roi_inputs(batch=2, count=3):
    raw = {"proto": torch.randn(batch, 4, 16, 24),
           "j6": {"detail": torch.randn(batch, 8, 64, 96), "semantic": torch.randn(batch, 8, 8, 12)}}
    boxes = torch.tensor([[[.05, .1, .45, .8], [.3, .2, .8, .9], [.6, .1, .98, .7]]]).expand(batch, -1, -1)
    boxes = boxes[:, :count].clone()
    coefficients = torch.randn(batch, count, 4)
    has_own = torch.ones(batch, count, dtype=torch.bool)
    has_own[:, -1:] = False
    scores = torch.rand(batch, count)
    sources = (~has_own).long()
    return raw, boxes, coefficients, has_own, scores, sources


def test_enabled_false_has_no_auxiliary_parameters_or_forward():
    model = head(enabled=False).train()
    assert not hasattr(model, "high_prefix") and not hasattr(model, "relation")
    output = model(features())
    assert "j6" not in output
    assert output["proto"].shape == (2, 4, 16, 24)
    assert len(output["feats"]) == 3


def test_native_high_one_shot_binding_and_fallback_flag():
    model = head().train()
    model.bind_high_view(torch.rand(2, 3, 128, 192))
    output = model(features())
    assert output["j6"]["native_high"]
    assert model._pending_high is None
    assert output["j6"]["detail"].shape == (2, 8, 64, 96)
    assert output["j6"]["scout"].shape == (2, 16, 32, 48)
    assert not model(features())["j6"]["native_high"]


def test_bad_high_geometry_consumed_before_error():
    model = head().train()
    model.bind_high_view(torch.rand(2, 3, 126, 192))
    with pytest.raises(ValueError, match="geometry"):
        model(features())
    assert model._pending_high is None


def test_native_eval_output_is_standard_segment_plus_raw_metadata():
    model = head().eval()
    model.bind_high_view(torch.rand(2, 3, 128, 192))
    with torch.no_grad():
        decoded, raw = model(features())
    assert decoded[0].shape == (2, 9, 126)  # 4 boxes + one class + four mask coefficients
    assert decoded[1].shape == (2, 4, 16, 24)
    assert raw["j6"]["native_high"] and len(raw["feats"]) == 3


def test_batched_roi_decode_matches_individual_images_and_zero_init():
    model = head().eval()
    data = roi_inputs()
    first = model.decode_rois(*data)
    assert torch.equal(first["logits"], first["own_logits"])
    assert torch.equal(first["logits"][:, -1], torch.zeros_like(first["logits"][:, -1]))
    torch.nn.init.normal_(model.roi_decoder.output.weight, std=.1)
    batched = model.decode_rois(*data)
    for image in range(2):
        raw = {"proto": data[0]["proto"][image:image + 1],
               "j6": {k: v[image:image + 1] for k, v in data[0]["j6"].items()}}
        individual = model.decode_rois(raw, *(x[image:image + 1] for x in data[1:]))
        for key in ("logits", "own_logits", "rival_logits", "query", "quality_logits"):
            assert torch.allclose(batched[key][image:image + 1], individual[key], atol=2e-6)


def test_residual_training_updates_without_zero_init_dead_end():
    model = head().train()
    output = model.decode_rois(*roi_inputs())
    output["logits"].square().mean().backward()
    assert model.roi_decoder.output.weight.grad.abs().sum() > 0
    assert model.roi_decoder.input.weight.grad.abs().sum() == 0


def test_rival_domain_support_not_neighbor_canonical_grid():
    model = head().eval()
    raw = {"proto": torch.ones(1, 4, 16, 16),
           "j6": {"detail": torch.randn(1, 8, 32, 32), "semantic": torch.randn(1, 8, 8, 8)}}
    boxes = torch.tensor([[[0., 0., .5, .5], [.25, 0., .75, .5], [.8, .8, 1., 1.]]])
    coefficients = torch.tensor([[[1., 0., 0., 0.], [3., 0., 0., 0.], [7., 0., 0., 0.]]])
    output = model.decode_rois(raw, boxes, coefficients, torch.ones(1, 3, dtype=torch.bool),
                              torch.ones(1, 3), torch.zeros(1, 3, dtype=torch.long))
    assert output["rival_known"][0, 0, 0, :, 8:].all()
    assert not output["rival_known"][0, 0, 0, :, :8].any()
    assert torch.equal(output["rival_logits"][0, 0, 0, :, 8:], torch.full((16, 8), 3.))
    assert not output["rival_known"][0, 2].any()
    assert torch.equal(output["rival_logits"][0, 2], torch.zeros_like(output["rival_logits"][0, 2]))


def test_shuffled_control_preserves_rival_availability():
    real, shuffled = head().eval(), head(shuffle=True).eval()
    shuffled.load_state_dict(real.state_dict())
    inputs = roi_inputs()
    correct, wrong = real.decode_rois(*inputs), shuffled.decode_rois(*inputs)
    assert torch.equal(correct["rival_known"], wrong["rival_known"])


def test_empty_roi_count_has_valid_empty_outputs():
    model = head().eval()
    raw, boxes, coeff, own, scores, sources = roi_inputs(count=0)
    output = model.decode_rois(raw, boxes, coeff, own, scores, sources)
    assert output["logits"].shape == (2, 0, 1, 16, 16)
    assert output["query"].shape == (2, 0, 32)
    assert output["quality_logits"].shape == (2, 0)


def test_readiness_state_is_persistent_and_not_set_by_forward():
    model = head().train()
    model(features())
    assert model.auxiliary_updates.item() == 0
    assert model.gain_updates.item() == 0
    assert not model.relation.relation_trained.item()
    model.auxiliary_updates.fill_(5)
    model.gain_updates.fill_(2)
    model.relation.mark_trained()
    restored = head()
    restored.load_state_dict(model.state_dict())
    assert restored.auxiliary_updates.item() == 5
    assert restored.gain_updates.item() == 2
    assert restored.relation.relation_trained.item()


def test_discovery_count_threshold_positive_boxes_and_no_global_dependency():
    model = head(gain_route=True).eval()
    existence = torch.full((2, 1, 6, 8), -8.)
    existence[1, 0, 1, 2], existence[1, 0, 4, 5] = 4., 3.
    raw = {"j6": {"discovery": {"existence": existence, "offset": torch.zeros(2, 2, 6, 8),
                                  "log_box_size": torch.full((2, 2, 6, 8), -3.)},
                   "gain": torch.zeros(2, 1, 6, 8)}}
    boxes, scores, ids = model.discover(raw, image=1, threshold=.1)
    assert len(boxes) == len(scores) == len(ids) == 2
    assert len(boxes) <= model.max_new
    assert (boxes >= 0).all() and (boxes <= 1).all() and (boxes[:, 2:] > boxes[:, :2]).all()
    assert not len(model.discover(raw, image=0, threshold=.1)[0])


def test_gain_routing_ignores_untrained_gain_head():
    model = head(gain_route=True).eval()
    raw = {"j6": {"discovery": {"existence": torch.randn(1, 1, 8, 8),
                                  "offset": torch.zeros(1, 2, 8, 8),
                                  "log_box_size": torch.full((1, 2, 8, 8), -3.)},
                   "gain": torch.zeros(1, 1, 8, 8)}}
    first = model.discover(raw, 0)
    raw["j6"]["gain"].fill_(100.)
    second = model.discover(raw, 0)
    assert torch.equal(first[2], second[2])


def test_expanded_roi_clips_without_zero_area():
    boxes = torch.tensor([[[0., 0., .002, .003], [.999, .999, 1., 1.]]])
    expanded = expand_roi(boxes)
    assert (expanded >= 0).all() and (expanded <= 1).all()
    assert (expanded[..., 2:] > expanded[..., :2]).all()


def test_registered_full_model_native_roi_graph_backward():
    model = YOLO(str(YAMLS / "J94_full.yaml"), task="segment", verbose=False).model.train()
    high = torch.rand(2, 3, 256, 320, requires_grad=True)
    model.model[-1].bind_high_view(high)
    raw = model(torch.rand(2, 3, 128, 160))
    boxes = torch.tensor([[[.05, .1, .45, .8], [.3, .2, .8, .9], [.6, .1, .98, .7]]]).expand(2, -1, -1)
    coefficients = raw["mask_coefficient"][:, :, :3].transpose(1, 2)
    own = torch.tensor([[True, True, False], [True, True, False]])
    output = model.model[-1].decode_rois(raw, boxes, coefficients, own, torch.ones(2, 3), (~own).long())
    loss = raw["proto"].square().mean() + raw["boxes"].square().mean() + raw["scores"].square().mean()
    loss = loss + sum(x.square().mean() for x in raw["j6"]["discovery"].values())
    loss = loss + output["logits"].square().mean() + output["quality_logits"].mean() + raw["j6"]["gain"].mean()
    loss.backward()
    assert torch.isfinite(high.grad).all() and high.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.model[-1].roi_decoder.output.weight.grad.abs().sum() > 0
    assert model.model[-1].auxiliary_updates.item() == 0  # A graph check is not a supervised trainer update.


@pytest.mark.parametrize("path", sorted(YAMLS.glob("*.yaml")), ids=lambda p: p.stem)
def test_official_yaml_build_rectangular_forward_and_three_strides(path):
    assert path.exists()
    model = YOLO(str(path), task="segment", verbose=False).model.eval()
    assert model.model[-1].stride.tolist() == [8., 16., 32.]
    assert model.model[-1].nl == 3
    with torch.no_grad():
        decoded, raw = model(torch.rand(1, 3, 128, 160))
    assert decoded[0].shape == (1, 37, 420)
    enabled = model.model[-1].enabled
    assert ("j6" in raw) == enabled
    if enabled:
        assert not raw["j6"]["native_high"]
