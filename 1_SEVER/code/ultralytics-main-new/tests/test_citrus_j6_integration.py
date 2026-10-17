"""Official API routing and EV6-raster contract tests for J6 adapters."""

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from citrus_j6_training import (
    J6Predictor,
    J6Trainer,
    J6Validator,
    find_j6_head,
    restore_j6_prediction,
)
from ultralytics.models.yolo.model import YOLO
from ultralytics.models.yolo.segment import SegmentationPredictor, SegmentationTrainer, SegmentationValidator
from ultralytics.utils import ops


class SegmentCitrusJ6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enabled = True
        self.scout_geometry = "bbox_center_offset_v2"
        self.scout_candidate_policy = "cell_topk_no_peak_nms_v2"
        self.scout_source1_policy = "zero_coeff_cell_heat_v2"

    def bind_high_view(self, image):
        self.bound = image


@pytest.mark.parametrize("key,expected", [("trainer", J6Trainer), ("validator", J6Validator), ("predictor", J6Predictor)])
def test_native_yolo_dispatches_only_j6(key, expected):
    wrapper = YOLO.__new__(YOLO)
    torch.nn.Module.__init__(wrapper)
    wrapper.task = "segment"
    wrapper.model = torch.nn.Sequential(SegmentCitrusJ6())
    assert wrapper._smart_load(key) is expected
    wrapper.model = torch.nn.Identity()
    standard = {"trainer": SegmentationTrainer, "validator": SegmentationValidator, "predictor": SegmentationPredictor}
    assert wrapper._smart_load(key) is standard[key]


def test_find_head_inside_unregistered_backend_and_ema():
    head = SegmentCitrusJ6()
    wrapped = SimpleNamespace(backend=SimpleNamespace(model=torch.nn.Sequential(head)))
    assert find_j6_head(wrapped) is head
    assert find_j6_head(SimpleNamespace(ema=torch.nn.Sequential(head))) is head
    assert find_j6_head(None, required=False) is None
    with pytest.raises(TypeError):
        find_j6_head(torch.nn.Identity())


@pytest.mark.parametrize("half", [False, True])
def test_validator_keeps_gt_and_decoded_raw_masks(monkeypatch, tmp_path, half):
    validator = J6Validator(args={"plots": False, "save_json": False, "half": half}, save_dir=tmp_path)
    validator.device = torch.device("cpu")
    validator._j6_head = SegmentCitrusJ6()
    validator.process = ops.process_mask
    target = torch.zeros(1, 8, 8)
    target[0, 2:5, 3:6] = 1
    batch = dict(
        img=torch.zeros(1, 3, 16, 16, dtype=torch.uint8),
        high_img=torch.full((1, 3, 32, 32), 255, dtype=torch.uint8),
        masks=target.clone(), cls=torch.zeros(1, 1), bboxes=torch.tensor([[0.5625, 0.4375, 0.375, 0.375]]),
        batch_idx=torch.zeros(1), ori_shape=((16, 16),), ratio_pad=(((1.0, 1.0), (0, 0)),), im_file=("test.png",),
    )
    processed = validator.preprocess(batch)
    torch.testing.assert_close(processed["masks"], target, rtol=0, atol=0)
    assert processed["img"].dtype == (torch.float16 if half else torch.float32)
    torch.testing.assert_close(validator._j6_head.bound, torch.ones_like(validator._j6_head.bound))
    assert validator._j6_head.bound.dtype == processed["img"].dtype
    pred = dict(bboxes=torch.tensor([[6.0, 4.0, 12.0, 10.0]]), conf=torch.tensor([0.9]), cls=torch.zeros(1),
                masks=target.to(torch.uint8))
    monkeypatch.setitem(sys.modules, "citrus_j6_inference", SimpleNamespace(postprocess_j6=lambda *a, **kw: [pred]))
    decoded = validator.postprocess(None)
    assert decoded[0]["masks"].data_ptr() == pred["masks"].data_ptr()
    prepared = validator._prepare_batch(0, processed)
    torch.testing.assert_close(prepared["masks"], target, rtol=0, atol=0)
    matches = validator._process_batch(decoded[0], prepared)
    assert matches["tp_m"].all()


def test_validator_rejects_wrong_mask_raster(monkeypatch, tmp_path):
    validator = J6Validator(args={"plots": False}, save_dir=tmp_path)
    validator._j6_head = SegmentCitrusJ6()
    validator.input_shape = (16, 16)
    pred = dict(bboxes=torch.empty(0, 4), conf=torch.empty(0), cls=torch.empty(0), masks=torch.empty(0, 16, 16))
    monkeypatch.setitem(sys.modules, "citrus_j6_inference", SimpleNamespace(postprocess_j6=lambda *a, **kw: [pred]))
    with pytest.raises(RuntimeError, match="input/2"):
        validator.postprocess(None)


def test_original_coordinate_restoration_uses_recorded_padding():
    mask = torch.zeros(1, 8, 8, dtype=torch.uint8)
    mask[:, 2:6, 2:6] = 1
    pred = dict(bboxes=torch.tensor([[4.0, 4.0, 12.0, 12.0]]), masks=mask,
                conf=torch.tensor([0.8]), cls=torch.zeros(1))
    # Original 4x8 maps to a 16 square: gain2, top4. The foreground
    # spans original x=[2,6], y=[0,4], not an assumed isotropic centre crop.
    affine = np.asarray([[2.0, 0, 0], [0, 2.0, 4.0], [0, 0, 1]], dtype=np.float32)
    restored = restore_j6_prediction(pred, (4, 8), (16, 16), affine)
    torch.testing.assert_close(restored["bboxes"], torch.tensor([[2.0, 0.0, 6.0, 4.0]]))
    expected = torch.zeros(1, 4, 8, dtype=torch.uint8)
    expected[:, :, 2:6] = 1
    torch.testing.assert_close(restored["masks"], expected, rtol=0, atol=0)
    compact = restore_j6_prediction(pred, (4, 8), (16, 16), affine, output_shape=(2, 4))
    torch.testing.assert_close(compact["masks"], torch.tensor([[[0, 1, 1, 0], [0, 1, 1, 0]]], dtype=torch.uint8))
    empty = {**pred, "bboxes": torch.empty(0, 4), "masks": torch.empty(0, 8, 8)}
    assert restore_j6_prediction(empty, (4, 8), (16, 16), affine, output_shape=(2, 4))["masks"] is None


def test_native_real_j6_head_routing_and_backend_discovery():
    from ultralytics.nn.modules.citrus_j6 import SegmentCitrusJ6 as RealJ6Head

    head = RealJ6Head(nc=1, enabled=False, ch=(16, 32, 64, 3))
    wrapper = YOLO.__new__(YOLO)
    torch.nn.Module.__init__(wrapper)
    wrapper.task = "segment"
    wrapper.model = torch.nn.Sequential(head)
    assert wrapper._smart_load("trainer") is J6Trainer
    assert find_j6_head(SimpleNamespace(backend=SimpleNamespace(model=wrapper.model))) is head


def test_real_yaml_official_yolo_routes_all_three_apis():
    yaml = Path(__file__).resolve().parents[1] / "0_orange_yaml/J_V6_series/J90_global_control.yaml"
    model = YOLO(str(yaml), task="segment", verbose=False)
    assert model._smart_load("trainer") is J6Trainer
    assert model._smart_load("validator") is J6Validator
    assert model._smart_load("predictor") is J6Predictor


@pytest.mark.parametrize("field,marker", [
    ("scout_geometry", None), ("scout_geometry", "visible_point_offset_v1"),
    ("scout_candidate_policy", None), ("scout_candidate_policy", "peak_nms_v1"),
    ("scout_source1_policy", None), ("scout_source1_policy", "nearest_anchor_v1"),
])
def test_legacy_or_wrong_scout_geometry_is_rejected_before_native_execution(monkeypatch, tmp_path, field, marker):
    from citrus_j6_inference import assert_j6_scout_geometry, postprocess_j6

    head = SegmentCitrusJ6()
    if marker is None:
        delattr(head, field)
    else:
        setattr(head, field, marker)
    wrapped = torch.nn.Sequential(head)
    trainer = object.__new__(J6Trainer)
    trainer.model = wrapped
    original_get_model = SegmentationTrainer.get_model

    def cannot_transfer(*args, **kwargs):
        pytest.fail("Legacy equal-shaped scout tensors must be rejected BEFORE weight transfer")

    monkeypatch.setattr(SegmentationTrainer, "get_model", cannot_transfer)
    with pytest.raises(RuntimeError, match="NEW PROJECT"):
        trainer.get_model(cfg={}, weights=wrapped, verbose=False)
    with pytest.raises(RuntimeError, match="prebuilt model"):
        trainer.setup_model()
    monkeypatch.setattr(SegmentationTrainer, "get_model", original_get_model)
    validator = J6Validator(args={"plots": False}, save_dir=tmp_path)
    with pytest.raises(RuntimeError, match="validation/checkpoint"):
        validator.init_metrics(wrapped)
    predictor = J6Predictor(overrides={"save": False, "project": str(tmp_path)})
    predictor.model = SimpleNamespace(format="pt", backend=SimpleNamespace(model=wrapped))
    with pytest.raises(RuntimeError, match="prediction/checkpoint"):
        predictor._check_options()
    with pytest.raises(RuntimeError, match="J6 postprocess"):
        postprocess_j6(head, None)
    # The control does not interpret scout offsets. Do not reject its legacy
    # global-only weights just because a disabled auxiliary branch has no marker.
    head.enabled = False
    assert_j6_scout_geometry(head)


def test_pickled_legacy_j6_checkpoint_cannot_silently_adopt_the_new_geometry(tmp_path):
    from citrus_j6_inference import assert_j6_scout_geometry, postprocess_j6

    root = Path(__file__).resolve().parents[1]
    for name, enabled in (("J94_full", True), ("J90_global_control", False)):
        model = YOLO(str(root / "0_orange_yaml/J_V6_series" / f"{name}.yaml"), task="segment", verbose=False)
        head = find_j6_head(model.model)
        assert head.scout_geometry == "bbox_center_offset_v2"
        del head.scout_geometry  # Simulate the old pickled instance, not its current class initializer.
        checkpoint = tmp_path / f"legacy_{name}.pt"
        model.save(checkpoint)
        restored = YOLO(str(checkpoint))
        active = find_j6_head(restored.model)
        assert not hasattr(active, "scout_geometry")
        for key, expected in (("trainer", J6Trainer), ("validator", J6Validator), ("predictor", J6Predictor)):
            assert restored._smart_load(key) is expected
        if enabled:
            with pytest.raises(RuntimeError, match="learned coordinate meaning"):
                assert_j6_scout_geometry(active)
            with pytest.raises(RuntimeError, match="NEW PROJECT"):
                postprocess_j6(active, None)
            trainer = object.__new__(J6Trainer)
            with pytest.raises(RuntimeError, match="training/checkpoint resume"):
                trainer.get_model(cfg=restored.model.yaml, weights=restored.model, verbose=False)
            with pytest.raises(RuntimeError, match="training/checkpoint resume"):
                trainer.get_model(cfg=restored.model.yaml, weights=str(checkpoint), verbose=False)
        else:
            assert_j6_scout_geometry(active)


def test_loading_old_scout_weights_into_fresh_yaml_cannot_launder_the_semantic_markers(tmp_path):
    root = Path(__file__).resolve().parents[1]
    yaml = root / "0_orange_yaml/J_V6_series/J94_full.yaml"
    legacy = YOLO(str(yaml), task="segment", verbose=False)
    legacy_head = find_j6_head(legacy.model)
    del legacy_head.scout_source1_policy
    checkpoint = tmp_path / "old_source_policy.pt"
    legacy.save(checkpoint)
    fresh = YOLO(str(yaml), task="segment", verbose=False)
    state_before = {name: tensor.detach().clone() for name, tensor in fresh.model.state_dict().items()}
    overrides_before, ckpt_before = dict(fresh.overrides), fresh.ckpt
    for weights in (str(checkpoint), legacy.model, {"model": legacy.model}):
        with pytest.raises(RuntimeError, match="YOLO.load/J6 source checkpoint"):
            fresh.load(weights)
        assert fresh.overrides == overrides_before and fresh.ckpt is ckpt_before
        assert all(torch.equal(tensor, state_before[name]) for name, tensor in fresh.model.state_dict().items())
    # The supported current checkpoint transfer remains available through the
    # same public API; the guard is not a blanket ban on loading J6 weights.
    valid = YOLO(str(yaml), task="segment", verbose=False)
    assert fresh.load(valid.model) is fresh


def test_non_j6_public_weight_loading_keeps_its_existing_path():
    root = Path(__file__).resolve().parents[1]
    path = root / "0_orange_yaml/J_V4_series/J60_yolo_control.yaml"
    source = YOLO(str(path), task="segment", verbose=False)
    target = YOLO(str(path), task="segment", verbose=False)
    assert target.load(source.model) is target
    assert all(torch.equal(left, right) for left, right in
               zip(target.model.state_dict().values(), source.model.state_dict().values()))


def test_ema_copies_discrete_trained_state_instead_of_staying_base_only():
    from ultralytics.nn.modules.citrus_j6 import SegmentCitrusJ6 as RealJ6Head
    from ultralytics.utils.torch_utils import ModelEMA

    head = RealJ6Head(nc=1, enabled=True, ch=(16, 32, 64, 3))
    model = torch.nn.Sequential(head)
    ema = ModelEMA(model)
    head.auxiliary_updates.fill_(7)
    head.gain_updates.fill_(3)
    head.identity_positive_updates.fill_(23)
    head.identity_negative_updates.fill_(24)
    head.relation.relation_trained.fill_(True)
    ema.update(model)
    target = find_j6_head(ema.ema)
    # Demonstrate the generic behaviour that makes this explicit sync necessary.
    assert target.auxiliary_updates.item() == 0
    assert not target.relation.relation_trained.item()
    J6Trainer._sync_j6_ema_training_state(SimpleNamespace(model=model, ema=ema))
    assert target.auxiliary_updates.item() == 7
    assert target.gain_updates.item() == 3
    assert target.identity_positive_updates.item() == 23
    assert target.identity_negative_updates.item() == 24
    assert target.relation.relation_trained.item()
    target.half()
    assert target.auxiliary_updates.dtype == torch.long
    assert target.relation.relation_trained.dtype == torch.bool


@pytest.mark.parametrize("before,after,expected", [(1.0, 1.0, True), (8.0, 4.0, False), (8.0, 16.0, True)])
def test_only_successful_optimizer_steps_commit_auxiliary_state(monkeypatch, before, after, expected):
    from ultralytics.nn.modules.citrus_j6 import SegmentCitrusJ6 as RealJ6Head
    from ultralytics.utils.torch_utils import ModelEMA

    head = RealJ6Head(nc=1, enabled=True, ch=(16, 32, 64, 3))
    model = torch.nn.Sequential(head)
    ema = ModelEMA(model)
    state, committed = {"scale": before}, []

    def commit(success):
        committed.append(success)
        if success:
            head.auxiliary_updates.add_(1)
            head.relation.relation_trained.fill_(True)

    model.criterion = SimpleNamespace(commit_updates=commit)
    trainer = object.__new__(J6Trainer)
    trainer.model, trainer.ema = model, ema
    trainer.scaler = SimpleNamespace(get_scale=lambda: state["scale"])

    def base_step(self):
        state["scale"] = after

    monkeypatch.setattr(SegmentationTrainer, "optimizer_step", base_step)
    trainer.optimizer_step()
    assert committed == [expected]
    assert find_j6_head(ema.ema).auxiliary_updates.item() == int(expected)
    assert find_j6_head(ema.ema).relation.relation_trained.item() is expected


@pytest.mark.parametrize("half", [False, True])
def test_checkpoint_routing_backend_and_discrete_state_survive_precision(tmp_path, half):
    from ultralytics.nn.autobackend import AutoBackend

    yaml = Path(__file__).resolve().parents[1] / "0_orange_yaml/J_V6_series/J94_full.yaml"
    model = YOLO(str(yaml), task="segment", verbose=False)
    head = find_j6_head(model.model)
    head.auxiliary_updates.fill_(7)
    head.identity_positive_updates.fill_(23)
    head.identity_negative_updates.fill_(24)
    head.relation.relation_trained.fill_(True)
    checkpoint = tmp_path / "best.pt"
    torch.save({"ema": model.model.half(), "model": None, "train_args": {"task": "segment", "imgsz": 640}}, checkpoint)
    restored = YOLO(str(checkpoint))
    assert restored._smart_load("trainer") is J6Trainer
    assert restored._smart_load("validator") is J6Validator
    assert restored._smart_load("predictor") is J6Predictor
    backend = AutoBackend(model=deepcopy(restored.model), device=torch.device("cpu"), fp16=half, fuse=False)
    active = find_j6_head(backend)
    assert next(active.parameters()).dtype == (torch.float16 if half else torch.float32)
    assert active.auxiliary_updates.dtype == torch.long and active.auxiliary_updates.item() == 7
    assert active.identity_positive_updates.item() == 23
    assert active.identity_negative_updates.item() == 24
    assert active.relation.relation_trained.item()


def test_auxiliary_epoch_log_uses_full_epoch_deltas(tmp_path):
    from ultralytics.nn.modules.citrus_j6 import SegmentCitrusJ6 as RealJ6Head

    head = RealJ6Head(nc=1, enabled=False, ch=(16, 32, 64, 3))
    model = torch.nn.Sequential(head)
    model.criterion = SimpleNamespace(cumulative={})
    trainer = SimpleNamespace(model=model, save_dir=tmp_path, epoch=0,
                              _j6_step_totals={"successful_optimizer_steps": 0, "amp_skipped_optimizer_steps": 0})
    J6Trainer._start_j6_epoch_log(trainer)
    model.criterion.cumulative.update(active_batches=4, annotated=11, local_loss_sum=3.0)
    J6Trainer._write_j6_epoch_log(trainer)
    trainer.epoch = 1
    model.criterion.cumulative.update(active_batches=7, annotated=19, local_loss_sum=4.0)
    J6Trainer._write_j6_epoch_log(trainer)
    with (tmp_path / "j6_auxiliary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    annotated = [row for row in rows if row["metric"] == "forward/annotated"]
    assert [float(row["delta"]) for row in annotated] == [11, 8]
    assert [float(row["total"]) for row in annotated] == [11, 19]


def test_decode_log_accumulates_without_changing_metrics(monkeypatch, tmp_path):
    from citrus_e_v6_training import EV6Validator

    validator = J6Validator(args={"plots": False}, save_dir=tmp_path)
    validator.seen = 3
    validator.training = True
    validator.j6_epoch = 2
    validator._j6_decode_batches = 2
    validator._j6_decode_totals = {"global_candidates": 11, "new_candidates": 4}
    metrics = {"metrics/mAP50(M)": 0.8}
    monkeypatch.setattr(EV6Validator, "get_stats", lambda self: metrics)
    assert validator.get_stats() is metrics
    summary = json.loads((tmp_path / "j6_decode_epoch0002.json").read_text(encoding="utf-8"))
    assert summary["counts"] == validator._j6_decode_totals
    assert summary["epoch"] == 2
    assert summary["batches"] == 2


def test_predictor_pairs_original_pixels_and_refuses_tensor_source(tmp_path):
    predictor = J6Predictor(overrides={"save": False, "project": str(tmp_path), "imgsz": 640})
    predictor.model = SimpleNamespace(format="pt", stride=32, fp16=False,
                                      backend=SimpleNamespace(model=torch.nn.Sequential(SegmentCitrusJ6())))
    predictor.device = torch.device("cpu")
    predictor.imgsz = (640, 640)
    raw = np.random.default_rng(1).integers(0, 256, size=(317, 563, 3), dtype=np.uint8)
    low = predictor.preprocess([raw])
    assert predictor._j6_high.shape[-2:] == tuple(s * 2 for s in low.shape[-2:])
    assert predictor._j6_high.dtype == torch.float32
    with pytest.raises(ValueError, match="raw high-resolution"):
        predictor.preprocess(torch.zeros(1, 3, 640, 640))


def test_predictor_uses_j6_decoder_and_returns_original_results(monkeypatch, tmp_path):
    predictor = J6Predictor(overrides={"save": False, "project": str(tmp_path)})
    predictor.model = SimpleNamespace(backend=SimpleNamespace(model=torch.nn.Sequential(SegmentCitrusJ6())),
                                      names={0: "orange_immature"})
    predictor.batch = (["test.png"], None, None)
    affine = np.asarray([[2, 0, 0], [0, 2, 4], [0, 0, 1]], dtype=np.float32)
    predictor._j6_original_geometry = [affine]
    mask = torch.zeros(1, 8, 8, dtype=torch.uint8)
    mask[:, 2:6, 2:6] = 1
    pred = dict(bboxes=torch.tensor([[4.0, 4.0, 12.0, 12.0]]), conf=torch.tensor([0.9]), cls=torch.zeros(1), masks=mask)
    called = []

    def decode(*args, **kwargs):
        called.append(kwargs["mask_ratio"])
        return [pred]

    monkeypatch.setitem(sys.modules, "citrus_j6_inference", SimpleNamespace(postprocess_j6=decode))
    results = predictor.postprocess(None, torch.zeros(1, 3, 16, 16), [np.zeros((4, 8, 3), dtype=np.uint8)])
    assert called == [2]
    torch.testing.assert_close(results[0].boxes.xyxy, torch.tensor([[2.0, 0.0, 6.0, 4.0]]))
    assert results[0].masks.data.shape == (1, 4, 8)


@pytest.mark.parametrize("retina,expected", [(False, (13, 16)), (True, (80, 100))])
def test_prediction_retina_is_explicit_and_default_masks_are_bounded(monkeypatch, tmp_path, retina, expected):
    predictor = J6Predictor(overrides={"save": False, "project": str(tmp_path), "retina_masks": retina})
    predictor.model = SimpleNamespace(backend=SimpleNamespace(model=torch.nn.Sequential(SegmentCitrusJ6())),
                                      names={0: "orange_immature"})
    predictor.batch = (["test.png"], None, None)
    predictor._j6_original_geometry = [np.asarray([[0.16, 0, 0], [0, 0.1625, 1], [0, 0, 1]])]
    pred = dict(bboxes=torch.tensor([[0.0, 1.0, 16.0, 14.0]]), conf=torch.tensor([0.9]), cls=torch.zeros(1),
                masks=torch.ones(1, 8, 8, dtype=torch.uint8))
    monkeypatch.setitem(sys.modules, "citrus_j6_inference", SimpleNamespace(postprocess_j6=lambda *a, **kw: [pred]))
    result = predictor.postprocess(None, torch.zeros(1, 3, 16, 16), [np.zeros((80, 100, 3), dtype=np.uint8)])[0]
    assert result.masks.data.shape[-2:] == expected
    torch.testing.assert_close(result.boxes.xyxy, torch.tensor([[0.0, 0.0, 100.0, 80.0]]), atol=1e-5, rtol=1e-5)


def test_default_prediction_of_large_original_never_allocates_original_mask_stack(monkeypatch, tmp_path):
    predictor = J6Predictor(overrides={"save": False, "project": str(tmp_path), "retina_masks": False})
    predictor.model = SimpleNamespace(backend=SimpleNamespace(model=torch.nn.Sequential(SegmentCitrusJ6())),
                                      names={0: "orange_immature"})
    predictor.batch = (["large.png"], None, None)
    predictor._j6_original_geometry = [np.asarray([[0.064, 0, 0], [0, 0.064, 64], [0, 0, 1]])]
    pred = dict(bboxes=torch.tensor([[0.0, 64.0, 640.0, 576.0]]).repeat(3, 1), conf=torch.full((3,), 0.9),
                cls=torch.zeros(3), masks=torch.ones(3, 320, 320, dtype=torch.uint8))
    monkeypatch.setitem(sys.modules, "citrus_j6_inference", SimpleNamespace(postprocess_j6=lambda *a, **kw: [pred]))
    original = np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8), (8000, 10000, 3))
    result = predictor.postprocess(None, torch.zeros(1, 3, 640, 640), [original])[0]
    assert result.masks.data.shape == (3, 512, 640)
    assert result.masks.data.numel() <= 3 * 640 * 640


@pytest.mark.parametrize("bad", [{"augment": True}, {"compile": True}, {"save_json": True}])
def test_validator_rejects_unimplemented_paths(tmp_path, bad):
    with pytest.raises(ValueError):
        J6Validator(args=bad, save_dir=tmp_path)
