"""Native J7R anchors: graph, paired scratch state and unchanged EV6/loss grids."""

import copy
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from citrus_j7r_data import pack_j7r_images
from citrus_j7r_initialization import initialize_native_pair, native_state_sha256
from citrus_j7r_predict import J7RPredictor
from citrus_j7r_training import J7RTrainer, J7RValidator
from ultralytics import YOLO
from ultralytics.nn.modules import J7REvidenceRead, J7RHighEvidence, Segment
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace, YAML
from ultralytics.utils.loss import v8SegmentationLoss

ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / "0_orange_yaml/J_V7R_series"
NATIVE_IDS = (116, 117, 118)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


def path_for(number):
    path, = YAML_DIR.glob("J{}_*.yaml".format(number))
    return path


def model_for(number, seed=42):
    model = YOLO(str(path_for(number)), task="segment", verbose=False).model
    initialize_native_pair(model, seed)
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    return model


def training_batch():
    masks = torch.zeros(2, 64, 64)
    masks[:, 12:52, 12:52] = 1
    masks[:, 9:12, 9:12] = 2
    return {
        "img": torch.rand(2, 15, 128, 128, requires_grad=True),
        "batch_idx": torch.tensor([0., 0., 1., 1.]), "cls": torch.zeros(4, 1),
        "masks": masks,
        "bboxes": torch.tensor([[.5, .5, .625, .625], [.1640625, .1640625, .046875, .046875]] * 2),
    }


@pytest.mark.parametrize("number", NATIVE_IDS)
def test_native_build_forward_backward_uses_proto4_and_loss_gt2(number, monkeypatch):
    model = model_for(number)
    assert type(model.model[-1]) is Segment
    assert model.stride.tolist() == [8., 16., 32.]
    assert len(model.yaml["citrus_native_layer_map"]) == 24
    assert not hasattr(model.model[-1], "high_mask")
    assert not any(type(module).__name__.startswith("EV11") for module in model.modules())
    assert sum(parameter.numel() for parameter in model.parameters()) == (2842803 if number == 116 else 2850739)
    sample = training_batch()
    raw = model(sample["img"])
    assert raw["proto"].shape == (2, 32, 32, 32)
    criterion = model.init_criterion()
    assert type(criterion) is v8SegmentationLoss
    observed = []
    original = criterion.calculate_segmentation_loss

    def observe(fg, masks, assigned, boxes, batch_index, proto, coefficients, imgsz):
        observed.append((tuple(masks.shape[-2:]), tuple(proto.shape[-2:])))
        return original(fg, masks, assigned, boxes, batch_index, proto, coefficients, imgsz)

    monkeypatch.setattr(criterion, "calculate_segmentation_loss", observe)
    losses, items = criterion(raw, sample)
    assert observed == [((64, 64), (64, 64))]  # Actual implementation interpolates Proto, NOT the GT.
    assert torch.isfinite(losses).all() and torch.isfinite(items).all()
    losses.sum().backward()
    assert sample["img"].grad is not None and torch.isfinite(sample["img"].grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    with torch.no_grad():
        prediction, raw = model.eval()(torch.rand(1, 15, 128, 160))
    assert prediction[0].shape[-1] == 420
    assert raw["proto"].shape == (1, 32, 32, 40)


def test_semantic_initialization_common_weights_hash_and_zero_start_outputs():
    models = [model_for(number).eval() for number in NATIVE_IDS]
    records = [model.j7r_native_initialization for model in models]
    assert len({record["common_state_sha256"] for record in records}) == 1
    for semantic in range(24):
        states = [model.model[record["map"][semantic]].state_dict() for model, record in zip(models, records)]
        for key, value in states[0].items():
            for state in states[1:]:
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    image = torch.rand(1, 15, 128, 160)
    with torch.no_grad():
        outputs = [model(image) for model in models]
    for prediction, raw in outputs[1:]:
        for key in ("boxes", "scores", "mask_coefficient", "proto"):
            torch.testing.assert_close(raw[key], outputs[0][1][key], rtol=0, atol=0)
        torch.testing.assert_close(prediction[0], outputs[0][0][0], rtol=0, atol=0)


def test_pseudo_raw_graph_and_full_initial_state_are_paired_at_same_seed():
    models = []
    for number in (117, 118):
        torch.manual_seed(73)
        models.append(model_for(number, seed=42))
    assert list(models[0].state_dict()) == list(models[1].state_dict())
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)


def test_global_has_no_high_forward_path():
    model = model_for(116).eval()
    assert not any(isinstance(module, (J7RHighEvidence, J7REvidenceRead)) for module in model.modules())
    image = torch.rand(1, 15, 128, 160, requires_grad=True)
    changed = image.detach().clone()
    changed[:, 3:] = 0
    expected, raw = model(image)
    with torch.no_grad():
        actual = model(changed)[0]
    for first, second in zip(expected, actual):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    raw["proto"].square().sum().backward()
    assert torch.count_nonzero(image.grad[:, 3:]) == 0


@pytest.mark.parametrize("number", (117, 118))
def test_nonzero_c3_read_uses_changed_high_evidence(number):
    model = model_for(number).eval()
    read, = (module for module in model.model if isinstance(module, J7REvidenceRead))
    with torch.no_grad():
        read.gain.fill_(.4)
    global_rgb = torch.rand(1, 3, 128, 160)
    high_rgb = torch.rand(1, 3, 256, 320)
    pseudo = pack_j7r_images(global_rgb, high_rgb, "upsampled")
    raw_input = pack_j7r_images(global_rgb, high_rgb, "raw")
    torch.testing.assert_close(pseudo[:, :3], raw_input[:, :3], rtol=0, atol=0)
    observed = []
    handle = read.register_forward_hook(lambda _module, _args, output: observed.append(output.clone()))
    with torch.no_grad():
        model(pseudo)
        model(raw_input)
    handle.remove()
    assert not torch.equal(observed[0], observed[1])


def test_initialization_preserves_cpu_python_numpy_rng_and_branch_state(monkeypatch):
    model = model_for(118)
    native_indices = set(model.yaml["citrus_native_layer_map"])
    before_branch = {index: copy.deepcopy(module.state_dict()) for index, module in enumerate(model.model)
                     if index not in native_indices}
    rng = torch.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()

    def forbidden_cuda(*args, **kwargs):
        raise AssertionError("native initialization must not initialize or seed CUDA")

    monkeypatch.setattr(torch.cuda, "init", forbidden_cuda)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden_cuda)
    first = initialize_native_pair(model, 43)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert random.getstate() == python_rng
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_rng[0] and np.array_equal(actual_numpy[1], numpy_rng[1])
    assert actual_numpy[2:] == numpy_rng[2:]
    for index, state in before_branch.items():
        for name, value in state.items():
            torch.testing.assert_close(value, model.model[index].state_dict()[name], rtol=0, atol=0)
    assert first["seed"] == 43 and "NOT a historical J90" in first["canonical"]["weights"]
    assert first["common_state_sha256"] == native_state_sha256(model, first["map"])


@pytest.mark.parametrize("invalid", ["unknown_policy", "duplicate_map", "wrong_class_map", "wrong_state_shape"])
def test_bad_initialization_metadata_fails_before_loading(invalid):
    model = model_for(118)
    if invalid == "unknown_policy":
        model.yaml["citrus_native_initialization"] = "unknown"
    elif invalid == "duplicate_map":
        model.yaml["citrus_native_layer_map"][1] = model.yaml["citrus_native_layer_map"][0]
    elif invalid == "wrong_class_map":
        model.yaml["citrus_native_layer_map"][0] = 7  # Index cannot masquerade as the original stem Conv.
    else:
        model.model[2].conv.weight = torch.nn.Parameter(torch.zeros(17, 3, 3, 3))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises((ValueError, TypeError)):
        initialize_native_pair(model, 45)
    for key, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("number", NATIVE_IDS)
def test_native_official_routes_and_trainer_initializes_scratch_not_weights(number):
    wrapper = YOLO(str(path_for(number)), task="segment", verbose=False)
    assert wrapper._smart_load("trainer") is J7RTrainer
    assert wrapper._smart_load("validator") is J7RValidator
    assert wrapper._smart_load("predictor") is J7RPredictor
    trainer = object.__new__(J7RTrainer)
    trainer.data = {"nc": 1, "channels": 3}
    trainer.args = SimpleNamespace(seed=42)
    scratch = trainer.get_model(cfg=str(path_for(number)), verbose=False)
    assert scratch.j7r_native_initialization["seed"] == 42
    # A supplied trained/native state must survive: no scratch reinitialization.
    with torch.no_grad():
        scratch.model[2].conv.weight.add_(.125)
    expected = scratch.model[2].conv.weight.clone()
    trainer.args.seed = 99
    supplied = trainer.get_model(cfg=copy.deepcopy(scratch.yaml), weights=scratch, verbose=False)
    torch.testing.assert_close(supplied.model[2].conv.weight, expected, rtol=0, atol=0)


def test_other_models_do_not_opt_into_native_initialization():
    config = YAML.load(ROOT / "ultralytics/cfg/models/11/yolo11-seg.yaml")
    config.update(nc=1, scale="n")
    model = SegmentationModel(config, nc=1, verbose=False)
    state = torch.get_rng_state().clone()
    assert initialize_native_pair(model, 42) is None
    torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
