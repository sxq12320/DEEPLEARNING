"""J2 structural graphs, exact deploy equivalence and native-validation contracts."""

import copy
import inspect

import pytest
import torch

import RUN_CITRUS_J2 as launcher
from citrus_e_v6_training import EV6Validator
from citrus_j2_batch import run, settings
from citrus_j2_data import make_trainer
from citrus_j2_structure_suite import STRUCTURE_NAMES
from citrus_j2_suite import FACTORS
from tests.test_citrus_j2 import batch_fixture, build
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.citrus_j2_structure import J2Contrast, J2RepMix, SegmentCitrusJ2Structural

torch.set_num_threads(2)


@pytest.mark.parametrize("name", STRUCTURE_NAMES)
@pytest.mark.parametrize("empty", [False, True])
def test_structural_yaml_train_backward_eval(name, empty):
    model = build(name).train()
    batch = batch_fixture(empty=empty)
    raw = model(batch["img"])
    loss, parts = model.loss(batch, raw)
    assert parts.shape == (5,) and torch.isfinite(loss).all()
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        output, _ = model(batch["img"])
    assert output[0].shape[:2] == (2, 37)
    if isinstance(model.model[-1], SegmentCitrusJ2Structural):
        assert model.stride.tolist() == [4, 8, 16]
        assert raw["proto"].shape == (2, 32, 96 // model.model[-1].proto_stride, 128 // model.model[-1].proto_stride)
        assert ("j2_density" in raw) == FACTORS[name]["density"]
    elif name != "J29_i20_structure_anchor":
        assert model.stride.tolist() == [8, 16, 32]


def test_rep_deploy_matches_frozen_branch_sum():
    mixer = J2RepMix(24).eval()
    with torch.no_grad():
        for module in mixer.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.uniform_(-1, 1)
                module.running_var.uniform_(0.1, 2)
                module.weight.uniform_(-1, 1)
                module.bias.uniform_(-1, 1)
        x = torch.randn(2, 24, 16, 20)
        expected = mixer(x)
        fused = copy.deepcopy(mixer).switch_to_deploy()
        torch.testing.assert_close(expected, fused(x), rtol=1e-5, atol=3e-6)
        assert len(list(fused.children())) == 1
    with pytest.raises(RuntimeError, match="eval"):
        J2RepMix(24).switch_to_deploy()


@pytest.mark.parametrize("name", ["J32_p2_discovery", "J36_structure_complete", "J41_subpixel_p3", "J42_subpixel_p2"])
def test_structural_export_forward_contract(name):
    model = build(name).eval()
    head = model.model[-1]
    head.export, head.format = True, "onnx"
    with torch.no_grad():
        boxes, proto = model(torch.rand(1, 3, 96, 128))
    assert boxes.shape[:2] == (1, 37)
    assert proto.shape == (1, 32, 96 // head.proto_stride, 128 // head.proto_stride)


def test_complete_model_fusion_and_serialization(tmp_path):
    model = build("J36_structure_complete").eval()
    x = torch.rand(1, 3, 96, 128)
    with torch.no_grad():
        reference = model(x)[0]
        fused = copy.deepcopy(model).fuse(verbose=False)
        predicted = fused(x)[0]
        torch.testing.assert_close(reference[0], predicted[0], rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(reference[1], predicted[1], rtol=2e-4, atol=2e-5)
        path = tmp_path / "model.pt"
        torch.save(fused, path)
        restored = torch.load(path, weights_only=False)
        torch.testing.assert_close(predicted[0], restored(x)[0][0])


def test_contrast_identity_at_initialization_and_constant_shift_invariance():
    module = J2Contrast(16).eval()
    x = torch.randn(2, 16, 12, 14)
    torch.testing.assert_close(module(x), x, rtol=0, atol=0)
    def surround(z):
        return torch.nn.functional.avg_pool2d(z, 5, 1, 2, count_include_pad=False)
    torch.testing.assert_close(x - surround(x), x + 3 - surround(x + 3), rtol=1e-4, atol=2e-6)


@pytest.mark.parametrize("name", STRUCTURE_NAMES)
def test_structure_uses_i_native_validator(name, tmp_path):
    trainer_type = make_trainer(FACTORS[name])
    trainer = trainer_type.__new__(trainer_type)
    trainer.args = get_cfg(overrides=settings(300, 42))
    trainer.test_loader, trainer.save_dir, trainer.callbacks = None, tmp_path, {}
    validator = trainer.get_validator()
    assert type(validator) is EV6Validator
    assert validator.evaluation_mask_ratio == 2
    assert validator.args.max_det == 300 and validator.args.plots


def test_launcher_does_not_add_validation_or_require_old_split():
    assert "grouped_dedup" in launcher.DATA and "NATIVEVAL" in launcher.PROJECT
    assert launcher.GAZE_EVAL is False and launcher.DIAGNOSE_PR is False
    assert inspect.signature(run).parameters["gaze_eval"].default is False
    assert inspect.signature(run).parameters["diagnose_pr"].default is False
    args = settings(300, 42)
    assert args["cache"] is True and args["amp"] is False and args["mask_ratio"] == 2
    assert args["optimizer"] == "AdamW" and args["lr0"] == 0.001 and args["conf"] == 0.001
