"""J5 analytical, gradient, reliability and deployment contracts; not a trained AP claim."""

import copy
import io
import csv
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from citrus_j5_distill import (J5DistillationCriterion, bernoulli_kl, composed_logits,
                                instance_distillation, make_trainer, ramp_factor, raw_predictions)
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_sampled_logits_invariant_to_basis_change():
    torch.manual_seed(42)
    prototype = torch.randn(4, 12, 16)
    coefficient = torch.randn(3, 4)
    basis = torch.eye(4) + torch.randn(4, 4) * 0.1
    grid = torch.rand(3, 25, 2) * 2 - 1
    before = composed_logits(prototype, coefficient, grid)
    changed = torch.einsum("dc,chw->dhw", basis, prototype)
    after = composed_logits(changed, coefficient @ torch.linalg.inv(basis), grid)
    assert torch.allclose(before, after, atol=2e-5, rtol=2e-5)
    assert not torch.allclose(prototype, changed)  # raw feature MSE would penalize equivalent masks


def test_bernoulli_kl_identity_and_gradient():
    logits = torch.randn(4, 8, requires_grad=True)
    same = bernoulli_kl(logits, logits.detach().sigmoid()).mean()
    assert same < 1e-6
    changed = bernoulli_kl(logits, torch.full_like(logits, 0.8)).mean()
    changed.backward()
    assert changed > 0 and torch.isfinite(logits.grad).all()


def test_packed_sampling_matches_expanded_sampling_and_its_gradients():
    torch.manual_seed(5)
    prototype = torch.randn(4, 12, 16, requires_grad=True)
    coefficients = torch.randn(3, 4, requires_grad=True)
    grid = torch.rand(3, 25, 2) * 2 - 1
    packed = composed_logits(prototype, coefficients, grid)
    features = F.grid_sample(prototype[None].expand(3, -1, -1, -1), grid[:, None],
                              padding_mode="border", align_corners=False)
    expanded = torch.einsum("ic,icp->ip", coefficients, features[:, :, 0])
    assert torch.allclose(packed, expanded, atol=1e-6)
    grad_packed = torch.autograd.grad(packed.square().sum(), (prototype, coefficients), retain_graph=True)
    grad_expanded = torch.autograd.grad(expanded.square().sum(), (prototype, coefficients))
    assert all(torch.allclose(a, b, atol=1e-5, rtol=1e-5) for a, b in zip(grad_packed, grad_expanded))


def example(empty=False, teacher_good=True):
    # 8x8 grid @stride8; one annotated 32x32 square.
    features = [torch.zeros(1, 4, 8, 8)]
    mask = torch.full((1, 1, 32, 32), -8.0)
    mask[:, :, 8:24, 8:24] = 8.0 if teacher_good else -8.0
    distribution = torch.full((1, 4 * 8, 64), -25.0)
    # Exact distances from anchor (28,28) to [16,16,48,48] in input pixels.
    anchor = 3 * 8 + 3
    for side, value in enumerate((1.5, 1.5, 2.5, 2.5)):
        low = int(value)
        distribution[0, side * 8 + low, anchor] = 0
        distribution[0, side * 8 + low + 1, anchor] = 0
    scores = torch.full((1, 1, 64), -8.0)
    scores[:, :, anchor] = 8
    teacher = dict(boxes=distribution, scores=scores, feats=features, proto=mask,
                   mask_coefficient=torch.ones(1, 1, 64))
    student = {key: value.detach().clone().requires_grad_() if torch.is_tensor(value) else value
               for key, value in teacher.items()}
    student["proto"] = (student["proto"] * 0.5).detach().requires_grad_()
    gt = (mask[:, 0] > 0).float() if teacher_good else torch.zeros(1, 32, 32)
    gt[:, 8:24, 8:24] = 1
    batch = dict(img=torch.rand(1, 3, 64, 64), batch_idx=torch.tensor([] if empty else [0.]),
                 cls=torch.zeros(0 if empty else 1, 1),
                 bboxes=torch.empty(0, 4) if empty else torch.tensor([[.5, .5, .5, .5]]), masks=gt)
    return student, teacher, batch


def test_verified_instance_loss_backprop_and_rejection():
    student, teacher, batch = example()
    mask, cls, counts = instance_distillation(student, teacher, batch, (8,), point_grid=12)
    assert counts["accepted"] == 1 and mask > 0
    (mask + cls).backward()
    assert student["proto"].grad.abs().sum() > 0
    assert student["mask_coefficient"].grad is not None
    assert all(value.grad is None for value in teacher.values() if torch.is_tensor(value))
    student, teacher, batch = example(teacher_good=False)
    mask, cls, counts = instance_distillation(student, teacher, batch, (8,))
    assert counts["accepted"] == 0 and mask == 0 and cls == 0


@pytest.mark.parametrize("empty", [True, False])
def test_no_teacher_validation_is_exact_original_loss(empty):
    student, _, batch = example(empty=empty)

    def base(predictions, received):
        value = predictions["proto"].sum() * 0 + torch.arange(5, dtype=torch.float)
        return value, value.detach()

    criterion = J5DistillationCriterion(base, (8,), {})
    expected = base(student, batch)
    actual = criterion(student, batch)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


def test_empty_batch_and_weak_teacher_do_not_create_pseudo_positives():
    student, teacher, batch = example(empty=True)
    mask, cls, counts = instance_distillation(student, teacher, batch, (8,))
    assert counts["annotated"] == 0 and counts["accepted"] == 0
    (mask + cls).backward()
    assert torch.isfinite(student["proto"].grad).all()
    student, teacher, batch = example()
    teacher["scores"].fill_(-20)
    assert instance_distillation(student, teacher, batch, (8,))[2]["accepted"] == 0


def test_ramp_is_prespecified_and_bounded():
    assert ramp_factor(0) == ramp_factor(9) == 0
    assert ramp_factor(10) == .05 and ramp_factor(29) == 1 and ramp_factor(100) == 1
    assert make_trainer(None, kd_mask=0, kd_cls=0).__name__ == "EV6TrainingTrainer"
    with pytest.raises(ValueError):
        make_trainer(None)
    with pytest.raises(ValueError):
        make_trainer(None, kd_box=.1)


def test_real_teacher_student_loss_and_checkpoint_contains_no_teacher():
    root = "0_orange_yaml/J_V4_series/"
    student = SegmentationModel(root + "J65_shared_towers.yaml", verbose=False)
    teacher = SegmentationModel(root + "J61_v12_replay.yaml", verbose=False).eval().requires_grad_(False)
    student.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    batch = dict(img=torch.rand(2, 3, 128, 128), batch_idx=torch.tensor([0., 1.]), cls=torch.zeros(2, 1),
                 bboxes=torch.tensor([[.5, .5, .5, .5]] * 2), masks=torch.zeros(2, 64, 64))
    batch["masks"][:, 16:48, 16:48] = 1
    with torch.no_grad():
        raw = raw_predictions(teacher(batch["img"]))
        batch["_j5_teacher"] = {key: raw[key] for key in ("boxes", "scores", "mask_coefficient", "proto")}
    batch["_j5_ramp"] = 1.
    student.criterion = J5DistillationCriterion(student.init_criterion(), student.stride,
        dict(kd_mask=.5, kd_cls=.1, point_grid=8, max_instances=8, teacher_min_iou=.5,
             teacher_min_score=.25, overlap=True))
    loss, parts = student.loss(batch)
    loss.sum().backward()
    assert len(parts) == 5 and torch.isfinite(loss).all()
    assert all(p.grad is None for p in teacher.parameters())
    assert not any("teacher" in name for name in student.state_dict())
    for value in vars(student.criterion).values():
        assert not isinstance(value, torch.nn.Module)
    stream = io.BytesIO()
    torch.save(copy.deepcopy(student).cpu(), stream)
    stream.seek(0)
    restored = torch.load(stream, map_location="cpu", weights_only=False).eval()
    with torch.no_grad():
        prediction = restored(torch.rand(1, 3, 128, 160))
    assert prediction[0][1].shape[-2:] == (64, 80)


def test_all_j5_yaml_build_backward_and_cost():
    from citrus_j5_suite import NAMES, YAML_DIR, jobs
    from ultralytics.utils.torch_utils import get_flops

    costs = []
    for name in NAMES:
        model = SegmentationModel(YAML_DIR / f"{name}.yaml", verbose=False)
        model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
        batch = dict(img=torch.rand(2, 3, 128, 128), batch_idx=torch.tensor([0., 1.]),
                     cls=torch.zeros(2, 1), bboxes=torch.tensor([[.5, .5, .5, .5]] * 2),
                     masks=torch.zeros(2, 64, 64))
        batch["masks"][:, 16:48, 16:48] = 1
        loss, _ = model.loss(batch)
        loss.sum().backward()
        assert torch.isfinite(loss).all()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        flops = get_flops(model, 640)
        assert 0 < flops < 15
        costs.append((sum(p.numel() for p in model.parameters()), flops))
    assert costs[2] == costs[3] == costs[4]
    assert costs[4][0] < costs[1][0] and costs[4][1] < costs[1][1]
    assert jobs("distill")[0] == ("J81_v12_teacher", 42)


def test_one_epoch_source_balanced_teacher_training(tmp_path):
    """Synthetic engineering smoke, NOT a citrus experiment or trained efficacy claim."""
    import numpy as np
    import yaml
    from PIL import Image
    from ultralytics import YOLO
    from citrus_j2_data import prepare_resampled_views

    dataset = tmp_path / "synthetic"
    for split, count in (("train", 2), ("val", 2)):
        for kind in ("images", "labels"):
            (dataset / split / kind).mkdir(parents=True)
        for i in range(count):
            image = np.full((128, 128, 3), 80 + i, np.uint8)
            image[32:96, 32:96] = (20, 150, 30)
            Image.fromarray(image).save(dataset / split / "images" / f"{split}{i}.png")
            (dataset / split / "labels" / f"{split}{i}.txt").write_text(
                "0 .25 .25 .75 .25 .75 .75 .25 .75\n", encoding="utf-8")
    data = dataset / "data.yaml"
    data.write_text(yaml.safe_dump(dict(path=str(dataset), train="train/images", val="val/images",
                                       names={0: "orange_immature"})), encoding="utf-8")
    prepared = prepare_resampled_views(data, tmp_path / "views", 128)
    teacher = YOLO("0_orange_yaml/J_V4_series/J61_v12_replay.yaml", task="segment", verbose=False)
    teacher_path = tmp_path / "random_teacher.pt"
    teacher.save(teacher_path)
    student = YOLO("0_orange_yaml/J_V5_series/J84_compact_joint_kd.yaml", task="segment", verbose=False)
    trainer = make_trainer(teacher_path, warmup_epochs=0, ramp_epochs=1,
                           teacher_min_iou=0.0, teacher_min_score=0.0, point_grid=4)
    student.train(trainer=trainer, data=str(prepared), project=str(tmp_path), name="smoke", epochs=1,
                  imgsz=128, batch=2, workers=0, device="cpu", amp=False, cache=True,
                  optimizer="AdamW", lr0=.001, val=True, save=True, plots=False, pretrained=False,
                  mask_ratio=2, overlap_mask=True, mosaic=0, copy_paste=0)
    assert student.trainer.train_loader.dataset.__class__.__name__ == "MultiScaleViewDataset"
    assert len(student.trainer.train_loader.dataset) == 2  # not number of cached views
    log = Path(student.trainer.save_dir) / "j5_distillation.csv"
    with log.open(encoding="utf-8", newline="") as stream:
        row = list(csv.DictReader(stream))[-1]
    assert int(row["active_batches"]) > 0 and int(row["accepted"]) > 0
    assert student.trainer.validator.__class__.__name__ == "EV6Validator"
    restored = YOLO(Path(student.trainer.save_dir) / "weights/last.pt", task="segment", verbose=False)
    assert not any("teacher" in key for key in restored.model.state_dict())
