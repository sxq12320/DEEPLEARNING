"""Regression checks for existing E V4 implementations, initialization and training wiring."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from train_citrus_yaml import custom_loss_overrides
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules.citrus_e_v4 import EV4ChromaFront, EV4IntegralContext, EV4ReverseRefine
from ultralytics.nn.modules.citrus_far import CARAFE, DySample
from ultralytics.nn.modules.smc_scheduler import SMCScheduler
from ultralytics.nn.modules.smcao_v22_scheduler import SMCAOV22Scheduler
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.loss import BboxLoss, v8SegmentationLoss

ROOT = Path(__file__).resolve().parents[1]
YAMLS = sorted((ROOT / "0_orange_yaml/E_V4_series").glob("*.yaml"))


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def pretrained():
    return YOLO(str(ROOT / "yolo11n-seg.pt"), verbose=False).model


def make_batch(empty=False):
    masks = torch.zeros(2, 32, 32)
    masks[:, 8:24, 8:24] = 1
    count = 0 if empty else 2
    return dict(
        img=torch.rand(2, 3, 128, 128),
        batch_idx=torch.tensor([0.0, 1.0])[:count],
        cls=torch.zeros(count, 1),
        bboxes=torch.tensor([[0.5, 0.5, 0.5, 0.5]] * 2)[:count],
        masks=masks * (not empty),
    )


@pytest.mark.parametrize("path", YAMLS, ids=lambda p: p.stem)
def test_existing_yaml_pretraining_loss_backward_and_reload(path, pretrained):
    model = SegmentationModel(path, nc=1, verbose=False)
    model.load(pretrained, verbose=False)
    source = pretrained.state_dict()
    state = model.state_dict()
    for target, origin in model.yaml["pretrained_layer_map"].items():
        if origin < 0:
            continue
        for key, value in source.items():
            prefix = f"model.{origin}."
            if key.startswith(prefix):
                mapped = f"model.{target}." + key[len(prefix) :]
                if mapped in state and value.shape == state[mapped].shape:
                    torch.testing.assert_close(state[mapped], value, rtol=0, atol=0)
    # The common segmentation predictors must transfer even when their layer index moved.
    head = model.model[-1]
    torch.testing.assert_close(head.cv4[0][0].conv.weight, pretrained.model[-1].cv4[0][0].conv.weight)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    for empty in (False, True):
        model.zero_grad(set_to_none=True)
        loss, items = model.loss(make_batch(empty))
        assert torch.isfinite(loss).all() and torch.isfinite(items).all()
        loss.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    restored = SegmentationModel(deepcopy(model.yaml), nc=1, verbose=False)
    restored.load(model, verbose=False)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    with torch.inference_mode():
        prediction = model.eval()(torch.rand(1, 3, 128, 160))[0]
        assert all(torch.isfinite(t).all() for t in prediction)


def check_equivalent(module, x, reference, atol=2e-5):
    actual = module(x)
    expected = reference(module, x)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=2e-5)
    cotangent = torch.randn_like(actual)
    parameters = (x, *module.parameters())
    a = torch.autograd.grad(actual, parameters, cotangent, retain_graph=True)
    b = torch.autograd.grad(expected, parameters, cotangent)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=atol, rtol=2e-5)


def old_chroma(m, x):
    mixed = x + torch.einsum("oc,bchw->bohw", m.ccm, x)
    basis = (1 - (mixed.clamp(0, 1).unsqueeze(2) - m.knots).abs() * (m.lut_points - 1)).clamp_min(0)
    return mixed + torch.einsum("cp,bcphw->bchw", m.tone, basis)


def test_lut_output_and_gradients():
    module = EV4ChromaFront(3)
    with torch.no_grad():
        module.tone.normal_(0, 0.04)
    check_equivalent(module, torch.rand(2, 3, 19, 23, requires_grad=True) * 0.98 + 0.01, old_chroma)
    # Output identity and endpoint/clamping behavior remain correct.
    module.tone.data.zero_()
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0]).expand(1, 3, 1, 5)
    torch.testing.assert_close(module(x), x, rtol=0, atol=0)


def test_context_output_and_gradients():
    module = EV4IntegralContext(8)
    module.gain.data.fill_(0.2)

    def old(m, x):
        cat = torch.cat([F.interpolate(p(x), x.shape[-2:], mode="nearest") for p in m.pools], 1)
        return x + m.gain * m.mix(cat)

    check_equivalent(module, torch.rand(2, 8, 9, 13, requires_grad=True), old)


@pytest.mark.parametrize("scale", [2, 3])
def test_carafe_output_and_gradients(scale):
    module = CARAFE(8, scale=scale, c_mid=4)

    def old(m, x):
        b, c, h, w = x.shape
        weights = m.pix_shf(m.enc(m.comp(x))).softmax(1)
        patches = m.unfold(m.upsmp(x)).view(b, c, -1, h * scale, w * scale)
        return torch.einsum("bkhw,bckhw->bchw", weights, patches)

    check_equivalent(module, torch.rand(2, 8, 7, 9, requires_grad=True), old, atol=1e-4)


def test_reverse_gate_has_unbounded_logits():
    module = EV4ReverseRefine([8, 16]).eval()
    assert isinstance(module.to_prob.act, torch.nn.Identity)
    module.to_prob.bn.bias.data.fill_(-10)
    with torch.no_grad():
        assert module.to_prob(torch.zeros(1, 16, 4, 4)).sigmoid().max() < 0.001


@pytest.mark.parametrize("shape", [(5, 5), (5, 9)])
def test_dysample_zero_offset_is_bilinear_not_transposed(shape):
    module = DySample(8)
    module.offset.weight.data.zero_()
    module.offset.bias.data.zero_()
    x = torch.rand(2, 8, *shape, requires_grad=True)
    torch.testing.assert_close(module(x), F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))


@pytest.mark.parametrize("shift", [0.0, 1e-5, 0.1])
def test_nwd_perfect_and_near_match_gradients(shift):
    loss = BboxLoss(reg_max=1, hyp=SimpleNamespace(nwd_ratio=0.5))
    pred = (torch.tensor([[[1.0, 1.0, 3.0, 3.0]]]) + shift).requires_grad_()
    target = torch.tensor([[[1.0, 1.0, 3.0, 3.0]]])
    out, _ = loss(
        torch.ones(1, 1, 4),
        pred,
        torch.tensor([[2.0, 2.0]]),
        target,
        torch.ones(1, 1, 1),
        torch.tensor(1.0),
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([128, 128]),
        torch.tensor([[8.0]]),
    )
    out.backward()
    assert torch.isfinite(out) and torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("scheduler_class", [SMCScheduler, SMCAOV22Scheduler])
def test_controller_duration_schedule_resume_and_relative_noise(scheduler_class):
    p = torch.nn.Parameter(torch.ones(100))
    optimizer = torch.optim.AdamW([p], lr=0.001, betas=(0.937, 0.999))
    scheduler = scheduler_class(
        optimizer, warmup_steps=0, surface_patience=1, escape_max_duration=5, escape_cooldown=10, verbose=False
    )
    scheduler.s_t_peak, scheduler.s_t = 10.0, 0.001
    scheduler.step(1.0)
    assert scheduler._in_escape
    # Time limit must still work when no noise is injected / noise budget is exhausted.
    for _ in range(5):
        scheduler.step(1.0)
    assert not scheduler._in_escape and scheduler._cooldown_counter > 0
    p.grad = torch.full_like(p, 0.3)
    original = p.grad.clone()
    scheduler._add_relative_noise(p.grad, torch.randn_like(p), 0.001)
    torch.testing.assert_close((p.grad - original).norm() / original.norm(), torch.tensor(0.001), atol=1e-7, rtol=0.001)
    assert scheduler._get_cosine_lr(100000) == pytest.approx(scheduler.min_lr_ratio)
    scheduler.set_training_context([0.0002], warmup_active=True)
    scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0002)
    restored = scheduler_class(optimizer, verbose=False)
    restored.load_state_dict(deepcopy(scheduler.state_dict()))
    assert restored.state_dict() == scheduler.state_dict()


def test_trainer_observes_unscaled_gradients_and_normalized_loss():
    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(module.parameters())
    scheduler = SMCScheduler(optimizer, warmup_steps=0, verbose=False)

    class Scaler:
        def unscale_(self, opt):
            module.weight.grad.div_(100)

        def get_scale(self):
            return 100

        def step(self, opt):
            opt.step()

        def update(self):
            pass

    trainer = SimpleNamespace(
        model=module,
        optimizer=optimizer,
        smc_scheduler=scheduler,
        scaler=Scaler(),
        ema=None,
        loss_items=torch.tensor([1.0, 2.0, 3.0]),
    )
    module.weight.grad = torch.ones_like(module.weight) * 100
    BaseTrainer.optimizer_step(trainer)
    assert scheduler.prev_grad_norm == pytest.approx(2**0.5)
    assert scheduler._last_loss == 6


def test_query_vectorization_preserves_centroids_and_ids():
    criterion = object.__new__(v8SegmentationLoss)
    criterion.overlap = True
    masks = torch.zeros(2, 16, 19)
    masks[0, :2, :2] = 2  # vanished ID 1 is deliberately absent
    masks[0, 8:10, 12:14] = 3
    masks[0, 3:12, 3:11] = 4  # large object, not a query
    masks[1, -1, -1] = 1
    expected = torch.zeros(2, 1, 16, 19)
    for i in range(2):
        for instance in criterion.image_instance_masks(masks, torch.tensor([]), i, (16, 19)):
            if 0 < instance.sum() < 64:
                y, x = torch.nonzero(instance).float().mean(0).round().long().tolist()
                expected[i, 0, max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2] = 1
    actual = criterion.build_small_query_targets(masks, torch.tensor([]), 2, (16, 19))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_invalid_loss_ratios_fail_early():
    with pytest.raises(ValueError):
        custom_loss_overrides(SimpleNamespace(nwd_ratio=2), "Segment")
    with pytest.raises(ValueError):
        custom_loss_overrides(SimpleNamespace(citrus_vfl=float("nan")), "Segment")


def test_pre_review_same_graph_checkpoint_does_not_get_remapped_again():
    path = next(p for p in YAMLS if p.stem == "E70_chroma_front")
    source = SegmentationModel(path, nc=1, verbose=False)
    source.yaml.pop("pretrained_map_family")
    source.yaml.pop("pretrained_layer_map")
    source.model[0].tone.data.fill_(.2)
    restored = SegmentationModel(path, nc=1, verbose=False)
    restored.load(source, verbose=False)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "gains",
    [
        {"nwd_ratio": 0.5},
        {"citrus_vfl": 0.5},
        {"citrus_boundary": 0.25},
        {"citrus_query": 0.1},
        {"citrus_boundary": 0.25, "citrus_query": 0.1},
    ],
)
def test_loss_ablations_change_objective_and_have_finite_gradients(gains):
    path = next(
        p
        for p in YAMLS
        if p.stem.startswith("E52_" if "nwd_ratio" not in gains and "citrus_vfl" not in gains else "E40_")
    )
    model = SegmentationModel(path, nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, **gains})
    batch = make_batch()
    # Ensure a tiny instance produces a nonempty P2 query target.
    batch["masks"][:, 8:24, 8:24] = 0
    batch["masks"][:, 14:18, 14:18] = 1
    batch["bboxes"][:, 2:] = 0.125
    preds = model(batch["img"])
    enabled, _ = model.init_criterion()(preds, batch)
    for key in gains:
        setattr(model.args, key, 0.0)
    disabled, _ = model.init_criterion()(preds, batch)
    assert not torch.allclose(enabled, disabled)
    enabled.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
