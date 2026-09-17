"""V11 standard YOLO entry, full gradients, transfer mapping, reparameterization and selective objectives."""

import ast
import importlib
import json
from copy import deepcopy

import pytest
import torch
import yaml

from citrus_e_v11_suite import FACTORS, NAMES, ROOT, RUN_OVERRIDES, YAML_DIR
from scripts.generate_citrus_e_v11_yaml import configs
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.nn.modules.citrus_e_v11 import EV11StablePool
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.citrus_e_v11_loss import EV11SegmentationLoss, EV11TinyAssigner, local_mask_contrast
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def make_batch(empty=False, overlap=True):
    masks = torch.zeros(2, 64, 64)
    masks[:, 12:52, 12:52] = 1
    masks[:, 3:6, 3:6] = 2
    if not overlap:
        masks = torch.cat([(masks[i] == torch.tensor([1, 2])[:, None, None]).float() for i in range(2)])
    return dict(
        img=torch.rand(2, 3, 128, 128),
        batch_idx=torch.tensor([0.0, 0.0, 1.0, 1.0])[: 0 if empty else 4],
        cls=torch.zeros(0 if empty else 4, 1),
        bboxes=torch.tensor([[0.5, 0.5, 0.625, 0.625], [0.0703125, 0.0703125, 0.046875, 0.046875]] * 2)[
            : 0 if empty else 4
        ],
        masks=(masks * 0 if empty else masks) if overlap or not empty else masks[:0],
    )


@pytest.mark.parametrize("i", range(len(NAMES)))
@pytest.mark.parametrize("empty", [False, True])
def test_backward_validation_fuse(i, empty):
    m = SegmentationModel(YAML_DIR / f"{NAMES[i]}.yaml", nc=1, verbose=False)
    m.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True, "nwd_ratio": 0.0})
    total, comp = m.loss(make_batch(empty))
    assert isinstance(m.criterion, EV11SegmentationLoss)
    assert len(comp) == 5 and torch.isfinite(total).all()
    total.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    if FACTORS[i][1] and not empty:
        assert m.model[3].mix.conv.weight.grad is not None
        assert m.model[10].gain.grad.abs().sum() > 0
        assert m.model[-1].persistent_gain.grad.abs().sum() > 0
    if i >= 10 and not empty:
        assert m.model[14].m[0].attn.context_gain.grad.abs().sum() > 0
    assert m.model[-1].stride.tolist() == [8.0, 16.0, 32.0]
    assert 0 < get_flops(m, 640) < 11
    with torch.no_grad():
        m.eval()
        batch = make_batch(empty)
        pred = m(batch["img"])
        val, _ = m.loss(batch, pred)
        assert torch.isfinite(val).all()
        assert pred[1]["proto"].shape == (2, 32, 64, 64)
    with torch.inference_mode():
        m.model[0].structure.gain.fill_(0.2) if FACTORS[i][0] else None
        x = torch.rand(1, 3, 128, 160)
        a = m(x)
        b = deepcopy(m).fuse(verbose=False)(x)
        assert a[0][0].shape[-1] == 420
        assert a[0][1].shape == (1, 32, 64, 80)
        torch.testing.assert_close(a[0][0], b[0][0], rtol=0.002, atol=0.02)
        torch.testing.assert_close(a[0][1], b[0][1], rtol=0.002, atol=0.02)


@pytest.mark.parametrize("i", [0, 1])
def test_exact_v10_replays(i):
    old_name = ["V10_05_structure_pair", "V10_04_detail_transport"][i]
    old = SegmentationModel(ROOT / f"0_orange_yaml/E_V10_series/{old_name}.yaml", nc=1, verbose=False).eval()
    new = SegmentationModel(YAML_DIR / f"{NAMES[i]}.yaml", nc=1, verbose=False).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    with torch.inference_mode():
        x = torch.rand(1, 3, 128, 128)
        a, b = old(x), new(x)
        torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
        torch.testing.assert_close(a[0][1], b[0][1], rtol=0, atol=0)
    assert old.model[-1].tiny_dice_gain == new.model[-1].tiny_dice_gain


@pytest.mark.parametrize("i", range(len(NAMES)))
def test_pretrained_semantic_mapping_reload(i, tmp_path):
    api = YOLO(str(YAML_DIR / f"{NAMES[i]}.yaml"), verbose=False).load(str(ROOT / "yolo11n-seg.pt"))
    source = YOLO(str(ROOT / "yolo11n-seg.pt"), verbose=False).model.state_dict()
    state = api.model.state_dict()
    if FACTORS[i][1]:
        pairs = [
            ("model.5.cv1.conv.weight", "model.4.cv1.conv.weight"),
            ("model.8.cv1.conv.weight", "model.6.cv1.conv.weight"),
            ("model.23.cv1.conv.weight", "model.19.cv1.conv.weight"),
            ("model.27.cv2.0.1.conv.weight", "model.23.cv2.0.1.conv.weight"),
        ]
    else:
        pairs = [
            ("model.6.cv1.conv.weight", "model.6.cv1.conv.weight"),
            ("model.19.cv1.conv.weight", "model.19.cv1.conv.weight"),
        ]
    for new, old in pairs:
        torch.testing.assert_close(state[new], source[old], rtol=0, atol=0)
    p = tmp_path / "model.pt"
    api.save(str(p))
    loaded = YOLO(str(p), verbose=False)
    assert type(loaded.model.model[-1]).__name__ == "SegmentCitrusEV11"
    assert loaded.model.yaml == api.model.yaml


def test_tiny_assignment_is_selective_and_zero_replays():
    gt = torch.tensor([[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 64.0, 64.0]])
    pred = torch.tensor([[1.0, 1.0, 5.0, 5.0], [1.0, 1.0, 63.0, 63.0]])
    base = TaskAlignedAssigner().iou_calculation(gt, pred)
    zero = EV11TinyAssigner(mix=0).iou_calculation(gt, pred)
    value = EV11TinyAssigner(mix=0.2).iou_calculation(gt, pred)
    torch.testing.assert_close(base, zero, rtol=0, atol=0)
    assert value[0] > base[0]
    assert value[1] == base[1]
    assert torch.isfinite(value).all() and (value <= 1).all()


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("empty", [True, False])
def test_ring_and_matching_with_both_mask_layouts(overlap, empty):
    m = SegmentationModel(YAML_DIR / f"{NAMES[8]}.yaml", nc=1, verbose=False)
    m.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": overlap, "nwd_ratio": 0.0})
    total, _ = m.loss(make_batch(empty, overlap))
    total.sum().backward()
    assert torch.isfinite(total).all()
    if not empty:
        assert m.criterion.last_ring > 0


def test_contrast_ring_excludes_other_fruit_and_empty():
    proto = torch.randn(2, 16, 16, requires_grad=True)
    coeff = torch.randn(1, 2, requires_grad=True)
    gt = torch.zeros(1, 16, 16)
    gt[:, 4:10, 4:10] = 1
    # Entire image labelled fruit -> no exterior negative, no fabricated label.
    loss = local_mask_contrast(coeff, proto, gt, torch.tensor([0]), torch.ones(16, 16))
    assert loss == 0
    loss.backward()
    assert torch.isfinite(proto.grad).all()
    empty = local_mask_contrast(coeff[:0], proto, gt[:0], torch.tensor([], dtype=torch.long), gt[0])
    assert empty == 0


def test_manifest_recipe_and_provenance():
    assert configs() == [yaml.safe_load((YAML_DIR / f"{n}.yaml").read_text()) for n in NAMES]
    assert len(list(YAML_DIR.glob("*.yaml"))) == len(NAMES) == 12
    assert len({json.dumps(d, sort_keys=True) for d in RUN_OVERRIDES.values()}) == 1
    entry = importlib.import_module("RUN_CITRUS_E_V11")
    assert not entry.DRY_RUN and entry.EPOCHS == 300
    assert FACTORS[6][4] == 0.2 and FACTORS[7][4] == 0
    runner = ROOT / "20260917_citrus_e_v11_batch.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "source_files" for t in n.targets)
    )
    files = eval(
        compile(ast.Expression(node.value), str(runner), "eval"),
        dict(ROOT=ROOT, YAML_DIR=YAML_DIR, NAMES=NAMES, Path=type(ROOT), __file__=str(runner)),
    )
    assert all(p.is_file() for p in files)
    for p in files:
        if p.suffix == ".py" and "v11" in p.stem.lower():
            ast.parse(p.read_text(encoding="utf-8"), feature_version=(3, 8))
    integrated = SegmentationModel(YAML_DIR / f"{NAMES[6]}.yaml", nc=1, verbose=False)
    anchor = SegmentationModel(YAML_DIR / f"{NAMES[0]}.yaml", nc=1, verbose=False)
    assert get_flops(integrated, 640) < get_flops(anchor, 640)
    assert sum(p.numel() for p in integrated.parameters()) < sum(p.numel() for p in anchor.parameters())


def test_selective_pool_matches_author_code_with_numerical_guard():
    torch.manual_seed(2)
    x = torch.randn(2, 16, 4, 5, requires_grad=True)
    pool = EV11StablePool(16)
    tokens = x.flatten(2).transpose(1, 2)
    spectrum = torch.fft.fftshift(torch.fft.fft(tokens.detach(), dim=-1), dim=-1)
    filtered = torch.fft.ifft(torch.fft.ifftshift(spectrum * pool.kernel, dim=-1), dim=-1).real
    score = tokens.detach() / (filtered - tokens.detach()).abs().clamp_min(1e-6)
    idx = score.topk(1, dim=1, sorted=False).indices
    expected = tokens.gather(1, idx).mean(1).unsqueeze(-1).unsqueeze(-1)
    value = pool(x)
    torch.testing.assert_close(value, expected, rtol=0, atol=0)
    value.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    # Flat/zero features must not create Inf/NaN stability scores.
    assert torch.isfinite(pool(torch.zeros_like(x))).all()
    torch.testing.assert_close(EV11StablePool(16, False)(x), x.mean((2, 3), keepdim=True))


def test_paper_pair_changes_only_aggregation():
    gap, selective = configs()[-2:]
    for cfg in (gap, selective):
        cfg.pop("pretrained_map_family")
    assert gap["backbone"][14][3] == [1024, False]
    assert selective["backbone"][14][3] == [1024, True]
    gap["backbone"][14][3][-1] = True
    assert gap == selective
    models = [SegmentationModel(YAML_DIR / f"{NAMES[i]}.yaml", nc=1, verbose=False) for i in (10, 11)]
    assert sum(p.numel() for p in models[0].parameters()) == sum(p.numel() for p in models[1].parameters())
    for m in models:
        assert type(m.model[14].m[0].attn).__name__ == "EV11ContextAttention"
