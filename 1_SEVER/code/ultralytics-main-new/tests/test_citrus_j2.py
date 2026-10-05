"""J V2 Glance-Gaze contracts: YAMLs, density supervision, selection geometry and data factors."""

import ast
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from citrus_j2_batch import SOURCES, settings
from citrus_j2_data import AntialiasYOLODataset, _AreaDownscale, _patched, make_trainer, small_count
from citrus_j2_suite import CACHE, FACTORS, LEGACY_NAMES as NAMES, SUITES, YAML_DIR, select_names
from eval_citrus_j2_gaze import box_density, density_peaks, fit_s50, greedy_windows, letterbox_geometry
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build as data_build
from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules.citrus_j2 import SegmentCitrusJ2
from ultralytics.utils.citrus_j2_loss import SMALL_AREA, J2SegmentationLoss, density_focal_loss

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def batch_fixture(overlap=True, empty=False):
    """96x128 input, masks at input/2: one 12x16 (large-ish) and one 4x4 (small) instance."""
    masks = torch.zeros((2, 48, 64)) if overlap else torch.zeros((0 if empty else 2, 48, 64))
    if not empty:
        masks[0, 12:24, 16:32] = 1
        masks[1, 22:26, 30:34] = 1
    return dict(
        img=torch.rand(2, 3, 96, 128),
        masks=masks,
        batch_idx=torch.tensor([] if empty else [0, 1]),
        cls=torch.zeros((0 if empty else 2, 1)),
        bboxes=torch.zeros((0, 4)) if empty else torch.tensor([[0.375, 0.375, 0.25, 0.25], [0.5, 0.5, 0.0625, 0.0833]]),
    )


def build(name):
    model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment", verbose=False).model
    model.args = get_cfg(overrides={"overlap_mask": True, "mask_ratio": 2})
    return model


def test_generated_yamls_are_current():
    from scripts.generate_citrus_j2_yaml import main

    main(check=True)


@pytest.mark.parametrize("name", NAMES)
def test_yaml_build_loss_backward_eval(name):
    model = build(name)
    density = FACTORS[name]["density"]
    assert isinstance(model.model[-1], SegmentCitrusJ2) == density
    model.train()
    batch = batch_fixture()
    preds = model(batch["img"])
    assert ("j2_density" in preds) == density
    if density:
        assert preds["j2_density"].shape == (2, 1, 12, 16)
        assert isinstance(model.init_criterion(), J2SegmentationLoss)
    loss, components = model.loss(batch, preds)
    assert components.shape == (5,) and torch.isfinite(loss).all()
    assert (components[4] > 0) == density
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        output, raw = model(batch["img"])
    assert output[0].shape == (2, 37, 252) and output[1].shape[1] == 32
    assert ("j2_density" in raw) == density


def test_graph_identical_to_j00_plus_density_only():
    ref = YOLO(str(ROOT / "0_orange_yaml/J_series/J00_yolo11n_control.yaml"), task="segment", verbose=False).model
    ref_sd = ref.state_dict()
    for name in NAMES:
        sd = build(name).state_dict()
        assert all(k in sd and sd[k].shape == v.shape for k, v in ref_sd.items())
        assert {k.split(".")[2] for k in sd if k not in ref_sd} == ({"density"} if FACTORS[name]["density"] else set())
        extra = sum(v.numel() for k, v in sd.items() if k not in ref_sd and "num_batches" not in k)
        assert extra < 30_000


def test_density_prior_and_readout_detach():
    for name, detach in (("J21_glance_density", False), ("J22_density_readout", True)):
        model = build(name).train()
        head = model.model[-1]
        assert torch.sigmoid(head.density[-1].bias).item() == pytest.approx(SegmentCitrusJ2.prior, rel=1e-4)
        model(batch_fixture()["img"])["j2_density"].sum().backward()
        backbone_grad = model.model[0].conv.weight.grad
        assert (backbone_grad is None or backbone_grad.abs().sum() == 0) == detach
        assert head.density[0].conv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("overlap", [True, False])
def test_density_targets_only_small_instances(overlap):
    model = build("J21_glance_density")
    criterion = J2SegmentationLoss(model)
    batch = batch_fixture(overlap)
    heat = criterion.density_targets(batch["masks"], batch["batch_idx"], 2, (12, 16), (96.0, 128.0))
    # Instance 0: 12x16 mask px -> 24x32 input px = 768 < 1024 -> small. Instance 1: 4x4 -> 64 input px -> small.
    assert heat.shape == (2, 1, 12, 16) and heat.max() == 1
    peaks = torch.nonzero(heat == 1)
    assert len(peaks) == 2
    # centre of instance 1 at mask (23.5, 31.5) -> grid ((23.5+.5)*12/48-.5, (31.5+.5)*16/64-.5) = (5.5, 7.5)
    assert any(int(p[0]) == 1 and int(p[2]) in (5, 6) and int(p[3]) in (7, 8) for p in peaks)
    big = batch_fixture(overlap)
    big["masks"][0 if overlap else 0, 4:44, 8:60] = 1  # 40x52 mask px = 8320 input px -> not small
    heat_big = criterion.density_targets(big["masks"], big["batch_idx"], 2, (12, 16), (96.0, 128.0))
    assert heat_big[0].max() < 1 if not overlap else True
    empty = criterion.density_targets(torch.zeros(2, 48, 64), torch.tensor([]), 2, (12, 16), (96.0, 128.0))
    assert empty.abs().sum() == 0
    assert SMALL_AREA == 1024


def test_focal_loss_monotone_and_empty_safe():
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 3, 3] = 1
    good = torch.full_like(target, -6.0)
    good[0, 0, 3, 3] = 6.0
    assert density_focal_loss(good, target) < density_focal_loss(torch.zeros_like(target), target)
    assert torch.isfinite(density_focal_loss(torch.randn(1, 1, 8, 8), torch.zeros(1, 1, 8, 8)))


def test_empty_batch_loss_finite():
    model = build("J25_glance_gaze").train()
    batch = batch_fixture(empty=True)
    loss, components = model.loss(batch, model(batch["img"]))
    assert torch.isfinite(loss).all() and components[4] >= 0


def test_greedy_coverage_order_threshold_and_no_zero_gain():
    windows = [(0, 0, 10, 10), (5, 0, 15, 10), (20, 0, 30, 10)]
    points = np.array([[1, 1], [6, 1], [7, 1], [25, 1]], float)
    weights = np.array([1.0, 1.0, 1.0, 0.4])
    # w0 covers 3.0; w1 then has ZERO marginal gain (its points are covered) and is never chosen; w2 adds 0.4.
    assert greedy_windows(points, weights, windows, 0.0) == [0, 2]
    chosen = greedy_windows(points, weights, windows, 0.5)
    assert chosen == [0]  # w0 covers 3.0; then w1 marginal 0, w2 marginal 0.4 < 0.5
    assert greedy_windows(points, weights, windows, 0.5, max_windows=0) == []
    assert greedy_windows(np.zeros((0, 2)), np.zeros(0), windows, 0.1) == []


@pytest.mark.parametrize("hw", [(3072, 3072), (3072, 4096), (4096, 3072), (1080, 1920)])
def test_letterbox_geometry_matches_predictor(hw):
    h, w = hw
    out = LetterBox(new_shape=640, auto=True, stride=32)(image=np.zeros((h, w, 3), np.uint8))
    shape = (out.shape[0] // 8, out.shape[1] // 8)
    r, left, top = letterbox_geometry(h, w, shape, 8, 640)
    assert r == pytest.approx(min(640 / h, 640 / w))
    assert 0 <= left < 32 and 0 <= top < 32
    prob = np.zeros(shape, np.float32)
    gy, gx = shape[0] // 3, shape[1] // 2
    prob[gy, gx] = 0.9
    points, values = density_peaks(prob, h, w, 8, 640)
    assert len(points) == 1 and values[0] == pytest.approx(0.9)
    x, y = points[0]
    assert box_density(prob, (x - 1, y - 1, x + 1, y + 1), h, w, 8, 640) == pytest.approx(0.9)


def test_s50_fit_recovers_threshold():
    rng = np.random.default_rng(1)
    sizes = np.exp(rng.uniform(np.log(4), np.log(80), 3000))  # log-uniform: enough samples near the threshold
    hits = (rng.random(3000) < 1 / (1 + np.exp(-(np.log(sizes) - np.log(15.0)) / 0.3))).astype(float)
    s50, tau = fit_s50(sizes, hits)
    assert abs(s50 - 15.0) < 1.0 and abs(tau - 0.3) < 0.08


def test_small_count_uses_global_glance_raster(tmp_path):
    label = tmp_path / "v.txt"
    # View = left half of a 3072x3072 image. Square of side .05 view-width -> 76.8 px orig -> 16 px @640 -> small.
    label.write_text("0 0.1 0.1 0.15 0.1 0.15 0.2 0.1 0.2\n0 0 0 1 0 1 1 0 1\n", encoding="utf-8")
    assert small_count(label, (0, 0, 1536, 3072), (3072, 3072)) == 1


def test_area_downscale_proxy_and_antialias_dataset(tmp_path):
    image = (np.indices((300, 300)).sum(0) % 2 * 255).astype(np.uint8)[..., None].repeat(3, 2)
    proxy = _AreaDownscale()
    torch.testing.assert_close(
        torch.from_numpy(proxy.resize(image, (60, 60), interpolation=cv2.INTER_LINEAR)),
        torch.from_numpy(cv2.resize(image, (60, 60), interpolation=cv2.INTER_AREA)),
    )
    up = proxy.resize(image[:30, :30], (60, 60), interpolation=cv2.INTER_LINEAR)
    torch.testing.assert_close(torch.from_numpy(up), torch.from_numpy(cv2.resize(image[:30, :30], (60, 60))))
    assert proxy.INTER_AREA == cv2.INTER_AREA
    original = data_build.YOLODataset
    with _patched(data_build, "YOLODataset", AntialiasYOLODataset):
        assert data_build.YOLODataset is AntialiasYOLODataset
    assert data_build.YOLODataset is original


def test_trainer_factors_and_suites():
    assert len(NAMES) == 8 and set(NAMES).issubset(FACTORS)
    assert set(select_names("all")) == set(FACTORS) and len(select_names("priority")) == 5
    assert set(SUITES["gaze"]) | set(SUITES["ablation"]) | set(SUITES["resolution"]) == set(NAMES)
    assert CACHE["J27_highres_1280"] is False and CACHE["J20_glance_anchor"] is True
    assert FACTORS["J25_glance_gaze"] == dict(density=True, detach=False, gaze_train=True, antialias=True, imgsz=640)
    for name in NAMES:
        trainer = make_trainer(FACTORS[name])
        assert trainer.__name__.startswith("J2Trainer") and trainer.get_validator.__qualname__.startswith("EV6")
    recipe = settings(300, 42, 1280, False)
    assert recipe["imgsz"] == 1280 and recipe["pretrained"] is False and recipe["amp"] is False
    assert recipe["mask_ratio"] == 2 and recipe["copy_paste"] == 0.3 and recipe["cache"] is False
    assert all((ROOT / s).is_file() for s in SOURCES)


@pytest.mark.parametrize(
    "path",
    ["citrus_j2_batch.py", "citrus_j2_data.py", "citrus_j2_suite.py", "eval_citrus_j2_gaze.py", "RUN_CITRUS_J2.py",
     "ultralytics/nn/modules/citrus_j2.py", "ultralytics/utils/citrus_j2_loss.py"],
)
def test_python38_syntax(path):
    ast.parse((ROOT / path).read_text(encoding="utf-8"), feature_version=(3, 8))


def test_model_index_registered():
    text = (ROOT / "0_orange_yaml/MODEL_INDEX.csv").read_text(encoding="utf-8-sig")
    assert all(f"J_V2_series/{n}.yaml" in text for n in NAMES)
    assert json.dumps(FACTORS)  # serialisable for the run record
