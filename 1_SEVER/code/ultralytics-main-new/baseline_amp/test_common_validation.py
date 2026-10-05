"""CPU contracts for common evaluation, selection, state preservation, and official trainer entry points."""

# ruff: noqa: E402 -- test fixtures intentionally select the baseline helper import path

import ast
import copy
import json
import random
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import validation
import validation_adapters as adapters
from registry import COMMON_VALIDATION, CONTROLLED_PROTOCOL_ID, make_queue
from validation import CommonValidation, cap_predictions, collect_metrics, original_due, prepare_quick_view


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "prepared"
    images = root / "coco/images/val"
    images.mkdir(parents=True)
    Image.new("RGB", (1280, 640), (50, 160, 80)).save(images / "one.jpg")
    annotations = root / "coco/annotations"
    annotations.mkdir(parents=True)
    coco = dict(info={},
                licenses=[],
                categories=[dict(id=1, name="orange_immature")],
                images=[dict(id=7, file_name="one.jpg", width=1280, height=640)],
                annotations=[
                    dict(id=31,
                         image_id=7,
                         category_id=1,
                         iscrowd=0,
                         area=6400,
                         bbox=[100, 100, 80, 80],
                         segmentation=[[100, 100, 180, 100, 180, 180, 100, 180]])
                ])
    (annotations / "instances_val.json").write_text(json.dumps(coco))
    (root / "summary.json").write_text(json.dumps(dict(names=["orange_immature"])))
    return root


def test_queue_all_fourteen_jobs_share_protocol_and_pairs():
    queue = make_queue("all", [42], 300, 4, validation=COMMON_VALIDATION)
    assert len(queue) == 14 and {j["protocol"] for j in queue} == {CONTROLLED_PROTOCOL_ID}
    for a, b in zip(queue[::2], queue[1::2]):
        assert {k: v for k, v in a.items() if k not in ("name", "amp")
               } == {k: v for k, v in b.items() if k not in ("name", "amp")}
        assert a["validation"] is not b["validation"]
    assert len(make_queue("all", [42, 43, 44], 300, 4, validation=COMMON_VALIDATION)) == 42
    with pytest.raises(ValueError, match="protocol"):
        make_queue("all", [42], 300, 4, validation=dict(COMMON_VALIDATION, quick_size=160))


@pytest.mark.parametrize("epochs,expected", [(1, [1]), (3, [1, 3]), (20, [1, 10, 20]), (23, [1, 10, 20, 23])])
def test_original_schedule_same_for_every_family(epochs, expected):
    assert [epoch for epoch in range(1, epochs + 1) if original_due(epoch, epochs, COMMON_VALIDATION)] == expected


def test_quick_view_geometry_ids_padding_and_original_unchanged(prepared):
    source = prepared / "coco/annotations/instances_val.json"
    before = source.read_bytes()
    view = prepare_quick_view(prepared, COMMON_VALIDATION)
    coco = json.loads((view / "instances_val.json").read_text())
    assert coco["images"] == [dict(id=7, file_name="one.jpg", width=640, height=640)]
    ann = coco["annotations"][0]
    assert ann["id"] == 31 and ann["image_id"] == 7 and ann["category_id"] == 1
    assert ann["bbox"] == [50, 210, 40, 40] and ann["area"] == 1600
    with Image.open(view / "images/one.jpg") as image:
        assert image.format == "PNG" and image.getpixel((0, 0)) == (114, 114, 114)
    assert source.read_bytes() == before and prepare_quick_view(prepared, COMMON_VALIDATION) == view


def test_quick_view_retains_tiny_annotation_even_if_rasterization_zero(prepared):
    path = prepared / "coco/annotations/instances_val.json"
    coco = json.loads(path.read_text())
    coco["annotations"].append(
        dict(id=32,
             image_id=7,
             category_id=1,
             iscrowd=0,
             area=0.01,
             bbox=[200, 100, 0.1, 0.1],
             segmentation=[[200, 100, 200.1, 100, 200.1, 100.1, 200, 100.1]]))
    path.write_text(json.dumps(coco))
    root = prepare_quick_view(prepared, COMMON_VALIDATION)
    assert len(json.loads((root / "instances_val.json").read_text())["annotations"]) == 2
    assert 32 in json.loads((root / "view.json").read_text())["zero_area_instance_ids"]


def test_common_coco_perfect_mask_original_and_quick(prepared):
    from pycocotools.coco import COCO

    quick = prepare_quick_view(prepared, COMMON_VALIDATION)
    for annotation, images in [(prepared / "coco/annotations/instances_val.json", prepared / "coco/images/val"),
                               (quick / "instances_val.json", quick / "images")]:
        coco = COCO(str(annotation))
        mask = coco.annToMask(coco.dataset["annotations"][0])
        result = collect_metrics(annotation, images, lambda _: ([mask], [0.9], [1]), COMMON_VALIDATION)
        assert result["mask_ap_50_95"] == pytest.approx(1)
        assert result["mask_precision"] == result["mask_recall"] == 1


def test_prediction_limit_and_score_order_shared_with_pr():
    predictions = [dict(image_id=1, category_id=1, score=i / 200) for i in range(180)]
    selected = cap_predictions(predictions)
    assert len(selected) == 100 and selected[0]["score"] == 179 / 200
    assert selected[-1]["score"] == 80 / 200
    with pytest.raises(ValueError, match="finite"):
        cap_predictions([dict(image_id=1, category_id=1, score=float("nan"))])


def test_fixed_threshold_pr_uses_same_cap_as_ap(prepared):
    from pycocotools.coco import COCO

    annotation = prepared / "coco/annotations/instances_val.json"
    coco = COCO(str(annotation))
    mask = coco.annToMask(coco.dataset["annotations"][0])
    result = collect_metrics(annotation, prepared / "coco/images/val", lambda _: ([mask] * 120, [0.9] * 120, [1] * 120),
                             COMMON_VALIDATION)
    assert result["prediction_count"] == 100 and result["false_positive"] == 99
    assert result["mask_precision"] == pytest.approx(0.01) and result["mask_recall"] == 1


def test_invalid_mask_coordinates_and_ids_fail(prepared):
    for mask, label in [(np.zeros((4, 4)), 1), (np.zeros((640, 1280)), 0)]:
        with pytest.raises(ValueError, match="contract"):
            collect_metrics(prepared / "coco/annotations/instances_val.json", prepared / "coco/images/val", lambda _:
                            ([mask], [0.9], [label]), COMMON_VALIDATION)


def test_validation_preserves_rng_modes_gradients_and_flags_even_on_error():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.train()
    model[1].eval()
    model[0].weight.grad = torch.ones_like(model[0].weight)
    modes = [m.training for m in model.modules()]
    py_state, np_state, ts = random.getstate(), np.random.get_state(), torch.get_rng_state()
    deterministic, warn_only = torch.are_deterministic_algorithms_enabled(
    ), torch.is_deterministic_algorithms_warn_only_enabled()
    with pytest.raises(RuntimeError, match="probe"):
        with validation.validation_state(model):
            assert not model.training and not torch.is_grad_enabled()
            assert torch.are_deterministic_algorithms_enabled()
            random.random(), np.random.rand(), torch.rand(2)
            raise RuntimeError("probe")
    assert [m.training for m in model.modules()] == modes
    assert random.getstate() == py_state and np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), ts) and torch.all(model[0].weight.grad == 1)
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only


def test_half_master_weights_are_not_mutated_for_validation():
    model = torch.nn.Linear(2, 1).half()
    with pytest.raises(RuntimeError, match="FP32"):
        with validation.validation_state(model):
            pass
    assert model.weight.dtype == torch.float16


def fake_metrics(score):
    return dict(mask_ap_50_95=score, mask_ap_50=score, mask_recall=score, seconds=0.1)


def test_checkpoint_selection_never_uses_quick_and_ties_keep_earlier(prepared, tmp_path, monkeypatch):
    job = dict(validation=copy.deepcopy(COMMON_VALIDATION), epochs=11)
    manager = CommonValidation(job, prepared, tmp_path / "run")
    original_scores = iter([0.7, 0.7, 0.8])
    monkeypatch.setattr(
        validation, "collect_metrics",
        lambda ann, *args: fake_metrics(0.99 if "validation_quick640" in str(ann) else next(original_scores)))
    model = torch.nn.Linear(2, 1)
    selected_epochs = []

    def save(epoch):
        selected_epochs.append(epoch)
        path = tmp_path / f"epoch_{epoch}.pth"
        path.touch()
        return path

    reports = [manager.run(epoch, model, None, save, "regular") for epoch in (1, 2, 10, 11)]
    assert selected_epochs == [1, 11] and manager.best["epoch"] == 11
    assert reports[1]["original"] is None and not reports[1]["selected"]
    assert not reports[2]["selected"]


def test_noneligible_regular_rfdetr_cannot_select(prepared, tmp_path, monkeypatch):
    manager = CommonValidation(dict(validation=COMMON_VALIDATION, epochs=1), prepared, tmp_path)
    monkeypatch.setattr(validation, "collect_metrics", lambda *args: fake_metrics(0.99))
    report = manager.run(1,
                         torch.nn.Linear(2, 1),
                         None,
                         lambda _: pytest.fail("must not save"),
                         "regular",
                         eligible=False)
    assert report["original"] is None and manager.best is None


def test_mmdet_loop_saves_evaluated_ema_not_double_swapped_weights(tmp_path, monkeypatch):
    pytest.importorskip("mmengine")
    from mmengine.config import Config
    from mmengine.registry import LOOPS

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    called = []

    class FakeManager:
        protocol = COMMON_VALIDATION

        def __init__(self, *args):
            pass

        def run(self, epoch, candidate, predictor, save, role):
            assert epoch == 1 and role == "ema" and candidate.weight.item() == 9
            save(epoch)
            return dict(quick=fake_metrics(0.7), original=fake_metrics(0.7))

    class EMAHook:
        pass

    def hook(event, **kwargs):
        called.append(event)
        if event == "before_val_epoch":
            model.weight.data.fill_(9)
        if event == "after_val_epoch":
            model.weight.data.fill_(1)

    cfg = Config(
        dict(val_cfg=dict(type="ValLoop"),
             train_cfg=dict(val_interval=1),
             custom_hooks=[dict(type="EarlyStoppingHook")],
             default_hooks=dict(checkpoint=dict(save_best="coco/segm_mAP"))))
    (tmp_path / "train").mkdir()
    monkeypatch.setattr(adapters, "CommonValidation", FakeManager)
    monkeypatch.setattr(adapters, "mmdet_predictor", lambda *args: None)
    adapters.install_mmdet_loop({}, tmp_path, tmp_path, cfg)
    loop = object.__new__(LOOPS.get("CitrusCommonValLoop"))
    loop._runner = types.SimpleNamespace(
        model=model,
        hooks=[EMAHook()],
        epoch=1,
        call_hook=hook,
        train_dataloader=types.SimpleNamespace(dataset=types.SimpleNamespace(metainfo=dict(classes=("fruit",)))))
    loop.run()
    saved = torch.load(tmp_path / "train/best_unified.pth", weights_only=False)
    assert saved["state_dict"]["weight"].item() == 9 and model.weight.item() == 1
    assert called == ["before_val", "before_val_epoch", "after_val_epoch", "after_val"]
    assert cfg.custom_hooks == [] and cfg.default_hooks.checkpoint.save_best is None
    assert not hasattr(model, "cfg")


@pytest.mark.parametrize("model_name", ["rtmdet_ins_tiny", "mask_rcnn_r50", "solov2_light_r18"])
def test_controlled_mmdet_preserves_rpn_training_proposals_and_caps_only_final(tmp_path, monkeypatch, model_name):
    pytest.importorskip("mmengine")
    import worker

    root = Path("E:/mastercode/_work/baseline_bug2_20260926/deps/mmdet/.mim")
    if not root.is_dir():
        pytest.skip("Pinned official MMDetection config fixture unavailable")
    monkeypatch.setattr(worker, "official_mmdet_root", lambda: root)
    (tmp_path / "summary.json").write_text(json.dumps(dict(names=["orange_immature"])))
    job = next(j for j in make_queue("mmdet", [42], 300, 4, validation=COMMON_VALIDATION)
               if j["model"] == model_name)
    cfg = worker.mmdet_config(job, tmp_path, tmp_path / "run", None)
    final = cfg.model.test_cfg.get("rcnn", cfg.model.test_cfg)
    assert final.score_thr == 0.001 and final.max_per_img == 100
    if model_name == "mask_rcnn_r50":
        assert cfg.model.train_cfg.rpn_proposal.max_per_img == 1000
        assert cfg.model.test_cfg.rpn.max_per_img == 1000


def test_file_only_mmdet_prediction_removes_gt_loader_without_changing_resize():
    from mmdet_common import prediction_pipeline

    original = [dict(type="LoadImageFromFile"), dict(type="Resize", scale=(640, 640), keep_ratio=True),
                dict(type="mmdet.LoadAnnotations", with_bbox=True, with_mask=False),
                dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"))]
    before = copy.deepcopy(original)
    pipeline = prediction_pipeline(original)
    assert original == before and [stage["type"] for stage in pipeline] == [
        "LoadImageFromFile", "Resize", "PackDetInputs"
    ]
    assert pipeline[1]["keep_ratio"] is True and pipeline[1]["scale"] == (640, 640)


def test_rf_pinned_entry_patch_restores_and_selects_only_ema(tmp_path, monkeypatch):
    current = {}
    raw, ema = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)

    def train(model, epoch):
        current["trained"] = epoch

    def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None):
        pass

    fake_main = types.ModuleType("rfdetr.main")
    fake_main.train_one_epoch, fake_main.evaluate = train, evaluate
    package = types.ModuleType("rfdetr")
    package.main = fake_main
    monkeypatch.setitem(sys.modules, "rfdetr", package)
    monkeypatch.setitem(sys.modules, "rfdetr.main", fake_main)
    calls = []

    class FakeManager:
        protocol = COMMON_VALIDATION

        def __init__(self, *args):
            pass

        def run(self, epoch, model, predictor, save, role, eligible):
            calls.append((epoch, role, eligible))
            if eligible:
                save(epoch)
            # All 12 named COCO stats are passed to the official loop.
            return dict(quick={
                f"mask_{s}": 0.7 for s in ("ap_50_95", "ap_50", "ap_75", "ap_small", "ap_medium", "ap_large", "ar_1",
                                           "ar_10", "ar_100", "ar_small", "ar_medium", "ar_large")
            },
                        original=None)

    (tmp_path / "train").mkdir()
    monkeypatch.setattr(adapters, "CommonValidation", FakeManager)
    monkeypatch.setattr(adapters, "rfdetr_predictor", lambda *args: None)
    owner = types.SimpleNamespace(model=types.SimpleNamespace(ema_m=types.SimpleNamespace(module=ema)))
    job = dict(recipe=dict(imgsz=624))
    with pytest.raises(RuntimeError, match="probe"):
        with adapters.rfdetr_validation(owner, job, tmp_path, tmp_path):
            fake_main.train_one_epoch(raw, 0)
            args = types.SimpleNamespace(use_ema=True)
            for candidate in (raw, ema):
                stats, evaluator = fake_main.evaluate(candidate,
                                                      None,
                                                      types.SimpleNamespace(num_select=300),
                                                      None,
                                                      None,
                                                      "cpu",
                                                      args=args)
                assert len(stats["coco_eval_masks"]) == 12 and "coco_eval_bbox" not in stats
                assert evaluator is None
            raise RuntimeError("probe")
    assert calls == [(1, "regular", False), (1, "ema", True)]
    assert fake_main.train_one_epoch is train and fake_main.evaluate is evaluate
    assert (tmp_path / "train/checkpoint_best_unified.pth").is_file()


def test_rf_decoder_matches_unmodified_official_postprocess(tmp_path):
    # Extract the official CPU-testable method, not a rewritten golden implementation.
    wheel = Path("E:/mastercode/_work/baseline_legacy78_source/rfdetr-1.4.0.post0-py3-none-any.whl")
    if not wheel.is_file():
        pytest.skip("Pinned RF-DETR wheel source fixture unavailable")
    from torchvision.transforms import functional as TF
    from torch.nn import functional as F
    with zipfile.ZipFile(wheel) as archive:
        source = archive.read("rfdetr/models/lwdetr.py").decode()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PostProcess")
    env = dict(torch=torch, nn=torch.nn, F=F, box_ops=types.SimpleNamespace(box_cxcywh_to_xyxy=lambda boxes: boxes))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "official_postprocess", "exec"), env)
    generator = torch.Generator().manual_seed(31)
    outputs = dict(pred_logits=torch.randn(1, 6, 2, generator=generator),
                   pred_boxes=torch.randn(1, 6, 4, generator=generator),
                   pred_masks=torch.randn(1, 6, 5, 7, generator=generator))

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, tensor):
            expected = TF.resize(TF.normalize(TF.to_tensor(Image.open(path)), [.485, .456, .406], [.229, .224, .225]),
                                 (24, 24))
            assert torch.equal(tensor[0], expected)
            return outputs

    path = tmp_path / "image.png"
    Image.new("RGB", (31, 23), (20, 80, 150)).save(path)
    model = Model()
    expected = env["PostProcess"](num_select=12)(outputs, torch.tensor([[23, 31]]))[0]
    keep = (expected["labels"] != 0) & (expected["scores"] >= 0.001)
    with validation.validation_state(model):
        masks, scores, labels = adapters.rfdetr_predictor(model, 24, 12, COMMON_VALIDATION)(path)
    assert np.array_equal(np.asarray(masks), expected["masks"][keep, 0].numpy())
    assert np.array_equal(scores, expected["scores"][keep].numpy())
    assert np.array_equal(labels, expected["labels"][keep].numpy())
