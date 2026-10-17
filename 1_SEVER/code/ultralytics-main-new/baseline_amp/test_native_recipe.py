"""CPU-only recipe/worker contracts; no real checkpoint download or backend training."""

import copy
import json
import sys
import types
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import pytest

from baseline_amp.native_recipe import (
    MMDET_REFERENCES,
    RFDETR_NATIVE_TRAIN,
    is_native,
    mmdet_apply_native,
    rfdetr_options,
    yolo_checkpoint,
    yolo_options,
)


def make_config(model):
    _, optimizer, lr = MMDET_REFERENCES[model]
    wrapper = dict(type="OptimWrapper", optimizer=dict(type=optimizer, lr=lr))
    if optimizer == "AdamW":
        wrapper["optimizer"]["weight_decay"] = 0.05
        wrapper["paramwise_cfg"] = dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
        scheduler = [dict(type="LinearLR", by_epoch=False, end=1000),
                     dict(type="CosineAnnealingLR", begin=150, end=300, T_max=150, eta_min=0.0002,
                          convert_to_iter_based=True)]
    else:
        wrapper["optimizer"].update(momentum=0.9, weight_decay=0.0001)
        if model == "solov2_light_r18":
            wrapper["clip_grad"] = dict(max_norm=35, norm_type=2)
        scheduler = [dict(type="LinearLR", by_epoch=False, end=500),
                     dict(type="MultiStepLR", by_epoch=True, milestones=[200, 275], end=300, gamma=0.1)]
    return dict(
        optim_wrapper=wrapper,
        param_scheduler=scheduler,
        auto_scale_lr=dict(enable=True),
        model=dict(backbone=dict(init_cfg=[dict(type="Pretrained", checkpoint="unused"), dict(type="Kaiming")],
                                 frozen_stages=1, norm_eval=True, norm_cfg=dict(type="BN", requires_grad=False))),
    )


def make_job(model, amp=True, batch=2):
    return dict(model=model, recipe_policy="official_transfer", amp=amp,
                recipe=dict(batch=batch, config="configs/example.py"))


def test_policy_is_explicit_and_old_jobs_remain_scratch():
    assert not is_native({})
    assert not is_native(dict(initialization="coco_pretrained"))
    assert not is_native(dict(recipe_policy="legacy78"))
    assert is_native(dict(recipe_policy="official_transfer"))


def test_yolo_transfer_preserves_checkpoint_and_caller_controls():
    original = dict(pretrained=False, optimizer="AdamW", lr0=0.001, dropout=0.1,
                    batch=16, epochs=300, custom=dict(value=3))
    options = yolo_options(original)
    assert options["pretrained"] is True
    assert options["optimizer"] == "auto" and options["lr0"] == 0.01
    assert options["dropout"] == 0 and options["batch"] == 16 and options["epochs"] == 300
    options["custom"]["value"] = 4
    assert original["pretrained"] is False and original["custom"]["value"] == 3


@pytest.mark.parametrize("name", ["yolo11n-seg", "yolov8n-seg", "yolo26n-seg"])
def test_yolo_official_checkpoint_names(name):
    assert Path(yolo_checkpoint(dict(yaml=f"{name}.yaml"))).name == f"{name}.pt"
    with pytest.raises(ValueError, match=".yaml"):
        yolo_checkpoint(dict(yaml=f"{name}.pt"))


@pytest.mark.parametrize("model,batch,expected_lr", [
    ("rtmdet_ins_tiny", 8, 0.000125),
    ("mask_rcnn_r50", 2, 0.0025),
    ("solov2_light_r18", 2, 0.00125),
])
@pytest.mark.parametrize("amp", [False, True])
def test_native_mmdet_scales_official_lr_and_preserves_recipe(model, batch, expected_lr, amp):
    cfg = make_config(model)
    before = copy.deepcopy(cfg)
    meta = mmdet_apply_native(cfg, make_job(model, amp, batch), Path("coco.pth"))
    wrapper = cfg["optim_wrapper"]
    assert wrapper["optimizer"]["lr"] == pytest.approx(expected_lr)
    assert wrapper["optimizer"]["type"] == before["optim_wrapper"]["optimizer"]["type"]
    assert wrapper["optimizer"]["weight_decay"] == before["optim_wrapper"]["optimizer"]["weight_decay"]
    assert wrapper.get("paramwise_cfg") == before["optim_wrapper"].get("paramwise_cfg")
    assert wrapper.get("clip_grad") == before["optim_wrapper"].get("clip_grad")
    assert wrapper["type"] == ("AmpOptimWrapper" if amp else "OptimWrapper")
    assert ("loss_scale" in wrapper) == ("dtype" in wrapper) == amp
    assert cfg["load_from"] == "coco.pth" and cfg["resume"] is False
    assert cfg["auto_scale_lr"]["enable"] is False
    backbone = cfg["model"]["backbone"]
    assert backbone["init_cfg"] == [dict(type="Kaiming")]
    assert backbone["frozen_stages"] == 1 and backbone["norm_eval"] is True
    assert backbone["norm_cfg"]["requires_grad"] is False
    assert meta["effective_lr"] == pytest.approx(expected_lr)
    assert meta["reference_optimizer"] == before["optim_wrapper"]["optimizer"]
    assert cfg["param_scheduler"][0] == before["param_scheduler"][0]
    if model == "rtmdet_ins_tiny":
        assert cfg["param_scheduler"][1]["eta_min"] == pytest.approx(0.00000625)
        assert cfg["param_scheduler"][1]["T_max"] == 150
        assert meta["cosine_minimums"] == [dict(reference=0.0002, effective=0.00000625)]
    else:
        assert cfg["param_scheduler"] == before["param_scheduler"]


def test_mmdet_fp32_removes_amp_only_arguments():
    cfg = make_config("mask_rcnn_r50")
    cfg["optim_wrapper"].update(type="AmpOptimWrapper", dtype="bfloat16", loss_scale="dynamic", use_fsdp=True)
    mmdet_apply_native(cfg, make_job("mask_rcnn_r50", amp=False), "coco.pth")
    assert not {"dtype", "loss_scale", "use_fsdp"} & set(cfg["optim_wrapper"])


@pytest.mark.parametrize("checkpoint", [None, "", "https://example.com/weights.pth"])
def test_mmdet_native_requires_local_checkpoint_without_mutating(checkpoint):
    cfg = make_config("mask_rcnn_r50")
    original = copy.deepcopy(cfg)
    with pytest.raises(ValueError, match="local COCO checkpoint"):
        mmdet_apply_native(cfg, make_job("mask_rcnn_r50"), checkpoint)
    assert cfg == original


def test_mmdet_refuses_scratch_and_legacy_optimizer():
    cfg = make_config("mask_rcnn_r50")
    job = make_job("mask_rcnn_r50")
    job["recipe_policy"] = "legacy78"
    with pytest.raises(ValueError, match="recipe_policy"):
        mmdet_apply_native(cfg, job, "coco.pth")
    job["recipe_policy"] = "official_transfer"
    cfg["optim_wrapper"]["optimizer"]["lr"] = 0.001
    with pytest.raises(ValueError, match="unmodified official"):
        mmdet_apply_native(cfg, job, "coco.pth")


def test_rf_optimizer_defaults_keep_fixed_citrus_resolution_and_common_controls():
    original = dict(epochs=300, batch_size=2, grad_accum_steps=8, early_stopping=False,
                    eval_max_dets=100, lr_drop=240, lr=0.001, lr_encoder=0.001,
                    dataset_dir="prepared/rfdetr", custom=dict(value=3))
    options = rfdetr_options(original)
    assert options["lr"] == 1e-4 and options["lr_encoder"] == 1.5e-4
    assert options["lr_vit_layer_decay"] == 0.8 and options["lr_component_decay"] == 0.7
    assert options["weight_decay"] == 1e-4 and options["warmup_epochs"] == 0
    assert options["multi_scale"] is options["expanded_scales"] is False
    for key in ("epochs", "batch_size", "grad_accum_steps", "early_stopping", "eval_max_dets", "lr_drop"):
        assert options[key] == original[key]
    options["custom"]["value"] = 4
    assert original["lr"] == 0.001 and original["custom"]["value"] == 3
    with pytest.raises(TypeError):
        RFDETR_NATIVE_TRAIN["lr"] = 1


def test_queue_tracks_are_independent_and_amp_only_pairs():
    from baseline_amp.registry import COMMON_VALIDATION, CONTROLLED_PROTOCOL_ID, TRANSFER_PROTOCOL_ID, make_queue

    scratch = make_queue("all", [42, 43], 50, 0, validation=COMMON_VALIDATION)
    transfer = make_queue("all", [42, 43], 50, 0, validation=COMMON_VALIDATION,
                          recipe_policy="official_transfer")
    assert len(scratch) == len(transfer) == 28
    for old, new in zip(scratch, transfer):
        assert old["protocol"] == CONTROLLED_PROTOCOL_ID and old["initialization"] == "scratch"
        assert "recipe_policy" not in old
        assert new["protocol"] == TRANSFER_PROTOCOL_ID and new["initialization"] == "official_coco_pretrained"
        assert new["name"] == "transfer_" + old["name"] and is_native(new)
        assert new["recipe"] == old["recipe"] and new["recipe"] is not old["recipe"]
        assert new["validation"] == old["validation"] and new["validation"] is not old["validation"]
    for left, right in zip(transfer[::2], transfer[1::2]):
        assert left["amp"] != right["amp"]
        assert {k: v for k, v in left.items() if k not in ("amp", "name")} == {
            k: v for k, v in right.items() if k not in ("amp", "name")}


def test_queue_rejects_uncontrolled_or_changed_native_validation():
    from baseline_amp.registry import COMMON_VALIDATION, make_queue

    with pytest.raises(ValueError, match="COMMON_VALIDATION"):
        make_queue("all", [42], 50, 0, recipe_policy="official_transfer")
    changed = dict(COMMON_VALIDATION, max_dets=300)
    with pytest.raises(ValueError, match="validation protocol"):
        make_queue("all", [42], 50, 0, validation=changed, recipe_policy="official_transfer")
    with pytest.raises(ValueError, match="recipe_policy"):
        make_queue("all", [42], 50, 0, validation=COMMON_VALIDATION, recipe_policy="unsupported")


def test_native_batch_dry_run_has_no_filesystem_or_environment_requirements(tmp_path, capsys):
    from baseline_amp.batch import run
    from baseline_amp.registry import COMMON_VALIDATION

    project = tmp_path / "unused"
    settings = dict(SUITE="anchor", SEEDS=[42], EPOCHS=1, WORKERS=0, BATCHES={}, DEVICE=0,
                    DATA="missing/data.yaml", PROJECT=str(project), PYTHONS={}, AMP_MODES=[1],
                    VALIDATION=COMMON_VALIDATION, RECIPE_POLICY="official_transfer")
    run(settings, dry_run=True)
    output = capsys.readouterr().out
    assert "COCO transfer + native optimizer" in output
    assert "transfer_yolo11n_seg" in output and "transfer_rtmdet_ins_tiny" in output
    assert not project.exists()


@pytest.mark.parametrize("failure", ["exception", "empty"])
def test_official_checkpoint_failure_is_visible_and_removes_partial_file(tmp_path, monkeypatch, failure):
    torch = pytest.importorskip("torch")
    from baseline_amp import worker

    def fake_download(url, destination, progress):
        Path(destination).write_bytes(b"partial" if failure == "exception" else b"")
        if failure == "exception":
            raise OSError("download denied")

    monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)
    expected = OSError if failure == "exception" else RuntimeError
    message = "download denied" if failure == "exception" else "empty"
    with pytest.raises(expected, match=message):
        worker.official_checkpoint("https://download.openmmlab.com/coco.pth", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_worker_rf_records_adapted_head_before_update_only_once(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from baseline_amp import worker
    from baseline_amp.registry import COMMON_VALIDATION, make_queue

    weights = tmp_path / "rf-detr-seg-nano.pt"
    weights.write_bytes(b"mock official checkpoint identity")
    (tmp_path / "summary.json").write_text(json.dumps(dict(names=["orange_immature"])))
    best = tmp_path / "best.pth"
    captured = {}

    class FakeRF:
        def __init__(self, **kwargs):
            captured["constructor"] = kwargs
            self.callbacks = defaultdict(list)
            self.model = types.SimpleNamespace(model=torch.nn.Linear(2, 91))
            self.model_config = types.SimpleNamespace(
                pretrain_weights=str(weights), patch_size=12, amp=kwargs["amp"], model_dump=lambda: kwargs)
            captured["constructor_sha"] = worker.state_sha256(self.model.model)

        def train(self, **options):
            captured["options"] = options
            coco = self.model.model
            adapted = torch.nn.Linear(2, 2)
            with torch.no_grad():
                adapted.weight.copy_(coco.weight[:2])
                adapted.bias.copy_(coco.bias[:2])
            self.model.model = adapted
            captured["adapted_sha"] = worker.state_sha256(adapted)
            assert not (tmp_path / "initialization.json").exists()
            for callback in self.callbacks["on_train_batch_start"]:
                callback(dict(step=0, epoch=0, model=adapted))
            with torch.no_grad():
                adapted.weight.add_(1)
            for callback in self.callbacks["on_train_batch_start"]:
                callback(dict(step=1, epoch=0, model=adapted))
            captured["updated_sha"] = worker.state_sha256(adapted)

    @contextmanager
    def fake_validation(*args):
        yield types.SimpleNamespace(checkpoint=lambda: best)

    monkeypatch.setitem(sys.modules, "rfdetr", types.SimpleNamespace(RFDETRSegNano=FakeRF))
    monkeypatch.setitem(sys.modules, "validation_adapters", types.SimpleNamespace(rfdetr_validation=fake_validation))
    job = make_queue("rfdetr", [42], 1, 0, amp_modes=[1], validation=COMMON_VALIDATION,
                     recipe_policy="official_transfer")[0]
    assert worker.train_rfdetr(job, tmp_path, tmp_path) == best
    initial = json.loads((tmp_path / "initialization.json").read_text())
    assert "pretrain_weights" not in captured["constructor"]
    assert initial["sha256"] == captured["adapted_sha"]
    assert initial["sha256"] != captured["constructor_sha"]
    assert initial["sha256"] != captured["updated_sha"]
    assert initial["checkpoint_sha256"] == worker.weight_sha256(weights)
    assert initial["mode"] == "official_coco_pretrained"
    assert captured["options"]["lr"] == 1e-4 and captured["options"]["early_stopping"] is False


class AttrConfig(dict):
    """Minimal Config/ConfigDict interface for testing actual config adaptation."""

    def __init__(self, values):
        super().__init__({key: self.convert(value) for key, value in values.items()})

    @classmethod
    def convert(cls, value):
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls.convert(item) for item in value]
        return value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setitem__(self, key, value):
        super().__setitem__(key, self.convert(value))

    def __setattr__(self, key, value):
        self[key] = self.convert(value)


def test_native_mmdet_worker_epoch1_keeps_cosine_and_official_optimizer(tmp_path, monkeypatch):
    from baseline_amp import worker
    from baseline_amp.registry import COMMON_VALIDATION, make_queue
    import mmdet_common

    original = make_config("rtmdet_ins_tiny")
    original["model"].update(data_preprocessor=dict(type="DetDataPreprocessor"),
                            bbox_head=dict(num_classes=80), test_cfg=dict(score_thr=0.05, max_per_img=100))
    original.update(
        train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=300, dynamic_intervals=[(280, 1)]),
        env_cfg=dict(cudnn_benchmark=True), visualizer=dict(vis_backends=[]),
        default_hooks=dict(checkpoint={}, logger={}),
        custom_hooks=[dict(type="EMAHook"), dict(type="PipelineSwitchHook", switch_epoch=280)],
    )
    for split in ("train", "val", "test"):
        original[f"{split}_dataloader"] = dict(batch_size=8, num_workers=2,
            dataset=dict(type="CocoDataset", pipeline=[dict(type="LoadAnnotations", with_mask=True)]))
    for split in ("val", "test"):
        original[f"{split}_evaluator"] = dict(type="CocoMetric", ann_file="old.json")
    cfg = AttrConfig(original)
    config_type = types.SimpleNamespace(fromfile=lambda path: cfg)
    monkeypatch.setattr(mmdet_common, "require_mmdet", lambda: config_type)
    official = tmp_path / "official"
    config_path = official / "configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py"
    config_path.parent.mkdir(parents=True)
    config_path.touch()
    monkeypatch.setattr(worker, "official_mmdet_root", lambda: official)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "summary.json").write_text(json.dumps(dict(names=["orange_immature"])))
    job = make_queue("mmdet", [42], 1, 0, amp_modes=[0], validation=COMMON_VALIDATION,
                     recipe_policy="official_transfer")[0]
    result = worker.mmdet_config(job, prepared, tmp_path / "run", tmp_path / "coco.pth")
    cosine = next(item for item in result.param_scheduler if item["type"] == "CosineAnnealingLR")
    assert cosine["begin"] == 0 and cosine["end"] == 1 and cosine["T_max"] == 1
    assert cosine["convert_to_iter_based"] is True and cosine["eta_min"] == pytest.approx(0.00000625)
    assert result.optim_wrapper.optimizer["type"] == "AdamW"
    assert result.optim_wrapper.optimizer["weight_decay"] == 0.05
    assert result.optim_wrapper.optimizer["lr"] == pytest.approx(0.000125)
    assert result.optim_wrapper.paramwise_cfg["bypass_duplicate"] is True
    assert "dtype" not in result.optim_wrapper and "loss_scale" not in result.optim_wrapper
    assert result.load_from == str(tmp_path / "coco.pth") and result.resume is False
    assert result.model.bbox_head["num_classes"] == 1
    assert result.train_cfg["max_epochs"] == 1 and "dynamic_intervals" not in result.train_cfg
    assert not any(hook["type"] == "PipelineSwitchHook" for hook in result.custom_hooks)
    assert result.train_dataloader["num_workers"] == 0
