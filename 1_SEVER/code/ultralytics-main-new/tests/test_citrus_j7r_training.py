"""J7R adapters use packed inputs while inheriting EV6 matching and selected-mask audit."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from citrus_e_v6_training import EV6Validator
from citrus_j7_pr_audit import J7FinalAuditValidator
from citrus_j7r_data import J7R_INPUT_PROTOCOL, build_j7r_dataset
from citrus_j7r_training import (J7RFinalAuditValidator, J7RTrainer, J7RValidator, record_first_batch,
                                 rgb_plot_batch)
from ultralytics.cfg import get_cfg
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[1]


def model_spec(observation="raw"):
    return {"channels": 15, "citrus_input_protocol": J7R_INPUT_PROTOCOL, "citrus_observation": observation}


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def batch():
    return dict(img=torch.full((1, 3, 32, 64), 128, dtype=torch.uint8),
                high_img=torch.full((1, 3, 64, 128), 255, dtype=torch.uint8),
                masks=torch.ones(1, 16, 32), cls=torch.zeros(1, 1), bboxes=torch.tensor([[.5, .5, .2, .2]]),
                batch_idx=torch.zeros(1), im_file=("synthetic.png",))


def test_warmup_hook_does_not_change_legacy_defaults(tmp_path):
    legacy = BaseValidator(args={"plots": False}, save_dir=tmp_path / "old")
    legacy.data = {"channels": 3}
    model = SimpleNamespace(yaml=model_spec(), channels=15)
    assert legacy.get_warmup_shape(model, 1, 640) == (1, 3, 640, 640)
    validator = J7RValidator(args={"plots": False}, save_dir=tmp_path / "new")
    validator.data = {"channels": 3}
    assert validator.get_warmup_shape(model, 2, 640) == (2, 15, 640, 640)
    assert validator.data["channels"] == 3


def test_validator_input_pack_and_gt_unchanged(tmp_path):
    validator = J7RValidator(args={"plots": False}, save_dir=tmp_path)
    validator.device = torch.device("cpu")
    validator._j7r_spec = {"protocol": J7R_INPUT_PROTOCOL, "channels": 15, "observation": "raw"}
    original = batch()
    gt = {key: original[key].clone() for key in ("masks", "cls", "bboxes", "batch_idx")}
    processed = validator.preprocess(original)
    assert processed["img"].shape == (1, 15, 32, 64)
    torch.testing.assert_close(processed["img"][:, :3], torch.full((1, 3, 32, 64), 128 / 255))
    torch.testing.assert_close(processed["img"][:, 3:], torch.ones(1, 12, 32, 64))
    for key, value in gt.items():
        torch.testing.assert_close(processed[key], value, rtol=0, atol=0)
    audit = json.loads((tmp_path / "j7r_val_first_batch.json").read_text(encoding="utf-8"))
    assert audit["sensor_channels"] == 3 and audit["model_channels"] == 15
    assert "same seed does not prove" in audit["limitation"]


@pytest.mark.parametrize("observation", ["raw", "upsampled", "global"])
def test_trainer_uses_explicit_model_observation(tmp_path, observation):
    trainer = object.__new__(J7RTrainer)
    trainer.device = torch.device("cpu")
    trainer.model = SimpleNamespace(yaml=model_spec(observation))
    trainer.args = SimpleNamespace(multi_scale=0.)
    trainer.save_dir = tmp_path
    actual = trainer.preprocess_batch(batch())
    assert actual["img"].shape[1] == 15
    if observation == "raw":
        assert torch.all(actual["img"][:, 3:] == 1)
    elif observation == "global":
        assert torch.count_nonzero(actual["img"][:, 3:]) == 0
    else:
        torch.testing.assert_close(actual["img"][:, 3:], torch.full((1, 12, 32, 64), 128 / 255))


def test_plot_views_and_metric_visualization_do_not_mutate_input(monkeypatch, tmp_path):
    original = {"img": torch.rand(2, 15, 32, 64), "masks": torch.ones(2, 16, 32)}
    plotted = rgb_plot_batch(original)
    assert plotted is not original and plotted["img"].shape[1] == 3
    assert original["img"].shape[1] == 15 and plotted["masks"] is original["masks"]
    validator = J7RValidator(args={"plots": False}, save_dir=tmp_path)
    seen = []
    for name in ("update_metrics", "plot_val_samples", "plot_predictions"):
        monkeypatch.setattr(EV6Validator, name, lambda self, *a, **kw: seen.append(a))
    validator.update_metrics([], original)
    validator.plot_val_samples(original, 0)
    validator.plot_predictions(original, [], 0)
    assert seen[0][1]["img"].shape[1] == 3
    assert seen[1][0]["img"].shape[1] == 3
    assert seen[2][0]["img"].shape[1] == 3
    assert original["img"].shape[1] == 15
    trainer = object.__new__(J7RTrainer)
    monkeypatch.setattr(SegmentationTrainer, "plot_training_samples", lambda self, value, ni: seen.append(value))
    trainer.plot_training_samples(original, 0)
    assert seen[-1]["img"].shape[1] == 3


def test_first_batch_audit_never_overwrites(tmp_path):
    owner = SimpleNamespace(save_dir=tmp_path)
    value = {"img": torch.zeros(1, 15, 32, 32), "masks": torch.zeros(1, 16, 16)}
    record_first_batch(owner, value, model_spec(), "train")
    path = tmp_path / "j7r_train_first_batch.json"
    original = path.read_bytes()
    value["img"].fill_(1)
    record_first_batch(owner, value, model_spec(), "train")
    assert path.read_bytes() == original


@pytest.mark.parametrize("options", [{"augment": True}, {"compile": True}, {"save_json": True},
                                     {"save_txt": True}, {"imgsz": 1280}])
def test_validator_rejects_unsupported_paths(tmp_path, options):
    with pytest.raises(ValueError):
        J7RValidator(args={"plots": False, **options}, save_dir=tmp_path)


def test_final_audit_mro_preserves_native_matching_and_postprocess():
    assert issubclass(J7RFinalAuditValidator, J7FinalAuditValidator)
    assert J7RFinalAuditValidator.get_stats is J7FinalAuditValidator.get_stats
    assert J7RValidator.postprocess is EV6Validator.postprocess
    assert J7RValidator._process_batch is EV6Validator._process_batch


def _small_val(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    raw = np.zeros((317, 563, 3), np.uint8)
    raw[60:250, 150:400] = (60, 160, 80)
    assert cv2.imwrite(str(images / "rect.png"), raw)
    (labels / "rect.txt").write_text("0 .25 .2 .7 .2 .7 .8 .25 .8\n", encoding="utf-8")
    data = {"path": str(tmp_path), "train": str(images), "val": str(images), "test": str(images),
            "names": {0: "orange_immature"}, "nc": 1, "channels": 3}
    path = tmp_path / "data.yaml"
    YAML.save(path, data)
    config = get_cfg(overrides=dict(task="segment", imgsz=640, mask_ratio=2, cache=False, rect=False,
                                   workers=0, plots=False, batch=1, data=str(path), conf=.25, half=False))
    dataset = build_j7r_dataset(config, str(images), 1, data, mode="val")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=dataset.collate_fn)
    return loader, data, config


def _packed_model():
    # Exercise the actual registered J7R evidence graph, not the earlier plumbing placeholder.
    cfg = YAML.load(ROOT / "0_orange_yaml/J_V7R_series/J112_raw_c3_evidence.yaml")
    model = SegmentationModel(cfg, ch=3, nc=1, verbose=False)
    model.args = get_cfg(overrides=dict(task="segment", mask_ratio=2, overlap_mask=True))
    return model


def test_standalone_final_with_preprovided_loader_calls_packed_warmup(monkeypatch, tmp_path):
    from ultralytics.nn.autobackend import AutoBackend

    loader, data, config = _small_val(tmp_path)
    recorded = []
    original = AutoBackend.warmup

    def warmup(self, imgsz):
        recorded.append(imgsz)
        assert imgsz[1] == 15
        return original(self, imgsz)

    monkeypatch.setattr(AutoBackend, "warmup", warmup)
    validator = J7RFinalAuditValidator(loader, args=config, save_dir=tmp_path / "final")
    results = validator(model=_packed_model())
    assert recorded and validator._j7r_spec["observation"] == "raw"
    assert "metrics/mAP50-95(M)" in results
    assert (tmp_path / "final" / "mask_pr_audit.json").is_file()
    assert loader.dataset.data["channels"] == 3


def test_official_get_model_keeps_rgb_data_and_actual_packed_graph():
    trainer = object.__new__(J7RTrainer)
    trainer.data = {"nc": 1, "channels": 3}
    path = ROOT / "0_orange_yaml/J_V7R_series/J112_raw_c3_evidence.yaml"
    model = trainer.get_model(cfg=str(path), verbose=False)
    assert model.yaml["channels"] == 15 and trainer.data["channels"] == 3
    assert type(model.model[-1]).__name__ == "SegmentCitrusJ7R"
    assert model.stride.tolist() == [8., 16., 32.]
    with pytest.raises(ValueError):
        trainer.get_model(cfg={**model.yaml, "citrus_observation": "upsampled"}, weights=model, verbose=False)


def test_actual_final_eval_keeps_selected_checkpoint_metrics_and_pr(tmp_path):
    loader, data, config = _small_val(tmp_path)
    trainer = object.__new__(J7RTrainer)
    trainer.save_dir = tmp_path / "run"
    (trainer.save_dir / "weights").mkdir(parents=True)
    selected = trainer.save_dir / "weights" / "best_mask.pt"
    native = trainer.save_dir / "weights" / "best.pt"
    native.write_bytes(b"native-fitness-file-is-not-overwritten")
    torch.save({"model": _packed_model(), "train_args": {"task": "segment"}}, selected)
    (trainer.save_dir / "best_mask_selection.json").write_text('{"epoch": 1}', encoding="utf-8")
    trainer.args, trainer.best, trainer.test_loader = config, native, loader
    trainer.callbacks, trainer.epoch = {}, 0
    events = []
    trainer.run_callbacks = lambda name: events.append(name)
    trainer.final_eval()
    assert trainer.best == selected and trainer.j7_native_best == native
    assert native.read_bytes() == b"native-fitness-file-is-not-overwritten"
    destination = trainer.save_dir / "selected_mask_validation"
    record = json.loads((destination / "selected_checkpoint.json").read_text(encoding="utf-8"))
    assert record["selection"]["epoch"] == 1 and record["mask_ratio"] == 2
    assert (destination / "mask_pr_audit.json").is_file()
    assert trainer.epoch == 0 and events == ["on_fit_epoch_end"]
    with pytest.raises(FileExistsError):
        trainer.final_eval()
    assert loader.dataset.data["channels"] == 3


@pytest.mark.parametrize("number", range(110, 124))
def test_every_j7r_yaml_routes_official_apis_and_rejects_reinterpretation(number):
    from ultralytics import YOLO
    from citrus_j7r_predict import J7RPredictor

    path, = (ROOT / "0_orange_yaml/J_V7R_series").glob("J{}_*.yaml".format(number))
    wrapper = YOLO(str(path), task="segment", verbose=False)
    assert wrapper._smart_load("trainer") is J7RTrainer
    assert wrapper._smart_load("validator") is J7RValidator
    assert wrapper._smart_load("predictor") is J7RPredictor
    wrapper.model.yaml["channels"] = 3
    with pytest.raises(ValueError):
        wrapper._smart_load("trainer")


def test_workers0_rng_reset_is_cpu_only_and_reproducible(tmp_path, monkeypatch):
    import random

    import numpy as np

    def unexpected_cuda(*args, **kwargs):
        raise AssertionError("Data RNG reset must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", unexpected_cuda)
    trainer = SimpleNamespace(args=SimpleNamespace(workers=0, seed=42), save_dir=tmp_path)
    J7RTrainer.seed_local_data_stream(trainer)
    observed = (random.random(), float(np.random.rand()), torch.rand(3))
    J7RTrainer.seed_local_data_stream(trainer)
    assert observed[:2] == (random.random(), float(np.random.rand()))
    torch.testing.assert_close(observed[2], torch.rand(3))
    assert (tmp_path / "j7r_local_data_rng.json").is_file()


def test_training_validation_uses_active_ema_observation(tmp_path):
    loader, data, config = _small_val(tmp_path)
    ema_model = _packed_model()
    raw_model = deepcopy(ema_model)
    raw_model.yaml["citrus_observation"] = "upsampled"
    trainer = SimpleNamespace(device=torch.device("cpu"), data=data, amp=False,
                              args=SimpleNamespace(compile=False), ema=SimpleNamespace(ema=ema_model), model=raw_model,
                              loss_items=torch.zeros(5), stopper=SimpleNamespace(possible_stop=False), epoch=0,
                              epochs=2, world_size=1,
                              label_loss_items=lambda losses, prefix: {prefix + "/box_loss": float(losses[0])})
    validator = J7RValidator(loader, args=config, save_dir=tmp_path / "epoch")
    results = validator(trainer=trainer)
    assert validator._j7r_spec["observation"] == "raw"
    assert "metrics/mAP50-95(M)" in results
