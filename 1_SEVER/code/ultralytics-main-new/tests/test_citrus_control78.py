"""Recipe fidelity, restart safety, data route and mask-ratio=4 gradient checks."""
import json
from types import SimpleNamespace

import pytest

from citrus_control78_batch import (
    MODELS, RECIPE78, ROOT, assert_resumable, main, parse_args, project_lock,
    queue_for, save_best_and_terminal, training_args, training_done,
)


def args(tmp_path, extra=()):
    return parse_args(["--data", str(tmp_path / "data.yaml"), "--project", str(tmp_path / "runs"), *extra])


def test_recipe_matches_original_when_available():
    import yaml
    path = ROOT.parent.parent / "results/A_baselines/old_data_runs/001_3_yolo11-seg_adamw/args.yaml"
    if not path.is_file():
        pytest.skip("Historical results are not uploaded with code.")
    original = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in RECIPE78.items():
        if key != "pretrained":
            assert value == original[key], key
    assert RECIPE78["pretrained"] is False


def test_dry_run_does_not_need_data_or_create_output(tmp_path):
    queued = main(["--data", str(tmp_path / "missing.yaml"), "--project", str(tmp_path / "out"), "--dry-run"])
    assert [j["kind"] for j in queued] == ["baseline", "candidate"]
    assert not (tmp_path / "out").exists()


def test_queue_and_shared_recipe(tmp_path):
    config = args(tmp_path, ["--seeds", "42,43,44"])
    assert len(queue_for(config)) == 6
    recipe = training_args(config, 42)
    assert recipe["amp"] and recipe["cache"]
    assert recipe["copy_paste"] == 0 and recipe["mask_ratio"] == 4
    assert recipe["patience"] == 100 and recipe["dropout"] == .1
    assert not recipe["pretrained"]
    assert training_args(config, 43) == dict(recipe, seed=43)


def test_smoke_separate_project(tmp_path):
    config = args(tmp_path, ["--smoke"])
    assert config.project.name.endswith("_SMOKE") and config.epochs == 1
    assert config.batch == 16


@pytest.mark.parametrize("extra", [
    ["--seeds", "42,42"], ["--device", "0,1"], ["--epochs", "0"], ["--workers", "-1"],
])
def test_invalid_arguments(tmp_path, extra):
    with pytest.raises(SystemExit):
        args(tmp_path, extra)


def test_output_lock_is_not_gpu_lock(tmp_path):
    with project_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with project_lock(tmp_path):
                pass
    with project_lock(tmp_path):
        pass


def make_trainer(tmp_path, epoch=0, score=.5, stop=False):
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    last = weights / "last.pt"
    last.write_bytes(b"checkpoint")
    return SimpleNamespace(
        metrics={"metrics/mAP50-95(M)": score}, last=last, save_dir=tmp_path,
        stop=stop, epoch=epoch, epochs=300,
    )


def test_best_checkpoint_not_overwritten_by_lower_score(tmp_path):
    trainer = make_trainer(tmp_path)
    save_best_and_terminal(trainer)
    trainer.last.write_bytes(b"worse checkpoint")
    trainer.metrics["metrics/mAP50-95(M)"] = .4
    save_best_and_terminal(trainer)
    assert (tmp_path / "weights/best_mask.pt").read_bytes() == b"checkpoint"
    assert json.loads((tmp_path / "best_mask_selection.json").read_text())["epoch"] == 1
    assert not training_done(tmp_path, 300)


def test_early_stop_is_completed_but_partial_is_not(tmp_path):
    trainer = make_trainer(tmp_path, epoch=154, stop=True)
    (tmp_path / "results.csv").write_text("epoch\n155\n")
    save_best_and_terminal(trainer)
    assert training_done(tmp_path, 300)
    assert not training_done(tmp_path, 301)


def test_resume_guard_and_invalid_metric(tmp_path):
    with pytest.raises(RuntimeError, match="resumable"):
        assert_resumable(SimpleNamespace(ckpt={"epoch": -1, "optimizer": None}), tmp_path)
    assert_resumable(SimpleNamespace(ckpt={"epoch": 6, "optimizer": {}}), tmp_path)
    with pytest.raises(RuntimeError, match="non-finite"):
        save_best_and_terminal(make_trainer(tmp_path, score=float("nan")))


def test_whole_image_training_dispatch(monkeypatch):
    from citrus_control78_training import Control78Trainer
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from citrus_e_v6_training import EV6TrainingTrainer
    monkeypatch.setattr(SegmentationTrainer, "build_dataset", lambda *a, **kw: "native")
    monkeypatch.setattr(EV6TrainingTrainer, "build_dataset", lambda *a, **kw: "ev6")
    trainer = object.__new__(Control78Trainer)
    assert trainer.build_dataset("images", "train") == "native"
    assert trainer.build_dataset("images", "val") == "ev6"


def test_empty_view_masks_are_valid_no_detection():
    import torch
    import numpy as np
    from eval_citrus_sliced import predict_view
    result = SimpleNamespace(masks=SimpleNamespace(data=torch.zeros(0, 64, 64)))
    model = SimpleNamespace(predict=lambda *a, **kw: [result])
    assert predict_view(model, np.zeros((64, 64, 3), dtype=np.uint8),
                        (0, 0, 64, 64), (64, 64), "cpu") == []


@pytest.mark.parametrize("kind", ["baseline", "candidate"])
def test_mask_ratio4_forward_backward(kind):
    import torch
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = SegmentationModel(str(ROOT / MODELS[kind][1]), nc=1, verbose=False)
        model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, **RECIPE78, "nwd_ratio": 0.0})
        masks = torch.zeros(2, 32, 32)
        masks[:, 8:24, 8:24] = 1
        batch = dict(
            img=torch.rand(2, 3, 128, 128), masks=masks,
            batch_idx=torch.tensor([0., 1.]), cls=torch.zeros(2, 1),
            bboxes=torch.tensor([[.5, .5, .5, .5], [.5, .5, .5, .5]]),
        )
        loss, components = model.loss(batch)
        assert torch.isfinite(loss).all() and len(components) == 5
        loss.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.eval()
        with torch.no_grad():
            prediction = model(torch.zeros(1, 3, 128, 128))
        assert prediction[0][0].shape[-1] == 336
    finally:
        torch.set_num_threads(old_threads)
