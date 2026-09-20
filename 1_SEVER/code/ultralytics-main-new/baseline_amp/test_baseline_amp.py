"""CPU-only contract tests; not a substitute for CUDA per-framework smoke runs."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from baseline_amp.batch import summarize
from baseline_amp.batch import run_visible
from prepare import inspect_source, prepare, split_files
from registry import MODELS, make_queue
from coco_utils import evaluate_predictions, prediction_from_mask


def make_source(tmp_path):
    for split in ("train", "val", "test"):
        images = tmp_path / split / "images"
        labels = tmp_path / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        Image.new("RGB", (24, 24), (100, 200, 80)).save(images / "same.png")
        (labels / "same.txt").write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    data.write_text(
        "path: .\ntrain: train/images\nval: val/images\ntest: test/images\nnames: [orange_immature]\n", encoding="utf-8"
    )
    return data


def test_all_pairs_only_amp_differs():
    queue = make_queue("all", [42], 300, 4)
    assert len(queue) == 14
    for left, right in zip(queue[::2], queue[1::2]):
        assert left["amp"] != right["amp"]
        assert {k: v for k, v in left.items() if k not in ("amp", "name")} == {
            k: v for k, v in right.items() if k not in ("amp", "name")
        }
    assert make_queue("all", [42, 43, 44], 300, 4).__len__() == 42


@pytest.mark.parametrize("batches", [{"typo": 1}, {"yolo11n_seg": 0}, {"yolo11n_seg": -1}])
def test_invalid_batch_rejected(batches):
    with pytest.raises(ValueError):
        make_queue("all", [42], 300, 4, batches)


def test_both_batches_override():
    queue = make_queue("all", [42], 300, 4, {"mask_rcnn_r50": 1})
    assert [j["recipe"]["batch"] for j in queue if j["model"] == "mask_rcnn_r50"] == [1, 1]
    assert MODELS["mask_rcnn_r50"]["batch"] == 2


def test_conversion_preserves_source_and_counts(tmp_path):
    data = make_source(tmp_path / "source")
    before = (data.parent / "train/labels/same.txt").read_bytes()
    output = tmp_path / "prepared"
    info = prepare(data, output)
    assert all(value == {"images": 1, "instances": 1} for value in info["splits"].values())
    assert before == (data.parent / "train/labels/same.txt").read_bytes()
    assert prepare(data, output) == info
    coco = json.loads((output / "coco/annotations/instances_val.json").read_text())
    assert coco["annotations"][0]["bbox"] == [6, 6, 12, 12]
    assert coco["annotations"][0]["area"] == 144
    assert coco["categories"][0]["id"] == 1
    rf_coco = json.loads((output / "rfdetr/valid/_annotations.coco.json").read_text())
    assert rf_coco["categories"][0]["id"] == 0
    assert rf_coco["annotations"][0]["category_id"] == 0


def test_source_change_rejected(tmp_path):
    data = make_source(tmp_path / "source")
    output = tmp_path / "prepared"
    prepare(data, output)
    (data.parent / "train/labels/same.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset changed"):
        prepare(data, output)


def test_missing_labels_not_repaired(tmp_path):
    data = make_source(tmp_path / "source")
    (data.parent / "val/labels/same.txt").unlink()
    with pytest.raises(FileNotFoundError):
        inspect_source(data)


def test_overlap_rejected(tmp_path):
    data = make_source(tmp_path / "source")
    data.write_text(data.read_text().replace("val/images", "train/images"))
    with pytest.raises(ValueError, match="overlapping"):
        inspect_source(data)


def test_nan_not_silently_dropped(tmp_path):
    data = make_source(tmp_path / "source")
    (data.parent / "train/labels/same.txt").write_text("0 nan 0.25 0.75 0.25 0.75 0.75\n")
    with pytest.raises(ValueError, match="Invalid polygon"):
        prepare(data, tmp_path / "prepared")


def test_common_coco_perfect_prediction(tmp_path):
    data = make_source(tmp_path / "source")
    output = tmp_path / "prepared"
    prepare(data, output)
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[6:18, 6:18] = 1
    metrics = evaluate_predictions(
        output / "coco/annotations/instances_val.json", [prediction_from_mask(1, 1, 0.9, mask)], evaluate_bbox=False
    )
    assert metrics["mask_ap_50_95"] == pytest.approx(1)
    assert metrics["mask_recall"] == 1
    empty = evaluate_predictions(output / "coco/annotations/instances_val.json", [], evaluate_bbox=False)
    assert empty["mask_recall"] == 0


def test_list_paths_resolved_relative_to_list(tmp_path):
    data = make_source(tmp_path / "source")
    file_list = data.parent / "val.txt"
    file_list.write_text("./val/images/same.png\n")
    assert split_files("val.txt", data.parent) == [(data.parent / "val/images/same.png").resolve()]


def test_empty_background_kept(tmp_path):
    data = make_source(tmp_path / "source")
    (data.parent / "train/labels/same.txt").write_text("")
    summary = prepare(data, tmp_path / "prepared")
    assert summary["splits"]["train"] == {"images": 1, "instances": 0}


def test_summary_does_not_call_one_seed_significant(tmp_path):
    for amp in (0, 1):
        run = tmp_path / f"model_amp{amp}"
        run.mkdir()
        job = dict(model="example", seed=42, epochs=300, amp=amp, recipe=dict(batch=2, imgsz=640))
        metrics = dict(
            mask_ap_50_95=0.6 + amp * 0.01,
            mask_ap_50=0.8 + amp * 0.02,
            mask_ap_small=0.3,
            mask_ar_100=0.7,
            mask_precision=0.9,
            mask_recall=0.6,
            params=100,
        )
        (run / "complete.json").write_text(json.dumps(dict(job=job, metrics=metrics)))
        (run / "amp_actual.json").write_text(json.dumps(dict(dtype="float16" if amp else "float32")))
        (run / "initialization.json").write_text(json.dumps(dict(sha256="sameweights")))
        (run / "dataset.json").write_text(json.dumps(dict(signature="same_split")))
    summarize(tmp_path)
    result = json.loads((tmp_path / "amp_paired_deltas.json").read_text())
    assert result["pairs"][0]["delta_AP50_pp"] == pytest.approx(2)


@pytest.mark.parametrize("model", ["rtmdet_ins_tiny", "mask_rcnn_r50", "solov2_light_r18"])
def test_official_mmdet_config_if_available(tmp_path, monkeypatch, model):
    pytest.importorskip("mmengine")
    import importlib.util
    import worker
    from mmdet_common import _walk_mappings

    spec = importlib.util.find_spec("mmdet")
    if spec is None:
        pytest.skip("mmdet package configs unavailable")
    root = Path(spec.origin).parent / ".mim"
    monkeypatch.setattr(worker, "official_mmdet_root", lambda: root)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "summary.json").write_text(json.dumps(dict(names=["orange_immature"])))
    jobs = [job for job in make_queue("all", [42], 300, 0) if job["model"] == model]
    for job in jobs:
        assert worker.model_zoo_weight(root, job["recipe"]["config"]).startswith("https://")
        cfg = worker.mmdet_config(job, prepared, tmp_path / job["name"], tmp_path / "pretrained.pth")
        assert cfg.optim_wrapper.type == ("AmpOptimWrapper" if job["amp"] else "OptimWrapper")
        assert cfg.train_cfg.max_epochs == 300
        assert cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt is False
        assert all(mapping["num_classes"] == 1 for mapping in _walk_mappings(cfg.model) if "num_classes" in mapping)
        assert not any(mapping.get("type") == "SyncBN" for mapping in _walk_mappings(cfg.model))
        cfg.dump(str(tmp_path / f"{job['name']}.py"))


def test_foreground_child_logs_and_failure(tmp_path):
    import os

    log = tmp_path / "visible.log"
    run_visible([sys.executable, "-c", "print('visible child output')"], tmp_path, os.environ.copy(), log)
    assert "visible child output" in log.read_text()
    with pytest.raises(RuntimeError, match="Queue stopped"):
        run_visible([sys.executable, "-c", "raise SystemExit(3)"], tmp_path, os.environ.copy(), log)
