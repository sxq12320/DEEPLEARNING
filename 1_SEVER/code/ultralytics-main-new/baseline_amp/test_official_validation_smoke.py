"""Small CPU training contract; run with the official pinned wheel ahead of the local research package."""

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


@pytest.mark.parametrize("model_name", ["yolo11n_seg", "yolov8n_seg", "yolo26n_seg"])
def test_official_yolo_training_and_selected_fp32_checkpoint(tmp_path, model_name):
    import ultralytics
    from ultralytics import YOLO
    from prepare import prepare
    from registry import COMMON_VALIDATION, make_queue
    from validation import collect_metrics, validation_state
    from validation_adapters import yolo_predictor
    from worker import train_yolo

    if ultralytics.__version__ != "8.4.60" or "baseline_common_validation_20260930" not in ultralytics.__file__:
        pytest.skip("Run explicitly with the extracted official 8.4.60 fixture, not the modified research package")
    torch.set_num_threads(2)
    for split in ("train", "val", "test"):
        images, labels = tmp_path / "source" / split / "images", tmp_path / "source" / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        for index in range(2):
            image = Image.new("RGB", (128, 96), (50, 90, 30))
            ImageDraw.Draw(image).ellipse((40, 20, 75, 55), fill=(130, 180, 60))
            image.save(images / f"{index}.png")
            (labels / f"{index}.txt").write_text("0 .3125 .2083 .5859 .2083 .5859 .5729 .3125 .5729\n")
    data = tmp_path / "source/data.yaml"
    data.write_text("path: .\ntrain: train/images\nval: val/images\ntest: test/images\nnames: [orange_immature]\n")
    prepared, run = tmp_path / "prepared", tmp_path / "run"
    prepare(data, prepared)
    run.mkdir()
    job = next(j for j in make_queue("yolo", [42], 2, 0, amp_modes=[0], validation=COMMON_VALIDATION)
               if j["model"] == model_name)
    job["recipe"]["batch"] = 2
    checkpoint = train_yolo(job, prepared, run)
    assert checkpoint.name == "best_unified.pt" and checkpoint.is_file()
    best = json.loads((run / "validation_common/best.json").read_text())
    loaded = YOLO(str(checkpoint)).model.cpu().float()
    with validation_state(loaded):
        metrics = collect_metrics(prepared / "coco/annotations/instances_val.json", prepared / "coco/images/val",
                                  yolo_predictor(loaded, 1, COMMON_VALIDATION), COMMON_VALIDATION)
    assert metrics["mask_ap_50_95"] == pytest.approx(best["mask_ap_50_95"], abs=1e-5)
    assert len(list((run / "validation_common").glob("epoch_*.json"))) == 2
    options = json.loads((run / "effective_train.json").read_text())
    assert options["patience"] == 0 and options["amp"] is False and options["cache"] is True
