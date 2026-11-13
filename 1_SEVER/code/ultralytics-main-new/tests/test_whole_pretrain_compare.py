"""Protocol and result-protection checks for the bounded whole-image comparison."""
import csv
import hashlib

import pytest
import yaml

import RUN_WHOLE_PRETRAIN_COMPARE as runner


def test_final_validation_cannot_silently_switch_label_population(tmp_path):
    runner.save_json(tmp_path / "effective_runtime.json", {"actual": {"val": {"targets": 1181}}})
    runner.verify_eval_population(tmp_path, {"targets": 1181})
    with pytest.raises(RuntimeError, match="Final validation targets"):
        runner.verify_eval_population(tmp_path, {"targets": 1077})


def test_candidate_changes_only_stem():
    baseline = runner.model_config("S42_sage42")
    candidate = runner.model_config("C01_sage42_contrast")
    assert candidate["backbone"][0][2] == "EV10ContrastStem"
    candidate["backbone"][0][2] = "Conv"
    assert candidate == baseline


def test_full_label_common_budget_and_smoke_isolation(tmp_path):
    args = runner.arguments(["--project", str(tmp_path / "formal"), "--smoke"])
    recipe = runner.recipe(args)
    assert args.project.name == "formal_SMOKE"
    assert recipe["epochs"] == 1 and recipe["patience"] == 0
    assert recipe["mask_ratio"] == 2 and recipe["rect"] is False
    assert recipe["pretrained"].endswith("yolo11n-seg.pt")
    assert recipe["copy_paste"] == 0.3


def test_early_stop_is_not_accepted_as_equal_budget(tmp_path):
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights/best_mask.pt").write_bytes(b"test")
    (run / "results.csv").write_text("epoch\n1\n")
    runner.save_json(run / "training_finished.json", dict(budget=300, last_epoch=1, reason="early_stop"))
    assert not runner.completed_run(run, 300)
    runner.save_json(run / "training_finished.json", dict(budget=1, last_epoch=1, reason="budget"))
    assert runner.completed_run(run, 1)


def dataset_fixture(tmp_path, crossed_group=False):
    rows = []
    for split in ("train", "val", "test"):
        images, labels = tmp_path / split / "images", tmp_path / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        # The audit hashes files; decoding is exercised by the real-image integration smoke run.
        payload = split.encode()
        (images / f"{split}.png").write_bytes(payload)
        (labels / f"{split}.txt").write_text("0 .1 .1 .3 .1 .3 .3 .1 .3\n")
        rows.append(dict(new_split=split, image=f"{split}.png", group_id="same" if crossed_group else split,
                         sha256=hashlib.sha256(payload).hexdigest()))
    (tmp_path / "audit").mkdir()
    with (tmp_path / "audit/split_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(dict(names={0: "orange_immature"}, path="/stale/server/path")))
    return path


def test_audit_uses_actual_packaged_members_and_freezes_labels(tmp_path):
    path = dataset_fixture(tmp_path)
    spec, audit = runner.audit_dataset(path)
    assert spec["val"] == str((tmp_path / "val/images").resolve())
    assert audit["counts"]["val"] == {"images": 1, "targets": 1}
    original = audit["records"][0]["label_sha256"]
    (tmp_path / "train/labels/train.txt").write_text("0 .1 .1 .4 .1 .4 .4 .1 .4\n")
    assert runner.audit_dataset(path)[1]["records"][0]["label_sha256"] != original


def test_group_crossing_is_rejected(tmp_path):
    path = dataset_fixture(tmp_path, crossed_group=True)
    with pytest.raises(ValueError, match="crossing splits"):
        runner.audit_dataset(path)


def test_changed_image_is_rejected(tmp_path):
    path = dataset_fixture(tmp_path)
    (tmp_path / "val/images/val.png").write_bytes(b"different image")
    with pytest.raises(ValueError, match="no longer matches manifest"):
        runner.audit_dataset(path)


def test_untracked_output_is_protected(tmp_path):
    (tmp_path / "existing.csv").write_text("keep me")
    args = runner.arguments(["--project", str(tmp_path)])
    with pytest.raises(FileExistsError, match="Nonempty project"):
        runner.prepare(args)
    assert (tmp_path / "existing.csv").read_text() == "keep me"
