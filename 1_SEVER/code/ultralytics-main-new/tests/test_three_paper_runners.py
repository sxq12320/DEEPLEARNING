"""The three public entries select the intended models and initialization recipes."""
import pytest
import json

import RUN_YOLO11_BASELINE78 as baseline
import RUN_V12_BEST as best
import RUN_V12_ABLATION as ablation
from citrus_control78_batch import parse_args as baseline_args, queue_for, training_args
from citrus_paper_batch import effective_training, main, parse_args, selected_queue


def test_baseline_runs_both_training_views():
    args = baseline_args(baseline.arguments())
    queue = queue_for(args)
    assert len(queue) == 2 and {j["kind"] for j in queue} == {"baseline"}
    assert [j["train_views"] for j in queue] == ["whole", "mixed"]
    assert len({j["name"] for j in queue}) == 2
    recipe = training_args(args, 42)
    assert recipe["amp"] is True and recipe["pretrained"] is False
    assert recipe["dropout"] == .1 and recipe["mask_ratio"] == 4


def test_best_and_ablations_share_recipe_and_reference():
    best_args = parse_args(best.arguments())
    ablation_args = parse_args(ablation.arguments())
    assert [j["model"] for j in selected_queue(best_args)[1]] == ["P01_v12_full"]
    jobs = selected_queue(ablation_args)[1]
    assert len(jobs) == 11
    assert {j["model"] for j in jobs}.issuperset({"P00_yolo11n_control", "P01_v12_full"})
    recipe = effective_training(best_args)
    assert recipe == effective_training(ablation_args)
    assert recipe["pretrained"] == str(best_args.pretrained.resolve())
    assert recipe["amp"] is False and recipe["mask_ratio"] == 2
    assert recipe["copy_paste"] == .3 and recipe["dropout"] == 0
    assert best_args.project != ablation_args.project


@pytest.mark.parametrize("extra", [["--seeds", ""], ["--seeds", "42,42"],
                                  ["--seeds", "-1"], ["--only", "wrong_model"]])
def test_invalid_selection_fails_before_training(extra):
    with pytest.raises(ValueError):
        selected_queue(parse_args(["--data", "missing.yaml", "--project", "unused", *extra]))


@pytest.mark.parametrize("module", [baseline, best, ablation])
def test_public_entry_dry_run_has_no_filesystem_side_effects(module, monkeypatch, tmp_path):
    monkeypatch.setattr(module, "DATA", str(tmp_path / "missing.yaml"))
    monkeypatch.setattr(module, "PROJECT", str(tmp_path / "new_output"))
    monkeypatch.setattr(module, "DRY_RUN", True)
    queue = module.main(module.arguments())
    assert queue and not (tmp_path / "new_output").exists()


def test_sliced_training_cannot_silently_have_zero_tiles(tmp_path, monkeypatch):
    import citrus_e_v5_slicing
    from citrus_control78_batch import prepare_training_views
    monkeypatch.setattr(citrus_e_v5_slicing, "prepare_multiscale_views", lambda *a: tmp_path / "data.yaml")
    group = {"views": [{"global_view": True}]}
    (tmp_path / "views.json").write_text(json.dumps({"groups": [group]}))
    with pytest.raises(ValueError, match="No usable training tiles"):
        prepare_training_views("unused", tmp_path)
    group["views"].append({"global_view": False})
    (tmp_path / "views.json").write_text(json.dumps({"groups": [group]}))
    assert prepare_training_views("unused", tmp_path) == tmp_path / "data.yaml"
