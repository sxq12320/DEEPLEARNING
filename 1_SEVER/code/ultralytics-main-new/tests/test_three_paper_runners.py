"""The three public entries select the intended models and initialization recipes."""
import csv
import json

import pytest

import RUN_YOLO11_BASELINE78 as baseline
import RUN_V12_BEST as best
import RUN_V12_ABLATION as ablation
from citrus_control78_batch import parse_args as baseline_args, queue_for, training_args
from citrus_paper_batch import (
    effective_training, finished_training, job_name, main, parse_args, selected_queue,
    summarize, summarize_recipe_comparison, training_for_job,
)


def test_baseline_runs_both_training_views():
    args = baseline_args(baseline.arguments())
    queue = queue_for(args)
    assert len(queue) == 2 and {j["kind"] for j in queue} == {"baseline"}
    assert [j["train_views"] for j in queue] == ["whole", "mixed"]
    assert len({j["name"] for j in queue}) == 2
    recipe = training_args(args, 42)
    assert recipe["amp"] is True and recipe["pretrained"] is False
    assert recipe["dropout"] == .1 and recipe["mask_ratio"] == 4


def test_best_and_ablation_candidates_share_recipe_while_p00_uses_78():
    best_args = parse_args(best.arguments())
    ablation_args = parse_args(ablation.arguments())
    assert [j["model"] for j in selected_queue(best_args)[1]] == ["P01_v12_full"]
    jobs = selected_queue(ablation_args)[1]
    assert len(jobs) == 12
    assert len({job_name(j) for j in jobs}) == 12
    assert {j["model"] for j in jobs}.issuperset({"P00_yolo11n_control", "P01_v12_full"})
    recipe = effective_training(best_args)
    assert recipe == effective_training(ablation_args)
    assert recipe["pretrained"] == str(best_args.pretrained.resolve())
    assert recipe["amp"] is False and recipe["mask_ratio"] == 2
    assert recipe["copy_paste"] == .3 and recipe["dropout"] == 0
    assert best_args.project != ablation_args.project
    base_jobs = [j for j in jobs if j["model"] == "P00_yolo11n_control"]
    full_job = next(j for j in jobs if j["model"] == "P01_v12_full")
    reference = training_args(baseline_args(baseline.arguments()), 42)
    assert {j["train_views"] for j in base_jobs} == {"whole", "mixed"}
    assert all(training_for_job(ablation_args, j) == reference for j in base_jobs)
    assert all(j["amp"] is True for j in base_jobs)
    assert full_job["amp"] is False and full_job["train_views"] == "mixed"
    assert training_for_job(ablation_args, full_job) == dict(recipe, seed=42, amp=False)
    for job in jobs:
        if job not in base_jobs:
            assert training_for_job(ablation_args, job) == dict(recipe, seed=42, amp=False)


@pytest.mark.parametrize("selection, expected", [
    ("whole", ["whole"]), ("mixed", ["mixed"]), ("both", ["whole", "mixed"]),
])
def test_p00_training_views_only_change_input_and_output_name(selection, expected):
    args = parse_args(["--data", "unused.yaml", "--project", "unused", "--seeds", "42,43",
                       "--baseline-recipe", "legacy78", "--baseline-train-views", selection,
                       "--only", "P00_yolo11n_control"])
    _, queue = selected_queue(args)
    assert len(queue) == len(expected) * 2
    assert len({job_name(j) for j in queue}) == len(queue)
    for seed in (42, 43):
        group = [j for j in queue if j["seed"] == seed]
        assert [j["train_views"] for j in group] == expected
        assert all(training_for_job(args, j) == training_args(args, seed) for j in group)


def test_matched_mode_remains_a_paired_recipe(monkeypatch):
    monkeypatch.setattr(ablation, "BASELINE_RECIPE", "matched")
    args = parse_args(ablation.arguments())
    _, jobs = selected_queue(args)
    assert {j["amp"] for j in jobs} == {False}
    assert all("recipe" not in j for j in jobs)
    assert all(training_for_job(args, j) == training_for_job(args, jobs[0]) for j in jobs)


def test_recipe_comparison_keeps_seed_split_and_modes_separate(tmp_path):
    rows = []
    for model, mode, ap in (("P00_yolo11n_control", "global", .70),
                             ("P00_yolo11n_control", "trustedmask", .75),
                             ("P01_v12_full", "global", .80),
                             ("P01_v12_full", "trustedmask", .85)):
        rows.append(dict(model=model, seed=42, split="val", mode=mode, run=model, training_recipe=model,
                         diagnostic_mask_ap50=ap, diagnostic_mask_ap5095=ap-.1,
                         diagnostic_mask_recall=ap-.2, r_at_p90=None))
    assert summarize_recipe_comparison(tmp_path, rows[:2]) == []
    assert not (tmp_path/"P00_P01_COMPARISON.csv").exists()
    comparisons = summarize_recipe_comparison(tmp_path, rows)
    assert {r["comparison"]: r["map50_delta_pp"] for r in comparisons} == pytest.approx({
        "requested_full_system": 15., "whole_vs_whole": 10., "ten_views_vs_ten_views": 10.,
    })
    assert all(r["recall_at_p90_delta_pp"] is None for r in comparisons)
    mismatched = [dict(r, seed=43) if r["model"] == "P01_v12_full" else r for r in rows]
    assert summarize_recipe_comparison(tmp_path, mismatched) == []


def test_two_p00_results_survive_aggregation_and_compare_same_evaluation_modes(tmp_path):
    args = parse_args(["--data", "unused.yaml", "--project", str(tmp_path), "--seeds", "42",
                       "--baseline-recipe", "legacy78", "--baseline-train-views", "both",
                       "--only", "P00_yolo11n_control,P01_v12_full"])
    _, queue = selected_queue(args)
    modes = ("global", "fixed_box", "crossmask", "trustedmask")
    for job in queue:
        ap = .80 if job["model"] == "P01_v12_full" else (.72 if job["train_views"] == "mixed" else .70)
        payload = {mode: {"metrics/mAP50(M)": ap + i*.01, "metrics/mAP50-95(M)": ap-.1,
                          "metrics/recall(M)": ap-.2, "operating_p90": {"recall": ap-.3}}
                   for i, mode in enumerate(modes)}
        output = tmp_path / job_name(job) / "eval_val/paired_metrics.json"
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps({"summary": payload}), encoding="utf-8")
    rows = summarize(tmp_path, queue)
    assert len(rows) == 12 and len({r["run"] for r in rows}) == 3
    with (tmp_path / "P00_SLICE_COMPARISON.csv").open(encoding="utf-8-sig", newline="") as handle:
        sliced = list(csv.DictReader(handle))
    assert {r["evaluation_mode"] for r in sliced} == set(modes)
    assert all(float(r["map50_delta_pp"]) == pytest.approx(2.) for r in sliced)
    assert all(r["whole_run"] != r["sliced_run"] for r in sliced)
    with (tmp_path / "P00_P01_COMPARISON.csv").open(encoding="utf-8-sig", newline="") as handle:
        comparisons = list(csv.DictReader(handle))
    assert len(comparisons) == 6
    whole_eval = {r["baseline_train_views"]: float(r["map50_delta_pp"]) for r in comparisons
                  if r["comparison"] == "whole_vs_whole"}
    assert whole_eval == pytest.approx({"whole": 10., "mixed": 8.})


def test_paper_loop_recognizes_only_explicit_legacy_early_stop(tmp_path):
    (tmp_path/"weights").mkdir()
    (tmp_path/"weights/best_mask.pt").touch()
    (tmp_path/"results.csv").write_text("epoch\n155\n")
    assert not finished_training(tmp_path, 300, legacy78=True)
    (tmp_path/"training_finished.json").write_text(json.dumps(dict(budget=300,last_epoch=155,reason="early_stop")))
    assert finished_training(tmp_path, 300, legacy78=True)
    assert not finished_training(tmp_path, 300)
    assert not finished_training(tmp_path, 301, legacy78=True)


def test_selected_removals_can_be_paired_across_seeds(monkeypatch):
    names = {"P01_v12_full", "P02_no_persistent_detail", "P06_bilinear_only"}
    monkeypatch.setattr(ablation, "SEEDS", [42, 43, 44])
    monkeypatch.setattr(ablation, "ONLY", ",".join(sorted(names)))
    monkeypatch.setattr(ablation, "REPEAT_ABLATIONS", True)
    _, queue = selected_queue(parse_args(ablation.arguments()))
    assert {(j["model"], j["seed"]) for j in queue} == {(n, s) for n in names for s in (42, 43, 44)}
    assert len(queue) == 9


def test_visualize_all_uses_same_project_seed_and_weight(tmp_path, monkeypatch):
    from argparse import Namespace
    import VIS_E12_ABLATION as vis

    names = ["P01_v12_full", *vis.CONTROLS.values()]
    paths = [tmp_path / (n + "_seed43_amp0") / "weights/best_mask.pt" for n in names]
    for p in paths:
        p.parent.mkdir(parents=True)
        p.touch()
    calls = []
    monkeypatch.setattr(vis, "main", calls.append)
    args = Namespace(all_retrained=True, control=None, full=paths[0], output=tmp_path/"figures")
    vis.run(args)
    assert [c.control for c in calls] == paths[1:]
    assert all(c.full == paths[0] and c.output.name == c.mechanism for c in calls)
    paths[-1].unlink()
    calls.clear()
    with pytest.raises(FileNotFoundError, match="P06_bilinear_only"):
        vis.run(args)
    assert not calls


def test_score_aligned_size_comparison_uses_each_models_threshold(tmp_path):
    import numpy as np
    from VIS_E12_ABLATION import precision90_by_size

    rows = [dict(image="pic", gt_index=i, area640=a) for i, a in enumerate((100, 300))]
    candidates = {"full": [(np.eye(2), np.array([.8, .5]))], "control": [(np.eye(2), np.array([.6, .4]))]}
    ranks = {k: {"confidence_at_precision90": v} for k, v in (("full", .7), ("control", .4))}
    precision90_by_size(tmp_path, rows, ["pic"], candidates, ranks)
    groups = json.loads((tmp_path/"precision90_by_size.json").read_text())["groups"]
    assert groups["tiny"]["full"] == groups["tiny"]["control"] == 1
    assert groups["small"]["lost"] == 1 and groups["all"]["delta_pp"] == -50.0


@pytest.mark.parametrize("extra", [["--seeds", ""], ["--seeds", "42,42"],
                                  ["--seeds", "-1"], ["--only", "wrong_model"],
                                  ["--baseline-recipe", "legacy78", "--amp-audit"]])
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
