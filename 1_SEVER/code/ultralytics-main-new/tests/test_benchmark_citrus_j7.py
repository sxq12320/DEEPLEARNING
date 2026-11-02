"""Pure argument/summary contracts; not a deployment timing result."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmark_citrus_j7 as benchmark


def test_bounded_cli_rejects_yaml_duplicate_or_existing_output(tmp_path):
    weights = tmp_path / "trained.pt"
    weights.write_bytes(b"fixture-only-not-loaded")
    source = tmp_path / "images"
    source.mkdir()
    output = tmp_path / "new.json"
    base = ["--weights", str(weights), "--source", str(source), "--output", str(output)]
    args = benchmark.parse_args(base)
    assert (args.samples, args.repeats, args.warmup) == (20, 2, 3)
    assert benchmark.PREDICT_ARGS["half"] is False
    assert benchmark.PREDICT_ARGS["conf"] == 0.001
    assert benchmark.PREDICT_ARGS["max_det"] == 300
    with pytest.raises(SystemExit):
        benchmark.parse_args(base + ["--samples", "31"])
    yaml = tmp_path / "random.yaml"
    yaml.write_text("nc: 1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--weights", str(yaml)] + base[2:])
    output.write_text(json.dumps({"keep": True}), encoding="utf-8")
    with pytest.raises(SystemExit):
        benchmark.parse_args(base)
    assert json.loads(output.read_text("utf-8")) == {"keep": True}


def test_latency_summary_linear_percentile_and_prediction_counts():
    rows = [{"wall_ms": value, "preprocess_ms": 1, "inference_ms": 2, "postprocess_ms": value - 3,
             "decoded_after_nms": 300, "returned_nonempty_instances": index}
            for index, value in enumerate([4, 5, 7, 12])]
    result = benchmark.summarize_rows(rows)
    assert result["prediction_calls"] == 4
    assert result["wall_ms"]["median"] == 6
    assert result["wall_ms"]["p95"] == pytest.approx(11.25)
    assert result["returned_nonempty_instances"] == {"min": 0, "max": 3, "mean": 1.5}
    assert result["decoded_after_nms"] == {"min": 300, "max": 300, "mean": 300}
    with pytest.raises(ValueError):
        benchmark.summarize_rows([])


def test_training_evidence_and_source_selection_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="no training"):
        benchmark.checkpoint_evidence(SimpleNamespace(ckpt={"epoch": -1}))
    evidence = benchmark.checkpoint_evidence(SimpleNamespace(ckpt={"epoch": 0, "train_metrics": {}}))
    assert evidence["saved_epoch_zero_based"] == 0
    assert evidence["integrity_or_quality_certified"] is False
    for name in ("a.jpg", "b.png", "not_an_image.txt"):
        (tmp_path / name).write_bytes(b"fixture-selection-only")
    assert [path.name for path in benchmark.source_files(tmp_path, 1)] == ["a.jpg"]
