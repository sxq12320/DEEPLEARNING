"""Bounded native-predict benchmark contract, not a latency/accuracy result."""

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark_citrus_j7 import PREDICT_ARGS, checkpoint_evidence, parse_args, percentile, source_files, summarize_rows


def test_fixed_native_prediction_protocol():
    assert PREDICT_ARGS["rect"] is False and PREDICT_ARGS["half"] is False
    assert PREDICT_ARGS["conf"] == .001 and PREDICT_ARGS["max_det"] == 300
    assert PREDICT_ARGS["imgsz"] == 640 and PREDICT_ARGS["batch"] == 1
    assert not any(PREDICT_ARGS[key] for key in ("save", "save_txt", "save_crop", "plots", "show"))


def test_summary_keeps_complete_wall_separate_from_component_speed():
    rows = [dict(wall_ms=v, preprocess_ms=1, inference_ms=2, postprocess_ms=3,
                 decoded_after_nms=300, returned_nonempty_instances=v) for v in (10, 20)]
    summary = summarize_rows(rows)
    assert summary["wall_ms"]["median"] == 15
    assert summary["wall_ms"]["p95"] == pytest.approx(19.5)
    assert summary["returned_nonempty_instances"]["mean"] == 15
    assert summary["decoded_after_nms"]["mean"] == 300
    assert percentile([1], 95) == 1
    json.dumps(summary, allow_nan=False)


def test_no_untrained_checkpoint_or_missing_evidence_is_benchmarked():
    with pytest.raises(ValueError, match="no training"):
        checkpoint_evidence(SimpleNamespace(ckpt={"date": "today"}))
    assert checkpoint_evidence(SimpleNamespace(ckpt={"epoch": 0}))["saved_epoch_zero_based"] == 0


def test_parsing_does_not_allow_yaml_or_overwrite_and_source_is_bounded(tmp_path):
    weights = tmp_path / "trained.pt"
    weights.write_bytes(b"parsing fixture only")
    image = tmp_path / "example.jpg"
    image.write_bytes(b"source selection fixture only")
    output = tmp_path / "new.json"
    argv = ["--weights", str(weights), "--source", str(image), "--output", str(output)]
    args = parse_args(argv)
    assert args.samples == 20 and args.repeats == 2 and args.warmup == 3
    assert source_files(tmp_path, 1) == [image]
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_args(argv)
    with pytest.raises(SystemExit):
        parse_args(argv + ["--samples", "999"])


def test_benchmark_help_and_new_sources_are_python38_compatible():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "benchmark_citrus_j7.py", "--help"], cwd=str(root),
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0 and "--weights" in result.stdout
    for source in ("benchmark_citrus_j7.py", "citrus_j7_pr_audit.py", "citrus_j7_training.py", "citrus_j7_batch.py"):
        ast.parse((root / source).read_text("utf-8"), feature_version=(3, 8))
