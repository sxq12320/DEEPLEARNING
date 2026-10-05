"""Microbenchmark contracts; CPU tests do not assert a latency improvement."""

import json

import pytest
import torch

from citrus_j_benchmark import run, synthetic_batch


def test_synthetic_targets_are_deterministic_and_in_bounds():
    batch = synthetic_batch(torch, 2, 96, "cpu")
    repeated = synthetic_batch(torch, 2, 96, "cpu")
    torch.testing.assert_close(batch["img"], repeated["img"])
    assert batch["masks"].shape == (2, 48, 48)
    assert batch["bboxes"].shape == (16, 4)
    assert set(batch["masks"].unique().tolist()) == set(range(9))
    assert (batch["bboxes"] > 0).all() and (batch["bboxes"] < 1).all()


@pytest.mark.parametrize("name", ["J00_yolo11n_control", "J11_synthesis", "J17_budget"])
def test_benchmark_inference_fusion_and_training_step(tmp_path, name):
    destination = tmp_path / name
    report = run(destination, "cpu", [name], size=96, batch_size=2, warmup=1, repeats=2, threads=2)
    row = report["results"][0]
    assert row["status"] == "ok"
    for phase in ("forward_unfused", "forward_fused", "training_step"):
        assert row[phase]["median_ms"] > 0
        assert len(row[phase]["samples_ms"]) == 2
        assert row[phase]["peak_allocated_mib"] is None
    assert json.loads((destination / "benchmark.json").read_text())["training_batch"] == 2
    assert (destination / "benchmark.csv").is_file()
    with pytest.raises(FileExistsError):
        run(destination, "cpu", [name], size=96, batch_size=2, warmup=1, repeats=2, threads=2)
