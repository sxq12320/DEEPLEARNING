"""Regression contracts added after the completed J96/partial J97 evidence."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from citrus_j7r_batch import record_failure
from citrus_j7r_training import J7RTrainer
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(number):
    path, = (ROOT / "0_orange_yaml/J_V7R_series").glob("J{}_*.yaml".format(number))
    return YOLO(str(path), task="segment", verbose=False)


@pytest.mark.parametrize("number", [112, 116, 118])
def test_all_packed_public_exports_explicitly_rejected(number):
    with pytest.raises(ValueError, match="export is not verified"):
        model(number).export(format="onnx")


def test_same_observation_cannot_reinterpret_mature_as_native_checkpoint():
    trainer = object.__new__(J7RTrainer)
    trainer.data = {"nc": 1, "channels": 3}
    trainer.args = SimpleNamespace(seed=42)
    with pytest.raises(ValueError, match="same architecture"):
        trainer.get_model(cfg=model(118).model.yaml, weights=model(112).model, verbose=False)


def test_failure_logs_without_gpu_initialization_or_protocol_change(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    record_failure(tmp_path, RuntimeError("CUDA out of memory: synthetic test"))
    destination = tmp_path / "j7r_failure.json"
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["status"] == "failed" and saved["cuda_bytes_this_process"] is None
    assert saved["automatic_batch_or_amp_change"] is False
    original = destination.read_bytes()
    record_failure(tmp_path, RuntimeError("later error must not overwrite evidence"))
    assert destination.read_bytes() == original
