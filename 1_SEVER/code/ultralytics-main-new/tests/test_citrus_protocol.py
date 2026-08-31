"""Tests for the formal paper-1 hyperparameter lock."""

from __future__ import annotations

from pathlib import Path

import pytest

from citrus_protocol import fixed_train_args, protocol_digest, validate_locked_runtime, write_protocol_snapshot


def test_formal_protocol_locks_accuracy_affecting_hyperparameters() -> None:
    """Core precision, optimization, resolution, and augmentation settings cannot drift."""
    fixed = fixed_train_args()
    assert fixed["batch"] == 16
    assert fixed["imgsz"] == 640
    assert fixed["optimizer"] == "AdamW"
    assert fixed["lr0"] == 0.001
    assert fixed["dropout"] == 0.0
    assert fixed["amp"] is False
    assert fixed["patience"] == 300
    assert fixed["mask_ratio"] == 4
    assert fixed["mosaic"] == 1.0
    assert fixed["copy_paste"] == 0.0
    assert len(protocol_digest()) == 64


def test_runtime_validator_rejects_silent_protocol_changes() -> None:
    """Batch and AMP changes must not enter formal structure comparisons silently."""
    assert validate_locked_runtime(
        batch=16, imgsz=640, workers=4, cache=False, amp=False
    ) == []
    with pytest.raises(ValueError, match="mismatches"):
        validate_locked_runtime(batch=8, imgsz=640, workers=4, cache=False, amp=False)
    with pytest.raises(ValueError, match="locks amp"):
        validate_locked_runtime(batch=16, imgsz=640, workers=4, cache=False, amp=True)
    assert validate_locked_runtime(
        batch=16, imgsz=640, workers=4, cache=False, amp=True, allow_amp_audit=True
    )


def test_protocol_snapshot_is_stable_and_conflict_safe(tmp_path: Path) -> None:
    """Every results project receives one immutable effective protocol snapshot."""
    path, digest = write_protocol_snapshot(tmp_path, {"series": "unit-test"})
    assert path.is_file()
    assert (tmp_path / "_protocol" / "formal_protocol.sha256").read_text().strip() == digest
    assert write_protocol_snapshot(tmp_path, {"series": "unit-test"}) == (path, digest)
    with pytest.raises(FileExistsError, match="Conflicting"):
        write_protocol_snapshot(tmp_path, {"series": "different"})
