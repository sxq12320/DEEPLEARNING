"""Tests for the clickable foreground citrus experiment launcher."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

from citrus_foreground import (
    ROOT,
    RUNNERS,
    DeviceRunLock,
    _build_argv,
    _run_module_in_current_process,
    resolve_runner,
)


def test_registry_covers_every_dated_batch_runner() -> None:
    registered = {spec.script for spec in RUNNERS.values()}
    discovered = {path.name for path in ROOT.glob("*citrus*batch.py")}
    assert registered == discovered
    assert resolve_runner("S")[0] == "SWIFT"
    assert resolve_runner("L")[0] == "TOPO"
    assert resolve_runner("sage-v3")[0] == "SAGE_V3"


@pytest.mark.parametrize("canonical", tuple(RUNNERS))
def test_every_runner_contract_builds_supported_argv(canonical: str) -> None:
    spec = RUNNERS[canonical]
    argv = _build_argv(
        spec,
        data="/tmp/data.yaml",
        suite=spec.suites[0],
        epochs=3,
        batch=16,
        imgsz=640,
        device="0",
        workers=4,
        project="/tmp/results",
        pretrained="/tmp/yolo11n-seg.pt",
        seeds="42",
        only="",
        cache="false",
        amp=None,
        dry_run=True,
        skip_completed=True,
        fail_fast=True,
    )
    assert argv[:2] == ["--data", "/tmp/data.yaml"]
    assert "--dry-run" in argv
    assert "--fail-fast" in argv
    assert ("--skip-completed" in argv) is spec.supports_skip_completed
    assert ("--cache" in argv) is spec.supports_cache
    module_name = "test_foreground_" + canonical
    definition = importlib.util.spec_from_file_location(module_name, ROOT / spec.script)
    module = importlib.util.module_from_spec(definition)
    sys.modules[module_name] = module
    original = sys.argv[:]
    try:
        definition.loader.exec_module(module)
        sys.argv = [spec.script, *argv]
        parsed = module.parse_args()
        assert parsed.epochs == 3
    finally:
        sys.argv = original
        sys.modules.pop(module_name, None)


def test_module_runs_in_current_process_and_restores_argv(tmp_path) -> None:
    marker = tmp_path / "marker.txt"
    script = tmp_path / "fake_batch.py"
    script.write_text(
        "import os\n"
        "import sys\n"
        "def main():\n"
        f"    open({str(marker)!r}, 'w', encoding='utf-8').write(str(os.getpid()) + '|' + '|'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    original = sys.argv[:]
    _run_module_in_current_process(script, ["--dry-run", "--epochs", "3"])
    pid, arguments = marker.read_text(encoding="utf-8").split("|", maxsplit=1)
    assert int(pid) == os.getpid()
    assert arguments == "--dry-run|--epochs|3"
    assert sys.argv == original


def test_device_lock_rejects_a_duplicate_live_launcher() -> None:
    with DeviceRunLock(("pytest-device",)):
        with pytest.raises(RuntimeError, match="already has a citrus foreground launcher lock"):
            with DeviceRunLock(("pytest-device",)):
                pass


def test_device_lock_releases_on_exception():
    with pytest.raises(KeyboardInterrupt):
        with DeviceRunLock(("pytest-interrupt",)):
            raise KeyboardInterrupt()
    with DeviceRunLock(("pytest-interrupt",)):
        pass


def test_interrupt_prevents_next_experiment_and_restores_argv(tmp_path):
    script = tmp_path / "interrupt_batch.py"
    script.write_text("def main():\n    raise KeyboardInterrupt()\n", encoding="utf-8")
    original = sys.argv[:]
    with pytest.raises(KeyboardInterrupt):
        _run_module_in_current_process(script, [])
    assert sys.argv == original


def test_partial_runs_are_not_marked_completed(tmp_path):
    import json

    definition = importlib.util.spec_from_file_location("test_sage4_completion", ROOT / RUNNERS["SAGE_V4"].script)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights/best.pt").touch()
    (tmp_path / "results.csv").write_text("epoch,metric\n10,0.5\n", encoding="utf-8")
    assert not module.completed(tmp_path, 300, 42)
    (tmp_path / "completed.json").write_text(json.dumps({"epochs": 300, "seed": 42}), encoding="utf-8")
    assert not module.completed(tmp_path, 300, 42)
    (tmp_path / "results.csv").write_text("epoch,metric\n300,0.5\n", encoding="utf-8")
    assert module.completed(tmp_path, 300, 42)
    assert not module.completed(tmp_path, 300, 43)
