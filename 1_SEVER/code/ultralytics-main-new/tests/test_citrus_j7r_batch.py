"""Foreground queue, fixed recipe and official routing contracts."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import citrus_j7r_batch as runner


@pytest.fixture
def queue(tmp_path, monkeypatch):
    import citrus_j2_batch
    import ultralytics
    import ultralytics.utils.torch_utils

    yaml_dir = tmp_path / "yaml"
    yaml_dir.mkdir()
    for stem in ("J110_global", "J112_raw"):
        (yaml_dir / (stem + ".yaml")).write_text("nc: 1\ncitrus_observation: raw\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    data.write_text("names: [orange_immature]\n", encoding="utf-8")
    state = SimpleNamespace(calls=[], error=None)

    class FakeYOLO:
        def __init__(self, path, **kwargs):
            self.path = Path(path)

        def add_callback(self, *args):
            pass

        def train(self, **kwargs):
            state.calls.append((self.path.stem, kwargs))
            if state.error is not None:
                raise state.error
            output = Path(kwargs["project"]) / kwargs["name"] / "selected_mask_validation"
            output.mkdir(parents=True)
            (output / "selected_checkpoint.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(runner, "YAML_DIR", yaml_dir)
    monkeypatch.setattr(runner, "source_manifest", lambda: {"source": "unchanged"})
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    monkeypatch.setattr(ultralytics.utils.torch_utils, "init_seeds", lambda *a, **k: None)
    monkeypatch.setattr(citrus_j2_batch, "settings", lambda *a, **k: {"amp": False, "cache": True})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return data, tmp_path / "runs", state


def test_entry_import_does_not_initialize_torch():
    result = subprocess.run([sys.executable, "-c", "import sys; import citrus_j7r_batch; assert 'torch' not in sys.modules"],
                            cwd=str(runner.ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_selection_and_dependencies():
    assert [p.stem.split("_")[0] for p in runner.select_models()] == list(runner.PRIORITY_IDS)
    assert len(runner.select_models("all")) == 14
    assert len(runner.select_models("native")) == 3
    assert len(runner.select_models("phase")) == 2
    assert [p.stem.split("_")[0] for p in runner.select_models("policy")] == ["J119", "J116"]
    assert [p.stem.split("_")[0] for p in runner.select_models("evidence")] == ["J118", "J123", "J120"]
    assert [p.stem.split("_")[0] for p in runner.select_models("legacy")] == list(runner.LEGACY_IDS)
    for invalid in ("J100", "J112,J112_raw_c3_evidence", "J112_wrong_name"):
        with pytest.raises(ValueError):
            runner.select_models(only=invalid)
    sources = runner.source_manifest()
    for source in runner.NEW_SOURCES:
        assert source in sources and len(sources[source]) == 64


def test_fixed_recipe_is_accuracy_protocol(tmp_path):
    path, = runner.YAML_DIR.glob("J112_*.yaml")
    record = runner.experiment_record(path, tmp_path / "actual_data.yaml", 300, 42, 16, 4, True, {})
    recipe = record["train"]
    assert recipe["amp"] is False and recipe["cache"] is True
    assert recipe["optimizer"] == "AdamW" and recipe["lr0"] == .001
    assert recipe["pretrained"] is False and recipe["imgsz"] == 640 and recipe["mask_ratio"] == 2
    assert recipe["epochs"] == 300 and recipe["dropout"] == 0
    assert record["input"]["observation"] == "raw"


def test_completed_skip_and_partial_protection(queue):
    data, project, state = queue
    runner.run(data, "cpu", project, epochs=1, only="J110", plots=False)
    assert len(state.calls) == 1
    assert runner.run(data, "cpu", project, epochs=1, only="J110", plots=False) == []
    assert len(state.calls) == 1
    completed = project / "J110_global_seed42/j7r_completed.json"
    saved = json.loads(completed.read_text(encoding="utf-8"))
    assert saved["train"]["amp"] is False and saved["train"]["cache"] is True
    with pytest.raises(FileExistsError):
        runner.run(data, "cpu", project, epochs=2, only="J110", plots=False)
    assert len(state.calls) == 1


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("train failed")])
def test_failure_does_not_start_next_arm(queue, error):
    data, project, state = queue
    state.error = error
    with pytest.raises(type(error)):
        runner.run(data, "cpu", project, epochs=1, only="J110,J112", plots=False)
    assert len(state.calls) == 1
    assert not (project / "J112_raw_seed42").exists()


def test_plot_failure_precedes_training(queue, monkeypatch):
    data, project, state = queue

    def fail():
        raise RuntimeError("incompatible plotting packages")

    monkeypatch.setattr(runner, "plotting_preflight", fail)
    with pytest.raises(RuntimeError, match="plotting packages"):
        runner.run(data, "cpu", project, epochs=300, only="J110", plots=True)
    assert state.calls == [] and not project.exists()


def test_dry_run_never_loads_dataset_or_creates_project(tmp_path, monkeypatch):
    checked = []
    monkeypatch.setattr(runner, "profile_model", lambda path: checked.append(path.stem) or {})
    project = tmp_path / "not_created"
    result = runner.run(tmp_path / "missing_data.yaml", "cpu", project, dry_run=True, only="J112", plots=True)
    assert len(result) == 1 and checked == ["J112_raw_c3_evidence"]
    assert not project.exists()
