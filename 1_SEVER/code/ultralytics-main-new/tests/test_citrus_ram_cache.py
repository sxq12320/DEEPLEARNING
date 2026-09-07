"""User-requested RAM defaults across protocol, CLI and clickable launchers."""

import importlib.util

import pytest
import yaml

from citrus_foreground import ROOT, RUNNERS, run_foreground
from citrus_protocol import fixed_train_args, load_protocol, normalize_cache, validate_locked_runtime


@pytest.mark.parametrize("value", [True, "true", "True", "ram", "RAM"])
def test_ram_aliases_match_the_protocol(value):
    assert normalize_cache(value) is True
    assert validate_locked_runtime(batch=16, imgsz=640, workers=4, cache=value, amp=False) == []


@pytest.mark.parametrize("value", [False, "false", "disk"])
def test_disabled_or_disk_cache_cannot_silently_enter_ram_protocol(value):
    with pytest.raises(ValueError, match="mismatches"):
        validate_locked_runtime(batch=16, imgsz=640, workers=4, cache=value, amp=False)


def test_v2_changes_only_cache_in_training_args():
    old = yaml.safe_load((ROOT / "protocols/citrus_paper1_formal_v1.yaml").read_text())
    new = load_protocol()
    assert old["fixed_train"]["cache"] is False
    assert new["fixed_train"]["cache"] is True
    before, after = dict(old["fixed_train"]), dict(new["fixed_train"])
    before.pop("cache")
    after.pop("cache")
    assert before == after
    assert old["fixed_validation"] == new["fixed_validation"]
    assert old["initialization"] == new["initialization"]
    default = yaml.safe_load((ROOT / "ultralytics/cfg/default.yaml").read_text(encoding="utf-8"))
    assert default["cache"] is True


@pytest.mark.parametrize("series", [k for k, v in RUNNERS.items() if v.supports_cache])
def test_batch_cli_defaults_to_ram(series, monkeypatch):
    import sys

    path = ROOT / RUNNERS[series].script
    spec = importlib.util.spec_from_file_location("cache_test_" + series, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(path), "--data", "unused.yaml"])
    assert normalize_cache(module.parse_args().cache) is True
    monkeypatch.setattr(sys, "argv", [str(path), "--data", "unused.yaml", "--cache", "True"])
    assert normalize_cache(module.parse_args().cache) is True


def test_foreground_accepts_literal_true(tmp_path, monkeypatch):
    import citrus_foreground

    data = tmp_path / "data.yaml"
    data.write_text("train: images\nval: images\nnames: [fruit]\n")
    captured = []
    monkeypatch.setattr(citrus_foreground, "_run_module_in_current_process", lambda script, argv: captured.extend(argv))
    run_foreground(series="SAGE_V7", data=str(data), suite="refusion", epochs=300, batch=16,
                   imgsz=640, device="cpu", workers=4, cache=True, amp=False, dry_run=True,
                   device_lock=False, refuse_busy_gpu=False)
    assert captured[captured.index("--cache") + 1] == "ram"
    assert fixed_train_args()["cache"] is True
