"""Dependency contract tests do not require installing legacy Torch/MMCV locally."""

from types import SimpleNamespace

import pytest

from baseline_amp import environment_check
from INSTALL_CITRUS_BASELINE_ENVS import commands
from REPAIR_CITRUS_MODERN_ENV import repair_commands


@pytest.mark.parametrize("failure", ["missing", "missing_packaging"])
def test_actionable_setuptools_failure(monkeypatch, failure):
    def load(name):
        assert name == "pkg_resources"
        if failure == "missing":
            raise ModuleNotFoundError("No module named pkg_resources")
        return SimpleNamespace()

    monkeypatch.setattr(environment_check.importlib, "import_module", load)
    with pytest.raises(RuntimeError, match="setuptools==69.5.1") as error:
        environment_check.check_mmdet_environment()
    assert error.value.__cause__ is not None


def test_exercises_actual_runner_collector(monkeypatch):
    seen = []
    def load(name):
        seen.append(name)
        return {
            "pkg_resources": SimpleNamespace(packaging=SimpleNamespace(version=SimpleNamespace(parse=lambda v: v))),
            "torch.utils.cpp_extension": SimpleNamespace(CUDA_HOME=None),
            "mmengine.utils.dl_utils": SimpleNamespace(collect_env=lambda: {"CUDA available": False}),
        }[name]

    monkeypatch.setattr(environment_check.importlib, "import_module", load)
    assert environment_check.check_mmdet_environment() == {"CUDA available": False}
    assert seen == ["pkg_resources", "torch.utils.cpp_extension", "mmengine.utils.dl_utils"]


def test_does_not_suppress_other_runtime_errors(monkeypatch):
    def load(name):
        if name == "pkg_resources":
            return SimpleNamespace(packaging=SimpleNamespace(version=SimpleNamespace(parse=lambda v: v)))
        raise OSError("CUDA library incompatible")

    monkeypatch.setattr(environment_check.importlib, "import_module", load)
    with pytest.raises(OSError, match="CUDA library incompatible"):
        environment_check.check_mmdet_environment()


def test_only_mmdet_installer_gets_legacy_runtime_probe():
    from pathlib import Path
    _, legacy = commands("mmdet", "conda")
    _, modern = commands("modern", "conda")
    assert legacy[-1][-1].endswith("environment_check.py")
    assert not any("environment_check.py" in part for command in modern for part in command)
    requirements = Path(__file__).with_name("requirements-mmdet.txt").read_text()
    assert "setuptools==69.5.1" in requirements


def test_modern_repair_is_exact_and_runs_real_api_probe(tmp_path):
    target = tmp_path / "citrus_baseline/bin/python"
    commands = repair_commands(target)
    install = commands[1]
    assert install[-1] == "rfdetr==1.4.0.post0"
    assert "--force-reinstall" in install and "--no-deps" in install
    assert install[:5] == [str(target), "-I", "-m", "pip", "--isolated"]
    assert "--no-cache-dir" in install
    assert install[install.index("--index-url") + 1] == "https://pypi.org/simple"
    assert all(command[0] == str(target) for command in commands)
    assert commands[2][-2:] == ["pip", "check"]
    assert commands[3][-4:] == ["--check", "yolo", "rfdetr", "--cpu-check"]


def test_yanked_rfdetr_error_is_actionable(monkeypatch):
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import worker

    monkeypatch.setattr(worker.importlib.metadata, "version", lambda package: "1.4.0")
    with pytest.raises(RuntimeError, match="yanked release") as error:
        worker.require_version("rfdetr", "1.4.0.post0")
    assert "REPAIR_CITRUS_MODERN_ENV.py" in str(error.value)


def test_modern_install_verifies_actual_worker_after_pip_check():
    _, steps = commands("modern", "conda")
    assert steps[-2][-2:] == ["pip", "check"]
    assert steps[-1][-4:] == ["--check", "yolo", "rfdetr", "--cpu-check"]


def test_modern_repair_stops_on_failure_without_starting_training(monkeypatch, tmp_path):
    import subprocess
    import sys
    import REPAIR_CITRUS_MODERN_ENV as repair

    target = tmp_path / "python"
    target.touch()
    monkeypatch.setattr(sys, "argv", ["repair", "--python", str(target)])
    monkeypatch.setattr(repair.platform, "system", lambda: "Linux")
    seen = []

    def run(command, check):
        assert check
        seen.append(command)
        if len(seen) == 2:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(repair.subprocess, "run", run)
    with pytest.raises(SystemExit, match="Install fixed RF-DETR only"):
        repair.main()
    assert len(seen) == 2


def test_modern_repair_dry_run_never_executes(monkeypatch):
    import sys
    import REPAIR_CITRUS_MODERN_ENV as repair

    monkeypatch.setattr(sys, "argv", ["repair", "--dry-run"])

    def unexpected(*args, **kwargs):
        pytest.fail("Dry run must not execute any environment command")

    monkeypatch.setattr(repair.subprocess, "run", unexpected)
    repair.main()
