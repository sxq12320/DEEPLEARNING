"""Shared foreground launcher utilities for citrus experiment suites.

This module deliberately runs a selected batch module in the current Python
process.  It does not create a shell, a detached process, or concurrent training
workers. DataLoader workers are still created by Ultralytics (``workers``).
Short-lived metadata probes (git/nvidia-smi) are not training processes.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunnerSpec:
    """Command-line contract for one historical citrus batch runner."""

    script: str
    suites: Tuple[str, ...]
    aliases: Tuple[str, ...]
    seed_flag: str = "--seeds"
    supports_cache: bool = False
    supports_amp: bool = False
    supports_skip_completed: bool = False


RUNNERS: Dict[str, RunnerSpec] = {
    "SAGE_V5": RunnerSpec(
        "20260904_citrus_sage_v5_batch.py",
        ("smoke", "screen", "structure", "geometry", "backbone", "all", "control"),
        ("SAGE5", "SAGE_V5", "SAGE-V5"),
        supports_cache=True,
        supports_amp=True,
        supports_skip_completed=True,
    ),
    "SAGE_V4R": RunnerSpec(
        "20260903_citrus_sage_v4r_batch.py",
        ("smoke", "screen", "structure", "geometry", "backbone", "all", "control"),
        ("SAGE4R", "SAGE_V4R", "SAGE-V4R"),
        supports_cache=True,
        supports_amp=True,
        supports_skip_completed=True,
    ),
    "SWIFT": RunnerSpec(
        "20260824_citrus_swift_batch.py",
        ("architectures", "losses", "all", "final"),
        ("S", "SWIFT"),
    ),
    "TOPO": RunnerSpec(
        "20260824_citrus_topo_batch.py",
        ("architectures", "losses", "all", "final"),
        ("L", "TOPO", "TOPOLOGY"),
        seed_flag="--seed",
    ),
    "B": RunnerSpec(
        "20260826_citrus_b_batch.py",
        ("architectures", "smoke", "screening", "losses", "all", "final"),
        ("B", "B_SERIES"),
    ),
    "C": RunnerSpec(
        "20260828_citrus_c_batch.py",
        ("smoke", "controls", "core", "architectures", "losses"),
        ("C", "C_SERIES"),
    ),
    "D": RunnerSpec(
        "20260828_citrus_d_batch.py",
        ("smoke", "controls", "core", "architectures", "losses"),
        ("D", "D_SERIES"),
    ),
    "T": RunnerSpec(
        "20260829_citrus_t_batch.py",
        ("smoke", "priority", "all"),
        ("T", "T_SERIES"),
    ),
    "G0830": RunnerSpec(
        "20260830_citrus_g0830_batch.py",
        ("smoke", "structure", "loss", "all", "final"),
        ("G0830", "G_0830"),
        supports_cache=True,
        supports_skip_completed=True,
    ),
    "G0839": RunnerSpec(
        "20260830_citrus_g0839_batch.py",
        ("smoke", "screen", "all", "final"),
        ("G0839", "G_0839"),
        supports_cache=True,
        supports_amp=True,
        supports_skip_completed=True,
    ),
    "LIGHT": RunnerSpec(
        "20260830_citrus_light_batch.py",
        ("smoke", "screen", "pareto", "pr", "all", "final"),
        ("LIGHT", "LIGHT_SERIES"),
        supports_cache=True,
        supports_amp=True,
        supports_skip_completed=True,
    ),
    "ORCHID": RunnerSpec(
        "20260901_citrus_orchid_batch.py",
        ("smoke", "screen", "pareto", "all", "control", "final"),
        ("ORCHID", "ORCHID_SERIES"),
        supports_cache=True,
        supports_amp=True,
        supports_skip_completed=True,
    ),
    "SAGE_V2": RunnerSpec(
        "20260902_citrus_sage_batch.py",
        ("smoke", "screen", "all", "control", "final", "aggressive"),
        ("SAGE", "SAGE2", "SAGE_V2"),
        supports_cache=True,
        supports_amp=True,
        supports_skip_completed=True,
    ),
    "SAGE_V3": RunnerSpec(
        "20260902_citrus_sage_v3_batch.py",
        ("smoke", "screen", "all", "control", "backbone", "fusion", "final"),
        ("SAGE3", "SAGE_V3", "SAGE-V3"),
        supports_cache=True,
        supports_amp=True,
        supports_skip_completed=True,
    ),
    "SAGE_V4": RunnerSpec(
        "20260903_citrus_sage_v4_batch.py",
        ("smoke", "screen", "all", "control", "backbone", "final"),
        ("SAGE4", "SAGE_V4", "SAGE-V4"),
        supports_cache=True,
        supports_amp=True,
        supports_skip_completed=True,
    ),
}


def _normalise_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def resolve_runner(series: str) -> Tuple[str, RunnerSpec]:
    """Resolve a user-facing series name or alias."""
    requested = _normalise_name(series)
    for canonical, spec in RUNNERS.items():
        names = (canonical, *spec.aliases)
        if requested in {_normalise_name(name) for name in names}:
            return canonical, spec
    choices = ", ".join(RUNNERS)
    raise ValueError(f"Unknown SERIES={series!r}. Available canonical names: {choices}")


def _parse_devices(device: str, single_gpu_only: bool) -> Tuple[str, ...]:
    value = str(device).strip().lower()
    if value in {"cpu", "mps"}:
        return ()
    devices = tuple(item.strip().replace("cuda:", "") for item in value.split(",") if item.strip())
    if not devices or any(not item.isdigit() for item in devices):
        raise ValueError(f"DEVICE must be one CUDA index such as '0', or 'cpu'; received {device!r}.")
    if single_gpu_only and len(devices) != 1:
        raise ValueError("This foreground launcher intentionally permits only one GPU per run.")
    return devices


class DeviceRunLock:
    """OS advisory lock, released by the OS even when the process is interrupted.

    Lock files persist intentionally: unlinking a locked inode permits a second
    process to lock a different inode at the same path. This protects only launchers
    that use this helper; it is not a server-wide GPU reservation system.
    """

    def __init__(self, devices: Sequence[str], enabled: bool = True) -> None:
        self.devices = tuple(devices)
        self.enabled = enabled
        self.handles = []

    @staticmethod
    def _path(device: str) -> Path:
        user = str(os.getuid()) if hasattr(os, "getuid") else os.environ.get("USERNAME", "user")
        return Path(tempfile.gettempdir()) / f"citrus_foreground_v2_{user}_gpu_{device}.lock"

    def __enter__(self) -> "DeviceRunLock":
        if not self.enabled:
            return self
        try:
            for device in self.devices:
                self._acquire(device)
        except Exception:
            self._release_all()
            raise
        return self

    def _acquire(self, device: str) -> None:
        path = self._path(device)
        handle = path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise RuntimeError(f"GPU {device} already has a citrus foreground launcher lock: {path}") from error
        self.handles.append(handle)

    def _release_all(self) -> None:
        for handle in reversed(self.handles):
            handle.close()
        self.handles.clear()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._release_all()


def gpu_processes(devices: Iterable[str]) -> Optional[Dict[str, list]]:
    """Return compute processes per selected GPU, or ``None`` when nvidia-smi is unavailable."""
    selected = tuple(devices)
    if not selected:
        return {}
    try:
        gpu_rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        process_rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    uuid_to_index = {}
    for row in gpu_rows.splitlines():
        fields = [item.strip() for item in row.split(",")]
        if len(fields) >= 2:
            uuid_to_index[fields[1]] = fields[0]
    busy = {device: [] for device in selected}
    for row in process_rows.splitlines():
        fields = [item.strip() for item in row.split(",")]
        if len(fields) < 2:
            continue
        index = uuid_to_index.get(fields[0])
        if index in busy:
            busy[index].append({"pid": fields[1], "used_gpu_memory_mb": fields[2] if len(fields) > 2 else "?"})
    return busy


def _build_argv(
    spec: RunnerSpec,
    *,
    data: str,
    suite: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    workers: int,
    project: str,
    pretrained: str,
    seeds: str,
    only: str,
    cache: str,
    amp: Optional[bool],
    dry_run: bool,
    skip_completed: bool,
    fail_fast: bool,
) -> list:
    if suite not in spec.suites:
        raise ValueError(f"Suite {suite!r} is invalid for {spec.script}; choose one of {spec.suites}.")
    arguments = [
        "--data",
        data,
        "--suite",
        suite,
        "--epochs",
        str(epochs),
        "--batch",
        str(batch),
        "--imgsz",
        str(imgsz),
        "--device",
        str(device),
        "--workers",
        str(workers),
    ]
    if project:
        arguments.extend(("--project", project))
    if pretrained:
        arguments.extend(("--pretrained", pretrained))
    if spec.seed_flag == "--seed":
        seed_values = [item.strip() for item in seeds.split(",") if item.strip()]
        if len(seed_values) != 1:
            raise ValueError(f"{spec.script} accepts one seed only; received SEEDS={seeds!r}.")
        arguments.extend((spec.seed_flag, seed_values[0]))
    else:
        arguments.extend((spec.seed_flag, seeds))
    if only:
        arguments.extend(("--only", only))
    if spec.supports_cache:
        arguments.extend(("--cache", cache))
    if spec.supports_amp and amp is not None:
        arguments.append("--amp" if amp else "--no-amp")
    if dry_run:
        arguments.append("--dry-run")
    if skip_completed and spec.supports_skip_completed:
        arguments.append("--skip-completed")
    if fail_fast:
        arguments.append("--fail-fast")
    return arguments


def _run_module_in_current_process(script: Path, argv: Sequence[str]) -> None:
    """Import a dated runner and call its ``main`` without spawning Python."""
    module_name = f"_citrus_foreground_{script.stem}_{os.getpid()}"
    specification = importlib.util.spec_from_file_location(module_name, str(script))
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load batch runner: {script}")
    module = importlib.util.module_from_spec(specification)
    previous_argv = sys.argv[:]
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
        main = getattr(module, "main", None)
        if not callable(main):
            raise AttributeError(f"Batch runner has no callable main(): {script}")
        sys.argv = [str(script), *argv]
        main()
    finally:
        sys.argv = previous_argv
        sys.modules.pop(module_name, None)


def run_foreground(
    *,
    series: str,
    data: str,
    suite: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    workers: int,
    project: str = "",
    pretrained: str = "",
    seeds: str = "42",
    only: str = "",
    cache: str = "false",
    amp: Optional[bool] = None,
    dry_run: bool = False,
    skip_completed: bool = True,
    fail_fast: bool = True,
    refuse_busy_gpu: bool = True,
    device_lock: bool = True,
    single_gpu_only: bool = True,
) -> None:
    """Validate resources and execute exactly one sequential suite in the foreground."""
    canonical, spec = resolve_runner(series)
    script = ROOT / spec.script
    if not script.is_file():
        raise FileNotFoundError(f"Batch runner not found: {script}")
    data_path = Path(data).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if epochs < 1 or batch < 1 or imgsz < 1 or workers < 0:
        raise ValueError("EPOCHS/BATCH/IMGSZ must be positive and WORKERS must be non-negative.")
    if cache not in {"false", "disk", "ram"}:
        raise ValueError("CACHE must be 'false', 'disk', or 'ram'.")

    devices = _parse_devices(device, single_gpu_only=single_gpu_only)
    arguments = _build_argv(
        spec,
        data=str(data_path),
        suite=suite,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        project=str(Path(project).expanduser().resolve()) if project else "",
        pretrained=str(Path(pretrained).expanduser().resolve()) if pretrained else "",
        seeds=seeds,
        only=only,
        cache=cache,
        amp=amp,
        dry_run=dry_run,
        skip_completed=skip_completed,
        fail_fast=fail_fast,
    )

    print("=" * 88, flush=True)
    print("CITRUS FOREGROUND SEQUENTIAL RUN", flush=True)
    print(f"Series       : {canonical}", flush=True)
    print(f"Runner       : {script.name}", flush=True)
    print(f"Python       : {sys.executable}", flush=True)
    print(f"Dataset      : {data_path}", flush=True)
    print(f"Suite/Epochs : {suite} / {epochs}", flush=True)
    print(f"Device       : {device} (single-GPU guard={single_gpu_only})", flush=True)
    print(f"Batch/Workers: {batch} / {workers}", flush=True)
    print(f"Project      : {project or '[runner default]'}", flush=True)
    print("Execution    : current Python process; foreground; sequential; no nohup", flush=True)
    print("Stop         : Ctrl+C in this terminal; no next model starts. Hard Stop may skip cleanup.", flush=True)
    print("=" * 88, flush=True)

    # This fork treats device numbers as physical CUDA indices. Reject an inherited
    # different visibility mask rather than probing one card and training on another.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if devices and not dry_run and visible and visible != ",".join(devices):
        raise ValueError(f"CUDA_VISIBLE_DEVICES={visible!r} conflicts with DEVICE={device!r}; use a fresh terminal.")
    with DeviceRunLock(devices, enabled=device_lock and not dry_run):
        if refuse_busy_gpu and devices and not dry_run:
            busy = gpu_processes(devices)
            if busy is None:
                print(
                    "WARNING: nvidia-smi unavailable; device lock is active but GPU occupancy was not verified.",
                    flush=True,
                )
            else:
                occupied = {key: value for key, value in busy.items() if value}
                if occupied:
                    raise RuntimeError(
                        f"Refusing to compete for an occupied GPU: {occupied}. "
                        "Stop the existing job, select another DEVICE, or explicitly set REFUSE_BUSY_GPU=False."
                    )
        try:
            _run_module_in_current_process(script, arguments)
        except KeyboardInterrupt:
            print("\nTraining interrupted by the user. No next experiment will be started.", flush=True)
            raise
