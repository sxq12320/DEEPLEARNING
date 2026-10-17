"""Process ownership/cleanup tests; no GPU, model imports or training."""

import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from baseline_amp import process


def test_real_command_preserves_interpreter_flags_environment_cwd_and_tee(tmp_path, capsys):
    logfile = tmp_path / "logs/run.log"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="7", PYTHONUNBUFFERED="1")
    code = (
        "import os,sys;sys.stdout.reconfigure(encoding='utf-8');print('hello 柑橘');"
        "print(sys.flags.isolated);print(os.getcwd());print(os.getenv('CUDA_VISIBLE_DEVICES'))"
    )
    process.run_visible([sys.executable, "-I", "-u", "-c", code], tmp_path, environment, logfile)
    logged = logfile.read_text(encoding="utf-8")
    assert "hello 柑橘\n1\n" in logged
    assert str(tmp_path) in logged and logged.endswith("7\n")
    assert logged in capsys.readouterr().out
    assert not any(thread.name == "baseline-output" for thread in threading.enumerate())


def test_real_nonzero_child_aborts_and_restores_signal_handlers(tmp_path):
    numbers = [signal.SIGINT, signal.SIGTERM] + ([signal.SIGBREAK] if os.name == "nt" else [])
    before = {number: signal.getsignal(number) for number in numbers}
    with pytest.raises(RuntimeError, match="code 3"):
        process.run_visible([sys.executable, "-u", "-c", "print('failure');raise SystemExit(3)"],
                            tmp_path, dict(os.environ), tmp_path / "failed.log")
    assert {number: signal.getsignal(number) for number in numbers} == before
    assert (tmp_path / "failed.log").read_text(encoding="utf-8") == "failure\n"


@pytest.mark.parametrize("failure", ["terminal", "log", "reader", "keyboard", "sigterm"])
def test_mock_io_or_signal_failure_stops_owned_tree_before_raising(tmp_path, monkeypatch, failure):
    fake = type("FakeProcess", (), {})()
    fake.pid = 12345
    fake.stdin = None
    fake.stdout = io.StringIO("payload\n")
    fake.poll = lambda: None
    fake.wait = lambda **kwargs: 0
    cleaned = []
    monkeypatch.setattr(process, "_WINDOWS", False)
    monkeypatch.setattr(process.subprocess, "Popen", lambda *args, **kwargs: fake)
    monkeypatch.setattr(process, "_stop_tree", lambda child, job, **kwargs: cleaned.append((child, kwargs)))
    original_print = print

    def output(*args, **kwargs):
        if args and args[0] == "payload\n":
            if failure == "terminal":
                raise BrokenPipeError("terminal failed")
            if failure == "keyboard":
                raise KeyboardInterrupt
            if failure == "sigterm":
                process.signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return original_print(*args, **kwargs)

    monkeypatch.setattr("builtins.print", output)
    if failure == "log":
        class FailedLog(io.StringIO):
            def write(self, value):
                raise OSError("log failed")

        monkeypatch.setattr(Path, "open", lambda *args, **kwargs: FailedLog())
    if failure == "reader":
        class FailedReader(io.StringIO):
            def __next__(self):
                raise OSError("reader failed")

        fake.stdout = FailedReader()
    expected = KeyboardInterrupt if failure == "keyboard" else SystemExit if failure == "sigterm" else OSError
    with pytest.raises(expected) as caught:
        process.run_visible(["unchanged-command"], tmp_path, {}, tmp_path / "mock.log")
    assert len(cleaned) == 1 and cleaned[0][0] is fake
    assert cleaned[0][1]["interrupted"] is (failure == "keyboard")
    if failure == "sigterm":
        assert caught.value.code == 128 + signal.SIGTERM
    assert fake.stdout.closed
    assert not any(thread.name == "baseline-output" for thread in threading.enumerate())


def test_linux_cleanup_handles_exit_races_and_remaining_descendant_group(monkeypatch):
    fake = type("FakeProcess", (), dict(pid=12345, poll=lambda self: 0, wait=lambda self, **kwargs: 0))()
    sent = []
    monkeypatch.setattr(process, "_WINDOWS", False)
    monkeypatch.setattr(process.signal, "SIGKILL", 9, raising=False)

    def disappeared(pid, number):
        sent.append((pid, number))
        raise ProcessLookupError

    monkeypatch.setattr(process.os, "killpg", disappeared, raising=False)
    process._stop_tree(fake, None)
    assert sent == [(fake.pid, signal.SIGKILL)]


def test_signal_during_launch_is_deferred_until_child_is_trackable():
    with process._SignalGuard() as guard:
        assigned = False
        with pytest.raises(SystemExit) as caught:
            with guard.defer():
                guard._handler(signal.SIGTERM, None)
                assigned = True
        assert assigned and caught.value.code == 128 + signal.SIGTERM


def test_linux_cleanup_escalates_with_bounded_waits(monkeypatch):
    class RunningProcess:
        pid = 23456

        def __init__(self):
            self.waits = []

        def poll(self):
            return None

        def wait(self, timeout):
            self.waits.append(timeout)
            if len(self.waits) < 3:
                raise subprocess.TimeoutExpired("test-owned", timeout)
            return -9

    child = RunningProcess()
    sent = []
    monkeypatch.setattr(process, "_WINDOWS", False)
    monkeypatch.setattr(process.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(process, "_kill_group", lambda owned, number: sent.append((owned.pid, number)))
    process._stop_tree(child, None, interrupted=True)
    assert sent == [(child.pid, signal.SIGINT), (child.pid, signal.SIGTERM), (child.pid, signal.SIGKILL)]
    assert child.waits == [process._GRACE_SECONDS, process._TERM_SECONDS, process._KILL_SECONDS]


def test_foreground_api_rejects_non_main_thread_before_starting_child(tmp_path, monkeypatch):
    launched, errors = [], []
    monkeypatch.setattr(process.subprocess, "Popen", lambda *args, **kwargs: launched.append(args))

    def run():
        try:
            process.run_visible(["never-launched"], tmp_path, {}, tmp_path / "thread.log")
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive() and not launched
    assert len(errors) == 1 and "main thread" in errors[0]


def _running(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = api.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:
                return False
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            code = wintypes.DWORD()
            if not api.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return code.value == 259
        finally:
            api.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    state = Path(f"/proc/{pid}/stat")
    return not state.is_file() or state.read_text().split(")", 1)[1].strip().split()[0] != "Z"


@pytest.mark.parametrize("root_exits", [False, True])
def test_real_owned_child_and_grandchild_are_stopped_even_after_root_exit(tmp_path, root_exits):
    """Run a separate controller so its real signals cannot terminate the pytest process."""
    child_file, grandchild_file = tmp_path / "child.pid", tmp_path / "grandchild.pid"
    grandchild = "import os,sys,time;open(sys.argv[1],'w').write(str(os.getpid()));time.sleep(60)"
    child = (
        "import os,sys,time,subprocess;open(sys.argv[1],'w').write(str(os.getpid()));"
        "p=subprocess.Popen([sys.executable,'-u','-c',sys.argv[3],sys.argv[2]]);"
        "print('tree started',flush=True);"
        + ("time.sleep(0.3);raise SystemExit(7)" if root_exits else "time.sleep(60)")
    )
    command = [sys.executable, "-I", "-u", "-c", child, str(child_file), str(grandchild_file), grandchild]
    controller = (
        "import json,os,sys;from pathlib import Path;from baseline_amp.process import run_visible;"
        "run_visible(json.loads(sys.argv[1]),Path(sys.argv[2]),dict(os.environ),Path(sys.argv[2])/'tree.log')"
    )
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent.parent), PYTHONUTF8="1")
    options = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", env=environment)
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    owner = subprocess.Popen([sys.executable, "-u", "-c", controller, json.dumps(command), str(tmp_path)], **options)
    owned = []
    try:
        deadline = time.monotonic() + 10
        while not (child_file.is_file() and grandchild_file.is_file()):
            if owner.poll() is not None:
                output = owner.communicate(timeout=2)[0]
                pytest.fail(f"Controller exited before creating tree: {output}")
            assert time.monotonic() < deadline, "Process tree did not become ready"
            time.sleep(0.02)
        owned = [int(path.read_text()) for path in (child_file, grandchild_file)]
        assert all(pid > 0 and pid != os.getpid() for pid in owned)
        if not root_exits:
            owner.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
        output = owner.communicate(timeout=15)[0]
        assert owner.returncode != 0, output
        deadline = time.monotonic() + 3
        while any(_running(pid) for pid in owned) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not any(_running(pid) for pid in owned), output
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.communicate(timeout=5)
        # Failure cleanup is restricted to exact test-owned recorded PID(s).
        if os.name != "nt":
            for pid in owned:
                if _running(pid):
                    os.kill(pid, signal.SIGKILL)
