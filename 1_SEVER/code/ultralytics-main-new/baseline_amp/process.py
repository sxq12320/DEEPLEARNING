"""Bounded foreground process-tree execution; never inspect or stop unrelated jobs."""

from __future__ import annotations

import contextlib
import os
import queue
import signal
import subprocess
import sys
import threading

_WINDOWS = os.name == "nt"
_GRACE_SECONDS = 10
_TERM_SECONDS = 5
_KILL_SECONDS = 5
_READER_JOIN_SECONDS = 2

# Windows needs the gate: assigning an already-running training process to a
# Job Object can miss descendants it spawned before assignment. Only this
# stdlib launcher runs before assignment; the original command/flags/env are
# passed unchanged and are released AFTER the whole tree is owned by our Job.
_WINDOWS_GATE = (
    "import subprocess,sys; "
    "allowed=sys.stdin.buffer.read(1)==b'G'; "
    "sys.exit(subprocess.Popen(sys.argv[1:],stdin=subprocess.DEVNULL).wait() if allowed else 125)"
)


class _SignalGuard:
    """Translate terminal/termination signals, deferring only the short launch race."""

    def __init__(self):
        self.previous = {}
        self.deferred = 0
        self.pending = None
        self.cleaning = False

    def _handler(self, number, frame):
        if self.cleaning:
            return  # Do not interrupt bounded cleanup and strand a child.
        if self.deferred:
            self.pending = number
            return
        if number in (signal.SIGINT, getattr(signal, "SIGBREAK", signal.SIGINT)):
            raise KeyboardInterrupt
        raise SystemExit(128 + number)

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Foreground process execution must run in the main thread for signal-safe cleanup")
        try:
            for number in (signal.SIGINT, signal.SIGTERM, *([signal.SIGBREAK] if _WINDOWS else [])):
                self.previous[number] = signal.signal(number, self._handler)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for number, handler in self.previous.items():
            signal.signal(number, handler)

    @contextlib.contextmanager
    def defer(self):
        self.deferred += 1
        try:
            yield
        finally:
            self.deferred -= 1
            if not self.deferred and self.pending is not None:
                number, self.pending = self.pending, None
                self._handler(number, None)


class _WindowsJob:
    """An unnamed, non-inheritable Job; close kills exactly its assigned process tree."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                        ("max_working_set", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                        ("scheduling", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", IoCounters), ("process_memory", ctypes.c_size_t),
                        ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t),
                        ("peak_job_memory", ctypes.c_size_t)]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.api.SetInformationJobObject.restype = wintypes.BOOL
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.AssignProcessToJobObject.restype = wintypes.BOOL
        self.api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.api.TerminateJobObject.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.handle = self.api.CreateJobObjectW(None, None)
        self.assigned = False
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        import ctypes

        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.assigned = True

    def terminate(self):
        import ctypes

        if self.handle and self.assigned and not self.api.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        import ctypes

        if self.handle:
            handle, self.handle = self.handle, None
            if not self.api.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())


def _kill_group(process, number):
    try:
        os.killpg(process.pid, number)
    except ProcessLookupError:
        pass  # The exact group already exited between poll and killpg.


def _stop_tree(process, job, graceful=True, interrupted=False):
    """Stop only our known new group/Job, including descendants after root exit."""
    if _WINDOWS:
        if graceful and process.poll() is None and job is not None and job.assigned:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
            except OSError:
                if process.poll() is None:
                    job.terminate()
            try:
                process.wait(timeout=_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        if job is not None:
            job.terminate()
        # Assignment failure happens before the gate is released; no target
        # or descendant can exist. Kill only that unassigned gate process.
        if process.poll() is None and (job is None or not job.assigned):
            process.kill()
    else:
        if graceful and process.poll() is None:
            stages = [(signal.SIGINT if interrupted else signal.SIGTERM, _GRACE_SECONDS),
                      (signal.SIGTERM, _TERM_SECONDS)]
            for number, timeout in stages:
                _kill_group(process, number)
                try:
                    process.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
        _kill_group(process, signal.SIGKILL)  # Also catches descendants of an already-exited root.
    process.wait(timeout=_KILL_SECONDS)


def _read_output(stream, events, stopped):
    def emit(kind, value=None):
        while not stopped.is_set():
            try:
                events.put((kind, value), timeout=0.1)
                return
            except queue.Full:
                continue

    try:
        for line in stream:
            if stopped.is_set():
                return
            emit("line", line)
        emit("eof")
    except BaseException as error:
        emit("error", error)


def run_visible(command, cwd, env, logfile):
    """Tee a foreground job; every exit/failure cleans our process tree before returning/raising.

    SIGTERM exits nonzero (128 + signal). Ctrl+C raises KeyboardInterrupt. IO
    failures/nonzero children propagate and must abort the caller's queue.
    Windows has one stdlib gate process but runs the original interpreter,
    arguments, working directory and environment unchanged. No GPU locks.
    """
    command = [str(argument) for argument in command]
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print("COMMAND:", subprocess.list2cmdline(command), flush=True)
    process = job = reader = None
    tree_stopped = False
    stopped = threading.Event()
    events = queue.Queue(maxsize=256)
    with _SignalGuard() as signals, logfile.open("a", encoding="utf-8") as log:
        failure = None
        try:
            with signals.defer():
                options = dict(cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)
                if _WINDOWS:
                    job = _WindowsJob()
                    options.update(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, stdin=subprocess.PIPE)
                    process = subprocess.Popen([sys.executable, "-I", "-u", "-c", _WINDOWS_GATE, *command], **options)
                    job.assign(process)
                    process.stdin.write("G")
                    process.stdin.flush()
                    process.stdin.close()
                else:
                    process = subprocess.Popen(command, start_new_session=True, **options)
            reader = threading.Thread(target=_read_output, args=(process.stdout, events, stopped),
                                      name="baseline-output", daemon=True)
            reader.start()
            while True:
                # A root can exit while its descendant still holds stdout open.
                # Poll separately instead of blocking forever in readline().
                if not tree_stopped and process.poll() is not None:
                    _stop_tree(process, job, graceful=False)
                    tree_stopped = True
                try:
                    kind, value = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if kind == "line":
                    print(value, end="", flush=True)
                    log.write(value)
                    log.flush()
                elif kind == "error":
                    raise value
                else:
                    break
            code = process.wait()
            if code:
                raise RuntimeError(f"Child exited with code {code}. Log: {logfile}. Queue stopped; no automatic changes.")
        except BaseException as error:
            failure = error
            raise
        finally:
            signals.cleaning = True
            try:
                try:
                    if process is not None and not tree_stopped:
                        _stop_tree(process, job, interrupted=isinstance(failure, KeyboardInterrupt))
                finally:
                    if job is not None:
                        job.close()  # Kill-on-close also covers cleanup failures/remaining descendants.
            finally:
                stopped.set()
                if process is not None and process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                if reader is not None:
                    reader.join(timeout=_READER_JOIN_SECONDS)
                    if reader.is_alive():
                        raise RuntimeError("Owned process output did not close after bounded process-tree cleanup") from failure
                if process is not None and process.stdout is not None:
                    process.stdout.close()
