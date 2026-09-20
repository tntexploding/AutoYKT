"""Process-scoped ownership and power requests for a classroom run."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from pathlib import Path
import sys
from typing import Iterator

from autoykt.core.config import ConfigError


@contextmanager
def run_lease(directory: Path) -> Iterator[None]:
    """Reject a second controller; the OS releases ownership after a crash."""
    if sys.platform == "win32":
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.CreateMutexW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_wchar_p,
        ]
        api.CreateMutexW.restype = ctypes.c_void_p
        api.CloseHandle.argtypes = [ctypes.c_void_p]
        api.CloseHandle.restype = ctypes.c_int
        ctypes.set_last_error(0)
        handle = api.CreateMutexW(None, False, "Local\\AutoYKT.Automation")
        if not handle:
            raise OSError("could not create the automation process lock")
        already_running = ctypes.get_last_error() == 183
        try:
            if already_running:
                raise ConfigError(
                    "another AutoYKT run is active; stop it first"
                )
            yield
        finally:
            api.CloseHandle(handle)
        return

    # flock is unavailable on Windows; importing it is deliberately deferred.
    import fcntl  # pylint: disable=import-outside-toplevel,import-error

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "runner.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ConfigError(
                "another AutoYKT run is active; stop it first"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def keep_awake() -> Iterator[None]:
    """Keep Windows and its display awake only while this process runs."""
    if sys.platform != "win32":
        yield
        return
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    request = api.SetThreadExecutionState
    request.argtypes = [ctypes.c_uint]
    request.restype = ctypes.c_uint
    continuous = 0x80000000
    previous = request(continuous | 0x00000001 | 0x00000002)
    if not previous:
        raise OSError("could not keep the classroom display awake")
    try:
        yield
    finally:
        request(continuous | previous)
