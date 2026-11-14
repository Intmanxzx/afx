"""Graceful stop requests; never swallow the interrupt as an ordinary case error."""
from contextlib import contextmanager
import signal
import threading


class SessionStopped(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(signal.Signals(signum).name)


@contextmanager
def stop_signals():
    # Signal handlers can only be installed by Python's main thread.
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}

    def stop(signum, frame):
        # Repeated Ctrl+C/TERM must not interrupt worker termination/checkpointing.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        raise SessionStopped(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, stop)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def guard_worker_parent(expected_pid):
    """On Linux, kill this worker group if its session leader disappears (kill -9)."""
    import os
    import sys
    if not expected_pid or sys.platform != "linux":
        return
    import ctypes

    def terminate_group(signum, frame):
        if os.getpgrp() == os.getpid():
            os.killpg(os.getpgrp(), signal.SIGKILL)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate_group)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "Cannot register parent-death signal")
    if os.getppid() != int(expected_pid):
        terminate_group(signal.SIGTERM, None)


@contextmanager
def defer_stop_signals():
    """Commit one completed case without a signal interrupting counter updates."""
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
