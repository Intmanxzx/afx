from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator


_PRINT_LOCK = threading.Lock()


def _seconds_from_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _rss_mib() -> float | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def progress_message(status: str, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _PRINT_LOCK:
        print(f"[AutofuseDebugger {timestamp}] {status} {message}", file=sys.stderr, flush=True)


@contextmanager
def progress_stage(
    name: str,
    *,
    detail: str | None = None,
    heartbeat_seconds: float | None = None,
    stall_seconds: float | None = None,
) -> Iterator[None]:
    heartbeat = heartbeat_seconds or _seconds_from_env("AUTOFUSEDEBUGGER_HEARTBEAT_SECONDS", 30.0)
    stall = stall_seconds or _seconds_from_env("AUTOFUSEDEBUGGER_STALL_SECONDS", 120.0)
    start = time.monotonic()
    start_cpu = time.process_time()
    last_cpu = start_cpu
    last_cpu_progress = start
    stop = threading.Event()
    suffix = f" {detail}" if detail else ""
    progress_message("START", f"{name}{suffix}")

    def emit_heartbeat() -> None:
        nonlocal last_cpu, last_cpu_progress
        while not stop.wait(heartbeat):
            now = time.monotonic()
            cpu_now = time.process_time()
            if cpu_now - last_cpu >= 0.05:
                last_cpu_progress = now
            last_cpu = cpu_now
            elapsed = now - start
            idle = now - last_cpu_progress
            rss = _rss_mib()
            metrics = f"elapsed={elapsed:.1f}s cpu={cpu_now - start_cpu:.1f}s"
            if rss is not None:
                metrics += f" rss={rss:.1f}MiB"
            if idle >= stall:
                progress_message(
                    "WARNING",
                    f"{name} {metrics} no measurable CPU progress for {idle:.1f}s; "
                    "the stage may be waiting or stalled",
                )
            else:
                progress_message("RUNNING", f"{name} {metrics}")

    monitor = threading.Thread(target=emit_heartbeat, name="autofusedebugger-progress", daemon=True)
    monitor.start()
    try:
        yield
    except BaseException as exc:
        elapsed = time.monotonic() - start
        progress_message("FAILED", f"{name} elapsed={elapsed:.1f}s error={type(exc).__name__}: {exc}")
        raise
    else:
        elapsed = time.monotonic() - start
        progress_message("DONE", f"{name} elapsed={elapsed:.1f}s")
    finally:
        stop.set()
        monitor.join(timeout=min(heartbeat, 1.0))
