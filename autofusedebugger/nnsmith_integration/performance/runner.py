"""Run exactly one backend per interpreter and clean up its process group."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .backends import BackendSpec
from .schema import BackendResult, ProfileConfig


def run_backend(*, spec: BackendSpec, request: dict, output_dir: Path,
                cache_dir: Path, profile: ProfileConfig) -> BackendResult:
    run_dir = output_dir / ("run-" + uuid.uuid4().hex[:12])
    run_dir.mkdir(parents=True)
    request = dict(request, backend=spec.name, profile=asdict(profile),
                   profile_dir=str(run_dir / "profiling"))
    request_path = run_dir / "request.json"
    result_path = run_dir / "result.json"
    log_path = run_dir / "worker.log"
    request_path.write_text(json.dumps(request, indent=2))
    env = spec.environment(os.environ, cache_dir.resolve())
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "autofusedebugger.nnsmith_integration.performance.worker",
             "--request", str(request_path), "--result", str(result_path)],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            code = process.wait(timeout=profile.timeout_s)
        except subprocess.TimeoutExpired:
            _stop_group(process)
            return BackendResult(spec.name, duration_s=time.monotonic() - started, status="timeout", log_path=str(log_path),
                                 error_stage="worker", error=f"Exceeded {profile.timeout_s} seconds")
    if not result_path.exists():
        return BackendResult(spec.name, duration_s=time.monotonic() - started, log_path=str(log_path), error_stage="worker",
                             error=f"Worker exited with {code}; see worker.log")
    try:
        result = BackendResult(**json.loads(result_path.read_text()))
    except (ValueError, TypeError) as e:
        return BackendResult(spec.name, duration_s=time.monotonic() - started, log_path=str(log_path), error_stage="worker_result", error=str(e))
    result.duration_s = time.monotonic() - started
    result.log_path = str(log_path)
    if code != 0 and result.status == "ok":
        result.status = "error"
        result.error = f"Worker exited with {code} after producing a result"
    return result


def _stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
