"""One backend per process group, with durable results and bounded artifacts."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .artifacts import finalize_artifacts
from .backends import BackendSpec
from .schema import BackendResult, ProfileConfig


def run_backend(*, spec: BackendSpec, request: dict, output_dir: Path,
                cache_dir: Path, profile: ProfileConfig) -> BackendResult:
    run_dir = output_dir / ("run-" + uuid.uuid4().hex[:12])
    run_dir.mkdir(parents=True)
    # Each worker owns its cache; cleanup cannot touch another measurement.
    owned_cache = cache_dir.resolve() / run_dir.name
    request = dict(request, backend=spec.name, profile=asdict(profile),
                   profile_dir=str(run_dir / "profiling"))
    request_path, result_path = run_dir / "request.json", run_dir / "result.json"
    log_path = run_dir / "worker.log"
    request_path.write_text(json.dumps(request, indent=2))
    env = spec.environment(os.environ, owned_cache)
    env["AF_PERF_PARENT_PID"] = str(os.getpid())
    started = time.monotonic()
    process = None
    result = None
    try:
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
                result = BackendResult(spec.name, status="timeout", error_stage="worker",
                                       error=f"Exceeded {profile.timeout_s} seconds")
        if result is None:
            if not result_path.exists():
                result = BackendResult(spec.name, error_stage="worker",
                                       error=f"Worker exited with {code}; see worker.log")
            else:
                try:
                    result = BackendResult(**json.loads(result_path.read_text()))
                except (ValueError, TypeError) as exc:
                    result = BackendResult(spec.name, error_stage="worker_result", error=str(exc))
                if code != 0 and result.status == "ok":
                    result.status = "error"
                    result.error = f"Worker exited with {code} after producing a result"
    except BaseException as exc:
        if process is not None:
            _stop_group(process)
        result = BackendResult(spec.name, status="error" if isinstance(exc, Exception) else "interrupted", log_path=str(log_path),
                               duration_s=time.monotonic() - started, error_stage="worker",
                               error=f"Worker did not complete: {type(exc).__name__}: {exc}")
        finalize_artifacts(result, run_dir, owned_cache, keep=profile.keep_artifacts)
        raise
    result.duration_s = time.monotonic() - started
    result.log_path = str(log_path)
    return finalize_artifacts(result, run_dir, owned_cache, keep=profile.keep_artifacts)


def _stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The leader may have exited while a profiler/compiler child still lives.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()
