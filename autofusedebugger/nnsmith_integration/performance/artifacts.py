"""Compact completed captures after persisting their statistics and diagnostics."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from .storage import atomic_write_text


class ArtifactCleanupError(RuntimeError):
    """Stop a long-running session rather than accumulating undeletable captures."""


def finalize_artifacts(result, run_dir: Path, cache_dir: Path, *, keep=False):
    result_path = run_dir / "result.json"
    result.runtime["artifact_policy"] = "keep" if keep else "compact"
    # Persist the result before touching any original data, even on timeouts/errors.
    atomic_write_text(result_path, json.dumps(asdict(result), indent=2, ensure_ascii=False))
    if keep:
        return result
    try:
        if not result.csv_path and (run_dir / "profiling").exists():
            from .statistics import find_op_statistic, ProfileDataError
            try:
                result.csv_path = str(find_op_statistic(run_dir / "profiling"))
            except ProfileDataError:
                pass
        if result.csv_path:
            source = Path(result.csv_path)
            # Only compact files owned by this run; never follow an external path.
            destination = run_dir / "op_statistic.csv"
            if source.resolve() != destination.resolve():
                source.resolve().relative_to((run_dir / "profiling").resolve())
                shutil.copy2(source, destination.with_suffix(".csv.tmp"))
                destination.with_suffix(".csv.tmp").replace(destination)
            result.csv_path = str(destination)
        # Save the new CSV location before removing the capture tree.
        atomic_write_text(result_path, json.dumps(asdict(result), indent=2, ensure_ascii=False))
        for directory in (run_dir / "profiling", cache_dir):
            if directory.is_symlink():
                raise ValueError(f"Refusing to remove artifact symlink: {directory}")
            if directory.exists():
                shutil.rmtree(directory)
    except (OSError, ValueError) as exc:
        result.cleanup_error = str(exc)
        atomic_write_text(result_path, json.dumps(asdict(result), indent=2, ensure_ascii=False))
        raise ArtifactCleanupError(f"Cannot compact {run_dir}: {exc}") from exc
    return result


def compact_pending_case(out: Path, case_id: str):
    """Recover remnants of a killed session after acquiring its directory lock."""
    from .schema import BackendResult
    for backend in ("ascendc", "triton-ascend"):
        for run_dir in (out / "cases" / case_id / backend).glob("run-*"):
            if run_dir.is_symlink():
                raise ArtifactCleanupError(f"Refusing artifact symlink: {run_dir}")
            try:
                result = BackendResult(**json.loads((run_dir / "result.json").read_text()))
            except (OSError, ValueError, TypeError):
                result = BackendResult(backend, status="interrupted", error_stage="worker",
                    error="Previous session stopped before worker result was saved",
                    log_path=str(run_dir / "worker.log"))
            finalize_artifacts(result, run_dir, out / "compile_cache" / backend / run_dir.name)
