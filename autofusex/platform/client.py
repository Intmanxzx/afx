"""Platform client: upload local fuzz results, heartbeat, run retest tasks (design 1, 2).

It reads the existing client result database (attempts/shapes/contexts written
by `afx fuzz run --save` or `afx fuzz replay --save`) and never writes to it.
Its own progress lives in a separate SQLite file next to it, so the old sync
path and this uploader never share state.

usage:
  afx platform client --endpoint https://host --token-file F --results DB setup
  afx platform client --endpoint https://host --token-file F --results DB run [--retest --device npu:0]

`setup` fixes --token-file permissions, creates the --results directory, confirms the
token/endpoint by sending one heartbeat, and prints the `fuzz run`/`client run` commands
to copy next. It never writes or invents a token: paste the one `afx platform token issue
USER` printed into --token-file first.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform as host_platform
import re
import socket
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

log = logging.getLogger("afx.platform.client")

BATCH = 100
UPLOAD_BUDGET = 1500 * 1024
RETEST_TRIES = 8
# Failures caused by the test machine, not by the backend under test. Such attempts are not uploaded
# (they would count as backend failures); they are recorded locally with the reason instead.
ENVIRONMENT_ERRORS = re.compile(
    r"module 'triton' has no attribute 'language'"
    r"|device busy: another managed test owns this physical NPU", re.I)
BUSY = re.compile(r"device busy|owns this physical NPU", re.I)  # stay below the proxy's 2 MiB request limit
MAX_LOG = 200 * 1024
AF_BACKENDS = ("af-mlir", "af-legacy")
_STAGE = re.compile(r"[^a-z0-9_.-]+")


class APIError(Exception):
    def __init__(self, status, code, message):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code = status, code


class API:
    def __init__(self, endpoint: str, token: str, timeout: float = 60):
        if not endpoint.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ValueError("endpoint must be https (plain http only for loopback)")
        self.endpoint, self.token, self.timeout = endpoint.rstrip("/"), token, timeout

    def call(self, method: str, path: str, body=None, *, attempts: int = 5):
        """One API call; proxy/server restarts (502/503/504, connection errors) are retried with backoff.
        Retries resend the identical body, so request_key idempotency keeps them safe."""
        for attempt in range(attempts):
            try:
                return self._call(method, path, body)
            except APIError as exc:
                if exc.status not in (502, 503, 504) or attempt == attempts - 1:
                    raise
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == attempts - 1:
                    raise
            time.sleep(min(60, 2 ** attempt * 2))

    def _call(self, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.endpoint + path, data=data, method=method, headers={
            "Authorization": "Bearer " + self.token, "Content-Type": "application/json",
            "User-Agent": "afx-platform-client"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                error = json.loads(exc.read()).get("error", {})
            except ValueError:
                error = {}
            raise APIError(exc.code, error.get("code", "http_error"), error.get("message", str(exc))) from None
        return payload["data"]


# -- mapping local results to platform items -----------------------------------------
def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


def _first_line(text: str | None) -> str | None:
    for line in (text or "").splitlines():
        line = line.strip()
        if line and not line.startswith(("Traceback", "File ")):
            return line[:500]
    return None


# Lines that only wrap the real exception (Inductor/Dynamo re-raise) and say nothing about the cause.
_WRAPPER = re.compile(r"raised:\s*$|^BackendCompilerFailed\b|^InductorError: (?:RuntimeError: )?Failed to execute graph code"
                      r"|^torch\._dynamo\.exc\.|^Set TORCH_LOGS|^You can suppress", re.I)
_EXCEPTION = re.compile(r"^\s*(?:[A-Za-z_][\w.]*\.)?[A-Z]\w*(?:Error|Exception)\b.*")


def summary_line(text: str | None) -> str | None:
    """Most informative single line: the last real exception if the first line is only a wrapper."""
    first = _first_line(text)
    if first is None or not _WRAPPER.search(first):
        return first
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if _EXCEPTION.match(line) and not _WRAPPER.search(line):
            return line[:500]
    return first

def plog_summary(plog: list[str]) -> str | None:
    """The single public failure summary: first CANN ERROR line."""
    for line in plog:
        line = str(line).strip()
        if line:
            return line[:500]
    return None


def os_label() -> str | None:
    try:
        info = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
        return (info.get("ID", "") + info.get("VERSION_ID", "").strip('"').split(".")[0]).strip() or None
    except OSError:
        return None


def gcc_version() -> str | None:
    try:
        out = subprocess.run(["gcc", "-dumpfullversion"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def nnsmith_version() -> str | None:
    import importlib.metadata
    try:
        return importlib.metadata.version("nnsmith-autofusex")
    except importlib.metadata.PackageNotFoundError:
        return None


def case_payload(case_spec_json: str, seed: str) -> dict:
    """Case description from the stored CaseSpec; never re-generates the graph.

    The case key is the short seed (E/F + base36), the same identity `afx fuzz replay` and
    the old pages use. `spec` is the old reporting metadata: graph nodes with operator
    display names, shape variants and weights, used to draw the case DAG.
    """
    from autofusex.fuzz.client.case_spec import decode
    from autofusex.fuzz.client.contracts import Seed
    from autofusex.fuzz.client.generation.categories import categories
    from autofusex.fuzz.server.reporting.report import spec_metadata
    spec = decode(case_spec_json)
    metadata = spec_metadata(spec)
    graph = metadata["graph"]
    ops = sorted({node["op"] for node in graph.get("nodes", [])})
    samples = spec.shape_samples or (spec.inputs,)
    tensors = [t for sample in samples for t in sample] + list(spec.weights)
    dtypes = sorted({t.dtype for t in (*spec.inputs, *spec.weights)})
    max_dim = max((d for t in tensors for d in t.shape), default=1)
    max_numel = max((_numel(t.shape) for t in tensors), default=1)
    display = sorted({str(name)[:64] for name in metadata["op_types"] if name})
    return {"case_key": Seed.parse(seed).short, "case_source": "nnsmith", "is_dynamic": spec.seed.dynamic,
            "shape_count": max(1, len(spec.shape_samples) or 1), "node_count": len(graph.get("nodes", [])),
            "op_types": [op[:64] for op in ops][:256], "dtypes": dtypes, "max_dim": max_dim, "max_numel": max_numel,
            "features": {"op_category": [c.lower() for c in categories(spec)], "op_display": display[:256]},
            "spec": metadata}


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _diagnostics(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _evidence(summary, diagnostics) -> str:
    diagnostics = _diagnostics(diagnostics)
    texts = [summary or ""]
    if isinstance(diagnostics, dict):
        texts.append(diagnostics.get("error") or "")
        for shape in (diagnostics.get("shapes") or {}).values():
            if isinstance(shape, dict):
                texts.append(shape.get("error") or "")
    return "\n".join(t for t in texts if t)


_AF_OWNER = ("autofuse", "ascir", "runmlircodegen", "af_mlir_strict")
_ERROR_WORD = ("error:", "failed", "failure", "shortfall=yes")


def af_error_line(evidence: str) -> str | None:
    """The first line where Autofuse itself reports a failure (not generated code that merely mentions it)."""
    from autofusex.fuzz.server.reporting.failure_origin import mlir_failed_line
    line = mlir_failed_line(evidence)
    if line:
        return line.strip()[:500]
    check = re.search(r"^.*Check (?:dtype|shape) failed.*$", evidence, re.I | re.M)
    if check:
        return check.group(0).strip()[:500]
    for raw in evidence.splitlines():
        lowered = raw.lower()
        if any(o in lowered for o in _AF_OWNER) and any(w in lowered for w in _ERROR_WORD) \
                and "= fuser." not in lowered and not lowered.lstrip().startswith(("from ", "import ")):
            return raw.strip()[:500]
    return None


# Source files of the Autofuse optimize module (autofuse/optimize). A GE error log line located
# in one of them means the failure happened in schedule (Optimize), before codegen. GE prints
# "[ERROR] GE(pid,proc):time [file.cpp:line]..." when plog goes to stdout
# (ASCEND_SLOG_PRINT_TO_STDOUT=1); the Python side raises "RuntimeError: Optimize fail".
OPTIMIZE_SOURCES = frozenset((
    'alignment_handler.cpp', 'alignment_handler.h', 'alignment_strategy.cpp', 'alignment_strategy.h', 'autoschedule.cpp', 'autoschedule_defs.h',
    'autoschedule.h', 'backend_spec.cpp', 'base_alignment_strategy.cpp', 'base_alignment_strategy.h', 'base_graph_pass.h', 'base_platform.h',
    'base_template_generator.cpp', 'base_template_generator.h', 'base_template.h', 'brc_inline_template.cpp', 'brc_inline_template.h', 'broadcast_backward_pass.cpp',
    'broadcast_backward_pass.h', 'broadcast_backward_shared_split.cpp', 'broadcast_backward_shared_split.h', 'broadcast_const_to_store.cpp', 'broadcast_const_to_store.h', 'buf_que_allocator.cpp',
    'buf_que_allocator.h', 'cast_optimization_pass.cpp', 'cast_optimization_pass.h', 'cast_reorder_pass.cpp', 'cast_reorder_pass.h', 'concat_group_partitioner.cpp',
    'concat_group_partitioner.h', 'concat_inputs_unification_pass.cpp', 'concat_inputs_unification_pass.h', 'concat_schedule_case_generator.cpp', 'concat_schedule_case_generator.h', 'concat_score_function_generator.cpp',
    'concat_score_function_generator.h', 'dtype_consistency.cpp', 'dtype_consistency.h', 'duplicate_elewise_cse_pass.cpp', 'duplicate_elewise_cse_pass.h', 'expand_dims_for_all_reduce.cpp',
    'expand_dims_for_all_reduce.h', 'fused_graph_modifier.cpp', 'fused_graph_modifier.h', 'fused_graph_unfolder.cpp', 'fused_graph_unfolder.h', 'graph_properties_cache.h',
    'improve_precision.cpp', 'improve_precision.h', 'indirect_load_schedule_case_generator.cpp', 'indirect_load_schedule_case_generator.h', 'masked_fill_input_reorder_pass.cpp', 'masked_fill_input_reorder_pass.h',
    'mem_reuse_manager.cpp', 'mem_reuse_manager.h', 'node_cache_marker.cpp', 'node_cache_marker.h', 'node_partition_utils.cpp', 'node_partition_utils.h',
    'optimize.cpp', 'optimize.h', 'pass_runner.h', 'pass_runner_handler.cpp', 'pass_runner_handler.h', 'pass_runner_v1.h',
    'pass_utils.cpp', 'pass_utils.h', 'platform_factory.cpp', 'platform_factory.h', 'platformv1.cpp', 'platformv1.h',
    'pow_equiv_substitution_pass.cpp', 'pow_equiv_substitution_pass.h', 'pre_process_config.h', 'pre_process.cpp', 'pre_process.h', 'recompute_case_generator.cpp',
    'recompute_case_generator.h', 'reduce_schedule_case_generator.cpp', 'reduce_schedule_case_generator.h', 'redundant_broadcast_eliminate_pass.cpp', 'redundant_broadcast_eliminate_pass.h', 'scalar_broadcast_insert.cpp',
    'scalar_broadcast_insert.h', 'scalar_broadcast_optimization.cpp', 'scalar_broadcast_optimization.h', 'scalar_to_1d_tensor.cpp', 'scalar_to_1d_tensor.h', 'schedule_case_generator.h',
    'schedule.cpp', 'schedule_group_partitioner.cpp', 'schedule_group_partitioner.h', 'schedule.h', 'schedule_task_generator.cpp', 'schedule_task_generator.h',
    'schedule_utils.cpp', 'schedule_utils.h', 'simt_boundary_sync.cpp', 'simt_boundary_sync.h', 'softmax_pattern_fusion_pass.cpp', 'softmax_pattern_fusion_pass.h',
    'softmax_pattern_fusion_utils.cpp', 'softmax_pattern_fusion_utils.h', 'split_group_partitioner.cpp', 'split_group_partitioner.h', 'split_schedule_case_generator.cpp', 'split_schedule_case_generator.h',
    'split_score_function_generator.cpp', 'split_score_function_generator.h', 'static_ub_template_filter.cpp', 'static_ub_template_filter.h', 'template_generator_handler.cpp', 'template_generator_handler.h',
    'template_generator_v1.h', 'tensor_mem_defs.h', 'tiling_group.cpp', 'tiling_group.h', 'transpose_schedule_case_generator.cpp', 'transpose_schedule_case_generator.h',
    'unaligned_template.cpp', 'unaligned_template.h',
))
_GE_ERROR = re.compile(r"\[ERROR\]\s+(?:GE|AUTOFUSE|ASCGEN)\b[^\n]*?\[([\w./+-]+\.(?:cpp|cc|h)):\d+\]")


def schedule_error_line(evidence: str) -> str | None:
    for match in _GE_ERROR.finditer(evidence):
        path = match.group(1)
        # GE prints the source path relative to the repo, e.g. [./autofuse/optimize/autoschedule/autoschedule.cpp:812]
        if "/optimize/" in path or ("/" not in path and path in OPTIMIZE_SOURCES):
            start = evidence.rfind("\n", 0, match.start()) + 1
            end = evidence.find("\n", match.end())
            return evidence[start:end if end >= 0 else None].strip()[:500]
    found = re.search(r"^.*\bOptimize fail\b.*$", evidence, re.M)
    return found.group(0).strip()[:500] if found else None


# --- CANN plog: GE errors are written to $ASCEND_PROCESS_LOG_PATH|~/ascend/log/debug/plog/plog-<pid>_*.log
# when ASCEND_GLOBAL_LOG_LEVEL=3 (ERROR only). Stage workers are reused across cases, so lines are
# attributed to an attempt by the worker pid that ran its NPU stage AND the attempt's wall-clock window.
_PLOG_TIME = re.compile(r"^\[ERROR\]\s+\S+\(\d+,[^)]*\):(\d{4})-(\d{2})-(\d{2})-(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
PLOG_MAX_LINES = 200
PLOG_MAX_BYTES = 32 * 1024


def plog_root() -> Path:
    return Path(os.environ.get("ASCEND_PROCESS_LOG_PATH") or Path.home() / "ascend" / "log") / "debug" / "plog"


def plog_errors(pids, start: float, end: float, root: Path | None = None) -> list:
    """ERROR lines written by the given worker pids between start and end (epoch seconds, local clock)."""
    root = root or plog_root()
    out, size = [], 0
    for pid in sorted({int(p) for p in pids if p}):
        for path in sorted(root.glob(f"plog-{pid}_*.log")):
            try:
                if path.stat().st_mtime < start - 5:
                    continue
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                m = _PLOG_TIME.match(line)
                if not m:
                    continue
                y, mo, d, h, mi, sec, ms = (int(g) for g in m.groups())
                ts = time.mktime((y, mo, d, h, mi, sec, 0, 0, -1)) + ms / 1000
                if start - 1 <= ts <= end + 1:
                    out.append(line[:1000])
                    size += len(line)
                    if len(out) >= PLOG_MAX_LINES or size >= PLOG_MAX_BYTES:
                        return out
    return out


def attempt_window(detail: dict | None):
    """(worker pids, start, end) of an attempt's NPU stages, from its stage timeline and lifecycle."""
    if not detail:
        return set(), None, None
    pids = {f.get("worker_pid") for f in detail.get("stage_timeline") or []
            if f.get("worker_pid") and (f.get("lease") or {}).get("device_uuid")}
    times = {}
    for event in detail.get("lifecycle") or []:
        try:
            times[event["event"]] = datetime.fromisoformat(str(event["wall_time"]).replace("Z", "+00:00")).timestamp()
        except (KeyError, ValueError, TypeError):
            continue
    start = times.get("dispatched") or times.get("started") or times.get("admitted")
    end = times.get("finished") or times.get("persisted")
    return pids, start, end


def function_status(backend, status, stage, summary, diagnostics, index, extra: str = ""):
    """Map the client's status + failure attribution to (function_status, fail_stage, summary line).
    `extra` carries GE ERROR lines from plog for the same attempt."""
    from autofusex.fuzz.server.reporting.failure_origin import classify, failure_path
    if status in ("ok", "fail"):
        return "pass", None, None
    if status == "unsupported":
        return "unsupported", None, None
    evidence = _evidence(summary, diagnostics) + ("\n" + extra if extra else "")
    if re.search(r"\btimed? ?out\b|\btimeout\b", summary or "", re.I):
        return "timeout", "timeout", None
    if backend in AF_BACKENDS:
        line = af_error_line(evidence) or ""
        schedule_line = schedule_error_line(evidence)
        if schedule_line and not re.search(r"check (?:dtype|shape) failed", line, re.I):
            return "compile_fail", "schedule", schedule_line  # optimize module failed: AF owns it
    kind = classify(backend, status, stage or "", summary or "", diagnostics, index)
    if kind == "environment_failed":
        return "runtime_fail", "environment", None
    if kind == "gen_failed":
        return "build_fail", _STAGE.sub("_", (stage or "build").lower())[:32] or "build", None
    if kind == "function_failed":
        line = af_error_line(evidence)
        lowered = (line or "").lower()
        # ASCIR graph construction (graph_check) runs before schedule; optimize errors mean schedule.
        if "check dtype failed" in lowered or "check shape failed" in lowered:
            return "compile_fail", "graph_check", line
        for word, label in (("tiling", "tiling"), ("lower", "lowering"), ("codegen", "codegen")):
            if word in lowered:
                return "compile_fail", label, line
        if failure_path(backend, status, stage or "", summary or "", diagnostics, index) == "mlir_failed":
            return "compile_fail", "mlir", line
        return "compile_fail", "autofuse", line
    return "runtime_fail", _STAGE.sub("_", (stage or "unknown").lower())[:32] or "unknown", None


DETAIL_BUDGET = 700 * 1024


def _cap_strings(value, limit):
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "\n…[truncated]"
    if isinstance(value, dict):
        return {k: _cap_strings(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_cap_strings(v, limit) for v in value]
    return value


class Mapper:
    def __init__(self, results_db: str):
        self.path = results_db
        self.conn = sqlite3.connect(f"file:{results_db}?mode=ro", uri=True, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self._contexts: dict = {}
        self._static_env = {"os": os_label(), "gcc": gcc_version(), "nnsmith": nnsmith_version()}

    def context(self, context_id):
        cached = self._contexts.get(context_id)
        if cached is None:
            row = self.conn.execute("SELECT payload FROM contexts WHERE id=?", (context_id,)).fetchone()
            cached = json.loads(row[0]) if row else {}
            if len(self._contexts) >= 2000:
                self._contexts.clear()
            self._contexts[context_id] = cached
        return cached

    def detail(self, attempt_id: str):
        """The same per-attempt view the old detail page rendered (ResultQueries.detail), trimmed:
        case metadata goes with the case, and long texts are capped so one item stays < 1 MiB."""
        from autofusex.fuzz.client.failure_details import diagnostic_view
        from autofusex.fuzz.client.storage.queries import ResultQueries
        try:
            result = ResultQueries(self.path).detail(attempt_id)
        except Exception as exc:  # detail is best effort; the execution itself still uploads
            log.warning("detail for %s unavailable: %s", attempt_id, exc)
            return None
        result.pop("metadata", None)
        execution = result.get("contexts", {}).get("execution", {})
        execution.pop("case_spec", None)
        diagnostics = _diagnostics(execution.get("diagnostics"))
        result["shape_views"] = {
            str(s["shape_index"]): diagnostic_view(diagnostics, s["shape_index"], s.get("error_summary") or "")
            for s in result.get("shape_results", [])[:8]}
        encoded = json.dumps(result, default=str)
        if len(encoded) > DETAIL_BUDGET:
            execution["diagnostics"] = None
            result = _cap_strings(result, 64 * 1024)
            encoded = json.dumps(result, default=str)
            if len(encoded) > DETAIL_BUDGET:
                result = _cap_strings(result, 8 * 1024)
        return json.loads(json.dumps(result, default=str))

    def attempts_after(self, rowid: int, limit: int = BATCH):
        return self.conn.execute("SELECT rowid, * FROM attempts WHERE rowid>? ORDER BY rowid LIMIT ?",
                                 (rowid, limit)).fetchall()

    def max_rowid(self) -> int:
        return self.conn.execute("SELECT COALESCE(MAX(rowid),0) FROM attempts").fetchone()[0]

    def item(self, attempt) -> tuple[dict, dict]:
        """Return (upload item, run info) for one attempt."""
        execution = self.context(attempt["execution_id"])
        environment = self.context(attempt["environment_id"])
        evaluation = self.context(attempt["evaluation_id"])
        backend = attempt["backend"]
        shapes = self.conn.execute("SELECT * FROM shapes WHERE attempt_id=? ORDER BY shape_index",
                                   (attempt["id"],)).fetchall()
        diagnostics = _diagnostics(execution.get("diagnostics"))
        detail = self.detail(attempt["id"])
        plog = []
        if backend in AF_BACKENDS and attempt["status"] not in ("ok", "fail"):
            pids, start, end = attempt_window(detail)
            if pids and start and end:
                plog = plog_errors(pids, start, end)
        plog_text = "\n".join(plog)
        status, stage, af_line = function_status(backend, attempt["status"], attempt["error_stage"],
                                                 attempt["error_summary"], diagnostics, -1, plog_text)
        shape_items, worst_shape = [], None
        for shape in shapes:
            s_status = status if shape["correctness"] != "error" else function_status(
                backend, "error", shape["error_stage"], shape["error_summary"], diagnostics, shape["shape_index"])[0]
            if shape["correctness"] in ("pass", "fail", "not_checked"):
                s_status = "pass"
            accuracy = {"pass": "pass", "fail": "fail"}.get(shape["correctness"])
            shape_items.append({"function_status": s_status, "accuracy_status": accuracy,
                                "kernel_ns": shape["mean_wall_ns"] if evaluation.get("performance") else None,
                                "e2e_ns": shape["first_call_combined_ns"]})
            if shape["error_summary"] and worst_shape is None:
                worst_shape = shape
        if status != "pass" and worst_shape is not None:
            status, stage, af_line = function_status(backend, "error", worst_shape["error_stage"],
                                                     worst_shape["error_summary"], diagnostics,
                                                     worst_shape["shape_index"], plog_text)
        accuracy = None
        if status == "pass" and backend != "cpu":
            corr = [s["correctness"] for s in shapes]
            accuracy = "fail" if "fail" in corr or attempt["status"] == "fail" else (
                "pass" if "pass" in corr else None)
        kernels = [s["mean_wall_ns"] for s in shapes if s["mean_wall_ns"]]
        summary_text = (worst_shape["error_summary"] if worst_shape is not None else None) or attempt["error_summary"]
        log_text = "\n\n".join(t for t in (attempt["error_summary"], *(s["error_summary"] for s in shapes),
                                           diagnostics.get("error"),
                                           ("# CANN plog (ERROR)\n" + plog_text) if plog_text else None)
                               if t)
        versions = {"af-mlir": environment.get("cann"), "af-legacy": environment.get("cann"),
                    "triton-npu": environment.get("triton"), "eager-npu": environment.get("cann"),
                    "cpu": environment.get("torch")}
        e2e = (attempt["prepare_ns"] or 0) + sum(s["first_call_combined_ns"] or 0 for s in shapes)
        item = {
            "exec_uuid": attempt["id"],
            "case": case_payload(execution["case_spec"], attempt["seed"]),
            "backend": backend,
            "npu_arch": "" if backend == "cpu" else (environment.get("device_model") or "unknown"),
            "env": {"cann": environment.get("cann"), "python": environment.get("python"),
                    "torch": environment.get("torch"), "torch_npu": environment.get("torch_npu"),
                    "backend_version": versions.get(backend), "afx": environment.get("package_version"),
                    "nnsmith": self._static_env["nnsmith"], "os": self._static_env["os"],
                    "cpu_arch": environment.get("machine"), "gcc": self._static_env["gcc"]},
            "function_status": status, "fail_stage": stage,
            "accuracy_status": accuracy, "accuracy_ref": "cpu" if accuracy else None,
            "compile_ns": attempt["prepare_ns"],
            "kernel_ns": int(sum(kernels) / len(kernels)) if kernels and evaluation.get("performance") else None,
            "e2e_ns": max(1, e2e),
            "log_summary": plog_summary(plog) or af_line or summary_line(log_text or summary_text),
            "log": log_text[:MAX_LOG] if log_text and status != "pass" else None,
            "shapes": shape_items[:8],
            "created_at": _ms(attempt["created_at"]),
        }
        if item["log"] and len(item["log"].encode()) > MAX_LOG:
            item["log"] = item["log"].encode()[:MAX_LOG].decode(errors="ignore")
        if detail is not None and plog:
            detail["plog_errors"] = plog
        item["detail"] = detail
        run = {"run_uuid": (execution.get("batch_id") or "local")[:64], "kind": "fuzz"}
        return item, {"machine_id": environment.get("installation_id") or socket.gethostname(),
                      "npu_arch": environment.get("device_model"), "afx_version": environment.get("package_version"),
                      "run": run, "env": environment}


# -- uploader -------------------------------------------------------------------------
class State:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS progress(key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS rejected(attempt_id TEXT PRIMARY KEY, reason TEXT, at INTEGER)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS sent_windows(window_id TEXT PRIMARY KEY, at INTEGER)")

    def get(self, key, default=0) -> int:
        row = self.conn.execute("SELECT value FROM progress WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set(self, key, value) -> None:
        self.conn.execute("INSERT INTO progress VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                          (key, value))


class Counters:
    def __init__(self):
        self.lock = threading.Lock()
        self.values = {"executions": 0, "uploads": 0, "upload_errors": 0, "retests": 0}

    def add(self, name, n=1):
        with self.lock:
            self.values[name] += n

    def take(self) -> dict:
        with self.lock:
            out, self.values = self.values, {k: 0 for k in self.values}
        return out

    def restore(self, values):
        with self.lock:
            for k, v in values.items():
                self.values[k] += v


def upload_new(api: API, mapper: Mapper, state: State, counters: Counters, *, key_prefix: str,
               run_override: dict | None = None, attempt_ids=None) -> dict:
    """Upload attempts after the saved cursor (or exactly attempt_ids). Returns {attempt_id: status}."""
    statuses = {}
    while True:
        if attempt_ids is not None:
            marks = ",".join("?" * len(attempt_ids))
            rows = mapper.conn.execute(f"SELECT rowid, * FROM attempts WHERE id IN ({marks})",
                                       list(attempt_ids)).fetchall()
        else:
            rows = mapper.attempts_after(state.get(key_prefix + ":cursor"))
        if not rows:
            return statuses
        groups: dict = {}
        for row in rows:
            env_error = environment_error(mapper, row)
            if env_error:
                state.conn.execute("INSERT OR REPLACE INTO rejected VALUES(?,?,?)",
                                   (row["id"], f"environment: {env_error}"[:500], int(time.time())))
                statuses[row["id"]] = "environment"
                continue
            try:
                item, info = mapper.item(row)
            except Exception as exc:  # malformed local record: remember and skip, never block the queue
                log.warning("skip attempt %s: %s", row["id"], exc)
                state.conn.execute("INSERT OR REPLACE INTO rejected VALUES(?,?,?)",
                                   (row["id"], f"map: {exc}"[:500], int(time.time())))
                statuses[row["id"]] = "skipped"
                continue
            run = run_override or info["run"]
            groups.setdefault((run["run_uuid"], run["kind"], info["machine_id"]), (run, info, []))[2].append(item)
        batches = []
        for key, (run, info, items) in groups.items():
            chunk, size = [], 0
            for entry in items:
                entry_size = len(json.dumps(entry, default=str))
                if chunk and size + entry_size > UPLOAD_BUDGET:
                    batches.append((key, run, info, chunk))
                    chunk, size = [], 0
                chunk.append(entry)
                size += entry_size
            if chunk:
                batches.append((key, run, info, chunk))
        for (run_uuid, kind, machine), run, info, items in batches:
            body = {"request_key": key_prefix + ":" + hashlib.sha256(
                        "|".join(sorted(i["exec_uuid"] for i in items)).encode()).hexdigest()[:40],
                    "client": {"machine_id": machine[:128], "hostname": socket.gethostname()[:128],
                               "npu_arch": info["npu_arch"], "afx_version": info["afx_version"]},
                    "run": run, "items": items}
            try:
                result = api.call("POST", "/api/client/executions", body)
            except APIError as exc:
                counters.add("upload_errors")
                if exc.status in (400, 409, 413):
                    for item in items:
                        state.conn.execute("INSERT OR REPLACE INTO rejected VALUES(?,?,?)",
                                           (item["exec_uuid"], str(exc)[:500], int(time.time())))
                        statuses[item["exec_uuid"]] = "rejected"
                    continue
                raise
            for entry in result["results"]:
                exec_uuid = items[entry["index"]]["exec_uuid"]
                statuses[exec_uuid] = entry["status"]
                if entry["status"] == "rejected":
                    state.conn.execute("INSERT OR REPLACE INTO rejected VALUES(?,?,?)",
                                       (exec_uuid, entry.get("message", "")[:500], int(time.time())))
            counters.add("uploads", result.get("accepted", 0))
        if attempt_ids is not None:
            return statuses
        state.set(key_prefix + ":cursor", rows[-1]["rowid"])
        counters.add("executions", len(rows))


def environment_error(mapper, attempt) -> str | None:
    texts = [attempt["error_summary"] or ""]
    texts += [r[0] or "" for r in mapper.conn.execute("SELECT error_summary FROM shapes WHERE attempt_id=?",
                                                       (attempt["id"],))]
    for text in texts:
        found = ENVIRONMENT_ERRORS.search(text)
        if found:
            return found.group(0)
    return None


def installed_packages(env: dict | None = None) -> dict:
    """What this client can verify: graph-autofusion version.info Version + pyautofuse.so sha256."""
    from autofusex.fuzz.client.environment import collect
    env = env or collect()
    if not env.get("cann") or not env.get("autofuse_binary"):
        return {}
    package = {"version": re.sub(r"[^A-Za-z0-9_.+-]", "_", env["cann"])[:64], "sha256": env["autofuse_binary"]}
    return {"af-mlir": package, "af-legacy": package}


class Client:
    def __init__(self, api: API, results_db: str, state_path: Path, *, devices=(), retest_db=None,
                 afx_command=None, interval=30.0):
        self.api, self.results_db, self.interval = api, results_db, interval
        self.state = State(state_path)
        self.counters = Counters()
        self.devices = list(devices)
        self.retest_db = retest_db or str(Path(state_path).with_name("retest-results.sqlite"))
        self.afx = afx_command or [sys.executable, "-m", "afx"]
        self.stop = threading.Event()
        self.env = None
        self.busy = False
        self._retried = False

    def environment(self) -> dict:
        if self.env is None:
            from autofusex.fuzz.client.environment import collect
            self.env = collect()
        return self.env

    def upload_once(self) -> dict:
        if not Path(self.results_db).exists():
            return {}
        mapper = Mapper(self.results_db)
        if not self._retried:
            self._retried = True
            self.retry_skipped(mapper)
        return upload_new(self.api, mapper, self.state, self.counters, key_prefix="fuzz")

    def retry_skipped(self, mapper) -> None:
        """Attempts skipped by a local mapping error are retried once per process start."""
        ids = [r[0] for r in self.state.conn.execute("SELECT attempt_id FROM rejected WHERE reason LIKE 'map:%'")]
        for i in range(0, len(ids), BATCH):
            chunk = ids[i:i + BATCH]
            statuses = upload_new(self.api, mapper, self.state, self.counters, key_prefix="fuzz", attempt_ids=chunk)
            done = [a for a, s in statuses.items() if s in ("accepted", "duplicate")]
            self.state.conn.executemany("DELETE FROM rejected WHERE attempt_id=?", [(a,) for a in done])

    def identity(self) -> dict:
        """Machine id, chip and afx version as recorded in the newest local result, so uploads and
        heartbeats describe the same client; falls back to the live environment."""
        env = dict(self.environment())
        if Path(self.results_db).exists():
            mapper = Mapper(self.results_db)
            row = mapper.conn.execute("SELECT environment_id FROM attempts WHERE backend<>'cpu' "
                                      "ORDER BY rowid DESC LIMIT 1").fetchone()
            if row is not None:
                recorded = mapper.context(row[0])
                for key in ("installation_id", "device_model", "package_version"):
                    env[key] = recorded.get(key) or env.get(key)
        return env

    def heartbeat(self) -> dict:
        env = self.identity()
        counters = self.counters.take()
        pending = 0
        if Path(self.results_db).exists():
            mapper = Mapper(self.results_db)
            pending = mapper.conn.execute("SELECT COUNT(*) FROM attempts WHERE rowid>?",
                                          (self.state.get("fuzz:cursor"),)).fetchone()[0]
        windows = self.unsent_windows()
        home = str(self.home().resolve())
        # installation_id is the stable identity shared with the managed
        # daemon. client_home is reported separately and must not become part
        # of machine_id, otherwise one installation creates duplicate clients.
        machine_id = (env.get("installation_id") or socket.gethostname())[:128]
        from autofusex.fuzz.client.scheduling.devices import device_inventory
        body = {"devices": device_inventory(), "machine_id": machine_id[:128], "client_home": home[:512],
                "hostname": socket.gethostname()[:128], "npu_arch": env.get("device_model"),
                "afx_version": env.get("package_version"),
                "backend_versions": {"af-mlir": env.get("cann"), "triton-npu": env.get("triton"),
                                     "cpu": env.get("torch")},
                "counters": counters, "busy": self.busy, "pending_uploads": pending,
                "runtime": self.runtime_snapshots(), "windows": windows}
        try:
            reply = self.api.call("POST", "/api/client/heartbeat", body)
        except Exception:
            self.counters.restore(counters)
            raise
        now = int(time.time())
        self.state.conn.executemany("INSERT OR IGNORE INTO sent_windows VALUES(?,?)",
                                    [(w, now) for w in reply.get("windows_accepted", [])])
        return reply

    def home(self) -> Path:
        return Path(os.environ.get("AUTOFUSEX_HOME") or Path(self.results_db).resolve().parent.parent)

    def runtime_snapshots(self) -> list:
        """Live run samples the fuzz scheduler publishes in HOME/runtime (the old fleet view's source)."""
        try:
            from autofusex.fuzz.client.runtime import boot_id
            boot = boot_id()
        except Exception:
            return []
        now, runs = time.monotonic(), []
        paths = sorted((self.home() / "runtime").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths[:64]:
            try:
                saved = json.loads(path.read_text())
                age = now - saved["monotonic"]
                if saved["boot"] != boot or age < 0 or age > 86400:
                    continue
                runs.append(dict(saved["sample"], age=age))
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if len(runs) >= 32:
                break
        return runs

    def unsent_windows(self, limit: int = 10) -> list:
        """Finished stage-window aggregates (tool efficiency) not yet acknowledged by the server."""
        if not Path(self.results_db).exists():
            return []
        mapper = Mapper(self.results_db)
        if not mapper.conn.execute("SELECT 1 FROM sqlite_master WHERE name='stage_window_aggregates_v1'").fetchone():
            return []
        sent = {r[0] for r in self.state.conn.execute("SELECT window_id FROM sent_windows")}
        out = []
        for window_id, payload in mapper.conn.execute(
                "SELECT window_id, payload FROM stage_window_aggregates_v1 ORDER BY rowid DESC LIMIT 200"):
            if window_id in sent:
                continue
            try:
                rows = json.loads(payload)
            except ValueError:
                continue
            out.append({"window_id": window_id[:64], "rows": rows if isinstance(rows, list) else [rows]})
            if len(out) >= limit or len(json.dumps(out)) > 400 * 1024:
                break
        return out

    def retest_once(self) -> int:
        """Claim and run retest tasks this client can serve. Returns the number of tasks finished."""
        installed = installed_packages(self.environment())
        env = self.identity()
        if not installed or not self.devices:
            return 0
        machine = (env.get("installation_id") or socket.gethostname())[:128]
        claim = self.api.call("POST", "/api/client/retests/claim", {
            "request_key": f"claim-{machine[:40]}-{time.time_ns()}", "machine_id": machine,
            "npu_arch": env.get("device_model") or "unknown", "installed": installed, "limit": len(self.devices)})
        tasks = claim["tasks"]
        if not tasks:
            return 0
        self.busy = True
        try:
            for task, device in zip(tasks, self.devices * len(tasks)):
                self._run_task(task, device, installed)
        finally:
            self.busy = False
        return len(tasks)

    def _run_task(self, task: dict, device: str, installed: dict) -> None:
        """Replay one retest task. Cards are shared with the running fuzz job: when the card is busy
        the replay waits its turn (each replay already waits up to 60 s for the card lock) and is
        retried with backoff on the next card, so busy-card attempts are never reported as results."""
        if task["case_source"] != "nnsmith":
            log.warning("task %s: %s cases are not executable by this client yet", task["task_id"], task["case_source"])
            return  # lease expires and the server marks it invalid after retries
        from autofusex.fuzz.client.contracts import Seed
        seed = str(Seed.parse(task["case_key"]))
        devices = [device] + [d for d in self.devices if d != device]
        ids = []
        for attempt in range(RETEST_TRIES):
            card = devices[attempt % len(devices)]
            before = Mapper(self.retest_db).max_rowid() if Path(self.retest_db).exists() else 0
            for backend in task["backends"]:
                if backend == "cpu":
                    continue  # replay always computes the shared CPU reference
                command = [*self.afx, "fuzz", "replay", task["case_key"], "--backend", backend,
                           "--database", self.retest_db, "--save", "--device", card]
                done = subprocess.run(command, capture_output=True, text=True, timeout=3600)
                if done.returncode not in (0, 1):
                    log.warning("replay %s exited %s: %s", task["case_key"], done.returncode, done.stderr[-2000:])
            mapper = Mapper(self.retest_db)
            rows = mapper.conn.execute("SELECT id, error_summary FROM attempts WHERE rowid>? AND seed=?",
                                       (before, seed)).fetchall()
            busy = any(BUSY.search(r[1] or "") for r in rows) or any(BUSY.search(r[0] or "") for r in mapper.conn.execute(
                "SELECT s.error_summary FROM shapes s JOIN attempts a ON a.id=s.attempt_id WHERE a.rowid>? AND a.seed=?",
                (before, seed)))
            if rows and not busy:
                ids = [r[0] for r in rows]
                break
            log.info("task %s: card %s busy, retrying", task["task_id"], card)
            if self.stop.wait(min(300, 20 * (attempt + 1))):
                return
        if not ids:
            log.warning("task %s produced no usable results; it will be re-queued when its lease expires", task["task_id"])
            return
        mapper = Mapper(self.retest_db)
        run = {"run_uuid": task["run_uuid"][:64], "kind": task["run_kind"]}
        statuses = upload_new(self.api, mapper, self.state, self.counters, key_prefix=f"task{task['task_id']}",
                              run_override=run, attempt_ids=ids)
        uploaded = [i for i, s in statuses.items() if s in ("accepted", "duplicate")]
        if not uploaded:
            log.warning("task %s: nothing accepted: %s", task["task_id"], statuses)
            return
        self.api.call("POST", f"/api/client/retests/{task['task_id']}/complete", {
            "request_key": f"complete-{task['task_id']}-{task['lease_id'][:8]}", "lease_id": task["lease_id"],
            "exec_uuids": uploaded[:8], "installed": installed})
        self.counters.add("retests")

    def run(self, *, retest: bool) -> None:
        # Ask the fuzz scheduler in this home to publish run snapshots for the fleet view.
        try:
            marker = self.home() / "platform-client.json"
            marker.write_text(json.dumps({"endpoint": self.api.endpoint, "pid": os.getpid()}))
        except OSError as exc:
            log.warning("cannot write %s: %s", marker, exc)
        last_beat = 0.0
        while not self.stop.is_set():
            try:
                self.upload_once()
                if time.monotonic() - last_beat > 60:
                    reply = self.heartbeat()
                    last_beat = time.monotonic()
                    if reply.get("update_to") and reply["update_to"] != self.environment().get("package_version"):
                        self._self_update(reply["update_to"])
                if retest:
                    self.retest_once()
            except Exception as exc:  # keep running through network and server errors
                log.warning("cycle failed: %s", exc)
            self.stop.wait(self.interval)

    def _self_update(self, version: str) -> None:
        """Install the requested exact version into this interpreter, then restart this process only."""
        if not re.fullmatch(r"[A-Za-z0-9_.+-]{1,64}", version):
            return
        log.warning("updating autofusex to %s as requested by the platform", version)
        done = subprocess.run([sys.executable, "-m", "pip", "install", f"autofusex=={version}"],
                              capture_output=True, text=True, timeout=1800)
        if done.returncode != 0:
            log.warning("update failed: %s", done.stderr[-2000:])
            return
        os.execv(sys.executable, [sys.executable, *sys.argv])


def _implied_data_dir(results_path: Path):
    """'afx fuzz run --save' always writes to <data-dir>/results/results.sqlite; it has
    no flag to redirect that to an arbitrary file. Return the --data-dir that makes
    results_path the one it actually writes to, or None if results_path doesn't follow
    that layout (in which case 'fuzz run --save' can never populate it)."""
    if results_path.name == "results.sqlite" and results_path.parent.name == "results":
        return results_path.parent.parent
    return None


def setup(args) -> int:
    """Bootstrap a new client host: fix --token-file permissions, create the --results
    directory, and confirm the token/endpoint actually work before anything runs
    unattended. Never generates or stores a token itself; it only fixes permissions
    on a file the operator already pasted an admin-issued token into."""
    token_path = Path(args.token_file)
    if not token_path.exists():
        print(json.dumps({"ok": False, "error":
            f"{token_path} does not exist; paste the token from "
            "`afx platform token issue USER` into it (chmod 600), then rerun setup"}),
            file=sys.stderr)
        return 2
    try:
        if stat.S_IMODE(token_path.stat().st_mode) != 0o600:
            token_path.chmod(0o600)
    except OSError as exc:
        print(json.dumps({"ok": False, "error": f"could not chmod 600 {token_path}: {exc}"}), file=sys.stderr)
        return 2
    token = token_path.read_text().strip()
    if not token:
        print(json.dumps({"ok": False, "error": f"{token_path} is empty"}), file=sys.stderr)
        return 2
    results_path = Path(args.results)
    try:
        results_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(json.dumps({"ok": False, "error": f"could not create {results_path.parent}: {exc}"}), file=sys.stderr)
        return 2
    try:
        api = API(args.endpoint, token)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    state = Path(args.state or results_path.with_name("platform-sync.sqlite"))
    client = Client(api, str(results_path), state, devices=args.device or [], interval=args.interval)
    try:
        beat = client.heartbeat()
    except APIError as exc:
        print(json.dumps({"ok": False, "error": f"{args.endpoint} rejected the token: {exc}"}), file=sys.stderr)
        return 2
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        print(json.dumps({"ok": False, "error": f"could not reach {args.endpoint}: {exc}"}), file=sys.stderr)
        return 2
    client_cmd = ["afx", "platform", "client", "run", "--endpoint", args.endpoint,
                  "--token-file", str(token_path), "--results", str(results_path)]
    for device in args.device or []:
        client_cmd += ["--device", device]
    if args.retest:
        client_cmd.append("--retest")
    data_dir = _implied_data_dir(results_path)
    warnings = []
    if data_dir is not None:
        run_cmd = ["afx", "fuzz", "run", "--save", "--data-dir", str(data_dir)]
    else:
        run_cmd = ["afx", "fuzz", "run", "--save"]
        warnings.append(
            f"{results_path} is not a .../results/results.sqlite path. 'afx fuzz run --save' "
            "always writes to <data-dir>/results/results.sqlite and has no flag to put results "
            "anywhere else, so the uploader above would keep watching an empty file. Point --results "
            f"at {results_path.parent / 'results' / 'results.sqlite'} (and run `fuzz run --save --data-dir "
            f"{results_path.parent}`), or pick whatever --data-dir you like and derive --results from it the "
            "same way.")
    result = {"ok": True, "client_id": beat["client_id"], "results": str(results_path),
              "token_file": str(token_path), "next_steps": [" ".join(run_cmd), " ".join(client_cmd)]}
    if warnings:
        result["warnings"] = warnings
    print(json.dumps(result, indent=2))
    return 0


def main(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.action == "setup":
        return setup(args)
    token = Path(args.token_file).read_text().strip()
    api = API(args.endpoint, token)
    state = Path(args.state or (Path(args.results).with_name("platform-sync.sqlite")))
    client = Client(api, args.results, state, devices=args.device or [], interval=args.interval)
    if args.action == "once":
        statuses = client.upload_once()
        beat = client.heartbeat()
        print(json.dumps({"uploaded": len(statuses), "client_id": beat["client_id"],
                          "update_to": beat["update_to"]}))
        return 0
    if args.action == "retest-once":
        print(json.dumps({"tasks": client.retest_once()}))
        return 0
    if args.action == "status":
        mapper = Mapper(args.results)
        print(json.dumps({"cursor": client.state.get("fuzz:cursor"), "local_attempts": mapper.max_rowid(),
                          "installed": installed_packages(client.environment())}))
        return 0
    client.run(retest=args.retest)
    return 0
