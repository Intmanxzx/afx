"""Platform client: upload local fuzz results, heartbeat, run retest tasks (design 1, 2).

It reads the existing client result database (attempts/shapes/contexts written
by `afx fuzz run --save` or `afx fuzz replay --save`) and never writes to it.
Its own progress lives in a separate SQLite file next to it, so the old sync
path and this uploader never share state.

usage:
  afx platform client --endpoint https://host --token-file F --results DB run [--retest --device npu:0]
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
UPLOAD_BUDGET = 1500 * 1024  # stay below the proxy's 2 MiB request limit
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

    def call(self, method: str, path: str, body=None):
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


def function_status(backend, status, stage, summary, diagnostics, index):
    """Map the client's status + failure attribution to (function_status, fail_stage, summary line)."""
    from autofusex.fuzz.server.reporting.failure_origin import classify, failure_path
    if status in ("ok", "fail"):
        return "pass", None, None
    if status == "unsupported":
        return "unsupported", None, None
    evidence = _evidence(summary, diagnostics)
    if re.search(r"\btimed? ?out\b|\btimeout\b", summary or "", re.I):
        return "timeout", "timeout", None
    kind = classify(backend, status, stage or "", summary or "", diagnostics, index)
    if kind == "gen_failed":
        return "build_fail", _STAGE.sub("_", (stage or "build").lower())[:32] or "build", None
    if kind == "function_failed":
        line = af_error_line(evidence)
        lowered = (line or "").lower()
        for word, label in (("check dtype failed", "graph_check"), ("check shape failed", "graph_check"),
                            ("schedul", "schedule"), ("tiling", "tiling"), ("lower", "lowering"),
                            ("codegen", "codegen")):
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
        status, stage, af_line = function_status(backend, attempt["status"], attempt["error_stage"],
                                                 attempt["error_summary"], diagnostics, -1)
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
                                                     worst_shape["shape_index"])
        accuracy = None
        if status == "pass" and backend != "cpu":
            corr = [s["correctness"] for s in shapes]
            accuracy = "fail" if "fail" in corr or attempt["status"] == "fail" else (
                "pass" if "pass" in corr else None)
        kernels = [s["mean_wall_ns"] for s in shapes if s["mean_wall_ns"]]
        summary_text = (worst_shape["error_summary"] if worst_shape is not None else None) or attempt["error_summary"]
        log_text = "\n\n".join(t for t in (attempt["error_summary"], *(s["error_summary"] for s in shapes),
                                           diagnostics.get("error"))
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
            "log_summary": af_line or summary_line(log_text or summary_text),
            "log": log_text[:MAX_LOG] if log_text and status != "pass" else None,
            "shapes": shape_items[:8],
            "created_at": _ms(attempt["created_at"]),
        }
        if item["log"] and len(item["log"].encode()) > MAX_LOG:
            item["log"] = item["log"].encode()[:MAX_LOG].decode(errors="ignore")
        item["detail"] = self.detail(attempt["id"])
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
        body = {"machine_id": (env.get("installation_id") or socket.gethostname())[:128],
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
        if task["case_source"] != "nnsmith":
            log.warning("task %s: %s cases are not executable by this client yet", task["task_id"], task["case_source"])
            return  # lease expires and the server marks it invalid after retries
        mapper_before = Mapper(self.retest_db).max_rowid() if Path(self.retest_db).exists() else 0
        for backend in task["backends"]:
            if backend == "cpu":
                continue  # replay always computes the shared CPU reference
            command = [*self.afx, "fuzz", "replay", task["case_key"], "--backend", backend,
                       "--database", self.retest_db, "--save", "--device", device]
            done = subprocess.run(command, capture_output=True, text=True, timeout=3600)
            if done.returncode not in (0, 1):
                log.warning("replay %s exited %s: %s", task["case_key"], done.returncode, done.stderr[-2000:])
        mapper = Mapper(self.retest_db)
        from autofusex.fuzz.client.contracts import Seed
        ids = [r[0] for r in mapper.conn.execute(
            "SELECT id FROM attempts WHERE rowid>? AND seed=?", (mapper_before, str(Seed.parse(task["case_key"]))))]
        if not ids:
            log.warning("task %s produced no results", task["task_id"])
            return
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


def main(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
