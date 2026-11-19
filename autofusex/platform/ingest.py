"""Execution upload: validation, case/environment upsert, heads and aggregates (design 4.3, 4.4)."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3

from . import aggregates, issues
from .auth import Principal
from .store import Store, canonical_json, now_ms, put_content

MAX_ITEMS = 500
MAX_LOG_BYTES = 256 * 1024
MAX_SPEC_BYTES = 512 * 1024
MAX_DETAIL_BYTES = 1024 * 1024
MAX_SUMMARY = 512
MAX_SHAPES = 8

CASE_SOURCES = ("nnsmith", "baize_tf", "baize_torch", "model")
RUN_KINDS = ("fuzz", "retest", "verify", "regress", "import")
# Candidate fix versions (verify/regress) never become the current result.
HEAD_RUN_KINDS = ("fuzz", "retest", "import")
FUNCTION_STATUS = ("pass", "build_fail", "compile_fail", "runtime_fail", "timeout", "unsupported")
ACCURACY_STATUS = ("pass", "fail", "skipped")
ACCURACY_REFS = ("cpu", "eager-npu")
ENV_FIELDS = ("cann", "python", "torch", "torch_npu", "backend_version", "afx", "nnsmith", "os", "cpu_arch", "gcc")
NUMERIC_FEATURES = ("node_count", "max_numel", "max_dim", "shape_count")

_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,31}$")
_SEED = re.compile(r"^[EF][0-9A-Z]{26}$")  # Seed.short: E static / F dynamic + base36 entropy
_KEY_PREFIX = {"baize_tf": "B:", "baize_torch": "B:", "model": "M:"}
_TEXT = re.compile(r"^[^\x00-\x08\x0b\x0c\x0e-\x1f]*$")


class UploadError(ValueError):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.status = code, status


# -- validation ---------------------------------------------------------------
def _str(obj, key, *, required=True, limit=128, pattern=None):
    value = obj.get(key)
    if value is None:
        if required:
            raise UploadError("invalid", f"{key} is required")
        return None
    if not isinstance(value, str) or len(value) > limit or not _TEXT.match(value):
        raise UploadError("invalid", f"{key} must be a string of at most {limit} chars")
    if pattern is not None and not pattern.match(value):
        raise UploadError("invalid", f"{key} has invalid format")
    return value


def _int(obj, key, *, required=False, minimum=0):
    value = obj.get(key)
    if value is None:
        if required:
            raise UploadError("invalid", f"{key} is required")
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > 2**62:
        raise UploadError("invalid", f"{key} must be an integer >= {minimum}")
    return value


def _float(obj, key):
    value = obj.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise UploadError("invalid", f"{key} must be a number")
    return float(value)


def _enum(obj, key, choices, *, required=True):
    value = obj.get(key)
    if value is None and not required:
        return None
    if value not in choices:
        raise UploadError("invalid", f"{key} must be one of {', '.join(choices)}")
    return value


def _labels(value, key):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 256:
        raise UploadError("invalid", f"{key} must be a list of at most 256 strings")
    out = []
    for item in value:
        if not isinstance(item, str) or not item or len(item) > 64 or not _TEXT.match(item):
            raise UploadError("invalid", f"{key} entries must be non-empty strings of at most 64 chars")
        out.append(item)
    return sorted(set(out))


def validate_case(case) -> dict:
    if not isinstance(case, dict):
        raise UploadError("invalid", "case must be an object")
    source = _enum(case, "case_source", CASE_SOURCES)
    key = _str(case, "case_key", limit=256)
    if source == "nnsmith":
        if not _SEED.match(key):
            raise UploadError("invalid", "nnsmith case_key must be a short seed (E/F + 26 base36 chars)")
    elif not key.startswith(_KEY_PREFIX[source]) or len(key) < 3:
        raise UploadError("invalid", f"{source} case_key must start with {_KEY_PREFIX[source]}")
    fingerprint = _str(case, "graph_fingerprint", required=source != "nnsmith", limit=64,
                       pattern=re.compile(r"^[0-9a-f]{64}$"))
    features = case.get("features") or {}
    if not isinstance(features, dict) or len(features) > 32:
        raise UploadError("invalid", "features must be an object with at most 32 entries")
    clean_features = {}
    for name, labels in features.items():
        if not isinstance(name, str) or not _NAME.match(name):
            raise UploadError("invalid", f"feature name {name!r} is invalid")
        clean_features[name] = _labels(labels, f"features.{name}")
    metrics = case.get("metrics") or {}
    if not isinstance(metrics, dict) or len(metrics) > 32:
        raise UploadError("invalid", "metrics must be an object with at most 32 entries")
    clean_metrics = {}
    for name in metrics:
        if not isinstance(name, str) or not _NAME.match(name):
            raise UploadError("invalid", f"metric name {name!r} is invalid")
        clean_metrics[name] = _float(metrics, name)
    spec = case.get("spec")
    spec_bytes = canonical_json(spec) if spec is not None else None
    if spec_bytes is not None and len(spec_bytes) > MAX_SPEC_BYTES:
        raise UploadError("too_large", "case spec exceeds 512 KiB")
    out = {
        "case_key": key, "case_source": source,
        "graph_fingerprint": bytes.fromhex(fingerprint) if fingerprint else None,
        "is_dynamic": 1 if case.get("is_dynamic") else 0,
        "shape_count": _int(case, "shape_count", required=True, minimum=1),
        "node_count": _int(case, "node_count"),
        "op_types": _labels(case.get("op_types"), "op_types"),
        "dtypes": _labels(case.get("dtypes"), "dtypes"),
        "max_dim": _int(case, "max_dim"),
        "max_numel": _int(case, "max_numel"),
        "external_ref": _str(case, "external_ref", required=False, limit=512),
        "features": clean_features, "metrics": clean_metrics, "spec": spec_bytes,
    }
    if out["shape_count"] > MAX_SHAPES:
        raise UploadError("invalid", f"shape_count must be at most {MAX_SHAPES}")
    return out


def validate_item(item) -> dict:
    if not isinstance(item, dict):
        raise UploadError("invalid", "item must be an object")
    function_status = _enum(item, "function_status", FUNCTION_STATUS)
    accuracy_status = _enum(item, "accuracy_status", ACCURACY_STATUS, required=False)
    if function_status != "pass" and accuracy_status not in (None, "skipped"):
        raise UploadError("invalid", "accuracy_status must be empty or skipped when function failed")
    env = item.get("env")
    if not isinstance(env, dict) or set(env) - set(ENV_FIELDS):
        raise UploadError("invalid", f"env must be an object with keys from {', '.join(ENV_FIELDS)}")
    env = {name: _str(env, name, required=False, limit=64) for name in ENV_FIELDS}
    log = item.get("log")
    if log is not None and (not isinstance(log, str) or len(log.encode()) > MAX_LOG_BYTES):
        raise UploadError("too_large", "log must be a string of at most 256 KiB")
    shapes = item.get("shapes") or []
    if not isinstance(shapes, list) or len(shapes) > MAX_SHAPES:
        raise UploadError("invalid", f"shapes must be a list of at most {MAX_SHAPES}")
    clean_shapes = []
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise UploadError("invalid", "shape must be an object")
        clean_shapes.append({
            "shape_index": index,
            "function_status": _enum(shape, "function_status", FUNCTION_STATUS),
            "accuracy_status": _enum(shape, "accuracy_status", ACCURACY_STATUS, required=False),
            "max_abs_err": _float(shape, "max_abs_err"), "max_rel_err": _float(shape, "max_rel_err"),
            "kernel_ns": _int(shape, "kernel_ns"), "e2e_ns": _int(shape, "e2e_ns"),
        })
    backend = _str(item, "backend", limit=32, pattern=_NAME)
    arch = _str(item, "npu_arch", required=False, limit=32) or ""
    if backend == "cpu":
        arch = ""
    detail = item.get("detail")
    detail_bytes = canonical_json(detail) if detail is not None else None
    if detail is not None and (not isinstance(detail, dict) or len(detail_bytes) > MAX_DETAIL_BYTES):
        raise UploadError("too_large", "detail must be an object of at most 1 MiB")
    return {
        "detail": detail_bytes,
        "exec_uuid": _str(item, "exec_uuid", limit=64, pattern=re.compile(r"^[A-Za-z0-9_-]{8,64}$")),
        "case": validate_case(item.get("case")),
        "backend": backend, "npu_arch": arch, "env": env,
        "function_status": function_status,
        "fail_stage": _str(item, "fail_stage", required=False, limit=32, pattern=_NAME),
        "accuracy_status": accuracy_status,
        "accuracy_ref": _enum(item, "accuracy_ref", ACCURACY_REFS, required=False),
        "max_abs_err": _float(item, "max_abs_err"), "max_rel_err": _float(item, "max_rel_err"),
        "compile_ns": _int(item, "compile_ns"), "kernel_ns": _int(item, "kernel_ns"),
        "e2e_ns": _int(item, "e2e_ns", required=True),
        "log_summary": (_str(item, "log_summary", required=False, limit=4096) or "")[:MAX_SUMMARY] or None,
        "log": log.encode() if log else None,
        "shapes": clean_shapes,
        "created_at": _int(item, "created_at", required=True, minimum=1),
    }


def validate_request(body) -> dict:
    if not isinstance(body, dict):
        raise UploadError("invalid", "body must be an object")
    client, run, items = body.get("client"), body.get("run"), body.get("items")
    if not isinstance(client, dict) or not isinstance(run, dict):
        raise UploadError("invalid", "client and run are required objects")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS:
        raise UploadError("invalid", f"items must be a list of 1..{MAX_ITEMS}")
    return {
        "request_key": _str(body, "request_key", limit=128),
        "client": {
            "machine_id": _str(client, "machine_id", limit=128),
            "hostname": _str(client, "hostname", required=False, limit=128),
            "npu_arch": _str(client, "npu_arch", required=False, limit=32),
            "afx_version": _str(client, "afx_version", required=False, limit=64),
        },
        "run": {"run_uuid": _str(run, "run_uuid", limit=64), "kind": _enum(run, "kind", RUN_KINDS)},
        "items": items,
    }


# -- ingest ------------------------------------------------------------------
class Ingestor:
    def __init__(self, store: Store):
        self.store = store
        self._feature_ids: dict = {}
        self._value_ids: dict = {}
        self._env_ids: dict = {}

    def upload(self, principal: Principal, body: dict, raw: bytes) -> dict:
        principal.require_write()
        request = validate_request(body)
        body_hash = hashlib.sha256(raw).digest()
        # Validate every item before taking the writer lock; per-item errors are reported, not fatal.
        prepared = []
        for index, item in enumerate(request["items"]):
            try:
                clean = validate_item(item)
                clean["content_hash"] = hashlib.sha256(canonical_json(item)).digest()
                prepared.append((index, clean, None))
            except UploadError as exc:
                prepared.append((index, None, {"index": index, "status": "rejected", "code": exc.code,
                                               "message": str(exc)}))
        try:
            with self.store.write() as conn:
                previous = conn.execute(
                    "SELECT body_hash, response FROM upload_requests WHERE user_id=? AND request_key=?",
                    (principal.user_id, request["request_key"])).fetchone()
                if previous is not None:
                    if previous["body_hash"] != body_hash:
                        raise UploadError("conflict", "request_key reused with a different body", 409)
                    return json.loads(previous["response"])
                tie = self.store.setting(conn, "perf.tie_threshold")
                self._issue_cfg = issues.config(conn)
                client_id = self._client(conn, principal.user_id, request["client"])
                run_id = self._run(conn, principal.user_id, client_id, request["run"])
                results, changed = [], False
                for index, clean, error in prepared:
                    if error is not None:
                        results.append(error)
                        continue
                    status, head_changed = self._item(conn, principal.user_id, client_id, run_id,
                                                      request["run"]["kind"], clean, tie)
                    changed |= head_changed
                    results.append({"index": index, "exec_uuid": clean["exec_uuid"], "status": status})
                if changed:
                    aggregates.advance_seq(conn)
                response = {"run_id": run_id, "results": results,
                            "accepted": sum(r["status"] == "accepted" for r in results)}
                conn.execute("INSERT INTO upload_requests VALUES(?,?,?,?,?)",
                             (principal.user_id, request["request_key"], body_hash,
                              json.dumps(response), now_ms()))
                return response
        except BaseException:
            # Ids cached during a rolled-back transaction may not exist.
            self._feature_ids.clear()
            self._value_ids.clear()
            self._env_ids.clear()
            raise

    # -- dimensions ------------------------------------------------------------
    def _client(self, conn, user_id, client) -> int:
        return upsert_client(conn, user_id, client)

    def _run(self, conn, user_id, client_id, run) -> int:
        row = conn.execute("SELECT run_id, user_id, client_id, kind FROM runs WHERE run_uuid=?",
                           (run["run_uuid"],)).fetchone()
        if row is not None:
            if (row["user_id"], row["client_id"], row["kind"]) != (user_id, client_id, run["kind"]):
                raise UploadError("conflict", "run_uuid belongs to another user, client or kind", 409)
            return row["run_id"]
        return conn.execute("INSERT INTO runs(run_uuid, user_id, client_id, kind, created_at) VALUES(?,?,?,?,?)",
                            (run["run_uuid"], user_id, client_id, run["kind"], now_ms())).lastrowid

    def _env(self, conn, env) -> int:
        digest = hashlib.sha256(canonical_json(env)).digest()
        cached = self._env_ids.get(digest)
        if cached is not None:
            return cached
        row = conn.execute("SELECT env_id FROM environments WHERE env_hash=?", (digest,)).fetchone()
        if row is None:
            env_id = conn.execute(
                f"INSERT INTO environments(env_hash, {', '.join(ENV_FIELDS)}) "
                f"VALUES(?{', ?' * len(ENV_FIELDS)})", (digest, *(env[k] for k in ENV_FIELDS))).lastrowid
        else:
            env_id = row[0]
        self._env_ids[digest] = env_id
        return env_id

    def _feature(self, conn, name, kind) -> int:
        key = (name, kind)
        if key in self._feature_ids:
            return self._feature_ids[key]
        row = conn.execute("SELECT feature_id, kind FROM feature_registry WHERE name=?", (name,)).fetchone()
        if row is None:
            feature_id = conn.execute(
                "INSERT INTO feature_registry(name, kind, groupable, axis, created_at) VALUES(?,?,?,?,?)",
                (name, kind, int(name in aggregates.GROUPABLE), int(kind == "numeric"), now_ms())).lastrowid
        elif row["kind"] != kind:
            raise UploadError("invalid", f"feature {name} is registered as {row['kind']}")
        else:
            feature_id = row["feature_id"]
        self._feature_ids[key] = feature_id
        return feature_id

    def _value(self, conn, feature_id, value) -> int:
        key = (feature_id, value)
        if key in self._value_ids:
            return self._value_ids[key]
        conn.execute("INSERT OR IGNORE INTO feature_values(feature_id, value) VALUES(?,?)", key)
        value_id = conn.execute("SELECT value_id FROM feature_values WHERE feature_id=? AND value=?",
                                key).fetchone()[0]
        if len(self._value_ids) < 200_000:
            self._value_ids[key] = value_id
        return value_id

    # -- cases -----------------------------------------------------------------
    def _case(self, conn, case, tie):
        """Return (case_id, revision of this upload, revision is current)."""
        now = now_ms()
        row = conn.execute("SELECT case_id, case_source, case_revision, graph_fingerprint FROM cases "
                           "WHERE case_key=?", (case["case_key"],)).fetchone()
        if row is not None and row["case_source"] != case["case_source"]:
            raise UploadError("conflict", f"{case['case_key']} already exists with another source")
        if row is not None:
            case_id, revision = row["case_id"], row["case_revision"]
            fingerprint = case["graph_fingerprint"]
            # Seeds are deterministic; only imported graphs can be replaced (design 3).
            if case["case_source"] == "nnsmith" or fingerprint is None or fingerprint == row["graph_fingerprint"]:
                return case_id, revision, True
            older = conn.execute("SELECT revision FROM case_revisions WHERE case_id=? AND graph_fingerprint=?",
                                 (case_id, fingerprint)).fetchone()
            if older is not None:
                return case_id, older[0], False
            self._retire_heads(conn, case_id, tie)
            revision += 1
            conn.execute("DELETE FROM case_features WHERE case_id=?", (case_id,))
            conn.execute("DELETE FROM case_metrics WHERE case_id=?", (case_id,))
            self._write_case(conn, case, case_id, revision, now)
            issues.mark_stale(conn, case_id, revision)
        else:
            case_id = self._write_case(conn, case, None, 1, now)
            revision = 1
        conn.execute("INSERT INTO case_revisions VALUES(?,?,?,?)", (case_id, revision, case["graph_fingerprint"], now))
        return case_id, revision, True

    def _write_case(self, conn, case, case_id, revision, now) -> int:
        spec_id = put_content(conn, case["spec"]) if case["spec"] is not None else None
        values = (case["case_source"], revision, case["graph_fingerprint"], case["is_dynamic"], case["shape_count"],
                  case["node_count"], ",".join(case["op_types"]) or None, ",".join(case["dtypes"]) or None,
                  case["max_dim"], case["max_numel"], spec_id, case["external_ref"], now)
        if case_id is None:
            case_id = conn.execute(
                "INSERT INTO cases(case_source, case_revision, graph_fingerprint, is_dynamic, shape_count, node_count, "
                "op_types, dtypes, max_dim, max_numel, spec_content_id, external_ref, updated_at, case_key, first_seen_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values + (case["case_key"], now)).lastrowid
            conn.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='case_count'")
        else:
            conn.execute(
                "UPDATE cases SET case_source=?, case_revision=?, graph_fingerprint=?, is_dynamic=?, shape_count=?, "
                "node_count=?, op_types=?, dtypes=?, max_dim=?, max_numel=?, spec_content_id=?, external_ref=?, "
                "updated_at=? WHERE case_id=?", values + (case_id,))
        labels = dict(case["features"])
        labels.setdefault("dtype", case["dtypes"])
        labels.setdefault("op", case["op_types"])
        labels["source"] = [case["case_source"]]
        labels["dynamic"] = ["dynamic" if case["is_dynamic"] else "static"]
        if case["max_numel"] is not None:
            labels["size_bucket"] = [size_bucket(case["max_numel"])]
        if case["max_dim"] is not None:
            labels["dim_bucket"] = [dim_bucket(case["max_dim"])]
        for name, values_ in labels.items():
            feature_id = self._feature(conn, name, "discrete")
            for value in values_:
                conn.execute("INSERT OR IGNORE INTO case_features VALUES(?,?,?)",
                             (case_id, feature_id, self._value(conn, feature_id, value)))
        metrics = dict(case["metrics"])
        for name in NUMERIC_FEATURES:
            if case.get(name) is not None:
                metrics.setdefault(name, float(case[name]))
        for name, value in metrics.items():
            if value is not None:
                conn.execute("INSERT OR REPLACE INTO case_metrics VALUES(?,?,?)",
                             (self._feature(conn, name, "numeric"), case_id, value))
        return case_id

    def _retire_heads(self, conn, case_id, tie):
        """A replaced graph invalidates every current result of the case (design 3)."""
        dims = aggregates.case_dims(conn, case_id)
        heads = conn.execute("SELECT exec_id FROM execution_heads WHERE case_id=?", (case_id,)).fetchall()
        for head in heads:
            row = aggregates.head_row(conn, head[0])
            conn.execute("DELETE FROM execution_heads WHERE case_id=? AND backend=? AND npu_arch=?",
                         (case_id, row["backend"], row["npu_arch"]))
            aggregates.apply_stat(conn, row, dims, -1)
            aggregates.apply_perf(conn, row, dims, -1, tie)

    # -- executions ------------------------------------------------------------
    def _item(self, conn, user_id, client_id, run_id, run_kind, item, tie):
        existing = conn.execute("SELECT content_hash FROM executions WHERE exec_uuid=?",
                                (item["exec_uuid"],)).fetchone()
        if existing is not None:
            return ("duplicate" if existing[0] == item["content_hash"] else "conflict"), False
        case_id, revision, current = self._case(conn, item["case"], tie)
        log_id = put_content(conn, item["log"]) if item["log"] else None
        detail_id = put_content(conn, item["detail"]) if item["detail"] else None
        exec_id = conn.execute(
            "INSERT INTO executions(exec_uuid, content_hash, case_id, case_revision, backend, npu_arch, user_id, "
            "client_id, run_id, run_kind, env_id, function_status, fail_stage, accuracy_status, accuracy_ref, "
            "max_abs_err, max_rel_err, compile_ns, kernel_ns, e2e_ns, log_summary, log_content_id, created_at, "
            "received_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (item["exec_uuid"], item["content_hash"], case_id, revision, item["backend"], item["npu_arch"], user_id,
             client_id, run_id, run_kind, self._env(conn, item["env"]), item["function_status"], item["fail_stage"],
             item["accuracy_status"], item["accuracy_ref"], item["max_abs_err"], item["max_rel_err"],
             item["compile_ns"], item["kernel_ns"], item["e2e_ns"], item["log_summary"], log_id,
             item["created_at"], now_ms())).lastrowid
        if detail_id is not None:
            conn.execute("UPDATE executions SET detail_content_id=? WHERE exec_id=?", (detail_id, exec_id))
        if item["shapes"]:
            conn.executemany(
                "INSERT INTO execution_shapes VALUES(?,?,?,?,?,?,?,?)",
                [(exec_id, s["shape_index"], s["function_status"], s["accuracy_status"], s["max_abs_err"],
                  s["max_rel_err"], s["kernel_ns"], s["e2e_ns"]) for s in item["shapes"]])
        if current and run_kind in HEAD_RUN_KINDS and self._advance_head(conn, case_id, exec_id, item, tie):
            issues.on_head(conn, case_id, exec_id, self._issue_cfg)
            return "accepted", True
        if issues.failure_dimension(item) is not None:
            if item["backend"] in self._issue_cfg["backends"]:
                row = conn.execute("SELECT * FROM executions WHERE exec_id=?", (exec_id,)).fetchone()
                conn.execute("UPDATE executions SET issue_signature=? WHERE exec_id=?",
                             (issues.execution_signature(conn, row), exec_id))
        return "accepted", False

    def _advance_head(self, conn, case_id, exec_id, item, tie) -> bool:
        head = conn.execute(
            "SELECT h.exec_id, e.created_at FROM execution_heads h JOIN executions e ON e.exec_id=h.exec_id "
            "WHERE h.case_id=? AND h.backend=? AND h.npu_arch=?",
            (case_id, item["backend"], item["npu_arch"])).fetchone()
        if head is not None and (head["created_at"], head["exec_id"]) > (item["created_at"], exec_id):
            return False  # a late upload of an older execution stays history only
        dims = aggregates.case_dims(conn, case_id)
        if head is not None:
            old = aggregates.head_row(conn, head["exec_id"])
            # Remove the old head first so the perf pass sees the other backends only.
            conn.execute("DELETE FROM execution_heads WHERE case_id=? AND backend=? AND npu_arch=?",
                         (case_id, item["backend"], item["npu_arch"]))
            aggregates.apply_stat(conn, old, dims, -1)
            aggregates.apply_perf(conn, old, dims, -1, tie)
        conn.execute("INSERT INTO execution_heads VALUES(?,?,?,?)",
                     (case_id, item["backend"], item["npu_arch"], exec_id))
        new = aggregates.head_row(conn, exec_id)
        aggregates.apply_stat(conn, new, dims, 1)
        aggregates.apply_perf(conn, new, dims, 1, tie)
        return True


def upsert_client(conn, user_id: str, client: dict) -> int:
    """Clients get a stable display name '<user display name>-npu<N>' in order of first appearance."""
    now = now_ms()
    exists = conn.execute("SELECT client_id FROM clients WHERE user_id=? AND machine_id=?",
                          (user_id, client["machine_id"])).fetchone()
    if exists is None:
        ordinal = conn.execute("SELECT COALESCE(MAX(ordinal)+1, 0) FROM clients WHERE user_id=?",
                               (user_id,)).fetchone()[0]
        user = conn.execute("SELECT COALESCE(display_name, user_id) FROM users WHERE user_id=?", (user_id,)).fetchone()
        conn.execute("INSERT INTO clients(user_id, machine_id, first_seen_at, last_seen_at, ordinal, display_name) "
                     "VALUES(?,?,?,?,?,?)", (user_id, client["machine_id"], now, now, ordinal,
                                             f"{user[0] if user else user_id}-npu{ordinal}"))
    conn.execute(
        "INSERT INTO clients(user_id, machine_id, hostname, npu_arch, afx_version, first_seen_at, last_seen_at) "
        "VALUES(?,?,?,?,?,?,?) ON CONFLICT(user_id, machine_id) DO UPDATE SET "
        "hostname=COALESCE(excluded.hostname, hostname), npu_arch=COALESCE(excluded.npu_arch, npu_arch), "
        "afx_version=COALESCE(excluded.afx_version, afx_version), last_seen_at=excluded.last_seen_at",
        (user_id, client["machine_id"], client.get("hostname"), client.get("npu_arch"), client.get("afx_version"),
         now, now))
    return conn.execute("SELECT client_id FROM clients WHERE user_id=? AND machine_id=?",
                        (user_id, client["machine_id"])).fetchone()[0]


def dim_bucket(dim: int) -> str:
    for limit in (8, 64, 512, 4096, 32768):
        if dim <= limit:
            return f"<={limit}"
    return ">32768"


def size_bucket(numel: int) -> str:
    for limit, label in ((1 << 10, "<=1K"), (1 << 16, "<=64K"), (1 << 20, "<=1M"), (1 << 24, "<=16M")):
        if numel <= limit:
            return label
    return ">16M"


def check_stats(store: Store) -> bool:
    """Rebuild stats.db when it is not in step with core.db. Returns True if rebuilt."""
    with store.write() as conn:
        if aggregates.consistent(conn):
            return False
        aggregates.rebuild(conn, store.setting(conn, "perf.tie_threshold"))
        return True


def update_settings(store: Store, actor: str, changes: dict, base_revisions: dict) -> dict:
    """Admin settings update with optimistic concurrency; a new tie threshold rebuilds perf aggregates."""
    from .store import DEFAULT_SETTINGS
    with store.write() as conn:
        for key, value in changes.items():
            if key not in DEFAULT_SETTINGS:
                raise UploadError("invalid", f"unknown setting {key}")
            _check_setting(key, value)
            row = conn.execute("SELECT revision FROM platform_settings WHERE key=?", (key,)).fetchone()
            if base_revisions.get(key) != row["revision"]:
                raise UploadError("conflict", f"{key} changed since revision {base_revisions.get(key)}", 409)
            conn.execute("UPDATE platform_settings SET value=?, revision=revision+1, updated_by=?, updated_at=? "
                         "WHERE key=?", (json.dumps(value), actor, now_ms(), key))
            conn.execute("INSERT INTO audit_log(at, actor, action, detail) VALUES(?,?,?,?)",
                         (now_ms(), actor, "settings.update", f"{key}={json.dumps(value)}"))
        if "perf.tie_threshold" in changes:
            aggregates.rebuild(conn, changes["perf.tie_threshold"])
    return store.settings()


def _check_setting(key, value):
    if key == "perf.tie_threshold":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 0.5
    elif key in ("regression.budget", "regression.max_budget"):
        ok = isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5000
    elif key == "regression.shares":
        ok = (isinstance(value, dict) and set(value) == {"neighbor", "sentinel", "history", "random"}
              and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value.values())
              and sum(value.values()) == 100)
    elif key == "issue.backends":
        ok = (isinstance(value, list) and 1 <= len(value) <= 8
              and all(isinstance(v, str) and _NAME.match(v) for v in value))
    elif key == "issue.perf_anchor":
        ok = isinstance(value, str) and bool(_NAME.match(value))
    elif key == "issue.perf_slowdown":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and 1.01 <= value <= 100
    elif key in ("issue.lease_minutes", "retest.lease_minutes"):
        ok = isinstance(value, int) and not isinstance(value, bool) and 5 <= value <= 1440
    elif key == "verify.max_cases":
        ok = isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5000
    else:
        ok = False
    if not ok:
        raise UploadError("invalid", f"invalid value for {key}")
