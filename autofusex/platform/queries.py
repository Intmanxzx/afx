"""Read models for the overview and the problem-discovery list (design 6.2, 6.3).

Overview queries only read stats.db aggregates, never scan executions.
"""
from __future__ import annotations

import json

from .aggregates import RATIO_BINS_PER_OCTAVE
from . import search
from .store import Store, get_content

BASELINES = ("cpu", "eager-npu")
# Backend families share early stages in the overview pipeline (design 6.2 module 1).
FAMILIES = {"af-legacy": "af", "af-mlir": "af"}
SHARED_STAGES = {"af": ("schedule",)}
MAX_PAGE = 200
MAX_EXECUTIONS_PER_CASE = 10


class QueryError(ValueError):
    pass


def _arch_clause(arch):
    return ("", ()) if arch is None else (" AND npu_arch=?", (arch,))


def pipeline(store: Store, arch: str | None = None, q: str | None = None) -> dict:
    conn = store.reader()
    query = search.parse(q)
    if query is not None:
        rows = search.filtered_buckets(conn, query, arch=arch)
    else:
        clause, args = _arch_clause(arch)
        rows = conn.execute(
            "SELECT backend, function_status, fail_stage, accuracy_status, SUM(count) AS n FROM stats.stat_buckets "
            f"WHERE dim='all'{clause} GROUP BY 1,2,3,4", args).fetchall()
    lanes = {}
    for row in rows:
        lane = lanes.setdefault(row["backend"], {"backend": row["backend"], "family": FAMILIES.get(row["backend"]),
                                                 "baseline": row["backend"] in BASELINES, "executed": 0,
                                                 "build_fail": 0, "stage_fail": {}, "function_fail": 0,
                                                 "accuracy_fail": 0, "unsupported": 0, "pass": 0})
        n = row["n"]
        lane["executed"] += n
        status = row["function_status"]
        if status == "pass":
            lane["accuracy_fail" if row["accuracy_status"] == "fail" else "pass"] += n
        elif status == "build_fail":
            lane["build_fail"] += n
        elif status == "unsupported":
            lane["unsupported"] += n
        else:
            lane["function_fail"] += n
            stage = row["fail_stage"] or "unknown"
            lane["stage_fail"][stage] = lane["stage_fail"].get(stage, 0) + n
    if query is not None:
        total = search.filtered_case_count(conn, query)
    elif arch:
        # Every case run on this chip has at least one head there; the busiest lane bounds it.
        total = max((lane["executed"] for lane in lanes.values()), default=0)
    else:
        total = int(conn.execute("SELECT value FROM meta WHERE key='case_count'").fetchone()[0])
    arches = [r[0] for r in conn.execute(
        "SELECT DISTINCT npu_arch FROM stats.stat_buckets WHERE npu_arch<>'' ORDER BY npu_arch")]
    return {"total_cases": total, "npu_arch": arch, "arches": arches, "q": query.text if query else None, "shared_stages": {k: list(v) for k, v in SHARED_STAGES.items()},
            "lanes": sorted(lanes.values(), key=lambda lane: (not lane["baseline"], lane["backend"]))}


def backend_stats(store: Store, backend: str, dim: str = "all", arch: str | None = None,
                  q: str | None = None) -> dict:
    conn = store.reader()
    query = search.parse(q)
    if query is not None:
        rows = search.filtered_buckets(conn, query, backend=backend, arch=arch, dim=dim)
    else:
        clause, args = _arch_clause(arch)
        rows = conn.execute(
            "SELECT dim_value, function_status, fail_stage, accuracy_status, SUM(count) AS n FROM stats.stat_buckets "
            f"WHERE backend=? AND dim=?{clause} GROUP BY 1,2,3,4", (backend, dim) + args).fetchall()
    groups = {}
    for row in rows:
        group = groups.setdefault(row["dim_value"], {"value": row["dim_value"], "total": 0, "pass": 0,
                                                     "function_fail": 0, "accuracy_fail": 0, "unsupported": 0,
                                                     "stage_fail": {}})
        n = row["n"]
        group["total"] += n
        if row["function_status"] == "pass":
            group["accuracy_fail" if row["accuracy_status"] == "fail" else "pass"] += n
        elif row["function_status"] == "unsupported":
            group["unsupported"] += n
        else:
            group["function_fail"] += n
            stage = row["fail_stage"] or row["function_status"]
            group["stage_fail"][stage] = group["stage_fail"].get(stage, 0) + n
    return {"backend": backend, "dim": dim, "npu_arch": arch,
            "groups": sorted(groups.values(), key=lambda g: -g["total"])}


def perf_pair(store: Store, x: str, y: str, dim: str = "all", arch: str | None = None) -> dict:
    """x measured against baseline y; buckets are stored once per unordered pair."""
    if x == y:
        raise QueryError("x and y must differ")
    flip = x > y
    bx, by = (y, x) if flip else (x, y)
    conn = store.reader()
    clause, args = _arch_clause(arch)
    rows = conn.execute(
        "SELECT dim_value, verdict, ratio_bin, SUM(count) AS n, SUM(sum_log) AS s FROM stats.perf_pair_buckets "
        f"WHERE backend_x=? AND backend_y=? AND dim=?{clause} GROUP BY 1,2,3", (bx, by, dim) + args).fetchall()
    swap = {"faster": "slower", "slower": "faster", "tie": "tie"}
    groups = {}
    for row in rows:
        verdict, rbin, s = row["verdict"], row["ratio_bin"], row["s"]
        if flip:
            verdict, rbin, s = swap[verdict], -rbin - 1, -s
        group = groups.setdefault(row["dim_value"], {"value": row["dim_value"], "total": 0, "faster": 0,
                                                     "tie": 0, "slower": 0, "_log": 0.0, "histogram": {}})
        group["total"] += row["n"]
        group[verdict] += row["n"]
        group["_log"] += s
        group["histogram"][rbin] = group["histogram"].get(rbin, 0) + row["n"]
    out = []
    for group in groups.values():
        group["geomean_ratio"] = 2 ** (group.pop("_log") / group["total"]) if group["total"] else None
        group["histogram"] = [{"bin": b, "log2_low": b / RATIO_BINS_PER_OCTAVE, "count": c}
                              for b, c in sorted(group["histogram"].items())]
        out.append(group)
    tie = store.setting(conn, "perf.tie_threshold")
    return {"x": x, "y": y, "dim": dim, "npu_arch": arch, "tie_threshold": tie,
            "verdict_rule": "ratio = kernel(x)/kernel(y); faster if ratio <= 1/(1+t), slower if >= 1+t",
            "groups": sorted(out, key=lambda g: -g["total"])}


def dims(store: Store) -> list:
    return ["all"] + [row[0] for row in store.reader().execute(
        "SELECT name FROM feature_registry WHERE groupable=1 ORDER BY name")]


_CASE_COLUMNS = ("case_id, case_key, case_source, case_revision, is_dynamic, shape_count, node_count, op_types, "
                 "dtypes, max_dim, max_numel, external_ref, first_seen_at, updated_at, "
                 "(SELECT group_concat(v.value, ',') FROM case_features f JOIN feature_registry r "
                 "ON r.feature_id=f.feature_id JOIN feature_values v ON v.value_id=f.value_id "
                 "WHERE f.case_id=cases.case_id AND r.name='op_display') AS op_display")
_EXEC_COLUMNS = ("e.exec_id, e.exec_uuid, e.case_revision, e.backend, e.npu_arch, e.user_id, e.client_id, "
                 "e.run_id, e.run_kind, e.function_status, e.fail_stage, e.accuracy_status, e.accuracy_ref, "
                 "e.max_abs_err, e.max_rel_err, e.compile_ns, e.kernel_ns, e.e2e_ns, e.log_summary, e.created_at, "
                 "(SELECT COALESCE(u.display_name, u.user_id) FROM users u WHERE u.user_id=e.user_id) AS user_name, "
                 "(SELECT c.display_name FROM clients c WHERE c.client_id=e.client_id) AS client_name, "
                 "(e.detail_content_id IS NOT NULL) AS has_detail")


def case_page(store: Store, *, after: int | None = None, limit: int = 50, backend: str | None = None,
              status: str | None = None, arch: str | None = None, case_key: str | None = None,
              case_id: int | None = None, q: str | None = None) -> dict:
    """Case groups newest first, each with its latest executions (design 6.3)."""
    if not 1 <= limit <= MAX_PAGE:
        raise QueryError(f"limit must be 1..{MAX_PAGE}")
    if status is not None and status not in ("pass", "fail", "function_fail", "accuracy_fail"):
        raise QueryError("status must be pass, fail, function_fail or accuracy_fail")
    if status is not None and backend is None:
        raise QueryError("status filter requires backend")
    conn = store.reader()
    cursor = after if after is not None else 1 << 62
    query = search.parse(q)
    if case_id is not None:
        ids = [row[0] for row in conn.execute("SELECT case_id FROM cases WHERE case_id=?", (case_id,))]
    elif case_key is not None:
        ids = [row[0] for row in conn.execute("SELECT case_id FROM cases WHERE case_key=?", (case_key,))]
    elif query is not None:
        if backend is not None or status is not None:
            raise QueryError("combine backend/status into q, e.g. backend:af-mlir status:fail")
        ids = search.filtered_cases(conn, query, cursor, limit)
    elif backend is None:
        ids = [row[0] for row in conn.execute(
            "SELECT case_id FROM cases WHERE case_id<? ORDER BY case_id DESC LIMIT ?", (cursor, limit))]
    else:
        where, args = ["h.backend=?", "h.case_id<?"], [backend, cursor]
        if arch is not None:
            where.append("h.npu_arch=?")
            args.append(arch)
        if status == "pass":
            where.append("e.function_status='pass' AND COALESCE(e.accuracy_status,'')<>'fail'")
        elif status == "function_fail":
            where.append("e.function_status NOT IN ('pass','unsupported')")
        elif status == "accuracy_fail":
            where.append("e.function_status='pass' AND e.accuracy_status='fail'")
        elif status == "fail":
            where.append("(e.function_status NOT IN ('pass','unsupported') OR e.accuracy_status='fail')")
        ids = [row[0] for row in conn.execute(
            "SELECT DISTINCT h.case_id FROM execution_heads h JOIN executions e ON e.exec_id=h.exec_id "
            f"WHERE {' AND '.join(where)} ORDER BY h.case_id DESC LIMIT ?", args + [limit])]
    groups = []
    for cid in ids:
        case = dict(conn.execute(f"SELECT {_CASE_COLUMNS} FROM cases WHERE case_id=?", (cid,)).fetchone())
        executions = [dict(row) for row in conn.execute(
            f"SELECT {_EXEC_COLUMNS} FROM executions e WHERE e.case_id=? ORDER BY e.created_at DESC, e.exec_id DESC "
            "LIMIT ?", (cid, MAX_EXECUTIONS_PER_CASE + 1))]
        heads = {row[0] for row in conn.execute("SELECT exec_id FROM execution_heads WHERE case_id=?", (cid,))}
        for execution in executions:
            execution["is_head"] = execution["exec_id"] in heads
            execution["stale_graph"] = execution["case_revision"] != case["case_revision"]
        groups.append({"case": case, "executions": executions[:MAX_EXECUTIONS_PER_CASE],
                       "more_executions": len(executions) > MAX_EXECUTIONS_PER_CASE})
    next_cursor = ids[-1] if len(ids) == limit and case_key is None and case_id is None else None
    return {"groups": groups, "next_cursor": next_cursor}


def execution_detail(store: Store, exec_uuid: str, *, log_limit: int = 64 * 1024) -> dict | None:
    conn = store.reader()
    row = conn.execute(f"SELECT {_EXEC_COLUMNS}, e.case_id, e.env_id, e.log_content_id FROM executions e "
                       "WHERE e.exec_uuid=?", (exec_uuid,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    env = conn.execute("SELECT * FROM environments WHERE env_id=?", (out.pop("env_id"),)).fetchone()
    out["env"] = {k: env[k] for k in env.keys() if k not in ("env_id", "env_hash")}
    out["case_key"] = conn.execute("SELECT case_key FROM cases WHERE case_id=?", (out["case_id"],)).fetchone()[0]
    out["shapes"] = [dict(r) for r in conn.execute(
        "SELECT shape_index, function_status, accuracy_status, max_abs_err, max_rel_err, kernel_ns, e2e_ns "
        "FROM execution_shapes WHERE exec_id=? ORDER BY shape_index", (out["exec_id"],))]
    detail_id = conn.execute("SELECT detail_content_id FROM executions WHERE exec_id=?",
                             (out["exec_id"],)).fetchone()[0]
    if detail_id is not None:
        data = get_content(conn, detail_id)
        out["detail"] = json.loads(data) if data is not None else None
    log_id = out.pop("log_content_id")
    if log_id is not None:
        data = get_content(conn, log_id)
        if data is None:  # restored from a core snapshot newer than its content snapshot
            out["log_missing"] = True
        else:
            out["log"] = data[:log_limit].decode(errors="replace")
            out["log_truncated"] = len(data) > log_limit
    return out


def case_history(store: Store, case_key: str, *, after=None, limit: int = 50) -> dict:
    """Every execution of one case, newest first, cursor = exec_id of the last row."""
    if not 1 <= limit <= 200:
        raise QueryError("limit must be 1..200")
    conn = store.reader()
    case = conn.execute(f"SELECT {_CASE_COLUMNS} FROM cases WHERE case_key=?", (case_key,)).fetchone()
    if case is None:
        raise QueryError("case not found")
    case = dict(case)
    rows = [dict(r) for r in conn.execute(
        f"SELECT {_EXEC_COLUMNS} FROM executions e WHERE e.case_id=? AND e.exec_id<? ORDER BY e.exec_id DESC LIMIT ?",
        (case["case_id"], after if after is not None else 1 << 62, limit))]
    heads = {r[0] for r in conn.execute("SELECT exec_id FROM execution_heads WHERE case_id=?", (case["case_id"],))}
    for row in rows:
        row["is_head"] = row["exec_id"] in heads
        row["stale_graph"] = row["case_revision"] != case["case_revision"]
    total = conn.execute("SELECT COUNT(*) FROM executions WHERE case_id=?", (case["case_id"],)).fetchone()[0]
    return {"case": case, "executions": rows, "total": total,
            "next_cursor": rows[-1]["exec_id"] if len(rows) == limit else None}


def case_spec(store: Store, case_key: str) -> dict:
    """Graph metadata saved with the case (nodes with display names, shape variants, weights)."""
    conn = store.reader()
    row = conn.execute("SELECT spec_content_id FROM cases WHERE case_key=?", (case_key,)).fetchone()
    if row is None:
        raise QueryError("case not found")
    if row[0] is None:
        return {}
    data = get_content(conn, row[0])
    return json.loads(data) if data is not None else {}


def clients(store: Store) -> list:
    return [dict(row) for row in store.reader().execute(
        "SELECT client_id, user_id, machine_id, display_name, hostname, npu_arch, afx_version, first_seen_at, "
        "last_seen_at FROM clients WHERE deleted_at IS NULL ORDER BY last_seen_at DESC")]


def settings(store: Store) -> dict:
    return store.settings()


def as_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
