"""Incrementally maintained aggregates in stats.db (design 4.8).

Every change of an execution head subtracts the old head's contribution and
adds the new one inside the ingest transaction. stats.db is derived data: if
its applied_seq differs from core's ingest_seq (crash between the two files'
commits, manual edits, threshold change) it is rebuilt from core.db.
"""
from __future__ import annotations

import math
import sqlite3

# Discrete features that become dimensions of the overview charts.
GROUPABLE = ("op_category", "dtype", "size_bucket", "dynamic", "source")
RATIO_BINS_PER_OCTAVE = 4
RATIO_BIN_LIMIT = 24  # +-6 octaves (64x) before clipping


def perf_verdict(ratio: float, tie: float) -> str:
    """ratio = kernel(x) / kernel(y); lower kernel time is better."""
    if ratio <= 1.0 / (1.0 + tie):
        return "faster"
    if ratio >= 1.0 + tie:
        return "slower"
    return "tie"


def ratio_bin(log2_ratio: float) -> int:
    value = math.floor(log2_ratio * RATIO_BINS_PER_OCTAVE)
    return max(-RATIO_BIN_LIMIT, min(RATIO_BIN_LIMIT, value))


def case_dims(conn: sqlite3.Connection, case_id: int):
    dims = [("all", "all")]
    dims.extend((row[0], row[1]) for row in conn.execute(
        "SELECT r.name, v.value FROM case_features f JOIN feature_registry r USING(feature_id) "
        "JOIN feature_values v ON v.value_id=f.value_id WHERE f.case_id=? AND r.groupable=1",
        (case_id,)))
    return dims


def head_row(conn: sqlite3.Connection, exec_id: int):
    return conn.execute(
        "SELECT exec_id, case_id, backend, npu_arch, function_status, fail_stage, accuracy_status, "
        "kernel_ns FROM executions WHERE exec_id=?", (exec_id,)).fetchone()


def apply_stat(conn: sqlite3.Connection, row, dims, sign: int) -> None:
    key = (row["backend"], row["npu_arch"], row["function_status"],
           row["fail_stage"] or "", row["accuracy_status"] or "")
    for dim, value in dims:
        _bump(conn, "stats.stat_buckets",
              ("backend", "npu_arch", "function_status", "fail_stage", "accuracy_status", "dim", "dim_value"),
              key + (dim, value), sign)


def perf_comparable(row) -> bool:
    return (row["function_status"] == "pass" and row["accuracy_status"] != "fail"
            and row["kernel_ns"] is not None and row["kernel_ns"] > 0)


def apply_perf(conn: sqlite3.Connection, row, dims, sign: int, tie: float) -> None:
    """Contribution of `row` paired with every other comparable head of the same case and arch."""
    if not perf_comparable(row):
        return
    others = conn.execute(
        "SELECT e.backend, e.function_status, e.accuracy_status, e.kernel_ns FROM execution_heads h "
        "JOIN executions e ON e.exec_id=h.exec_id WHERE h.case_id=? AND h.npu_arch=? AND h.backend<>?",
        (row["case_id"], row["npu_arch"], row["backend"])).fetchall()
    for other in others:
        if not perf_comparable(other):
            continue
        x, y = (row, other) if row["backend"] < other["backend"] else (other, row)
        _apply_pair(conn, x["backend"], y["backend"], row["npu_arch"], x["kernel_ns"], y["kernel_ns"],
                    dims, sign, tie)


def _apply_pair(conn, bx, by, arch, kx, ky, dims, sign, tie):
    ratio = kx / ky
    log2r = math.log2(ratio)
    verdict, rbin = perf_verdict(ratio, tie), ratio_bin(log2r)
    for dim, value in dims:
        conn.execute(
            "INSERT INTO stats.perf_pair_buckets(backend_x, backend_y, npu_arch, dim, dim_value, verdict, "
            "ratio_bin, count, sum_log) VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(backend_x, backend_y, npu_arch, dim, dim_value, verdict, ratio_bin) DO UPDATE SET count=count+excluded.count, sum_log=sum_log+excluded.sum_log",
            (bx, by, arch, dim, value, verdict, rbin, sign, sign * log2r))
    conn.execute("DELETE FROM stats.perf_pair_buckets WHERE backend_x=? AND backend_y=? AND npu_arch=? "
                 "AND verdict=? AND ratio_bin=? AND count=0", (bx, by, arch, verdict, rbin))


def _bump(conn, table, columns, values, sign):
    placeholders = ",".join("?" * len(columns))
    conn.execute(f"INSERT INTO {table}({','.join(columns)}, count) VALUES({placeholders}, ?) "
                 f"ON CONFLICT({','.join(columns)}) DO UPDATE SET count=count+excluded.count",
                 values + (sign,))
    if sign < 0:
        where = " AND ".join(f"{c}=?" for c in columns)
        conn.execute(f"DELETE FROM {table} WHERE {where} AND count=0", values)


# -- full rebuild -------------------------------------------------------------
_DIMS_CTE = """
WITH dims(case_id, dim, dim_value) AS (
  SELECT case_id, 'all', 'all' FROM cases
  UNION ALL
  SELECT f.case_id, r.name, v.value FROM case_features f
  JOIN feature_registry r ON r.feature_id=f.feature_id AND r.groupable=1
  JOIN feature_values v ON v.value_id=f.value_id
)
"""


def rebuild(conn: sqlite3.Connection, tie: float) -> None:
    """Recompute all aggregates from core.db inside the caller's write transaction."""
    conn.create_function("afx_ln2", 1, lambda r: math.log2(r), deterministic=True)
    conn.create_function("afx_verdict", 2, perf_verdict, deterministic=True)
    conn.create_function("afx_bin", 1, ratio_bin, deterministic=True)
    conn.execute("DELETE FROM stats.stat_buckets")
    conn.execute("DELETE FROM stats.perf_pair_buckets")
    conn.execute(_DIMS_CTE + """
INSERT INTO stats.stat_buckets(backend, npu_arch, dim, dim_value, function_status, fail_stage,
  accuracy_status, count)
SELECT e.backend, h.npu_arch, d.dim, d.dim_value, e.function_status, COALESCE(e.fail_stage,''),
  COALESCE(e.accuracy_status,''), COUNT(*)
FROM execution_heads h JOIN executions e ON e.exec_id=h.exec_id JOIN dims d ON d.case_id=h.case_id
GROUP BY 1,2,3,4,5,6,7""")
    conn.execute(_DIMS_CTE + """
, pairs AS (
  SELECT hx.case_id, hx.npu_arch, ex.backend AS bx, ey.backend AS by_,
         CAST(ex.kernel_ns AS REAL)/ey.kernel_ns AS ratio
  FROM execution_heads hx
  JOIN execution_heads hy ON hy.case_id=hx.case_id AND hy.npu_arch=hx.npu_arch AND hy.backend>hx.backend
  JOIN executions ex ON ex.exec_id=hx.exec_id
  JOIN executions ey ON ey.exec_id=hy.exec_id
  WHERE ex.function_status='pass' AND ey.function_status='pass'
    AND COALESCE(ex.accuracy_status,'')<>'fail' AND COALESCE(ey.accuracy_status,'')<>'fail'
    AND ex.kernel_ns>0 AND ey.kernel_ns>0
)
INSERT INTO stats.perf_pair_buckets(backend_x, backend_y, npu_arch, dim, dim_value, verdict, ratio_bin,
  count, sum_log)
SELECT p.bx, p.by_, p.npu_arch, d.dim, d.dim_value, afx_verdict(p.ratio, ?), afx_bin(afx_ln2(p.ratio)),
  COUNT(*), SUM(afx_ln2(p.ratio))
FROM pairs p JOIN dims d ON d.case_id=p.case_id
GROUP BY 1,2,3,4,5,6,7""", (tie,))
    seq = conn.execute("SELECT value FROM meta WHERE key='ingest_seq'").fetchone()[0]
    conn.execute("UPDATE stats.stats_meta SET value=? WHERE key='applied_seq'", (seq,))


def consistent(conn: sqlite3.Connection) -> bool:
    core = conn.execute("SELECT value FROM meta WHERE key='ingest_seq'").fetchone()[0]
    applied = conn.execute("SELECT value FROM stats.stats_meta WHERE key='applied_seq'").fetchone()[0]
    return str(core) == str(applied)


def advance_seq(conn: sqlite3.Connection) -> None:
    """Called at the end of each write that changed heads; keeps both files in step."""
    conn.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='ingest_seq'")
    seq = conn.execute("SELECT value FROM meta WHERE key='ingest_seq'").fetchone()[0]
    conn.execute("UPDATE stats.stats_meta SET value=? WHERE key='applied_seq'", (str(seq),))
