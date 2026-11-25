"""Global search expression (design 6.8): parsed, whitelisted, parameter-bound, time-limited.

Syntax: space separated terms, each `field:value`, `field:v1,v2` (any of) or
`-field:value` (exclude). A bare number is a case_key prefix. Examples:
    backend:af-mlir status:fail op:aten.add.Tensor dtype:bfloat16
    category:reduce -stage:schedule error:tiling
No SQL is ever built from user text: fields map to fixed SQL fragments and
values are bound parameters.
"""
from __future__ import annotations

import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

MAX_LENGTH = 256
MAX_TERMS = 8
MAX_VALUES = 50
TIMEOUT_SECONDS = 2.0

HEAD_FIELDS = {"backend": "e.backend", "arch": "e.npu_arch", "user": "e.user_id", "stage": "e.fail_stage"}
FEATURE_FIELDS = {"op": ("op", "op_display"), "dtype": ("dtype",), "category": ("op_category",),
                  "source": ("source",), "dynamic": ("dynamic",), "size": ("size_bucket",), "dim": ("dim_bucket",)}
STATUS = {
    "pass": "(e.function_status='pass' AND COALESCE(e.accuracy_status,'')<>'fail')",
    "fail": "(e.function_status NOT IN ('pass','unsupported') OR e.accuracy_status='fail')",
    "function_fail": "e.function_status NOT IN ('pass','unsupported')",
    "accuracy_fail": "(e.function_status='pass' AND e.accuracy_status='fail')",
    "unsupported": "e.function_status='unsupported'",
}
_VALUE = re.compile(r"^[A-Za-z0-9_.:/+<>=*-]{1,64}$")
_TERM = re.compile(r"^(-?)([a-z]+):(.+)$")


class SearchError(ValueError):
    def __init__(self, message, status=400, code="invalid_query"):
        super().__init__(message)
        self.status, self.code = status, code


@dataclass
class Query:
    head: list = field(default_factory=list)      # (sql, args) needing executions alias e
    case: list = field(default_factory=list)      # (sql, args) on cases alias c
    text: str = ""

    @property
    def needs_heads(self) -> bool:
        return bool(self.head)

    def head_sql(self):
        return _join(self.head)

    def case_sql(self):
        return _join(self.case)


def _join(parts):
    if not parts:
        return "1", []
    return " AND ".join(p[0] for p in parts), [a for p in parts for a in p[1]]


def parse(text: str | None) -> Query | None:
    if text is None or not text.strip():
        return None
    if len(text) > MAX_LENGTH:
        raise SearchError(f"query longer than {MAX_LENGTH} characters")
    terms = text.split()
    if len(terms) > MAX_TERMS:
        raise SearchError(f"at most {MAX_TERMS} terms")
    query = Query(text=text.strip())
    for position, term in enumerate(terms):
        if re.fullmatch(r"[EFef][0-9A-Za-z]{3,26}", term) or re.fullmatch(r"[BM]:\S{1,200}", term):
            prefix = term if term[1] == ":" else term.upper()
            query.case.append(("c.case_key >= ? AND c.case_key < ?", [prefix, prefix + "\uffff"]))
            continue
        match = _TERM.match(term)
        if not match:
            raise SearchError(f"term {position + 1} must look like field:value")
        negate, name, raw = match.groups()
        if name == "error":
            if len(raw) < 3 or len(raw) > 64 or "\\" in raw:
                raise SearchError("error text must be 3..64 characters")
            pattern = "%" + raw.replace("%", "\\%").replace("_", "\\_") + "%"
            query.head.append((("NOT " if negate else "") + "COALESCE(e.log_summary,'') LIKE ? ESCAPE '\\'",
                               [pattern]))
            continue
        values = raw.split(",")
        if len(values) > MAX_VALUES or not all(_VALUE.match(v) for v in values):
            raise SearchError(f"invalid value in term {position + 1}")
        marks = ",".join("?" * len(values))
        if name == "status":
            if not all(v in STATUS for v in values):
                raise SearchError(f"status must be one of {', '.join(STATUS)}")
            sql = "(" + " OR ".join(STATUS[v] for v in values) + ")"
            query.head.append(((f"NOT {sql}" if negate else sql), []))
        elif name in HEAD_FIELDS:
            column = HEAD_FIELDS[name]
            # exclusion keeps rows where the field is empty (NULL NOT IN (...) would drop them)
            query.head.append(((f"COALESCE({column},'') NOT IN ({marks})" if negate else f"{column} IN ({marks})"), values))
        elif name in FEATURE_FIELDS:
            names = FEATURE_FIELDS[name]
            sql = ("EXISTS (SELECT 1 FROM case_features f JOIN feature_registry r ON r.feature_id=f.feature_id "
                   "JOIN feature_values v ON v.value_id=f.value_id WHERE f.case_id=c.case_id "
                   f"AND r.name IN ({','.join('?' * len(names))}) AND v.value IN ({marks}))")
            query.case.append(((f"NOT {sql}" if negate else sql), [*names, *values]))
        elif name == "case":
            query.case.append(("c.case_key " + ("<>" if negate else "=") + " ?", [values[0]]))
        else:
            raise SearchError(f"unknown field {name}; use one of "
                              f"{', '.join(sorted({*HEAD_FIELDS, *FEATURE_FIELDS, 'status', 'error', 'case'}))}")
    return query


@contextmanager
def time_limit(conn: sqlite3.Connection, seconds: float = TIMEOUT_SECONDS):
    deadline = time.monotonic() + seconds
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 20_000)
    try:
        yield
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc):
            raise SearchError("query too broad, add more conditions", 408, "query_timeout") from None
        raise
    finally:
        conn.set_progress_handler(None, 0)


def filtered_cases(conn, query: Query, cursor: int, limit: int) -> list:
    case_sql, case_args = query.case_sql()
    with time_limit(conn):
        if query.needs_heads:
            head_sql, head_args = query.head_sql()
            rows = conn.execute(
                "SELECT DISTINCT h.case_id FROM execution_heads h JOIN executions e ON e.exec_id=h.exec_id "
                f"JOIN cases c ON c.case_id=h.case_id WHERE h.case_id<? AND {head_sql} AND {case_sql} "
                "ORDER BY h.case_id DESC LIMIT ?", [cursor, *head_args, *case_args, limit]).fetchall()
        else:
            rows = conn.execute(f"SELECT c.case_id FROM cases c WHERE c.case_id<? AND {case_sql} "
                                "ORDER BY c.case_id DESC LIMIT ?", [cursor, *case_args, limit]).fetchall()
    return [r[0] for r in rows]


def filtered_buckets(conn, query: Query, *, backend=None, arch=None, dim="all"):
    """Head-level status counts for the filtered case set, same columns as stat_buckets."""
    # These predicates depend only on columns already present in stat_buckets.
    # Keep visual filters and status/stage/backend/arch searches on aggregates.
    if aggregate_compatible(query):
        sql, args = query.head_sql()
        where = [sql, "e.dim=?"]
        args.append(dim)
        if backend:
            where.append("e.backend=?"); args.append(backend)
        if arch:
            where.append("e.npu_arch=?"); args.append(arch)
        return conn.execute(
            "SELECT e.dim_value, e.backend, e.function_status, e.fail_stage, e.accuracy_status, SUM(e.count) AS n "
            "FROM stats.stat_buckets e WHERE " + ' AND '.join(where) + " GROUP BY 1,2,3,4,5", args).fetchall()
    head_sql, head_args = query.head_sql()
    case_sql, case_args = query.case_sql()
    join, value = "", "'all'"
    args = []
    if dim != "all":
        join = ("JOIN case_features df ON df.case_id=h.case_id JOIN feature_registry dr ON dr.feature_id=df.feature_id "
                "AND dr.name=? JOIN feature_values dv ON dv.value_id=df.value_id ")
        value = "dv.value"
        args.append(dim)
    where = [head_sql, case_sql]
    args.extend([*head_args, *case_args])
    if backend:
        where.append("e.backend=?")
        args.append(backend)
    if arch:
        where.append("h.npu_arch=?")
        args.append(arch)
    with time_limit(conn):
        return conn.execute(
            f"SELECT {value} AS dim_value, e.backend AS backend, e.function_status AS function_status, "
            "COALESCE(e.fail_stage,'') AS fail_stage, COALESCE(e.accuracy_status,'') AS accuracy_status, "
            "COUNT(*) AS n FROM execution_heads h JOIN executions e ON e.exec_id=h.exec_id "
            f"JOIN cases c ON c.case_id=h.case_id {join}WHERE {' AND '.join(where)} GROUP BY 1,2,3,4,5",
            args).fetchall()


def field_values(store) -> dict:
    """Known values for the search fields whose vocabulary isn't a small fixed set in the UI.

    feature_values is a dedup dictionary (one row per distinct value ever seen), so these reads
    are cheap regardless of how many executions exist. backend/status/stage/source/dynamic are
    left out: their vocabulary is already a short fixed list the UI keeps alongside its labels.
    """
    conn = store.reader()

    def feature(name):
        return [r[0] for r in conn.execute(
            "SELECT v.value FROM feature_values v JOIN feature_registry r USING(feature_id) "
            "WHERE r.name=? ORDER BY v.value", (name,))]

    return {
        "op": feature("op"),
        "dtype": feature("dtype"),
        "category": feature("op_category"),
        "size": feature("size_bucket"),
        "dim": feature("dim_bucket"),
        "arch": [r[0] for r in conn.execute(
            "SELECT DISTINCT npu_arch FROM stats.stat_buckets WHERE npu_arch<>'' ORDER BY npu_arch")],
        "user": [{"id": r[0], "name": r[1] or r[0]} for r in conn.execute("SELECT user_id, display_name FROM users ORDER BY COALESCE(display_name, user_id), user_id")],
    }


def filtered_case_count(conn, query: Query) -> int:
    case_sql, case_args = query.case_sql()
    with time_limit(conn):
        if query.needs_heads:
            head_sql, head_args = query.head_sql()
            return conn.execute("SELECT COUNT(DISTINCT h.case_id) FROM execution_heads h JOIN executions e "
                                f"ON e.exec_id=h.exec_id JOIN cases c ON c.case_id=h.case_id WHERE {head_sql} "
                                f"AND {case_sql}", [*head_args, *case_args]).fetchone()[0]
        return conn.execute(f"SELECT COUNT(*) FROM cases c WHERE {case_sql}", case_args).fetchone()[0]


def visual_query(q=None, results=None, selected_backends=None, chips=None):
    """Compose visual predicates with the search on the same execution head."""
    query = parse(q) or Query()
    for text, column in ((selected_backends, 'e.backend'),):
        if text is not None:
            values = text.split(',')
            if len(values) > MAX_VALUES or not all(_VALUE.fullmatch(v) for v in values):
                raise SearchError('invalid visual filter')
            query.head.append((f"{column} IN ({','.join('?' for _ in values)})", values))
    if chips is not None:
        values = chips.split(',')
        if len(values) > MAX_VALUES or not all(_VALUE.fullmatch(v) for v in values):
            raise SearchError('invalid chip filter')
        parts, args = [], []
        for value in values:
            if value in ('a2a3', 'a5'):
                parts.append("lower(e.npu_arch) LIKE ?")
                args.append('%910%' if value == 'a2a3' else '%950%')
            else:
                parts.append('e.npu_arch=?'); args.append(value)
        query.head.append(('(' + ' OR '.join(parts) + ')', args))
    if results is not None:
        values = results.split(',')
        from autofusex.server.result_categories import RESULT_KEYS, category_sql
        if not values or not set(values) <= RESULT_KEYS | {'none'}:
            raise SearchError('invalid result filter')
        query.head.append((f"({category_sql()}) IN ({','.join('?' for _ in values)})", values))
    return query if query.head or query.case else None


def filtered_perf(conn, query, bx, by, dim, arch, tie):
    import math
    from autofusex.server.aggregates import perf_verdict, ratio_bin
    head_sql, head_args = query.head_sql()
    case_sql, case_args = query.case_sql()
    where, args = [case_sql], list(case_args)
    if arch:
        where.append('hx.npu_arch=?'); args.append(arch)
    # Filter the cohort with one matching head; compare both measured backends
    # for those cases on that same chip, without multiplying feature rows.
    where.append('EXISTS (SELECT 1 FROM execution_heads h JOIN executions e ON e.exec_id=h.exec_id '
                 'WHERE h.case_id=c.case_id AND h.npu_arch=hx.npu_arch AND ' + head_sql + ')')
    args.extend(head_args)
    join, value, prefix = '', "'all'", []
    if dim != 'all':
        join = ('JOIN case_features f ON f.case_id=c.case_id JOIN feature_registry r ON r.feature_id=f.feature_id '
                'AND r.name=? JOIN feature_values v ON v.value_id=f.value_id ')
        value = 'v.value'; prefix = [dim]
    with time_limit(conn):
        rows = conn.execute(
            f'SELECT {value} AS value, ex.kernel_ns AS kx, ey.kernel_ns AS ky FROM cases c '
            'JOIN execution_heads hx ON hx.case_id=c.case_id AND hx.backend=? '
            'JOIN execution_heads hy ON hy.case_id=c.case_id AND hy.npu_arch=hx.npu_arch AND hy.backend=? '
            'JOIN executions ex ON ex.exec_id=hx.exec_id JOIN executions ey ON ey.exec_id=hy.exec_id ' + join +
            "WHERE ex.function_status='pass' AND ey.function_status='pass' "
            "AND COALESCE(ex.accuracy_status,'')<>'fail' AND COALESCE(ey.accuracy_status,'')<>'fail' "
            "AND ex.kernel_ns>0 AND ey.kernel_ns>0 AND " + ' AND '.join(where),
            [bx, by, *prefix, *args]).fetchall()
    buckets = {}
    for row in rows:
        ratio = row['kx'] / row['ky']; log = math.log2(ratio)
        key = (row['value'], perf_verdict(ratio, tie), ratio_bin(log))
        bucket = buckets.setdefault(key, dict(dim_value=key[0], verdict=key[1], ratio_bin=key[2], n=0, s=0.0))
        bucket['n'] += 1; bucket['s'] += log
    return list(buckets.values())


def aggregate_compatible(query):
    if query.case:
        return False
    return not any("e.user_id" in sql or "e.log_summary" in sql for sql, _ in query.head)
