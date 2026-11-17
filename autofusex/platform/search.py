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
FEATURE_FIELDS = {"op": "op", "dtype": "dtype", "category": "op_category", "source": "source",
                  "dynamic": "dynamic", "size": "size_bucket"}
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
        if term.isdigit():
            if len(term) < 4 or len(term) > 60:
                raise SearchError("case number prefix must be 4..60 digits")
            query.case.append(("c.case_key >= ? AND c.case_key < ?", [term, term[:-1] + chr(ord(term[-1]) + 1)]))
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
            query.head.append((f"{HEAD_FIELDS[name]} {'NOT ' if negate else ''}IN ({marks})", values))
        elif name in FEATURE_FIELDS:
            sql = ("EXISTS (SELECT 1 FROM case_features f JOIN feature_registry r ON r.feature_id=f.feature_id "
                   "JOIN feature_values v ON v.value_id=f.value_id WHERE f.case_id=c.case_id AND r.name=? "
                   f"AND v.value IN ({marks}))")
            query.case.append(((f"NOT {sql}" if negate else sql), [FEATURE_FIELDS[name], *values]))
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


def filtered_case_count(conn, query: Query) -> int:
    case_sql, case_args = query.case_sql()
    with time_limit(conn):
        if query.needs_heads:
            head_sql, head_args = query.head_sql()
            return conn.execute("SELECT COUNT(DISTINCT h.case_id) FROM execution_heads h JOIN executions e "
                                f"ON e.exec_id=h.exec_id JOIN cases c ON c.case_id=h.case_id WHERE {head_sql} "
                                f"AND {case_sql}", [*head_args, *case_args]).fetchone()[0]
        return conn.execute(f"SELECT COUNT(*) FROM cases c WHERE {case_sql}", case_args).fetchone()[0]
