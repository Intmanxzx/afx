"""Issue clustering and collaborative ownership for the problem-class page.

Immutable attempt/result tables remain authoritative. This module keeps a
rebuildable issue-case index and stores only mutable ownership state.
"""
from contextlib import closing
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid

from ...client.storage.repository import _connect

DIMENSIONS = {"function", "accuracy", "performance"}
CLAIM_TTL = 7 * 24 * 60 * 60
MAX_CASES = 50_000
DEFAULT_CATEGORY_PRIORITY = ("MatMul", "Norm", "Gather", "View", "Transpose", "Reduce", "Elementwise", "Other")


def _category_priority():
    value = os.environ.get("AF_FUZZ_ISSUE_CATEGORY_PRIORITY")
    if not value:
        return DEFAULT_CATEGORY_PRIORITY
    priorities = tuple(item.strip() for item in value.split(",") if item.strip())
    if not priorities:
        raise ValueError("issue category priority cannot be empty")
    return priorities


def initialize(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS issue_cases_v1 (
        seed TEXT NOT NULL, shape_index INTEGER NOT NULL, attempt_id TEXT NOT NULL,
        status TEXT NOT NULL, function_status TEXT NOT NULL, core_error TEXT NOT NULL,
        correctness TEXT NOT NULL, performance TEXT NOT NULL, error_summary TEXT NOT NULL,
        op_types TEXT NOT NULL, dimension TEXT NOT NULL, category TEXT NOT NULL,
        title TEXT NOT NULL, issue_key TEXT NOT NULL, PRIMARY KEY(seed,shape_index))""")
    conn.execute("CREATE INDEX IF NOT EXISTS issue_cases_key ON issue_cases_v1(issue_key,dimension)")
    conn.execute("""CREATE TABLE IF NOT EXISTS issue_groups_v1 (
        issue_key TEXT PRIMARY KEY, dimension TEXT NOT NULL, category TEXT NOT NULL, title TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS issue_claims_v1 (
        issue_key TEXT PRIMARY KEY, dimension TEXT NOT NULL, category TEXT NOT NULL,
        owner_id TEXT NOT NULL REFERENCES users(id), owner_name TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('claimed','retesting')),
        claimed REAL NOT NULL, updated REAL NOT NULL, batch_id TEXT
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS issue_claims_dimension ON issue_claims_v1(dimension,status)")


def _text(value):
    return str(value or "").strip()


def _first_error_line(value):
    for line in _text(value).splitlines():
        line = line.strip()
        if line:
            return line[:240]
    return "未提供错误摘要"


def _issue(dimension, category, message):
    if dimension == "function":
        label = _first_error_line(message) if message else category
    else:
        label = category
    key = hashlib.sha256((dimension + "\0" + category + "\0" + label).encode()).hexdigest()[:24]
    return key, label


def _operator_category(row):
    from ...client.generation.categories import OP_CATEGORY
    operations = row.get("op_types") or []
    categories = {OP_CATEGORY.get(str(op).removeprefix("aten."), "Other") for op in operations}
    if not categories:
        return "Other"
    non_elementwise = categories - {"Elementwise"}
    if not non_elementwise:
        return "Elementwise"
    for category in _category_priority():
        if category in non_elementwise:
            return category
    return sorted(non_elementwise)[0]


def _dimension(row):
    function = _text(row["function_status"])
    if function in {"function_failed", "gen_failed", "error", "Failed", "构图Failed"}:
        return "function", function or "function_failed", row["core_error"] or row["error_summary"]
    if row["status"] in {"fail", "accuracy_failed"} or row["correctness"] == "fail":
        return "accuracy", _operator_category(row), row["core_error"] or row["error_summary"]
    if row["performance"] in {"error", "fail", "failed"} or row["measurement_status"] in {"error", "failed"}:
        return "performance", _operator_category(row), row["core_error"] or row["error_summary"]
    if row["status"] in {"error", "gen_failed"}:
        return "function", function or "function_failed", row["core_error"] or row["error_summary"]
    return None


def _source_rows(conn, seeds=None):
    where = '''cp.backend='af-mlir' AND (
        cp.function_status IN ('function_failed','gen_failed','error','Failed','构图Failed') OR
        cp.status IN ('fail','error','gen_failed') OR s.correctness='fail' OR
        s.performance IN ('error','fail','failed'))'''
    values = ()
    if seeds:
        where += ' AND cp.seed IN (' + ','.join('?' for _ in seeds) + ')'
        values = tuple(seeds)
    return conn.execute('''SELECT cp.seed,cp.shape_index,ch.id AS attempt_id,cp.status,
        cp.function_status,COALESCE(cd.core_error,'') AS core_error,
        COALESCE(s.correctness,'') AS correctness,COALESCE(s.performance,'') AS performance,
        COALESCE(s.error_summary,a.error_summary,'') AS error_summary,cp.op
        FROM case_projection_v1 cp
        JOIN case_heads_v1 ch ON ch.record_id=cp.record_id AND ch.shape_index=cp.shape_index
        LEFT JOIN case_display_v1 cd ON cd.attempt_id=ch.id AND cd.shape_index=cp.shape_index
        LEFT JOIN shapes s ON s.attempt_id=ch.id AND s.shape_index=cp.shape_index
        LEFT JOIN attempts a ON a.id=ch.id WHERE ''' + where, values)


def refresh_index(conn, seeds=None):
    """Refresh the issue index from the current materialized case heads."""
    from ..reporting.materialized import installed
    if not installed(conn):
        return {'skipped': 'materialized case model not installed'}
    conn.row_factory = sqlite3.Row
    initialize(conn)
    seeds = None if seeds is None else tuple(dict.fromkeys(seeds))
    if seeds == ():
        return {'cases': conn.execute('SELECT count(*) FROM issue_cases_v1').fetchone()[0],
                'groups': conn.execute('SELECT count(*) FROM issue_groups_v1').fetchone()[0]}
    if seeds is None:
        conn.execute('DELETE FROM issue_cases_v1')
    elif seeds:
        conn.executemany('DELETE FROM issue_cases_v1 WHERE seed=?', ((seed,) for seed in seeds))
    groups = {}
    for row in _source_rows(conn, seeds):
        value = dict(row)
        try:
            value['op_types'] = json.loads(value.get('op') or '[]')
        except (TypeError, ValueError):
            value['op_types'] = []
        item = _dimension(value)
        if not item:
            continue
        dimension, category, message = item
        key, title = _issue(dimension, category, message)
        conn.execute('''INSERT INTO issue_cases_v1 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (value['seed'], value['shape_index'], value['attempt_id'], value['status'],
             value['function_status'], value['core_error'], value['correctness'], value['performance'],
             value['error_summary'], json.dumps(value['op_types']), dimension, category, title, key))
        groups[key] = (dimension, category, title)
    conn.execute('DELETE FROM issue_groups_v1')
    conn.executemany('INSERT INTO issue_groups_v1 VALUES (?,?,?,?)',
        ((key, *value) for key, value in groups.items()))
    return {'cases': conn.execute('SELECT count(*) FROM issue_cases_v1').fetchone()[0],
            'groups': conn.execute('SELECT count(*) FROM issue_groups_v1').fetchone()[0]}


def rebuild(path):
    """Rebuild the current issue index atomically."""
    from ..reporting.materialized import refresh as refresh_cases
    with closing(_connect(path, writable=True)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        initialize(conn)
        refresh_cases(conn)
        return refresh_index(conn)


def _index_current(conn, *, include_targets=False):
    claims = {row['issue_key']: dict(row) for row in conn.execute('SELECT * FROM issue_claims_v1')} \
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='issue_claims_v1'").fetchone() else {}
    groups = {}
    for row in conn.execute('''SELECT issue_key,dimension,category,title,seed,shape_index,
        attempt_id,status,error_summary FROM issue_cases_v1'''):
        group = groups.setdefault(row['issue_key'], dict(issue_key=row['issue_key'],
            dimension=row['dimension'], category=row['category'], title=row['title'], count=0,
            examples=[], _targets=[], status='open', owner_id=None, owner_name=None,
            claimed_at=None, updated=None, batch_id=None))
        group['count'] += 1
        group['_targets'].append(dict(attempt_id=row['attempt_id'], shape_index=row['shape_index']))
        if len(group['examples']) < 5:
            group['examples'].append(dict(seed=row['seed'], shape_index=row['shape_index'],
                attempt_id=row['attempt_id'], status=row['status'], message=_text(row['error_summary'])[:500]))
    now = time.time()
    result = []
    for key, group in groups.items():
        claim = claims.get(key)
        if claim and now - float(claim['updated']) <= CLAIM_TTL:
            state = claim['status']
            if state == 'retesting' and claim['batch_id']:
                batch_ids = json.loads(claim['batch_id'])
                placeholders = ','.join('?' for _ in batch_ids)
                pending = conn.execute(
                    f"SELECT count(*) FROM retest_tasks_v1 WHERE batch_id IN ({placeholders}) "
                    "AND state IN ('queued','leased','running','cancel_requested')",
                    batch_ids).fetchone()[0] if batch_ids else 0
                if pending == 0:
                    state = 'claimed'
            group.update(status='retesting' if state == 'retesting' else 'claimed',
                owner_id=claim['owner_id'], owner_name=claim['owner_name'],
                claimed_at=claim['claimed'], updated=claim['updated'], batch_id=claim['batch_id'])
        group['target_count'] = len(group['_targets'])
        if not include_targets:
            group.pop('_targets', None)
        result.append(group)
    result.sort(key=lambda item: (-item['count'], item['title'], item['issue_key']))
    return result


def _current(path, *, include_targets=False):
    with closing(_connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        from ..reporting.materialized import ready
        if not ready(conn):
            raise ValueError('case index is not ready; rebuild it before serving issues')
        return _index_current(conn, include_targets=include_targets)


def list_groups(path, dimension=None):
    if dimension is not None and dimension not in DIMENSIONS:
        raise ValueError("invalid issue dimension")
    items = [group for group in _current(path) if dimension is None or group['dimension'] == dimension]
    return {"dimension": dimension or "all", "items": items, "count": len(items)}


def claim(path, user, issue_key):
    if not isinstance(issue_key, str) or not re.fullmatch(r"[0-9a-f]{24}", issue_key):
        raise ValueError("invalid issue key")
    group = next((group for group in _current(path) if group["issue_key"] == issue_key), None)
    if group is None:
        raise LookupError("issue not open")
    with closing(_connect(path, writable=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        from .repairs import locked_for_agent
        if locked_for_agent(conn, issue_key):
            raise PermissionError('issue managed by agent repair lease')
        identity = conn.execute("SELECT id,display_name FROM users WHERE id=? AND active=1", (user,)).fetchone()
        if identity is None:
            raise PermissionError("user not found")
        now = time.time()
        current = conn.execute("SELECT * FROM issue_claims_v1 WHERE issue_key=?", (issue_key,)).fetchone()
        if current and current["owner_id"] != user and now - float(current["updated"]) <= CLAIM_TTL:
            raise PermissionError("issue already claimed")
        if current:
            conn.execute("""UPDATE issue_claims_v1
                SET dimension=?,category=?,owner_id=?,owner_name=?,status='claimed',
                    claimed=?,updated=?,batch_id=NULL WHERE issue_key=?""",
                (group["dimension"], group["category"], user, identity["display_name"], now, now, issue_key))
        else:
            conn.execute("INSERT INTO issue_claims_v1 VALUES (?,?,?,?,?,?,?,?,?)",
                (issue_key, group["dimension"], group["category"], user, identity["display_name"],
                 "claimed", now, now, None))
        conn.commit()
    return next(group for group in _current(path) if group["issue_key"] == issue_key)


def mark_retesting(path, user, issue_key, batch_id):
    with closing(_connect(path, writable=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        from .repairs import locked_for_agent
        if locked_for_agent(conn, issue_key):
            raise PermissionError('issue managed by agent repair lease')
        row = conn.execute("SELECT owner_id FROM issue_claims_v1 WHERE issue_key=?", (issue_key,)).fetchone()
        if not row or row["owner_id"] != user:
            raise PermissionError("issue not owned")
        now = time.time()
        conn.execute("UPDATE issue_claims_v1 SET status='retesting',batch_id=?,updated=? WHERE issue_key=?",
                     (batch_id, now, issue_key))
        conn.commit()
    return next(group for group in _current(path) if group["issue_key"] == issue_key)


def retest(path, user, issue_key, client_id, repetitions=1, request_key=None):
    from . import retests
    if not isinstance(client_id, str):
        raise ValueError("client_id required")
    group = next((group for group in _current(path, include_targets=True) if group["issue_key"] == issue_key), None)
    if group is None:
        raise LookupError("issue not open")
    if group.get("owner_id") != user:
        raise PermissionError("claim the issue first")
    with closing(_connect(path)) as conn:
        from .repairs import locked_for_agent
        if locked_for_agent(conn, issue_key):
            raise PermissionError('issue managed by agent repair lease')
    targets = group.pop("_targets", [])
    if not targets:
        raise LookupError("issue not open")
    try:
        request_base = uuid.UUID(request_key) if request_key is not None else uuid.uuid4()
    except (ValueError, TypeError, AttributeError):
        raise ValueError("request_key must be a UUID") from None
    batches = []
    for offset in range(0, len(targets), 200):
        chunk = offset // 200
        payload = dict(client_id=client_id,
                       request_key=str(uuid.uuid5(request_base, str(chunk))),
                       targets=targets[offset:offset + 200], repetitions=repetitions)
        answer = retests.create(path, user, payload)
        batches.append(answer["batch_id"])
    mark_retesting(path, user, issue_key, json.dumps(batches))
    return {"issue_key": issue_key, "batch_ids": batches, "target_count": len(targets)}


def release(path, user, issue_key):
    with closing(_connect(path, writable=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        from .repairs import locked_for_agent
        if locked_for_agent(conn, issue_key):
            raise PermissionError('issue managed by agent repair lease')
        row = conn.execute("SELECT owner_id FROM issue_claims_v1 WHERE issue_key=?", (issue_key,)).fetchone()
        if not row or row["owner_id"] != user:
            raise PermissionError("issue not owned")
        conn.execute("DELETE FROM issue_claims_v1 WHERE issue_key=?", (issue_key,))
        conn.commit()
    return {"released": True, "issue_key": issue_key}
