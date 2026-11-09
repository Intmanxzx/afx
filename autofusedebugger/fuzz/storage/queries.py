"""Read-only viewer queries shared with future central UI; bounded responses."""
from contextlib import contextmanager
import json
import sqlite3
import time
import uuid

from .repository import _connect


@contextmanager
def reading(path):
    conn = _connect(path)
    try:
        deadline = time.monotonic() + 3
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


def filters(seed=None, status=None):
    clauses, values = [], []
    if seed is not None:
        if not seed.isascii() or not seed.isdecimal() or len(seed) > 128:
            raise ValueError('invalid seed filter')
        clauses.append('seed=?'); values.append(seed)
    if status is not None:
        if status not in {'ok', 'fail', 'error', 'unsupported'}:
            raise ValueError('invalid status filter')
        clauses.append('status=?'); values.append(status)
    return clauses, values


class ResultQueries:
    def __init__(self, path):
        self.path = path
        with reading(path):
            pass

    def page(self, *, limit=50, after=0, seed=None, status=None):
        if type(limit) is not int or not 1 <= limit <= 200 or type(after) is not int or not 0 <= after < 2**63:
            raise ValueError('invalid pagination')
        clauses, values = filters(seed, status)
        clauses.append('rowid>?'); values.extend((after, limit + 1))
        with reading(self.path) as conn:
            rows = [dict(row) for row in conn.execute(
                'SELECT rowid,id,seed,workload_id,backend,status,created_at FROM attempts WHERE '
                + ' AND '.join(clauses) + ' ORDER BY rowid LIMIT ?', values)]
        more = len(rows) > limit
        return {'items': rows[:limit], 'next_cursor': rows[limit-1]['rowid'] if more else None}

    def summary(self, *, seed=None, status=None):
        clauses, values = filters(seed, status)
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        with reading(self.path) as conn:
            counts = {row['status']: row['count'] for row in conn.execute(
                'SELECT status,count(*) AS count FROM attempts' + where + ' GROUP BY status', values)}
        return {'total': sum(counts.values()), 'by_status': counts}

    def detail(self, identity):
        identity = str(uuid.UUID(identity))
        with reading(self.path) as conn:
            conn.execute('BEGIN')
            row = conn.execute('SELECT * FROM attempts WHERE id=?', (identity,)).fetchone()
            if row is None:
                raise LookupError('attempt not found')
            result = dict(row)
            result['shape_results'] = [dict(r) for r in conn.execute(
                'SELECT * FROM shapes WHERE attempt_id=? ORDER BY shape_index LIMIT 1025', (identity,))]
            if len(result['shape_results']) > 1024:
                raise ValueError('excessive shape results')
            result['contexts'] = {}
            for kind in ('execution','evaluation','environment'):
                payload = conn.execute('SELECT payload FROM contexts WHERE id=?', (row[kind+'_id'],)).fetchone()
                if payload is None or len(payload[0]) > 16384:
                    raise ValueError('invalid context')
                result['contexts'][kind] = json.loads(payload[0])
        # Copy command is based only on validated decimal seed, never stored shell text.
        filters(seed=result['seed'])
        result['replay_command'] = 'autofusedbg fuzz replay ' + result['seed']
        return result
