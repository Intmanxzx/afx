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


def filters(seed=None, status=None, user_id=None, q=None):
    clauses, values = [], []
    if seed is not None:
        from ..contracts import Seed
        seed = str(Seed.parse(seed))
        if not seed.isascii() or not seed.isdecimal() or len(seed) > 128:
            raise ValueError('invalid seed filter')
        clauses.append('seed=?'); values.append(seed)
    if status is not None:
        if status not in {'ok', 'fail', 'error', 'unsupported'}:
            raise ValueError('invalid status filter')
        clauses.append('status=?'); values.append(status)
    if user_id is not None:
        if not isinstance(user_id, str) or not 1 <= len(user_id) <= 256:
            raise ValueError('invalid user filter')
        clauses.append('id IN (SELECT attempt_id FROM attempt_facts WHERE user_id=?)')
        values.append(user_id)
    if q:
        if not isinstance(q, str) or len(q)>256:
            raise ValueError('invalid search')
        clauses.append('id IN (SELECT attempt_id FROM report_search WHERE instr(lower(content),lower(?))>0)')
        values.append(q)
    return clauses, values


class ResultQueries:
    def __init__(self, path):
        self.path = path
        with reading(path):
            pass

    def page(self, *, limit=50, after=0, seed=None, status=None, user_id=None, q=None, newest=False):
        if type(limit) is not int or not 1 <= limit <= 200 or type(after) is not int or not 0 <= after < 2**63:
            raise ValueError('invalid pagination')
        if type(newest) is not bool:
            raise ValueError('invalid ordering')
        clauses, values = filters(seed, status, user_id, q)
        clauses.append('rowid<?' if newest else 'rowid>?')
        values.extend((after if after or not newest else 2**63-1, limit + 1))
        with reading(self.path) as conn:
            rows = [dict(row) for row in conn.execute(
                'SELECT rowid,id,seed,workload_id,backend,status,created_at FROM attempts WHERE '
                + ' AND '.join(clauses) + (' ORDER BY rowid DESC LIMIT ?' if newest else ' ORDER BY rowid LIMIT ?'), values)]
        from ..contracts import Seed
        from ..backend_names import display_backend
        for row in rows:
            row['seed_short'] = Seed.parse(row['seed']).short
            row['backend_display'] = display_backend(row['backend'])
        more = len(rows) > limit
        return {'items': rows[:limit], 'next_cursor': rows[limit-1]['rowid'] if more else None}

    def summary(self, *, seed=None, status=None, user_id=None, q=None):
        clauses, values = filters(seed, status, user_id, q)
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
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='attempt_facts'").fetchone():
                fact = conn.execute('SELECT device_name,user_id,source_id,total_ns,duration_scope FROM attempt_facts WHERE attempt_id=?',(identity,)).fetchone()
                result['provenance'] = dict(fact) if fact else {}
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='case_facts'").fetchone():
                fact = conn.execute('SELECT metadata FROM case_facts WHERE workload_id=?',(row['workload_id'],)).fetchone()
                result['metadata'] = json.loads(fact[0]) if fact else {}
            result['contexts'] = {}
            for kind in ('execution','evaluation','environment'):
                payload = conn.execute('SELECT payload FROM contexts WHERE id=?', (row[kind+'_id'],)).fetchone()
                if payload is None or len(payload[0]) > 16384:
                    raise ValueError('invalid context')
                result['contexts'][kind] = json.loads(payload[0])
        # Copy command is based only on validated decimal seed, never stored shell text.
        filters(seed=result['seed'])
        from ..contracts import Seed
        from ..backend_names import display_backend
        result['seed_short'] = Seed.parse(result['seed']).short
        result['backend_display'] = display_backend(result['backend'])
        result['replay_command'] = 'autofusedbg fuzz replay ' + result['seed_short']
        return result

    def case_page(self, *, limit=50, after=0, seed=None, status=None, user_id=None, q=None):
        """Group comparable runs, retaining every attempt in backend history.

        Status selects groups, not their members: a failure filter must still show
        the passing peer backend and later retries. Cursor is a group high-water rowid.
        """
        if type(limit) is not int or not 1 <= limit <= 200 or type(after) is not int or not 0 <= after < 2**63:
            raise ValueError('invalid pagination')
        clauses, values = filters(seed, status, user_id, q)
        where = ' WHERE ' + ' AND '.join('a.'+c for c in clauses) if clauses else ''
        with reading(self.path) as conn:
            conn.execute('BEGIN')
            groups = conn.execute('SELECT f.group_id, max(a.rowid) AS cursor FROM attempts a '
                'JOIN attempt_facts f ON f.attempt_id=a.id WHERE f.group_id IN '
                '(SELECT f.group_id FROM attempts a JOIN attempt_facts f ON f.attempt_id=a.id'+where+') '
                'GROUP BY f.group_id HAVING max(a.rowid)<? ORDER BY cursor DESC LIMIT ?',
                (*values, after or 2**63-1, limit+1)).fetchall()
            items = []
            from ..contracts import Seed
            from ..backend_names import display_backend
            for group in groups[:limit]:
                rows = conn.execute('SELECT a.*,f.*,a.rowid AS cursor FROM attempt_facts f '
                    'JOIN attempts a ON a.id=f.attempt_id WHERE group_id=? ORDER BY a.rowid DESC LIMIT 1001', (group['group_id'],)).fetchall()
                if len(rows)>1000: raise ValueError('case history exceeds query limit')
                first = rows[0]
                meta = conn.execute('SELECT metadata FROM case_facts WHERE workload_id=?',(first['workload_id'],)).fetchone()
                item = {key:first[key] for key in ('seed','workload_id','source_id','device_name','user_id','batch_id','created_at')}
                item.update(user_name=first['device_name'], case_id=group['group_id'], seed_short=Seed.parse(item['seed']).short,
                            execution=json.loads(first['execution']), evaluation=json.loads(first['evaluation']),
                            versions=json.loads(first['versions']), metadata=json.loads(meta[0]) if meta else {}, backends={})
                for row in rows:
                    backend = display_backend(row['backend'])
                    member = {key:row[key] for key in ('id','status','created_at','total_ns','duration_scope','error_stage','error_summary')}
                    member['device'] = json.loads(row['execution']).get('device')
                    member['stages'] = [dict(r) for r in conn.execute('SELECT shape_index,stage,duration_ns FROM stage_metrics WHERE attempt_id=? ORDER BY shape_index,stage',(row['id'],))]
                    member['shapes'] = [dict(r) for r in conn.execute('SELECT * FROM shapes WHERE attempt_id=? ORDER BY shape_index',(row['id'],))]
                    member['compile_ns'] = next((m['duration_ns'] for m in member['stages'] if m['stage']=='compile' and m['shape_index']==-1),None)
                    member['kernel_ns'] = [{'shape_index':m['shape_index'],'duration_ns':m['duration_ns']} for m in member['stages'] if m['stage']=='kernel_mean'] if backend!='cpu' else []
                    from ..reporting.outcomes import outcomes
                    member.update(outcomes(member, json.loads(row['evaluation'])))
                    item['backends'].setdefault(backend, []).append(member)
                items.append(item)
        return {'items':items,'next_cursor':groups[limit-1]['cursor'] if len(groups)>limit else None}
