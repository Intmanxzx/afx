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

    def case_summary(self, **params):
        from ..reporting.cases import summary
        return summary(self.path, **params)

    def detail(self, identity, shape_index=None):
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
        if shape_index is not None:
            if type(shape_index) is not int or not -1 <= shape_index < 1024:
                raise ValueError('invalid shape index')
            chosen = [s for s in result['shape_results'] if s['shape_index'] == shape_index]
            if not chosen and (shape_index != -1 or result['shape_results']):
                raise LookupError('shape result not found')
            result['shape_index'] = shape_index
            result['shape_results'] = chosen
            variants = result.get('metadata', {}).get('shape_variants', [])
            result['sample'] = variants[shape_index] if 0 <= shape_index < len(variants) else {}
            if chosen:
                result['status'] = {'pass':'ok','fail':'fail','error':'error'}.get(chosen[0]['correctness'], 'unknown')
                result['error_summary'] = chosen[0]['error_summary']
                result['error_stage'] = chosen[0]['error_stage']
            execution = result['contexts']['execution']
            if execution.get('diagnostics'):
                diag = json.loads(execution['diagnostics'])
                diag['shapes'] = {k:v for k,v in diag.get('shapes',{}).items() if k == str(shape_index)}
                execution['diagnostics'] = json.dumps(diag)
            if execution.get('telemetry'):
                telemetry = json.loads(execution['telemetry'])
                telemetry['shapes'] = {k:v for k,v in telemetry.get('shapes',{}).items() if k == str(shape_index)}
                execution['telemetry'] = json.dumps(telemetry)
        return result

    def case_page(self, *, limit=50, after=0, seed=None, status=None, user_id=None, q=None):
        """Group comparable runs, retaining every attempt in backend history.

        Filters select current shape cases, retaining passing CPU peers and backend
        history in each row. Cursor combines the group high-water mark and shape.
        """
        if type(limit) is not int or not 1 <= limit <= 200 or type(after) is not int or not 0 <= after < 2**63:
            raise ValueError('invalid pagination')
        with reading(self.path) as conn:
            conn.execute('BEGIN')
            from ..reporting.cases import rollup
            cte, values = rollup(seed, status, user_id, q)
            groups = conn.execute(cte + 'SELECT * FROM cases WHERE cursor<? ORDER BY cursor DESC LIMIT ?',
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
                # List pages carry only previews. Full logs/timing payloads are
                # fetched on detail navigation, not repeated for every shape.
                item['execution'] = {k:v for k,v in item['execution'].items() if k not in {'diagnostics','telemetry'}}
                index = group['shape_index']
                item['shape_index'] = index
                item['case_id'] += ':' + str(index)
                variants = item['metadata'].get('shape_variants', [])
                item['sample'] = variants[index] if 0 <= index < len(variants) else {}
                item['shape_mode'] = 'static' if item['execution'].get('static_only') else 'dynamic'
                for row in rows:
                    backend = display_backend(row['backend'])
                    member = {key:row[key] for key in ('id','status','created_at','total_ns','duration_scope','error_stage','error_summary')}
                    member['has_diagnostics'] = bool(json.loads(row['execution']).get('diagnostics'))
                    member['device'] = json.loads(row['execution']).get('device')
                    member['stages'] = [dict(r) for r in conn.execute('SELECT shape_index,stage,duration_ns FROM stage_metrics WHERE attempt_id=? ORDER BY shape_index,stage',(row['id'],))]
                    member['shapes'] = [dict(r) for r in conn.execute('SELECT * FROM shapes WHERE attempt_id=? ORDER BY shape_index',(row['id'],))]
                    member['shapes'] = [s for s in member['shapes'] if s['shape_index'] == index]
                    if not member['shapes'] and index >= 0:
                        continue
                    member['stages'] = [m for m in member['stages'] if m['shape_index'] == index]
                    # Profiler values are nested measurements, not additional
                    # elapsed phases. Never add kernel/compile time twice.
                    elapsed = [m['duration_ns'] for m in member['stages'] if m['stage'] not in {'kernel_mean','compile'}]
                    member['total_ns'] = sum(elapsed) if elapsed else None
                    member['duration_scope'] = 'shape_execution'
                    if member['shapes']:
                        shape = member['shapes'][0]
                        member['status'] = {'pass':'ok','fail':'fail','error':'error'}.get(shape['correctness'], 'unknown')
                        member['error_stage'] = shape['error_stage']
                        member['error_summary'] = shape['error_summary']
                    from ..failure_details import core_error
                    diagnostics = json.loads(json.loads(row['execution']).get('diagnostics', '{}'))
                    member['core_error'] = core_error(diagnostics, index, member['error_summary'])
                    member['compile_ns'] = next((m['duration_ns'] for m in member['stages'] if m['stage']=='compile' and m['shape_index']==-1),None)
                    member['kernel_ns'] = [{'shape_index':m['shape_index'],'duration_ns':m['duration_ns']} for m in member['stages'] if m['stage']=='kernel_mean'] if backend!='cpu' else []
                    from ..reporting.outcomes import outcomes
                    member.update(outcomes(member, json.loads(row['evaluation'])))
                    item['backends'].setdefault(backend, []).append(member)
                items.append(item)
        return {'items':items,'next_cursor':groups[limit-1]['cursor'] if len(groups)>limit else None}
