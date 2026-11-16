"""Bounded retest control plane, separate from immutable result facts.

All ownership checks and lease transitions occur in the same write transaction.
This module has no model dependencies and does not execute submitted commands.
"""
from contextlib import closing, contextmanager, nullcontext
import json
import sqlite3
import time
import uuid

from ...client.storage.repository import _connect
from ...client.storage.records import canonical, digest
from ...client.contracts import Seed, EvaluationConfig
from ...client.backends import BACKEND_NAMES

CLAIM_LIMIT = 8
BATCH_LIMIT = 200
QUEUE_LIMIT = 1000
LEASE_SECONDS = 120
MAX_GENERATIONS = 3


def initialize(conn):
    ddl = '''
    CREATE TABLE IF NOT EXISTS retest_clients_v1 (
      id TEXT PRIMARY KEY REFERENCES sources(id), user_id TEXT NOT NULL REFERENCES users(id),
      machine_id TEXT NOT NULL, name TEXT NOT NULL, capabilities TEXT NOT NULL, seen REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS retest_batches_v1 (
      id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), client_id TEXT NOT NULL,
      request_key TEXT NOT NULL, request_hash TEXT NOT NULL, created REAL NOT NULL,
      UNIQUE(user_id,request_key));
    CREATE TABLE IF NOT EXISTS retest_tasks_v1 (
      sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
      batch_id TEXT NOT NULL REFERENCES retest_batches_v1(id), spec TEXT NOT NULL,
      repetition INTEGER NOT NULL, state TEXT NOT NULL DEFAULT 'queued',
      generation INTEGER NOT NULL DEFAULT 0, lease_id TEXT UNIQUE, expires REAL,
      attempt_id TEXT REFERENCES attempts(id), updated REAL NOT NULL);
    CREATE INDEX IF NOT EXISTS retest_tasks_queue ON retest_tasks_v1(state,sequence);
    CREATE INDEX IF NOT EXISTS retest_tasks_batch ON retest_tasks_v1(batch_id,sequence);
    CREATE TABLE IF NOT EXISTS retest_targets_v1 (
      task_id TEXT NOT NULL REFERENCES retest_tasks_v1(id),
      attempt_id TEXT NOT NULL REFERENCES attempts(id), shape_index INTEGER NOT NULL,
      selected INTEGER NOT NULL, PRIMARY KEY(task_id,attempt_id,shape_index));
    CREATE INDEX IF NOT EXISTS retest_targets_attempt ON retest_targets_v1(attempt_id,task_id);
    CREATE TABLE IF NOT EXISTS retest_result_annotations_v1 (
      task_id TEXT NOT NULL REFERENCES retest_tasks_v1(id), generation INTEGER NOT NULL,
      attempt_id TEXT NOT NULL REFERENCES attempts(id), shape_index INTEGER NOT NULL,
      summary_id TEXT CHECK(summary_id IS NULL OR length(summary_id)<=256),
      PRIMARY KEY(task_id,generation,attempt_id,shape_index));
    CREATE TABLE IF NOT EXISTS retest_lease_history_v1 (
      lease_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES retest_tasks_v1(id),
      generation INTEGER NOT NULL, created REAL NOT NULL);
    CREATE INDEX IF NOT EXISTS retest_lease_task ON retest_lease_history_v1(task_id,generation);
    CREATE TABLE IF NOT EXISTS retest_completions_v1 (
      lease_id TEXT PRIMARY KEY REFERENCES retest_lease_history_v1(lease_id),
      attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(id), accepted INTEGER NOT NULL,
      reason TEXT NOT NULL, created REAL NOT NULL);
    '''
    # Do not implicitly commit an ingestion or migration transaction.
    for statement in ddl.split(';'):
        if statement.strip():
            conn.execute(statement)


@contextmanager
def transaction(path):
    with closing(_connect(path, writable=True)) as conn, conn:
        conn.row_factory = sqlite3.Row
        conn.execute('BEGIN IMMEDIATE')
        yield conn


def identity(value):
    if not isinstance(value, str):
        raise ValueError('UUID required')
    return str(uuid.UUID(value))


def positive(value, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError('integer outside limit')
    return value


def _client(conn, user, client):
    row = conn.execute('SELECT * FROM retest_clients_v1 WHERE id=? AND user_id=?', (client, user)).fetchone()
    if row is None:
        raise PermissionError('client not owned')
    return row


def register(path, user, payload):
    if not isinstance(payload, dict) or set(payload) != {'client_id', 'machine_id', 'name', 'backends'}:
        raise ValueError('invalid registration')
    client = identity(payload['client_id'])
    machine, name, backends = payload['machine_id'], payload['name'], payload['backends']
    if (not isinstance(machine, str) or not 1 <= len(machine) <= 128 or
            not isinstance(name, str) or not 1 <= len(name) <= 80 or
            not isinstance(backends, list) or not backends or len(backends) > 16 or
            any(not isinstance(b, str) or b not in BACKEND_NAMES for b in backends)):
        raise ValueError('invalid capabilities')
    with transaction(path) as conn:
        source = conn.execute('SELECT user_id FROM sources WHERE id=?', (client,)).fetchone()
        if source is None:
            conn.execute('INSERT INTO sources(id,user_id) VALUES (?,?)',(client,user))
        elif source[0] != user:
            raise PermissionError('source belongs to another identity')
        old = conn.execute('SELECT machine_id FROM retest_clients_v1 WHERE id=?', (client,)).fetchone()
        if old and old[0] != machine:
            raise PermissionError('client identity moved to another machine')
        conn.execute('INSERT INTO retest_clients_v1 VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET '
                     'name=excluded.name,capabilities=excluded.capabilities,seen=excluded.seen',
                     (client, user, machine, name, canonical(sorted(set(backends))), time.time()))
    return {'client_id': client}


def clients(path, user):
    with closing(_connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM retest_clients_v1 WHERE user_id=? ORDER BY seen DESC LIMIT 100', (user,)).fetchall()
        return {'items': [dict(id=r['id'], name=r['name'], machine_id=r['machine_id'],
                               backends=json.loads(r['capabilities']), online=time.time()-r['seen'] < LEASE_SECONDS)
                          for r in rows]}


def _spec(conn, attempt):
    row = conn.execute('''SELECT a.*,e.payload AS execution,v.payload AS evaluation
        FROM attempts a JOIN contexts e ON e.id=a.execution_id
        JOIN contexts v ON v.id=a.evaluation_id WHERE a.id=?''', (attempt,)).fetchone()
    if row is None:
        raise LookupError('attempt not found')
    ex, ev = json.loads(row['execution']), json.loads(row['evaluation'])
    Seed.parse(row['seed'])
    if row['backend'] != 'af-mlir' or 'static_only' not in ex or not ev:
        raise ValueError('original replay configuration unavailable')
    if type(ex['static_only']) is not bool:
        raise ValueError('invalid original execution mode')
    if set(ev) != {'correctness','performance','warmup_steps','measured_steps'}:
        raise ValueError('original evaluation configuration unavailable')
    if (any(type(ev[k]) is not bool for k in ('correctness','performance')) or
            any(type(ev[k]) is not int for k in ('warmup_steps','measured_steps'))):
        raise ValueError('invalid original evaluation')
    EvaluationConfig(**ev)
    # Preserve only replay semantics, never commands, local paths or diagnostics.
    execution = {k: ex[k] for k in ('static_only',) if k in ex}
    return dict(seed=row['seed'], workload_id=row['workload_id'], oracle_id=row['oracle_id'],
                backend=row['backend'], execution=execution, evaluation=ev)


def create(path, user, payload, *, preview=False, connection=None):
    if not isinstance(payload, dict) or set(payload) != {'client_id', 'request_key', 'targets', 'repetitions'}:
        raise ValueError('invalid batch')
    client, key = identity(payload['client_id']), identity(payload['request_key'])
    repetitions = positive(payload['repetitions'], 10)
    targets = payload['targets']
    if not isinstance(targets, list) or not 1 <= len(targets) <= BATCH_LIMIT:
        raise ValueError('bounded selection required')
    request_hash = digest(payload)
    with (transaction(path) if connection is None else nullcontext(connection)) as conn:
        existing = conn.execute('SELECT id,request_hash FROM retest_batches_v1 WHERE user_id=? AND request_key=?', (user,key)).fetchone()
        if existing:
            if existing['request_hash'] != request_hash:
                raise ValueError('idempotency key reused with different request')
            return {'batch_id': existing['id'], 'reused': True}
        worker = _client(conn, user, client)
        if time.time() - worker['seen'] >= LEASE_SECONDS:
            raise ValueError('client offline')
        groups = {}
        for target in targets:
            if not isinstance(target, dict) or set(target) != {'attempt_id','shape_index'}:
                raise ValueError('invalid target')
            attempt, shape = identity(target['attempt_id']), target['shape_index']
            if type(shape) is not int or not -1 <= shape < 1024:
                raise ValueError('invalid shape')
            spec = _spec(conn, attempt)
            if spec['backend'] not in json.loads(worker['capabilities']):
                raise ValueError('backend not supported by client')
            indices = [r[0] for r in conn.execute('SELECT shape_index FROM shapes WHERE attempt_id=?', (attempt,))]
            if shape not in (indices or [-1]):
                raise ValueError('shape not found')
            group = groups.setdefault(canonical(spec), {})
            for index in indices or [-1]:
                group[(attempt,index)] = group.get((attempt,index),False) or index == shape
            if len(group) > 4096:
                raise ValueError('too many covered Shape targets')
        count = len(groups)*repetitions
        pending = conn.execute("SELECT count(*) FROM retest_tasks_v1 t JOIN retest_batches_v1 b ON b.id=t.batch_id WHERE b.user_id=? AND t.state IN ('queued','leased','running','cancel_requested')", (user,)).fetchone()[0]
        if count + pending > QUEUE_LIMIT:
            raise ValueError('queue quota exceeded')
        if preview:
            return {'task_count':count,'selected_count':len(targets)}
        batch, now = str(uuid.uuid4()), time.time()
        conn.execute('INSERT INTO retest_batches_v1 VALUES (?,?,?,?,?,?)', (batch,user,client,key,request_hash,now))
        for spec, associations in groups.items():
            for repetition in range(repetitions):
                task = str(uuid.uuid4())
                conn.execute('INSERT INTO retest_tasks_v1(id,batch_id,spec,repetition,updated) VALUES (?,?,?,?,?)',
                             (task,batch,spec,repetition,now))
                conn.executemany('INSERT INTO retest_targets_v1 VALUES (?,?,?,?)',
                                 [(task,attempt,shape,int(selected)) for (attempt,shape),selected in associations.items()])
    return {'batch_id': batch, 'task_count': count, 'selected_count': len(targets), 'reused': False}


def claim(path, user, payload):
    if not isinstance(payload, dict) or set(payload) != {'client_id','limit'}:
        raise ValueError('invalid claim')
    client, limit = identity(payload['client_id']), positive(payload['limit'], CLAIM_LIMIT)
    now = time.time()
    with transaction(path) as conn:
        _client(conn,user,client)
        conn.execute('UPDATE retest_clients_v1 SET seen=? WHERE id=?', (now,client))
        scope = 'SELECT id FROM retest_batches_v1 WHERE user_id=? AND client_id=?'
        conn.execute(f"UPDATE retest_tasks_v1 SET state=CASE WHEN state='cancel_requested' THEN 'cancelled' WHEN generation>=? THEN 'interrupted' ELSE 'queued' END, updated=? WHERE batch_id IN ({scope}) AND state IN ('leased','running','cancel_requested') AND expires<=?", (MAX_GENERATIONS,now,user,client,now))
        active = conn.execute(f"SELECT count(*) FROM retest_tasks_v1 WHERE batch_id IN ({scope}) AND state IN ('leased','running','cancel_requested')", (user,client)).fetchone()[0]
        rows = conn.execute(f"SELECT * FROM retest_tasks_v1 WHERE batch_id IN ({scope}) AND state='queued' ORDER BY sequence LIMIT ?", (user,client,min(limit,max(0,CLAIM_LIMIT-active)))).fetchall()
        items = []
        for row in rows:
            lease = str(uuid.uuid4())
            conn.execute("UPDATE retest_tasks_v1 SET state='leased',generation=generation+1,lease_id=?,expires=?,updated=? WHERE id=?", (lease,now+LEASE_SECONDS,now,row['id']))
            conn.execute('INSERT INTO retest_lease_history_v1 VALUES (?,?,?,?)', (lease,row['id'],row['generation']+1,now))
            items.append(dict(task_id=row['id'], lease_id=lease, generation=row['generation']+1,
                              expires=now+LEASE_SECONDS, spec=json.loads(row['spec']), repetition=row['repetition']))
        return {'items':items}


def renew(path, user, payload):
    if not isinstance(payload, dict) or set(payload) != {'client_id','lease_id','state'}:
        raise ValueError('invalid lease event')
    client, lease = identity(payload['client_id']), identity(payload['lease_id'])
    state = payload['state']
    if state not in {'leased','running','cancelled','interrupted'}:
        raise ValueError('invalid state')
    now = time.time()
    with transaction(path) as conn:
        _client(conn,user,client)
        conn.execute('UPDATE retest_clients_v1 SET seen=? WHERE id=?',(now,client))
        row = conn.execute('''SELECT t.* FROM retest_tasks_v1 t JOIN retest_batches_v1 b ON b.id=t.batch_id
            WHERE t.lease_id=? AND b.user_id=? AND b.client_id=?''', (lease,user,client)).fetchone()
        if row is None:
            raise PermissionError('lease not owned')
        if row['state'] in {'completed','cancelled','interrupted'}:
            return {'state':row['state'], 'expires':row['expires']}
        if row['expires'] <= now or row['state'] == 'queued':
            raise ValueError('lease expired')
        if state == 'cancelled' and row['state'] != 'cancel_requested':
            raise ValueError('cancellation not requested')
        if row['state'] == 'cancel_requested' and state != 'cancelled':
            state = 'cancel_requested'
        elif row['state'] == 'running' and state == 'leased':
            state = 'running'
        conn.execute('UPDATE retest_tasks_v1 SET state=?,expires=?,updated=? WHERE id=?', (state,now+LEASE_SECONDS,now,row['id']))
        return {'state':state, 'expires':now+LEASE_SECONDS}


def cancel(path, user, batch):
    batch = identity(batch)
    with transaction(path) as conn:
        if conn.execute('SELECT 1 FROM retest_batches_v1 WHERE id=? AND user_id=?', (batch,user)).fetchone() is None:
            raise PermissionError('batch not owned')
        conn.execute("UPDATE retest_tasks_v1 SET state=CASE WHEN state='queued' THEN 'cancelled' ELSE 'cancel_requested' END,updated=? WHERE batch_id=? AND state IN ('queued','leased','running')", (time.time(),batch))
    return {'batch_id':batch, 'cancel_requested':True}


def batches(path, user, before=2**63-1):
    if type(before) is not int or not 0 < before <= 2**63-1:
        raise ValueError('invalid cursor')
    with closing(_connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT rowid AS cursor,* FROM retest_batches_v1 WHERE user_id=? AND rowid<? ORDER BY rowid DESC LIMIT 50', (user,before)).fetchall()
        result=[]
        for row in rows:
            item={k:row[k] for k in ('id','client_id','created','cursor')}
            item['states']={r[0]:r[1] for r in conn.execute('SELECT state,count(*) FROM retest_tasks_v1 WHERE batch_id=? GROUP BY state',(row['id'],))}
            result.append(item)
        return {'items':result, 'next_cursor':result[-1]['cursor'] if len(result)==50 else None}


def detail(path, user, batch, after=0):
    """Paginated owner-only task progress; never expose lease capabilities."""
    batch = identity(batch)
    if type(after) is not int or not 0 <= after < 2**63:
        raise ValueError('invalid cursor')
    with closing(_connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        if conn.execute('SELECT 1 FROM retest_batches_v1 WHERE id=? AND user_id=?', (batch,user)).fetchone() is None:
            raise PermissionError('batch not owned')
        rows = conn.execute('SELECT * FROM retest_tasks_v1 WHERE batch_id=? AND sequence>? ORDER BY sequence LIMIT 51', (batch,after)).fetchall()
        items=[]
        for row in rows[:50]:
            item={k:row[k] for k in ('id','sequence','repetition','state','generation','attempt_id','updated')}
            item['spec']=json.loads(row['spec'])
            item['targets']=[dict(r) for r in conn.execute('SELECT attempt_id,shape_index,selected FROM retest_targets_v1 WHERE task_id=? ORDER BY attempt_id,shape_index LIMIT 4097', (row['id'],))]
            if len(item['targets']) > 4096:
                raise ValueError('target detail exceeds limit')
            item['summaries']=[dict(r) for r in conn.execute('SELECT attempt_id,shape_index,summary_id FROM retest_result_annotations_v1 WHERE task_id=? ORDER BY generation,shape_index LIMIT 4096', (row['id'],))]
            items.append(item)
        return {'batch_id':batch,'items':items,'next_cursor':items[-1]['sequence'] if len(rows)>50 else None}


def _publishable(row, shape, execution):
    """Use existing structured failure attribution; facility errors stay Unknown."""
    if shape and shape['correctness'] in {'pass','fail'}:
        return True
    from ..reporting.failure_origin import sql_status
    stage = shape['error_stage'] if shape else row['error_stage']
    summary = shape['error_summary'] if shape else row['error_summary']
    status = sql_status(row['backend'],stage,summary,execution.get('diagnostics'),
                        shape['shape_index'] if shape else -1)
    return status in {'error','gen_failed'} and stage not in {'worker','generation','environment','check'}


def complete(path, user, payload):
    """Publish only uploaded, same-source facts bound to this lease's execution.

    Result arrival and control arrival may be reordered. Missing facts reject the
    request for retry; lost responses replay the same outcome without mutations.
    """
    if not isinstance(payload,dict) or set(payload) != {'client_id','lease_id','attempt_id'}:
        raise ValueError('invalid completion')
    client, lease, attempt = (identity(payload[k]) for k in ('client_id','lease_id','attempt_id'))
    now = time.time()
    with transaction(path) as conn:
        _client(conn,user,client)
        task = conn.execute('''SELECT t.*,h.generation AS submitted_generation FROM retest_lease_history_v1 h
            JOIN retest_tasks_v1 t ON t.id=h.task_id JOIN retest_batches_v1 b ON b.id=t.batch_id
            WHERE h.lease_id=? AND b.user_id=? AND b.client_id=?''',(lease,user,client)).fetchone()
        if task is None:
            raise PermissionError('lease not owned')
        prior = conn.execute('SELECT * FROM retest_completions_v1 WHERE lease_id=?',(lease,)).fetchone()
        if prior:
            if prior['attempt_id'] != attempt:
                raise ValueError('completion changed')
            return dict(accepted=bool(prior['accepted']),reason=prior['reason'],task_id=task['id'],reused=True)
        row = conn.execute('''SELECT a.*,f.source_id,f.user_id,e.payload AS execution,v.payload AS evaluation
            FROM attempts a JOIN attempt_facts f ON f.attempt_id=a.id
            JOIN contexts e ON e.id=a.execution_id JOIN contexts v ON v.id=a.evaluation_id WHERE a.id=?''',(attempt,)).fetchone()
        if row is None:
            raise LookupError('result not uploaded yet')
        if row['source_id'] != client or row['user_id'] != user:
            raise PermissionError('result not from assigned source')
        ex, ev, spec = json.loads(row['execution']),json.loads(row['evaluation']),json.loads(task['spec'])
        # A historical result cannot be retroactively presented as this execution.
        if ex.get('batch_id') != lease:
            raise ValueError('result not bound to lease')
        if (row['seed'] != spec['seed'] or row['backend'] != spec['backend'] or
                ex.get('static_only') != spec['execution']['static_only'] or ev != spec['evaluation']):
            raise ValueError('replay configuration mismatch')
        reason = 'published'
        if task['lease_id'] != lease or task['expires'] <= now:
            reason = 'late_result'
        elif task['state'] not in {'leased','running'}:
            reason = 'cancelled_or_interrupted'
        elif row['workload_id'] != spec['workload_id'] or row['oracle_id'] != spec['oracle_id']:
            reason = 'reproduction_mismatch'
        shapes = conn.execute('SELECT * FROM shapes WHERE attempt_id=? ORDER BY shape_index',(attempt,)).fetchall()
        eligible = [s for s in shapes if _publishable(row,s,ex)]
        shared = not shapes and row['status'] in {'error','unsupported'} and _publishable(row,None,ex)
        if reason == 'published' and not eligible and not shared:
            reason = 'unverified'
        accepted = reason == 'published'
        conn.execute('INSERT INTO retest_completions_v1 VALUES (?,?,?,?,?)',(lease,attempt,int(accepted),reason,now))
        for index in [s['shape_index'] for s in shapes] or [-1]:
            conn.execute('INSERT INTO retest_result_annotations_v1 VALUES (?,?,?,?,NULL)',
                         (task['id'],task['submitted_generation'],attempt,index))
        if task['lease_id'] == lease and task['state'] in {'leased','running'}:
            conn.execute("UPDATE retest_tasks_v1 SET state=?,attempt_id=?,updated=? WHERE id=?",
                         ('completed' if accepted else 'interrupted',attempt,now,task['id']))
        return dict(accepted=accepted,reason=reason,task_id=task['id'],reused=False)


def history(path, attempt, after=2**63-1):
    """Public evidence only; no lease capabilities or private machine controls."""
    attempt = identity(attempt)
    if type(after) is not int or not 0 < after <= 2**63-1:
        raise ValueError('invalid cursor')
    with closing(_connect(path)) as conn:
        conn.row_factory=sqlite3.Row
        source = conn.execute('''SELECT f.group_id FROM retest_completions_v1 c
            JOIN retest_lease_history_v1 h ON h.lease_id=c.lease_id JOIN retest_targets_v1 x ON x.task_id=h.task_id
            JOIN attempt_facts f ON f.attempt_id=x.attempt_id WHERE c.attempt_id=? ORDER BY f.group_id LIMIT 1''',(attempt,)).fetchone()
        if source is None:
            source = conn.execute('SELECT group_id FROM attempt_facts WHERE attempt_id=?',(attempt,)).fetchone()
        if source is None:
            raise LookupError('attempt not found')
        rows=conn.execute('''SELECT t.sequence,t.id,t.state,t.repetition,c.attempt_id,c.accepted,c.reason,c.created,
                u.display_name AS executor FROM retest_tasks_v1 t
            JOIN retest_batches_v1 b ON b.id=t.batch_id JOIN users u ON u.id=b.user_id
            LEFT JOIN retest_completions_v1 c ON c.lease_id=t.lease_id
            WHERE t.sequence<? AND EXISTS (SELECT 1 FROM retest_targets_v1 x
                JOIN attempt_facts f ON f.attempt_id=x.attempt_id WHERE x.task_id=t.id AND f.group_id=?)
            ORDER BY t.sequence DESC LIMIT 51''',(after,source[0])).fetchall()
        items=[]
        for row in rows[:50]:
            item=dict(row)
            item['baseline']=[dict(r) for r in conn.execute('''SELECT x.attempt_id,x.shape_index,s.correctness,s.error_stage
                FROM retest_targets_v1 x LEFT JOIN shapes s ON s.attempt_id=x.attempt_id AND s.shape_index=x.shape_index
                WHERE x.task_id=? ORDER BY x.attempt_id,x.shape_index LIMIT 4096''',(row['id'],))]
            item['late_results']=[dict(r) for r in conn.execute('''SELECT c.attempt_id,c.reason,c.created,h.generation
                FROM retest_completions_v1 c JOIN retest_lease_history_v1 h ON h.lease_id=c.lease_id
                WHERE h.task_id=? AND c.lease_id!=(SELECT lease_id FROM retest_tasks_v1 WHERE id=?)
                ORDER BY h.generation''',(row['id'],row['id']))]
            if row['attempt_id']:
                environment=conn.execute('SELECT c.payload FROM contexts c JOIN attempts a ON a.environment_id=c.id WHERE a.id=?',(row['attempt_id'],)).fetchone()
                env=json.loads(environment[0]) if environment else {}
                item['environment']={k:env.get(k) for k in ('cann','torch','torch_npu','autofuse_binary','device_model')}
            else:
                item['environment']={}
            item['shapes']=[dict(r) for r in conn.execute('''SELECT s.shape_index,s.correctness,s.performance,s.error_stage,
                s.error_summary,n.summary_id FROM shapes s LEFT JOIN retest_result_annotations_v1 n
                ON n.attempt_id=s.attempt_id AND n.shape_index=s.shape_index AND n.task_id=? WHERE s.attempt_id=?''',
                (row['id'],row['attempt_id']))] if row['attempt_id'] else []
            items.append(item)
        return {'items':items,'next_cursor':items[-1]['sequence'] if len(rows)>50 else None}
