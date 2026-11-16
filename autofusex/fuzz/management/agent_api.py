"""Small owner-authenticated HTTP facade over durable managed operations."""
import copy
import json
import sqlite3
import time
from urllib.parse import parse_qs, urlsplit

from .central import Conflict
from .config import validate
from ..central.repository import authenticate
from ..central.maintenance import MaintenanceBlocked

TERMINAL = {'completed', 'failed', 'stopped', 'cancelled', 'interrupted', 'needs_attention'}


def initialize(db):
    db.executescript('''
      CREATE TABLE IF NOT EXISTS agent_run_observations(
        device_id TEXT NOT NULL,run_id TEXT NOT NULL,state TEXT NOT NULL,
        result_count INTEGER,observed_at REAL NOT NULL,PRIMARY KEY(device_id,run_id));
    ''')


def observe(db, device_id, status):
    """Persist bounded device reports; absence is never evidence of completion."""
    runs = status.get('runs', [])
    if not isinstance(runs, list) or len(runs) > 100:
        raise ValueError('invalid run status')
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get('id'), str) or len(run['id']) > 80:
            raise ValueError('invalid run identity')
        state = run.get('state')
        if state not in TERMINAL | {'preparing', 'running', 'paused', 'pausing', 'draining', 'cancelling'}:
            raise ValueError('invalid run state')
        count = run.get('result_count')
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError('invalid result count')
        db.execute('INSERT OR REPLACE INTO agent_run_observations VALUES (?,?,?,?,?)',
                   (device_id, run['id'], state, count, time.time()))


class AgentAPI:
    def __init__(self, store, allowlist):
        self.store = store
        self.allowlist = allowlist
        with store.connect() as db:
            initialize(db)
            db.execute('CREATE INDEX IF NOT EXISTS agent_result_batch ON attempt_provenance(batch_id,attempt_id)')

    def operation(self, user, identity):
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM managed_operations WHERE id=? AND kind='task.submit'", (identity,)).fetchone()
            if not row:
                raise LookupError('task not found')
            self.store.owner(db, user, row['device_id'])
        return dict(row)

    def task(self, user, identity):
        operation = self.operation(user, identity)
        result = json.loads(operation['result']) if operation['result'] else {}
        run_id = result.get('run_id')
        state = operation['state']
        state = 'starting' if state == 'succeeded' else state
        observed_at = None
        expected = None
        received = 0
        with self.store.connect() as db:
            device = self.store.owner(db, user, operation['device_id'])
            if run_id:
                report = db.execute('SELECT * FROM agent_run_observations WHERE device_id=? AND run_id=?',
                                    (device['id'], run_id)).fetchone()
                if report:
                    state, expected, observed_at = report['state'], report['result_count'], report['observed_at']
                received = db.execute('SELECT count(*) FROM attempt_facts f JOIN owners o ON o.attempt_id=f.attempt_id '
                                      'JOIN managed_sources s ON s.source_id=f.source_id '
                                      'WHERE o.user_id=? AND s.device_id=? AND f.batch_id GLOB ?',
                                      (user, device['id'], run_id + ':[0-9]*')).fetchone()[0]
        terminal = state in TERMINAL | {'expired'}
        return dict(id=identity, device_id=operation['device_id'], state=state, terminal=terminal,
                    observed_at=observed_at, stale=not observed_at or time.time()-observed_at > 60,
                    results=dict(received=received, expected=expected,
                                 complete=terminal and expected is not None and received == expected))

    def submit(self, user, body):
        if set(body) != {'device_id', 'spec', 'request_key'}:
            raise ValueError('device_id, spec and request_key required')
        self.allowlist.browser(user)
        self.allowlist.client(user, body['device_id'])
        spec = validate(body['spec'])
        if spec['kind'] != 'fuzz_task':
            raise ValueError('fuzz_task required')
        # Remote workflows always persist and upload their results.
        spec['results'] = dict(save=True, upload=True)
        operation = self.store.submit(user, body['device_id'], 'task.submit', spec, body['request_key'])
        return self.task(user, operation['id'])

    def stop(self, user, identity, body):
        if set(body) != {'request_key'}:
            raise ValueError('request_key required')
        operation = self.operation(user, identity)
        self.allowlist.browser(user)
        self.allowlist.client(user, operation['device_id'])
        if not isinstance(body['request_key'], str) or not 1 <= len(body['request_key']) <= 128:
            raise ValueError('invalid request key')
        result = json.loads(operation['result']) if operation['result'] else {}
        cancelled = False
        with self.store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT state FROM managed_operations WHERE id=?', (identity,)).fetchone()
            if row['state'] == 'queued':
                # Undelivered submission can be cancelled atomically with poll.
                db.execute("UPDATE managed_operations SET state='cancelled',result='{}',updated=? WHERE id=?",
                           (time.time(), identity))
                self.store.audit(db, user, operation['device_id'], 'agent.stop', dict(id=identity))
                cancelled = True
        if cancelled:
            return self.task(user, identity)
        if self.task(user, identity)['terminal']:
            return self.task(user, identity)
        if not result.get('run_id'):
            raise Conflict('submission is in flight; retry stop with the same key')
        self.store.submit(user, operation['device_id'], 'task.action',
                          dict(run_id=result['run_id'], action='stop'), body['request_key'])
        return dict(self.task(user, identity), stop_requested=True)

    def replay(self, user, identity, body):
        if set(body) != {'seeds', 'request_key'}:
            raise ValueError('seeds and request_key required')
        original = self.operation(user, identity)
        spec = copy.deepcopy(json.loads(original['payload']))
        spec['workload'] = dict(mode='replay', seeds=body['seeds'])
        spec.pop('termination', None)
        return self.submit(user, dict(device_id=original['device_id'], spec=spec, request_key=body['request_key']))

    def results(self, user, identity, params):
        if set(params) - {'after', 'limit', 'status', 'attempt'}:
            raise ValueError('unknown results parameter')
        limit, after = int(params.get('limit', 50)), int(params.get('after', 0))
        if not 1 <= limit <= 200 or not 0 <= after < 2**63:
            raise ValueError('invalid pagination')
        status = params.get('status')
        if status is not None and status not in {'ok', 'fail', 'error', 'unsupported'}:
            raise ValueError('invalid status')
        operation = self.operation(user, identity)
        result = json.loads(operation['result']) if operation['result'] else {}
        task = self.task(user, identity)
        if not result.get('run_id'):
            if 'attempt' in params:
                raise LookupError('attempt not found')
            return dict(task_id=identity, **task['results'], items=[], next_cursor=None)
        with self.store.connect() as db:
            sql = ('SELECT a.rowid,a.id,a.seed,a.backend,a.status FROM attempts a '
                   'JOIN attempt_facts f ON f.attempt_id=a.id JOIN owners o ON o.attempt_id=a.id '
                   'JOIN managed_sources s ON s.source_id=f.source_id '
                   'WHERE o.user_id=? AND s.device_id=? AND f.batch_id GLOB ? AND a.rowid>?')
            values = [user, operation['device_id'], result['run_id'] + ':[0-9]*', after]
            if status is not None:
                sql += ' AND a.status=?'; values.append(status)
            if 'attempt' in params:
                sql += ' AND a.id=?'; values.append(params['attempt'])
            rows = [dict(row) for row in db.execute(sql + ' ORDER BY a.rowid LIMIT ?', (*values, limit+1))]
        if 'attempt' in params:
            if not rows:
                raise LookupError('attempt not found')
            from ..storage.queries import ResultQueries
            return dict(task_id=identity, attempt=ResultQueries(self.store.path).detail(rows[0]['id']))
        from ..contracts import Seed
        for row in rows:
            row['seed_short'] = Seed.parse(row['seed']).short
        return dict(task_id=identity, **task['results'], unit='attempt', items=rows[:limit],
                    next_cursor=rows[limit-1]['rowid'] if len(rows)>limit else None)

    def handle(self, handler, method):
        url = urlsplit(handler.path)
        if not url.path.startswith('/api/agent/'):
            return False
        def send(code, data=None, error=None, headers=None):
            handler.send(code, dict(schema_version=1, ok=error is None,
                                   **({'data': data} if error is None else {'error': error})), headers=headers)
        try:
            if len(handler.path) > 2048:
                raise ValueError('URL too long')
            auth = handler.headers.get('Authorization', '')
            if not auth.startswith('Bearer ') or len(auth) > 256:
                raise PermissionError('user Bearer token required')
            with self.store.connect() as db:
                user = authenticate(db, auth[7:])
            params = parse_qs(url.query, keep_blank_values=True, max_num_fields=4)
            if any(len(value) != 1 for value in params.values()):
                raise ValueError('duplicate query parameter')
            params = {key:value[0] for key,value in params.items()}
            body = {}
            if method == 'POST':
                if params or handler.headers.get('Transfer-Encoding') or handler.headers.get('Content-Type','').split(';')[0] != 'application/json':
                    raise ValueError('JSON body required without query parameters')
                length = int(handler.headers.get('Content-Length', '0'))
                if not 1 <= length <= 65536:
                    raise ValueError('body must be between 1 and 65536 bytes')
                raw = handler.rfile.read(length)
                if len(raw) != length:
                    raise ValueError('incomplete body')
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError('object required')
            parts = url.path.removeprefix('/api/agent/').split('/')
            code = 200
            if parts == ['experience'] and method == 'GET' and not params:
                from ..central import repairs
                with self.store.connect() as db:
                    repairs.initialize(db)
                    data=dict(items=[dict(r) for r in db.execute('SELECT s.*,r.issue_key FROM repair_summaries_v1 s JOIN repairs_v1 r ON r.id=s.repair_id WHERE r.user_id=? ORDER BY s.created DESC LIMIT 100',(user,))])
            elif parts == ['repairs', 'clients'] and method == 'GET' and not params:
                from ..central import retests
                data = retests.clients(self.store.path, user)
            elif parts == ['repairs', 'claim'] and method == 'POST':
                from ..central import repairs
                data = repairs.mutate(self.store.path, user, 'claim', body, allowlist=self.allowlist)
            elif len(parts) == 2 and parts[0] == 'repairs' and method == 'GET' and not params:
                from ..central import repairs
                data = repairs.get(self.store.path, user, parts[1])
            elif len(parts) == 3 and parts[0] == 'repairs' and method == 'POST':
                from ..central import repairs
                data = repairs.mutate(self.store.path, user, parts[2], body, parts[1], self.allowlist)
            elif parts == ['devices'] and method == 'GET' and not params:
                data = dict(items=[dict(id=d['id'], name=d['name'], online=time.time()-d['seen']<=60,
                                        seen=d['seen'], environments=d['status'].get('environments', []))
                                   for d in self.store.devices(user) if d['active']])
            elif parts == ['tasks'] and method == 'POST':
                data = self.submit(user, body); code = 202
            elif len(parts) == 2 and parts[0] == 'tasks' and method == 'GET' and not params:
                data = self.task(user, parts[1])
            elif len(parts) == 3 and parts[0] == 'tasks':
                identity, action = parts[1:]
                if action == 'results' and method == 'GET': data = self.results(user, identity, params)
                elif action == 'stop' and method == 'POST': data = self.stop(user, identity, body); code = 202
                elif action == 'replay' and method == 'POST': data = self.replay(user, identity, body); code = 202
                else: raise LookupError('route not found')
            else:
                raise LookupError('route not found')
            send(code, data=data)
        except MaintenanceBlocked:
            send(503, error=dict(code='maintenance_write_blocked', message='writes temporarily restricted'), headers={'Retry-After':'60'})
        except PermissionError:
            send(401, error=dict(code='unauthorized', message='invalid credential or resource ownership'))
        except Conflict as exc:
            send(409, error=dict(code='conflict', message=str(exc)))
        except LookupError:
            send(404, error=dict(code='not_found', message='resource not found'))
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
            send(400, error=dict(code='invalid_request', message='invalid request fields or configuration'))
        except sqlite3.Error:
            send(503, error=dict(code='unavailable', message='database temporarily unavailable'), headers={'Retry-After':'5'})
        return True
