"""Durable refresh jobs; completion is committed atomically with the changes."""
import json
import threading
import uuid
from collections import Counter
from .store import now_ms
from .classification_rules import merge_builtins, plan
from .reclassification import apply_rule_changes


class RefreshJobs:
    def __init__(self, store):
        self.store = store
        self.lock = threading.Lock()
        self.thread = None
        self.phase = {}

    def list(self):
        return [self.get(r[0]) for r in self.store.reader().execute(
            'SELECT job_id FROM classification_jobs ORDER BY created_at DESC LIMIT 30')]

    def get(self, job_id):
        row = self.store.reader().execute('SELECT * FROM classification_jobs WHERE job_id=?', (job_id,)).fetchone()
        if row is None: raise LookupError('refresh job not found')
        result = dict(row)
        result.pop('rules')
        result['result'] = json.loads(result['result']) if result['result'] else None
        if job_id in self.phase: result['phase'] = self.phase[job_id]
        return result

    def submit(self, actor, request_key, rules):
        rules = json.dumps(merge_builtins(rules), sort_keys=True)
        with self.store.write() as conn:
            old = conn.execute('SELECT job_id,rules FROM classification_jobs WHERE actor=? AND request_key=?', (actor, request_key)).fetchone()
            if old:
                if old['rules'] != rules: raise ValueError('request key reused with different rules')
                job_id = old['job_id']
            else:
                if conn.execute("SELECT 1 FROM classification_jobs WHERE status IN ('queued','running')").fetchone():
                    raise ValueError('已有分类刷新任务，请在任务列表查看状态')
                job_id = str(uuid.uuid4())
                conn.execute('INSERT INTO classification_jobs(job_id,actor,request_key,rules,status,phase,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)',
                             (job_id, actor, request_key, rules, 'queued', 'queued', now_ms(), now_ms()))
        self.start()
        return self.get(job_id)

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive(): return
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self):
        # Queued/running rows survive process restarts. No edits can have been
        # committed without the completed status, so retry is safe.
        try:
            while True:
                with self.store.connect() as conn:
                    row = conn.execute("SELECT * FROM classification_jobs WHERE status IN ('queued','running') ORDER BY created_at LIMIT 1").fetchone()
                if row is None: return
                job_id = row['job_id']
                try:
                    self._update(job_id, status='running', phase='scanning')
                    changes = plan(self.store, json.loads(row['rules']), progress=lambda n,m: self._update(job_id, scanned=n, matched=m))
                    self._update(job_id, phase='applying', matched=len(changes))
                    result = json.dumps({'count':len(changes), 'batch_id':job_id,
                                         'rules':dict(Counter(c['rule_id'] for c in changes)), 'samples':changes[:50]})
                    def complete(conn):
                        conn.execute("UPDATE classification_jobs SET status='completed',phase='completed',result=?,updated_at=? WHERE job_id=?", (result,now_ms(),job_id))
                    apply_rule_changes(self.store, job_id, changes,
                                       progress=lambda phase: self.phase.update({job_id:phase}), complete=complete)
                except Exception as exc:
                    self._update(job_id, status='failed', phase='failed', error=str(exc)[:2000])
                finally:
                    self.phase.pop(job_id, None)
        finally:
            # Serialize worker exit with submit so a just-enqueued row cannot
            # be left without a worker.
            with self.lock:
                self.thread = None
                with self.store.connect() as conn:
                    pending = conn.execute("SELECT 1 FROM classification_jobs WHERE status IN ('queued','running')").fetchone()
                if pending:
                    self.thread = threading.Thread(target=self._run, daemon=True)
                    self.thread.start()

    def _update(self, job_id, **values):
        with self.store.write() as conn:
            values['updated_at'] = now_ms()
            conn.execute('UPDATE classification_jobs SET '+','.join(k+'=?' for k in values)+' WHERE job_id=?', (*values.values(),job_id))
