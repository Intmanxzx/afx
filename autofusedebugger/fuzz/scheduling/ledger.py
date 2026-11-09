"""Durable bounded task queue. Claim tokens fence stale completions, NOT devices.

No wall-clock lease stealing: a supervisor must terminate/reap an old worker
before interrupting its claim. Explicit resume creates a new attempt identity.
This module never imports or executes a backend.
"""
from contextlib import contextmanager
from dataclasses import asdict
import os
from pathlib import Path
import sqlite3
import tempfile
import uuid

from ..contracts import Seed, ExecutionConfig, EvaluationConfig, canonical_json

APPLICATION_ID = 0x41465451
VERSION = 2
DDL = '''
CREATE TABLE queue_config (id INTEGER PRIMARY KEY CHECK(id=1), max_active INTEGER NOT NULL CHECK(max_active>0));
CREATE TABLE tasks (
 id TEXT PRIMARY KEY,
 seed TEXT NOT NULL,
 execution TEXT NOT NULL,
 evaluation TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('pending','running','stop_requested','interrupted','completed','cancelled')),
 claim TEXT,
 attempt_id TEXT,
 revision INTEGER NOT NULL DEFAULT 0,
 reason TEXT NOT NULL DEFAULT ''
);
CREATE INDEX task_queue ON tasks(state);
'''


class TaskLedger:
    def __init__(self, path, *, max_active=None):
        if max_active is not None and (type(max_active) is not int or max_active < 1):
            raise ValueError('positive active task limit required')
        self.path = Path(path).expanduser().resolve()
        self.max_active = max_active

    @contextmanager
    def _connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            fd, name = tempfile.mkstemp(prefix='.fuzz-tasks-', dir=self.path.parent)
            os.close(fd)
            try:
                initial = sqlite3.connect(name)
                try:
                    initial.executescript(DDL)
                    initial.execute('INSERT INTO queue_config VALUES (1,?)', (self.max_active or 1000,))
                    initial.execute(f'PRAGMA application_id={APPLICATION_ID}')
                    initial.execute(f'PRAGMA user_version={VERSION}')
                    initial.commit()
                finally:
                    initial.close()
                try:
                    os.link(name, self.path)
                except FileExistsError:
                    pass
            finally:
                Path(name).unlink(missing_ok=True)
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            conn.execute('PRAGMA trusted_schema=OFF')
            if (conn.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID or
                    conn.execute('PRAGMA user_version').fetchone()[0] != VERSION):
                raise ValueError('not a supported task ledger')
            columns = tuple(row[1] for row in conn.execute('PRAGMA table_info(tasks)'))
            if columns != ('id','seed','execution','evaluation','state','claim','attempt_id','revision','reason'):
                raise ValueError('invalid task ledger schema')
            capacity = conn.execute('SELECT max_active FROM queue_config WHERE id=1').fetchone()
            if capacity is None or type(capacity[0]) is not int or capacity[0] < 1:
                raise ValueError('invalid persisted task capacity')
            if self.max_active is not None and self.max_active != capacity[0]:
                raise ValueError('task capacity differs from persisted configuration')
            conn.execute('PRAGMA journal_mode=WAL')
            conn.row_factory = sqlite3.Row
            with conn:
                conn.execute('BEGIN IMMEDIATE')
                yield conn
        finally:
            conn.close()

    def enqueue(self, seed, execution, evaluation, *, task_id=None):
        return self.enqueue_many([(seed, execution, evaluation, task_id)])[0]

    def enqueue_many(self, specifications):
        """All-or-nothing bounded batch; canonicalize every spec before SQL writes."""
        prepared = {}
        identities = []
        for seed, execution, evaluation, task_id in specifications:
            if len(identities) >= 10000:
                raise ValueError('batch exceeds 10000 tasks')
            if not isinstance(execution, ExecutionConfig) or not isinstance(evaluation, EvaluationConfig):
                raise TypeError('task requires execution/evaluation contracts')
            value = (str(Seed.parse(seed)), canonical_json(asdict(execution)), canonical_json(asdict(evaluation)))
            identity = str(uuid.UUID(task_id)) if task_id else str(uuid.uuid4())
            if identity in prepared and prepared[identity] != value:
                raise ValueError('task ID conflicts within batch')
            prepared[identity] = value
            identities.append(identity)
        with self._connection() as conn:
            additions = []
            for identity, value in prepared.items():
                row = conn.execute('SELECT seed,execution,evaluation FROM tasks WHERE id=?', (identity,)).fetchone()
                if row is not None:
                    if tuple(row) != value:
                        raise ValueError('task ID conflicts with immutable specification')
                else:
                    additions.append((identity, *value))
            count = conn.execute("SELECT count(*) FROM tasks WHERE state NOT IN ('completed','cancelled')").fetchone()[0]
            capacity = conn.execute('SELECT max_active FROM queue_config WHERE id=1').fetchone()[0]
            if count + len(additions) > capacity:
                raise ValueError('active task budget exhausted; entire batch rejected')
            conn.executemany("INSERT INTO tasks(id,seed,execution,evaluation,state) VALUES (?,?,?,?,'pending')", additions)
        return identities

    def claim(self):
        """Atomically claim one case/backend task; None means no pending work."""
        with self._connection() as conn:
            row = conn.execute("SELECT id FROM tasks WHERE state='pending' ORDER BY rowid LIMIT 1").fetchone()
            if row is None:
                return None
            conn.execute("UPDATE tasks SET state='running',reason='',claim=?,attempt_id=?,revision=revision+1 WHERE id=?",
                         (str(uuid.uuid4()), str(uuid.uuid4()), row['id']))
            return dict(conn.execute('SELECT * FROM tasks WHERE id=?', (row['id'],)).fetchone())

    def get(self, identity):
        with self._connection() as conn:
            row = conn.execute('SELECT * FROM tasks WHERE id=?', (identity,)).fetchone()
            if row is None:
                raise ValueError('task not found')
            return dict(row)

    def stop(self, identity):
        with self._connection() as conn:
            row = conn.execute('SELECT state FROM tasks WHERE id=?', (identity,)).fetchone()
            if row is None:
                raise ValueError('task not found')
            state = {'pending': 'cancelled', 'interrupted': 'cancelled',
                     'running': 'stop_requested'}.get(row['state'])
            if state:
                conn.execute("UPDATE tasks SET state=?,reason='user_stop',revision=revision+1 WHERE id=?", (state, identity))

    def _finish_claim(self, identity, claim, state, reason=''):
        if not isinstance(reason, str) or len(reason) > 500:
            raise ValueError('reason must be short text')
        if not isinstance(claim, str) or not claim:
            raise ValueError('claim token required')
        with self._connection() as conn:
            count = conn.execute("UPDATE tasks SET state=?,reason=?,claim=NULL,revision=revision+1 "
                                 "WHERE id=? AND claim=? AND state IN ('running','stop_requested')",
                                 (state, reason, identity, claim)).rowcount
            if count != 1:
                raise ValueError('stale claim or invalid task transition')

    def complete(self, identity, claim):
        """Caller must durably save the claim's attempt result first.

        A stop request racing with a finished result may complete. A crash between
        result save and this acknowledgement needs supervisor reconciliation.
        """
        self._finish_claim(identity, claim, 'completed')

    def interrupt(self, identity, claim, reason='worker_lost'):
        """Supervisor-only: call AFTER worker death/reaping, not on timeout alone."""
        self._finish_claim(identity, claim, 'interrupted', reason)

    def resume(self, identity):
        with self._connection() as conn:
            row = conn.execute('SELECT state FROM tasks WHERE id=?', (identity,)).fetchone()
            if row is not None and row['state'] == 'cancelled':
                count = conn.execute("SELECT count(*) FROM tasks WHERE state NOT IN ('completed','cancelled')").fetchone()[0]
                if count >= conn.execute('SELECT max_active FROM queue_config WHERE id=1').fetchone()[0]:
                    raise ValueError('active task budget exhausted; apply backpressure')
            changed = conn.execute("UPDATE tasks SET state='pending',reason='',claim=NULL,attempt_id=NULL,revision=revision+1 "
                                   "WHERE id=? AND state IN ('interrupted','cancelled')", (identity,)).rowcount
            if changed != 1:
                raise ValueError('only interrupted or cancelled tasks can resume')

    def page(self, *, limit=100, after=0):
        if type(limit) is not int or not 1 <= limit <= 200 or type(after) is not int or not 0 <= after < 2**63:
            raise ValueError('invalid task pagination')
        with self._connection() as conn:
            rows = [dict(row) for row in conn.execute(
                'SELECT rowid,* FROM tasks WHERE rowid>? ORDER BY rowid LIMIT ?', (after, limit+1))]
        return {'items': rows[:limit], 'next_cursor': rows[limit-1]['rowid'] if len(rows)>limit else None}
