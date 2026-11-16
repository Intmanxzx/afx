"""Late binding for campaign tasks only. Explicit replay placement stays pinned.

Device candidates live in the private task ledger, not in ExecutionConfig: a
claimed execution always records its actual logical device. The backend still
holds the existing physical UUID lease across its combined invocation.
"""
import json


def initialize(conn):
    conn.execute('CREATE TABLE IF NOT EXISTS task_devices (task_id TEXT PRIMARY KEY, devices TEXT NOT NULL)')


def candidates(ledger, task):
    with ledger._connection() as conn:
        initialize(conn)
        row = conn.execute('SELECT devices FROM task_devices WHERE task_id=?', (task['id'],)).fetchone()
    return json.loads(row[0]) if row else [json.loads(task['execution'])['device']]


def claim_on(ledger, task, device):
    # Under supervisor ownership; cancel may race but cannot revive a task.
    with ledger._connection() as conn:
        row = conn.execute("SELECT execution FROM tasks WHERE id=? AND state='pending'", (task['id'],)).fetchone()
        if row is None:
            return None
        execution = json.loads(row[0])
        execution['device'] = device
        conn.execute('UPDATE tasks SET execution=? WHERE id=?', (json.dumps(execution, sort_keys=True, separators=(',', ':')), task['id']))
    return ledger.claim(task['id'])


def admit(conn, identities):
    from ..storage.lifecycle import stamp
    observed = json.dumps(stamp())
    conn.execute('CREATE TABLE IF NOT EXISTS task_admission (task_id TEXT PRIMARY KEY, observed TEXT NOT NULL)')
    conn.executemany('INSERT OR IGNORE INTO task_admission VALUES (?,?)', ((identity, observed) for identity in identities))


def admission(ledger, identity):
    with ledger._connection() as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='task_admission'").fetchone():
            return None
        row = conn.execute('SELECT observed FROM task_admission WHERE task_id=?', (identity,)).fetchone()
        return json.loads(row[0]) if row else None
