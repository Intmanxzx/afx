"""Bounded campaign feeder; no preallocation of an infinite task queue."""
import json
import secrets
import uuid
from dataclasses import asdict
from ..contracts import canonical_json


def initialize(ledger, configs, evaluation, count):
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError('campaign count must be positive or unlimited')
    with ledger._connection() as conn:
        if not configs or len(configs) > conn.execute('SELECT max_active FROM queue_config').fetchone()[0]:
            raise ValueError('campaign backend matrix exceeds task capacity')
        conn.execute('CREATE TABLE IF NOT EXISTS campaign (id INTEGER PRIMARY KEY, payload TEXT NOT NULL, remaining INTEGER, stopped INTEGER NOT NULL)')
        conn.execute('INSERT OR REPLACE INTO campaign VALUES (1,?,?,0)',
                     (canonical_json({'configs':[asdict(c) for c in configs],'evaluation':asdict(evaluation)}),count))


def stop(ledger):
    with ledger._connection() as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='campaign'").fetchone():
            conn.execute('UPDATE campaign SET stopped=1')
        conn.execute("UPDATE tasks SET state=CASE WHEN state='running' THEN 'stop_requested' ELSE 'cancelled' END,reason='user_stop',revision=revision+1 WHERE state IN ('pending','running','interrupted')")


def refill(ledger):
    with ledger._connection() as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='campaign'").fetchone():return False
        row=conn.execute('SELECT payload,remaining,stopped FROM campaign WHERE id=1').fetchone()
        if not row or row[2] or row[1]==0:return False
        if conn.execute("SELECT 1 FROM tasks WHERE state IN ('pending','running','stop_requested')").fetchone():return False
        config=json.loads(row[0]);seed=str(2*2**128+secrets.randbelow(36**9))
        active = conn.execute("SELECT count(*) FROM tasks WHERE state NOT IN ('completed','cancelled')").fetchone()[0]
        capacity = conn.execute('SELECT max_active FROM queue_config').fetchone()[0]
        if active + len(config['configs']) > capacity:
            return False
        for execution in config['configs']:
            conn.execute('INSERT INTO tasks(id,seed,execution,evaluation,state,revision,reason) VALUES (?,?,?,?,?,0,?)',
                         (str(uuid.uuid4()),seed,canonical_json(execution),canonical_json(config['evaluation']),'pending',''))
        if row[1] is not None:conn.execute('UPDATE campaign SET remaining=remaining-1 WHERE id=1')
        return True
