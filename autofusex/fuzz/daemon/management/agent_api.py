"""Bounded local persistence for device-reported run status.

Used by the owner-scoped device management control plane (central.py,
ControlStore.poll) to remember run state reported by a device between polls.
Absence of a report is never evidence of completion.
"""
import time

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
