"""Lifecycle v1, outside immutable result hashes; independent fact replication.

At most seven facts per attempt. Missing post-commit facts stay missing (a crash
cannot be repaired by inventing a persisted timestamp). No remote clock math.
"""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import socket
import sqlite3
import time
import uuid
import warnings

EVENTS = {'admitted', 'ready', 'dispatched', 'started', 'finished', 'persisted', 'uploaded_ack'}
# Linux boot ID is shared by supervisor, workers and uploader. Else a process
# domain intentionally prevents cross-process subtraction.
try:
    _boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
except OSError:
    _boot = str(uuid.uuid4())
DOMAIN = hashlib.sha256((socket.gethostname() + _boot).encode()).hexdigest()


def stamp():
    return {'domain': DOMAIN, 'monotonic_ns': time.monotonic_ns(),
            'wall_time': datetime.now(timezone.utc).isoformat()}


def initialize(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS lifecycle_v1 (
        attempt_id TEXT NOT NULL, event TEXT NOT NULL, domain TEXT NOT NULL,
        monotonic_ns INTEGER NOT NULL, wall_time TEXT NOT NULL,
        PRIMARY KEY(attempt_id,event))''')


def append(path, identity, event, observed=None):
    if event not in EVENTS:
        raise ValueError('invalid lifecycle event')
    identity = str(uuid.UUID(identity))
    observed = observed or stamp()
    # Supplemental observations must not turn an already committed execution
    # into a failed save. A warning is preferable to a fabricated later time.
    try:
        from autofusex.execution.storage.repository import _connect
        with closing(_connect(path, writable=True)) as conn, conn:
            initialize(conn)
            cursor = conn.execute('INSERT OR IGNORE INTO lifecycle_v1 VALUES (?,?,?,?,?)',
                         (identity, event, observed['domain'], observed['monotonic_ns'], observed['wall_time']))
            if cursor.rowcount and conn.execute('SELECT 1 FROM attempts WHERE id=?',(identity,)).fetchone():
                from autofusex.execution.storage.stage_facts import initialize as init_facts, publish
                init_facts(conn)
                for name,domain,mono,wall in conn.execute('SELECT event,domain,monotonic_ns,wall_time FROM lifecycle_v1 WHERE attempt_id=?',(identity,)):
                    publish(conn,'lifecycle',identity+':'+name,{'domain':domain,'monotonic_ns':mono,'wall_time':wall})
    except sqlite3.Error as exc:
        warnings.warn('lifecycle observation not saved: ' + type(exc).__name__, RuntimeWarning)


def read(conn, identity):
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='lifecycle_v1'").fetchone():
        return []
    rows = conn.execute('SELECT event,domain,monotonic_ns,wall_time FROM lifecycle_v1 WHERE attempt_id=? ORDER BY monotonic_ns', (identity,))
    return [dict(zip(('event', 'domain', 'monotonic_ns', 'wall_time'), row)) for row in rows]


def elapsed(events, begin, end):
    by_name = {row['event']: row for row in events}
    a, b = by_name.get(begin), by_name.get(end)
    if a and b and a['domain'] == b['domain'] and b['monotonic_ns'] >= a['monotonic_ns']:
        return b['monotonic_ns'] - a['monotonic_ns']
    return None
