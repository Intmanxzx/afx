"""Private durable operation journal. External installers are never blindly retried."""
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid
from contextlib import contextmanager
from .config import canonical, fingerprint

TERMINAL = {'succeeded', 'failed', 'cancelled', 'needs_attention'}


class Store:
    def __init__(self, home):
        self.home = Path(home).expanduser().resolve()
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.home / 'manager.sqlite'
        with self.connect() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS operations(id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL,
                hash TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, state TEXT NOT NULL,
                result TEXT, created REAL NOT NULL, updated REAL NOT NULL);
              CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, operation_id TEXT,
                time REAL NOT NULL, message TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS environments(id TEXT PRIMARY KEY, name TEXT NOT NULL,
                manifest TEXT NOT NULL, state TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, spec TEXT NOT NULL, state TEXT NOT NULL,
                pid INTEGER, result TEXT);
            ''')
        os.chmod(self.path, 0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, key, default=None):
        with self.connect() as db:
            row = db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
            return json.loads(row[0]) if row else default

    def put(self, key, value):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key, canonical(value)))

    def submit(self, kind, payload, request_key, operation_id=None):
        digest = fingerprint(dict(kind=kind, payload=payload))
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT * FROM operations WHERE request_key=?', (request_key,)).fetchone()
            if operation_id:
                existing_id = db.execute('SELECT * FROM operations WHERE id=?',(operation_id,)).fetchone()
                if existing_id and existing_id['hash'] != digest: raise ValueError('operation ID payload conflict')
                if existing_id: return self.operation(operation_id)
            if old:
                if old['hash'] != digest:
                    raise ValueError('idempotency key payload conflict')
                return self.operation(old['id'])
            identity = operation_id or str(uuid.uuid4())
            now = time.time()
            db.execute('INSERT INTO operations VALUES (?,?,?,?,?,?,?,?,?)',
                       (identity, request_key, digest, kind, canonical(payload), 'queued', None, now, now))
        return self.operation(identity)

    def operation(self, identity):
        with self.connect() as db:
            row = db.execute('SELECT * FROM operations WHERE id=?', (identity,)).fetchone()
        if not row:
            raise LookupError('operation not found')
        value = dict(row)
        for key in ('payload', 'result'):
            value[key] = json.loads(value[key]) if value[key] else None
        return value

    def pending(self):
        with self.connect() as db:
            rows = db.execute("SELECT id FROM operations WHERE state='queued' ORDER BY created LIMIT 1").fetchall()
        return [self.operation(row[0]) for row in rows]

    def state(self, identity, state, result=None):
        with self.connect() as db:
            db.execute('UPDATE operations SET state=?, result=?,updated=? WHERE id=?',
                       (state, canonical(result), time.time(), identity))

    def event(self, identity, message):
        # Only structured, non-secret progress labels belong here, not raw installer output.
        with self.connect() as db:
            db.execute('INSERT INTO events(operation_id,time,message) VALUES (?,?,?)',
                       (identity, time.time(), message[:1000]))
            db.execute('DELETE FROM events WHERE seq < (SELECT coalesce(max(seq),0)-10000 FROM events)')

    def recover(self):
        with self.connect() as db:
            db.execute("UPDATE operations SET state='needs_attention',result=? WHERE state='running'",
                       (canonical(dict(code='interrupted', message='Inspect external effects before retrying')),))

    def events(self, identity, after=0):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM events WHERE operation_id=? AND seq>? ORDER BY seq LIMIT 100', (identity, after))]
