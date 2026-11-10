"""Transactional lightweight results, bounded reads and consistent safe snapshots."""
from contextlib import closing
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import tempfile
import uuid

from .records import canonical, digest, normalize
from .schema import APPLICATION_ID, SCHEMA_VERSION, DDL, TABLE_COLUMNS


def _check(conn):
    if conn.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID:
        raise ValueError('not an Autofuse fuzz result database')
    if conn.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION:
        raise ValueError('unsupported result schema; refusing implicit migration')
    for table, expected in TABLE_COLUMNS.items():
        kind = conn.execute('SELECT type FROM sqlite_master WHERE name=?', (table,)).fetchone()
        if kind is None or kind[0] != 'table':
            raise ValueError('invalid result schema table')
        columns = tuple(row[1] for row in conn.execute(f'PRAGMA table_info({table})'))
        if columns != expected:
            raise ValueError('invalid result schema columns')


def _initialize(conn):
    conn.executescript(DDL)
    conn.execute(f'PRAGMA application_id={APPLICATION_ID}')
    conn.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
    from ..reporting.index import initialize
    initialize(conn)
    conn.commit()


def _connect(path, writable=False):
    path = Path(path).expanduser().resolve()
    if writable:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Publish a fully initialized file atomically, so concurrent first writers
        # cannot observe the empty/half-created schema. Do not overwrite files.
        if not path.exists():
            fd, name = tempfile.mkstemp(prefix='.fuzz-init-', dir=path.parent)
            os.close(fd)
            temporary = Path(name)
            try:
                with closing(sqlite3.connect(str(temporary))) as initial:
                    _initialize(initial)
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    pass  # Another writer won; validate its database below.
            finally:
                temporary.unlink(missing_ok=True)
        new = False
        conn = sqlite3.connect(str(path), timeout=10)
    else:
        new = False
        conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=10)
    try:
        conn.execute('PRAGMA trusted_schema=OFF')
        conn.execute('PRAGMA foreign_keys=ON')
        if new:
            _initialize(conn)
        _check(conn)
        if writable:
            from .sqlite_utils import enable_wal
            enable_wal(conn)
        else:
            conn.execute('PRAGMA query_only=ON')
        return conn
    except Exception:
        conn.close()
        raise


class ResultRepository:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()

    def save(self, result, *, environment=None, attempt_id=None):
        record = normalize(result, environment or {})
        attempt_id = str(uuid.UUID(attempt_id)) if attempt_id else str(uuid.uuid4())
        result_hash = digest(record)
        with closing(_connect(self.path, writable=True)) as conn, conn:
            # IMMEDIATE prevents same-ID concurrent writers racing the idempotency check.
            conn.execute('BEGIN IMMEDIATE')
            existing = conn.execute('SELECT result_hash FROM attempts WHERE id=?', (attempt_id,)).fetchone()
            if existing:
                if existing[0] != result_hash:
                    raise ValueError('attempt ID already exists with different content')
                return attempt_id
            context_ids = {}
            for kind, payload in record['contexts'].items():
                identity = digest({'kind': kind, 'payload': payload})
                conn.execute('INSERT OR IGNORE INTO contexts VALUES (?,?,?)', (identity,kind,canonical(payload)))
                context_ids[kind] = identity
            row = dict(record['attempt'], id=attempt_id, created_at=datetime.now(timezone.utc).isoformat(),
                       result_hash=result_hash, **{kind+'_id': value for kind,value in context_ids.items()})
            columns = TABLE_COLUMNS['attempts']
            conn.execute(f"INSERT INTO attempts VALUES ({','.join('?' for _ in columns)})", tuple(row[c] for c in columns))
            columns = TABLE_COLUMNS['shapes']
            for shape in record['shapes']:
                row = dict(shape, attempt_id=attempt_id)
                conn.execute(f"INSERT INTO shapes VALUES ({','.join('?' for _ in columns)})", tuple(row[c] for c in columns))
            from ..reporting.index import index_attempt
            index_attempt(conn, attempt_id)
            from ..sync.outbox import enqueue
            enqueue(conn, attempt_id)
        return attempt_id

    def list(self, *, limit=50, after=None, seed=None):
        # Queries reuse the validated connection; import lazily to avoid a cycle.
        from .queries import ResultQueries
        return ResultQueries(self.path).page(limit=limit, after=0 if after is None else after,
                                             seed=seed)

    def detail(self, attempt_id):
        from .queries import ResultQueries
        try:
            return ResultQueries(self.path).detail(attempt_id)
        except LookupError as exc:
            raise ValueError('attempt not found') from exc

    def snapshot(self, destination):
        """Read transaction includes committed WAL content; export owned tables only.

        Not a raw file copy. Extra tables (credentials, outbox, leases) are excluded.
        Uses a same-directory temporary file and atomic no-overwrite hard link.
        """
        destination = Path(destination).expanduser().resolve()
        if destination == self.path or destination.exists():
            raise ValueError('snapshot destination must be a new file')
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.fuzz-snapshot-', dir=destination.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            with closing(_connect(self.path)) as source, closing(sqlite3.connect(str(temporary))) as target:
                source.execute('BEGIN')
                _initialize(target)
                for table, columns in TABLE_COLUMNS.items():
                    reader = source.execute(f"SELECT {','.join(columns)} FROM {table} ORDER BY rowid")
                    while True:
                        rows = reader.fetchmany(200)
                        if not rows:
                            break
                        target.executemany(f"INSERT INTO {table} VALUES ({','.join('?' for _ in columns)})", rows)
                if source.execute("SELECT 1 FROM sqlite_master WHERE name='report_schema'").fetchone():
                    from ..reporting.index import initialize
                    initialize(target)
                    for table in ('case_facts','attempt_facts','stage_metrics','case_dimensions','report_devices','report_search'):
                        if not source.execute('SELECT 1 FROM sqlite_master WHERE name=?',(table,)).fetchone(): continue
                        rows=source.execute('SELECT * FROM '+table)
                        for values in rows:
                            target.execute('INSERT INTO '+table+' VALUES ('+','.join('?' for _ in values)+')',tuple(values))
                target.commit()
                if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok' or target.execute('PRAGMA foreign_key_check').fetchone():
                    raise ValueError('snapshot integrity validation failed')
                source.rollback()
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            os.link(temporary, destination)  # Fails safely rather than replacing an existing file.
        finally:
            temporary.unlink(missing_ok=True)
        return str(destination)
