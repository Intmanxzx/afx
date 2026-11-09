"""Explicit trusted-result restore/merge, never SQL script or legacy summary import.

Source rows are revalidated against the lightweight record contract. Hashes detect
accidental corruption, not authenticity: only accept snapshots from trusted peers.
"""
from contextlib import closing
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import uuid

from .records import canonical, digest, normalize
from .repository import _connect, _initialize
from .schema import SCHEMA_VERSION, TABLE_COLUMNS


def _integrity(conn):
    if conn.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
        raise ValueError('result database integrity check failed')
    if conn.execute('PRAGMA foreign_key_check').fetchone():
        raise ValueError('result database has broken references')


def _context(conn, identity, kind):
    row = conn.execute('SELECT kind,payload FROM contexts WHERE id=?', (identity,)).fetchone()
    if row is None or row[0] != kind or not isinstance(row[1], str) or len(row[1]) > 16384:
        raise ValueError('invalid result context')
    payload = json.loads(row[1])
    if canonical(payload) != row[1] or digest({'kind': kind, 'payload': payload}) != identity:
        raise ValueError('context fingerprint mismatch')
    return payload


def _record(conn, row):
    """Validate stored bytes/fields rather than trusting a supplied result_hash."""
    try:
        if str(uuid.UUID(row['id'])) != row['id']:
            raise ValueError('noncanonical attempt ID')
        if not isinstance(row['created_at'], str) or len(row['created_at']) > 64:
            raise ValueError('invalid creation time')
        if datetime.fromisoformat(row['created_at']).tzinfo is None:
            raise ValueError('creation time must include timezone')
        contexts = {kind: _context(conn, row[kind+'_id'], kind)
                    for kind in ('execution', 'evaluation', 'environment')}
        shapes = [dict(r) for r in conn.execute(
            'SELECT * FROM shapes WHERE attempt_id=? ORDER BY rowid LIMIT 1025', (row['id'],))]
        raw = {key: row[key] for key in ('seed','workload_id','oracle_id','backend','status',
                                         'prepare_ns','error_stage','error_summary')}
        raw.update(execution=contexts['execution'], evaluation=contexts['evaluation'],
                   shape_results=shapes)
        record = normalize(raw, contexts['environment'])
        stored = {'attempt': {key: row[key] for key in record['attempt']},
                  'contexts': contexts,
                  'shapes': [{key: shape[key] for key in record['shapes'][i]}
                             for i, shape in enumerate(shapes)]}
        if record != stored or digest(record) != row['result_hash']:
            raise ValueError('result fingerprint or field contract mismatch')
        return contexts, shapes
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('invalid result fields') from exc


def _copy(source, target):
    """Stream bounded rows inside caller-owned read/write transactions."""
    added = skipped = 0
    for row in source.execute('SELECT * FROM attempts ORDER BY rowid'):
        contexts, shapes = _record(source, row)
        existing = target.execute('SELECT * FROM attempts WHERE id=?', (row['id'],)).fetchone()
        if existing is not None:
            _, existing_shapes = _record(target, existing)
            if dict(row) != dict(existing) or shapes != existing_shapes:
                raise ValueError('attempt ID conflict; entire import rolled back')
            skipped += 1
            continue
        for kind, payload in contexts.items():
            values = (row[kind+'_id'], kind, canonical(payload))
            existing_context = target.execute('SELECT * FROM contexts WHERE id=?', (values[0],)).fetchone()
            if existing_context is not None and tuple(existing_context) != values:
                raise ValueError('context ID conflict')
            target.execute('INSERT OR IGNORE INTO contexts VALUES (?,?,?)', values)
        for table, records in (('attempts', [row]), ('shapes', shapes)):
            columns = TABLE_COLUMNS[table]
            target.executemany(f"INSERT INTO {table} VALUES ({','.join('?' for _ in columns)})",
                               (tuple(record[key] for key in columns) for record in records))
        added += 1
    return {'added': added, 'skipped': skipped}


def check_database(path):
    """Full contract/integrity audit; no migration or database mutation."""
    with closing(_connect(path)) as source:
        source.row_factory = sqlite3.Row
        source.execute('BEGIN')
        _integrity(source)
        count = 0
        for row in source.execute('SELECT * FROM attempts ORDER BY rowid'):
            _record(source, row)
            count += 1
        return {'schema_version': SCHEMA_VERSION, 'attempts': count, 'integrity': 'ok',
                'upgrade': 'not_needed'}


def merge(source, destination):
    """All-or-nothing import into an EXISTING owned result database."""
    source, destination = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
    if source == destination or (destination.exists() and os.path.samefile(source, destination)):
        raise ValueError('source and destination must differ')
    # Existing only: restore is the explicit new-file operation. URI rw closes the
    # exists-check race without accidentally creating a blank destination.
    with closing(_connect(source)) as src, closing(sqlite3.connect(
            destination.as_uri()+'?mode=rw', uri=True, timeout=10)) as dst:
        from .repository import _check
        dst.execute('PRAGMA trusted_schema=OFF')
        dst.execute('PRAGMA foreign_keys=ON')
        _check(dst)
        # No triggers in owned tables: do not execute extra destination code.
        if dst.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND tbl_name IN ('contexts','attempts','shapes')").fetchone():
            raise ValueError('result-table triggers are not supported')
        src.row_factory = dst.row_factory = sqlite3.Row
        src.execute('BEGIN')
        _integrity(src)
        with dst:
            dst.execute('BEGIN IMMEDIATE')
            result = _copy(src, dst)
            _integrity(dst)
        return result


def restore(source, destination):
    """Validate and atomically publish a new whitelist-only DB; never overwrite."""
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise ValueError('restore destination must be a new file')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.fuzz-restore-', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with closing(sqlite3.connect(temporary)) as conn:
            _initialize(conn)
        result = merge(source, temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        return dict(result, database=str(destination))
    finally:
        temporary.unlink(missing_ok=True)
