"""Explicit offline operation-ledger migration, with required no-overwrite backup.

Results schema is independent and remains unchanged. Never migrate a live queue.
"""
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import tempfile

from .ledger import APPLICATION_ID, VERSION


def upgrade_queue(path, backup):
    if os.name != 'posix':
        raise ValueError('queue upgrade requires POSIX ownership locking')
    import fcntl
    path, backup = Path(path).expanduser().resolve(), Path(backup).expanduser().resolve()
    fd = os.open(path, os.O_RDONLY)
    name = None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('queue still owned; stop supervisor/workers before upgrade') from None
        with closing(sqlite3.connect(path.as_uri()+'?mode=rw', uri=True, timeout=10)) as conn:
            conn.execute('PRAGMA trusted_schema=OFF')
            if conn.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID:
                raise ValueError('not a task ledger')
            version = conn.execute('PRAGMA user_version').fetchone()[0]
            if version != 1:
                raise ValueError('only task schema 1 -> 2 upgrade is supported')
            if tuple(row[1] for row in conn.execute('PRAGMA table_info(tasks)')) != (
                    'id','seed','execution','evaluation','state','claim','attempt_id','revision'):
                raise ValueError('invalid task schema 1')
            # Freeze writers while taking a backup via a separate reader. WAL readers
            # see the exact committed pre-migration state; IMMEDIATE blocks enqueues.
            conn.execute('BEGIN IMMEDIATE')
            try:
                backup.parent.mkdir(parents=True, exist_ok=True)
                tempfd, name = tempfile.mkstemp(prefix='.fuzz-queue-backup-', dir=backup.parent)
                os.close(tempfd)
                with closing(sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)) as reader, closing(sqlite3.connect(name)) as target:
                    reader.backup(target)
                    if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                        raise ValueError('queue backup integrity failure')
                with open(name, 'rb') as stream:
                    os.fsync(stream.fileno())
                os.link(name, backup)
                conn.execute("ALTER TABLE tasks ADD COLUMN reason TEXT NOT NULL DEFAULT ''")
                conn.execute(f'PRAGMA user_version={VERSION}')
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return {'schema_version': VERSION, 'backup': str(backup)}
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)
        os.close(fd)
