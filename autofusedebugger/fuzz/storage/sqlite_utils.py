"""Small shared SQLite connection operations, no result schema dependencies."""
import sqlite3
import time


def enable_wal(conn, timeout=10):
    """Concurrent first-open journal transitions can bypass SQLite busy_timeout.

    Retry only BUSY/LOCKED, outside any transaction, within a finite deadline.
    Existing WAL connections avoid an unnecessary exclusive transition.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            if conn.execute('PRAGMA journal_mode').fetchone()[0].lower() != 'wal':
                if conn.execute('PRAGMA journal_mode=WAL').fetchone()[0].lower() != 'wal':
                    raise ValueError('WAL mode unavailable')
            return
        except sqlite3.OperationalError as exc:
            code = getattr(exc, 'sqlite_errorcode', 0) & 0xff
            if code not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
