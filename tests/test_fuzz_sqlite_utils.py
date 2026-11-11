import sqlite3
import pytest
from autofusedebugger.fuzz.storage.sqlite_utils import enable_wal


def test_wal_retries_transient_busy_only():
    class Connection:
        count = 0
        def execute(self, sql):
            self.count += 1
            if self.count == 1:
                error = sqlite3.OperationalError('database is locked')
                error.sqlite_errorcode = sqlite3.SQLITE_BUSY
                raise error
            return self
        def fetchone(self):
            return ('wal',)
    conn = Connection()
    enable_wal(conn)
    assert conn.count == 2


def test_wal_does_not_mask_non_busy_failure():
    class Connection:
        def execute(self, sql):
            raise sqlite3.OperationalError('bad database')
    with pytest.raises(sqlite3.OperationalError, match='bad database'):
        enable_wal(Connection())
