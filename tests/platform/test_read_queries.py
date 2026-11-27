import json

import pytest

from autofusex.server.read_queries import response, snapshot
from test_platform import env


def test_response_snapshot_cache_and_invalidation(env):
    store, _, _ = env
    calls = []
    def compute():
        calls.append(1)
        return len(calls)
    def read():
        return json.loads(response(store, 'test', {}, compute))['data']
    assert read() == read() == 1
    # Another connection must share the same cache key, not its data_version.
    with store.connect() as other:
        previous = store._local.reader
        store._local.reader = other
        try:
            assert read() == 1
        finally:
            store._local.reader = previous
    with store.write() as conn:
        conn.execute("UPDATE users SET display_name='changed'")
    assert read() == 2
    with pytest.raises(RuntimeError):
        with store.write() as conn:
            conn.execute("UPDATE users SET display_name='rolled back'")
            raise RuntimeError()
    assert read() == 2
    with store.write() as conn:
        conn.execute("UPDATE users SET role=role")
    assert read() == 2
    with store.write() as conn:
        conn.execute("INSERT INTO content.contents VALUES(1,?, 'raw', 1, ?, 0)", (b'hash', b'x'))
    assert read() == 3
    with store.write() as conn:
        conn.execute("UPDATE content.contents SET data=? WHERE content_id=1", (b'y',))
    assert read() == 4
    assert not store.reader().in_transaction


def test_nested_snapshot_and_failure_release(env):
    store, _, _ = env
    conn = store.reader()
    with pytest.raises(ValueError):
        with snapshot(conn):
            with snapshot(conn):
                assert conn.in_transaction
            assert conn.in_transaction
            raise ValueError()
    assert not conn.in_transaction
