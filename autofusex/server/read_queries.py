"""Snapshot-safe public list responses using the shared bounded cache."""
import json
from contextlib import contextmanager


@contextmanager
def snapshot(conn):
    owner = not conn.in_transaction
    if owner:
        conn.execute('BEGIN')
    try:
        yield conn
    finally:
        if owner:
            conn.execute('ROLLBACK')


def response(store, kind, parameters, compute):
    # These lists are public and independent of the principal. Authentication
    # and admission still run before this function; never use it for user data.
    conn = store.reader()
    with snapshot(conn):
        revision = conn.execute('SELECT revision FROM read_revision WHERE id=1').fetchone()[0]
        # Restoring previously missing logs must invalidate discovery responses.
        content_version = conn.execute('SELECT revision FROM content.read_revision WHERE id=1').fetchone()[0]
        key = (kind, revision, content_version, tuple(sorted(parameters.items())))
        return store.response_cache.get(key, lambda: json.dumps(
            {'schema_version': 1, 'ok': True, 'data': compute()}, ensure_ascii=False,
            allow_nan=False, separators=(',', ':')).encode())
