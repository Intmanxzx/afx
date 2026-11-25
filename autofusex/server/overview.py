"""Overview computations shared by the API and dedicated read-only processes."""
import json

from autofusex.server import queries
from autofusex.server.store import Store


def version(conn):
    return tuple(conn.execute(
        "SELECT (SELECT value FROM meta WHERE key='ingest_seq'), "
        "(SELECT value FROM stats.stats_meta WHERE key='applied_seq'), "
        "(SELECT revision FROM platform_settings WHERE key='perf.tie_threshold')").fetchone())


def compute(store, key):
    kind, *args = key
    if kind == 'pipeline':
        return queries.pipeline(store, *args)
    if kind == 'backend':
        return queries.backend_stats(store, *args)
    if kind == 'perf':
        return queries.perf_pair(store, *args)
    if kind == 'dims':
        return queries.dims(store)
    if kind == 'matrix':
        names, *filters = args
        dimensions = queries.dims(store)
        return {'dims': dimensions, 'backends': names,
                'cells': {b: {d: queries.backend_stats(store, b, d, *filters)
                              for d in dimensions} for b in names}}
    raise ValueError('unknown overview query')


def encode(store, key):
    return json.dumps({'schema_version': 1, 'ok': True, 'data': compute(store, key)},
                      ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


_worker_store = None


def initialize_worker(data_dir):
    global _worker_store
    _worker_store = Store.open_readonly(data_dir)
    _worker_store.reader()


def read_overview(key, expected_version):
    """Never put a response from a different snapshot under a cached version."""
    conn = _worker_store.reader()
    conn.execute('BEGIN')
    try:
        if version(conn) != expected_version:
            return None
        return encode(_worker_store, key)
    finally:
        conn.execute('ROLLBACK')


def worker_ready():
    return _worker_store.readonly
