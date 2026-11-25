import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from autofusex.server.read_cache import ReadCache
from autofusex.server import queries
from autofusex.server.store import put_content
from test_platform import env, http, body, item, upload


def test_concurrent_miss_computes_once_and_errors_can_retry():
    cache = ReadCache(max_entries=2, max_bytes=10, size=len)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def load():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return b'result'
    with ThreadPoolExecutor(4) as pool:
        first = pool.submit(cache.get, 'same', load)
        assert entered.wait(5)
        others = [pool.submit(cache.get, 'same', load) for _ in range(3)]
        release.set()
        assert [f.result() for f in [first, *others]] == [b'result'] * 4
    assert len(calls) == 1
    def fail():
        raise ValueError('bad query')
    with pytest.raises(ValueError):
        cache.get('fail', fail)
    assert cache.get('fail', lambda: b'ok') == b'ok'
    cache.get('big', lambda: b'x' * 11)
    assert cache.get('same', lambda: b'wrong') == b'result'
    cache.get('new', lambda: b'12345')
    assert cache.get('same', lambda: b'new') == b'new'


def test_summary_cache_avoids_blob_reads_and_tracks_evidence(env, monkeypatch):
    from autofusex.server import store as storage
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('cached-summary', status='runtime_fail',
        detail={'error_summary': 'ValueError: original failure'})]))
    assert queries.case_page(store)['groups'][0]['executions'][0]['log_summary'] == 'ValueError: original failure'
    original = storage.get_content
    def forbid(*args):
        raise AssertionError('warm summary must not read content')
    monkeypatch.setattr(storage, 'get_content', forbid)
    assert queries.case_page(store)['groups'][0]['executions'][0]['log_summary'] == 'ValueError: original failure'
    monkeypatch.setattr(storage, 'get_content', original)
    with store.write() as conn:
        identity = put_content(conn, json.dumps({'error_summary': 'TypeError: replaced evidence'}).encode())
        conn.execute('UPDATE executions SET detail_content_id=?', (identity,))
    assert queries.case_page(store)['groups'][0]['executions'][0]['log_summary'] == 'TypeError: replaced evidence'


def test_missing_blob_is_not_cached(env):
    from autofusex.server.plog_summary import attach
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('missing-summary', status='runtime_fail', log_summary='wrapper')]))
    with store.write() as conn:
        conn.execute('UPDATE executions SET detail_content_id=99999')
    row = {'exec_id': store.reader().execute('SELECT exec_id FROM executions').fetchone()[0]}
    attach(store.reader(), row, store.summary_cache)
    assert row['log_summary'] == 'wrapper'
    with store.write() as conn:
        identity = put_content(conn, json.dumps({'error_summary': 'ValueError: restored'}).encode())
        conn.execute('UPDATE content.contents SET content_id=99999 WHERE content_id=?', (identity,))
    attach(store.reader(), row, store.summary_cache)
    assert row['log_summary'] == 'ValueError: restored'


def test_overview_cache_invalidates_after_upload_and_separates_filters(http, monkeypatch):
    calls = []
    original = queries.backend_stats
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(queries, 'backend_stats', counted)
    path = '/api/overview/matrix?backends=af-mlir'
    first = http('GET', path)
    assert http('GET', path) == first
    initial = len(calls)
    assert initial == 1  # Empty fixture has only the all dimension.
    assert http('GET', '/api/overview/backend?backend=af-mlir&dim=all')[0] == 200
    assert len(calls) == initial + 1
    assert http('GET', path + '&q=stage:mlir')[0] == 200
    assert len(calls) == initial * 2 + 1
    code, result = http('POST', '/api/client/executions', body([item('cache-invalidation')]))
    assert code == 200 and result['data']['accepted'] == 1
    code, result = http('GET', path)
    assert code == 200
    assert result['data']['cells']['af-mlir']['all']['groups'][0]['total'] == 1
    assert len(calls) > initial * 2 + 1


def test_perf_cache_invalidates_after_threshold_change(env):
    from fastapi.testclient import TestClient
    from autofusex.server.server import create_app
    from autofusex.server.ingest import update_settings
    store, _, _ = env
    with TestClient(create_app(store)) as client:
        path = '/api/overview/perf?x=af-mlir&y=af-legacy'
        assert client.get(path).json()['data']['tie_threshold'] == .05
        update_settings(store, 'root', {'perf.tie_threshold': .2}, {'perf.tie_threshold': 1})
        assert client.get(path).json()['data']['tie_threshold'] == .2


def test_readonly_store_cannot_write_or_create_missing_data(env, tmp_path):
    from autofusex.server.store import Store
    store, _, _ = env
    reader = Store.open_readonly(store.dir)
    assert reader.reader().execute('PRAGMA query_only').fetchone()[0] == 1
    assert not hasattr(reader, 'secret')
    with pytest.raises(RuntimeError, match='read worker'):
        with reader.write():
            pass
    missing = tmp_path / 'missing'
    readonly = Store.open_readonly(missing)
    with pytest.raises(Exception):
        readonly.reader()
    assert not missing.exists()


def test_process_pool_matches_local_and_observes_new_uploads(env):
    from autofusex.server.server import PlatformServer
    from autofusex.server.overview import read_overview, version
    from autofusex.server.store import Store
    import urllib.request
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('pool-first')]))
    server = PlatformServer(store, ('127.0.0.1', 0), read_processes=2)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def get(path):
        with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}' + path) as response:
            return json.load(response)['data']
    try:
        key = ('matrix', ('af-mlir',), None, None, None, None, None)
        before = version(store.reader())
        out = get('/api/overview/matrix?backends=af-mlir')
        assert out['cells']['af-mlir']['all'] == queries.backend_stats(store, 'af-mlir')
        upload(ingestor, principal, body([item('pool-second', created=2000)], key='pool-2', run='pool-2'))
        # Workers must reject a changed snapshot rather than mislabel its cache.
        assert server._read_pool.submit(read_overview, key, before).result(timeout=10) is None
        assert get('/api/overview/matrix?backends=af-mlir')['cells']['af-mlir']['all'] == queries.backend_stats(store, 'af-mlir')
        for path in ['/api/overview/pipeline', '/api/overview/backend?backend=af-mlir',
                     '/api/overview/perf?x=af-mlir&y=af-legacy', '/api/overview/dims']:
            assert get(path) is not None
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as error:
            get('/api/overview/matrix?backends=af-mlir&q=invalid:value')
        assert error.value.code == 400
        assert json.load(error.value)['error']['code'] == 'invalid_query'
        # The API retains the sole write path; process readers remain read-only.
        readonly = Store.open_readonly(store.dir)
        assert readonly.reader().execute('SELECT count(*) FROM executions').fetchone()[0] == 2
    finally:
        server.server_close()
        thread.join(timeout=10)


def test_query_timeout_keeps_status_across_process_boundary():
    import pickle
    from autofusex.server.search import SearchError
    error = pickle.loads(pickle.dumps(SearchError('too broad', 408, 'query_timeout')))
    assert (str(error), error.status, error.code) == ('too broad', 408, 'query_timeout')
