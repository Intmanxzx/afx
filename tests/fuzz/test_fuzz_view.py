"""Offline viewer contract, transport and read-only boundary tests (stdlib client)."""
import http.client
import json
import sqlite3
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from urllib.parse import urlsplit

import pytest

from autofusex.fuzz.storage import ResultRepository
from autofusex.fuzz.storage.queries import ResultQueries
from autofusex.fuzz.view.server import create_server


def record(seed='680564733841876926926749214863536422954', status='ok'):
    import hashlib
    return {'seed': seed, 'workload_id': hashlib.sha256(seed.encode()).hexdigest(), 'oracle_id': 'ref/1',
            'backend': 'af-mlir', 'status': status,
            'execution': {'backend': 'af-mlir', 'device': 'cpu'},
            'evaluation': {'correctness': True, 'performance': False},
            'shape_results': [{'shape_index': 0, 'correctness': {'ok':'pass','fail':'fail','error':'error'}[status],
                               'performance': 'not_requested'}]}


@pytest.fixture
def database(tmp_path):
    repo = ResultRepository(tmp_path/'source.sqlite')
    for seed, status in [('680564733841876926926749214863536422954', 'ok'), ('680564733841876926926749214863536422954', 'fail'), (str(3*2**128-1), 'error')]:
        repo.save(record(seed, status))
    return repo.snapshot(tmp_path/'snapshot.sqlite')


@contextmanager
def running(database):
    with create_server(database) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
        thread.start()
        try:
            yield urlsplit(server.viewer_url)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def request(url, path='', *, method='GET', headers=None, absolute=False):
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
    try:
        conn.request(method, path if absolute else url.path+path, headers=headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def test_paging_filters_summary_and_detail(database):
    queries = ResultQueries(database)
    first = queries.page(limit=2)
    assert len(first['items']) == 2
    last = queries.page(limit=2, after=first['next_cursor'])
    assert len(last['items']) == 1 and last['next_cursor'] is None
    assert last['items'][0]['seed'] == str(3*2**128-1)
    assert queries.page(limit=3)['next_cursor'] is None
    assert queries.summary() == {'total': 3, 'by_status': {'ok': 1, 'fail': 1, 'error': 1}}
    assert queries.summary(seed='680564733841876926926749214863536422954', status='ok')['total'] == 1
    assert len(queries.page(seed='680564733841876926926749214863536422954')['items']) == 2
    detail = queries.detail(first['items'][0]['id'])
    assert detail['replay_command'] == ''  # Retired seeds remain readable, not executable.
    assert len(detail['shape_results']) == 1
    assert set(detail['contexts']) == {'execution', 'evaluation', 'environment'}


def test_http_assets_api_and_read_only(database):
    from pathlib import Path
    before = Path(database).read_bytes()
    files = set(Path(database).parent.iterdir())
    with running(database) as url:
        for asset, mime in [('', 'text/html'), ('app.js', 'text/javascript'), ('style.css', 'text/css')]:
            status, headers, body = request(url, asset)
            assert status == 200 and headers['Content-Type'].startswith(mime) and body
            assert headers['Cache-Control'] == 'no-store'
            assert "frame-ancestors 'none'" in headers['Content-Security-Policy']
        status, _, body = request(url, 'api/results?seed=C000000016&limit=1')
        assert status == 200
        item = json.loads(body)['items'][0]
        status, _, body = request(url, 'api/attempts/'+item['id'])
        assert status == 200 and json.loads(body)['replay_command'] == ''
        assert json.loads(request(url, 'api/summary?status=fail')[2])['total'] == 1
    assert Path(database).read_bytes() == before
    assert set(Path(database).parent.iterdir()) == files


@pytest.mark.parametrize('path,expected', [
    ('api/results?limit=201', 400), ('api/results?after=-1', 400),
    ('api/results?after=9223372036854775808', 400),
    ('api/results?limit=x', 400), ('api/results?limit=1&limit=2', 400),
    ('api/results?unknown=x', 400), ('api/results?seed=42%3Brm', 400),
    ('api/results?status=bad', 400), ('api/summary?limit=1', 400),
    ('api/attempts/not-a-uuid', 400), ('api/attempts/'+str(uuid.UUID(int=0)), 404),
    ('../source.sqlite', 404), ('%2e%2e/source.sqlite', 404),
])
def test_bad_queries_and_paths(database, path, expected):
    with running(database) as url:
        assert request(url, path)[0] == expected


def test_origin_capability_and_mutation_boundaries(database):
    with running(database) as url:
        assert request(url, '/', absolute=True)[0] == 404
        assert request(url, headers={'Host': 'evil.example'})[0] == 403
        assert request(url, headers={'Origin': 'https://evil.example'})[0] == 403
        assert request(url, 'x'*2049)[0] == 414
        for method in ('POST', 'PUT', 'PATCH', 'DELETE'):
            assert request(url, 'api/results', method=method)[0] == 405


@pytest.mark.parametrize('kind', ['missing', 'corrupt', 'unknown', 'foreign'])
def test_invalid_databases_fail_before_listening(tmp_path, kind):
    path = tmp_path/'invalid.sqlite'
    if kind == 'corrupt':
        path.write_bytes(b'not sqlite')
    elif kind in {'unknown', 'foreign'}:
        with sqlite3.connect(path) as conn:
            conn.execute('PRAGMA user_version=999')
            if kind == 'unknown':
                from autofusex.fuzz.storage.schema import APPLICATION_ID
                conn.execute(f'PRAGMA application_id={APPLICATION_ID}')
    before = path.read_bytes() if path.exists() else None
    with pytest.raises((ValueError, sqlite3.Error, OSError)):
        create_server(path)
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.parametrize('port', [-1, 65536])
def test_invalid_port(database, port):
    with pytest.raises(ValueError, match='port'):
        create_server(database, port=port)


def test_view_import_and_help_are_lightweight():
    code = """
import sys
from autofusex.fuzz.view.server import create_server
assert not set(sys.modules) & {'torch', 'torch_npu', 'nnsmith', 'numpy'}
from autofusex.fuzz.cli import main
main(['view', '--help'])
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--no-browser' in result.stdout


def test_untrusted_text_is_data_and_never_a_replay_command(database):
    payload = '<img src=x onerror=alert(1)>'
    with sqlite3.connect(database) as conn:
        conn.execute('UPDATE attempts SET backend=?,error_summary=?', (payload, payload))
    with running(database) as url:
        item = json.loads(request(url, 'api/results')[2])['items'][0]
        assert item['backend'] == payload
        detail = json.loads(request(url, 'api/attempts/'+item['id'])[2])
        assert detail['error_summary'] == payload
        assert detail['replay_command'] == ''  # Retired seeds remain readable, not executable.
        assert payload.encode() not in request(url)[2]
    with sqlite3.connect(database) as conn:
        conn.execute("UPDATE attempts SET seed='42; echo unsafe'")
    with running(database) as url:
        assert request(url, 'api/attempts/'+item['id'])[0] == 400


def test_cli_dispatch_and_missing_database(tmp_path, monkeypatch, capsys):
    from autofusex.fuzz.cli import main
    from autofusex.fuzz.view import server
    calls = []
    monkeypatch.setattr(server, 'serve', lambda *a, **kw: calls.append((a, kw)) or 0)
    assert main(['view', 'example.sqlite', '--no-browser', '--port', '12345']) == 0
    assert calls == [(('example.sqlite',), {'port': 12345, 'open_browser': False})]
    monkeypatch.undo()
    assert main(['view', str(tmp_path/'missing.sqlite'), '--no-browser']) == 2
    assert 'ERROR:' in capsys.readouterr().err


def test_central_chart_requests_share_bounded_cache(tmp_path, monkeypatch):
    import threading
    from urllib.request import urlopen
    from autofusex.fuzz.central.server import create_server
    calls=[]
    monkeypatch.setattr('autofusex.fuzz.reporting.incremental.IncrementalStatistics.refresh',
                        lambda *args,**kwargs: calls.append(kwargs) or {'attempts':1,'overall':[]})
    server=create_server(tmp_path/'db',0)
    assert server.public_cache.ready.wait(5)
    thread=threading.Thread(target=server.serve_forever);thread.start()
    try:
        for _ in range(2):
            with urlopen(f'http://127.0.0.1:{server.server_port}/api/statistics') as response:
                assert response.status==200
        assert len(calls)==1
    finally:
        server.shutdown();thread.join();server.server_close()
