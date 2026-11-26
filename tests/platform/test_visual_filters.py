"""Visual predicates intersect search on a single execution head."""
import itertools
import sqlite3

import pytest

from autofusex.server import search, queries
from autofusex.server.result_categories import category, category_sql
from test_platform import env, upload, body, item, case, SEED_A, SEED_B


def test_category_sql_matches_authoritative_classifier():
    conn = sqlite3.connect(':memory:')
    statuses = ['pass', 'build_fail', 'gen_fail', 'ascgraph_fail', 'unsupported', 'timeout', 'compile_fail', 'runtime_fail', 'unknown']
    stages = [None, '', 'unknown_retest', 'environment', 'nnsmith', 'mlir', 'schedule', 'legacy_codegen', 'triton_ascend', 'af_common', 'aclnn', 'aicpu', 'fused_kernel']
    for status, stage, accuracy in itertools.product(statuses, stages, [None, '', 'pass', 'fail']):
        actual = conn.execute('SELECT ' + category_sql() + ' FROM (SELECT ? function_status, ? fail_stage, ? accuracy_status) e',
                              (status, stage, accuracy)).fetchone()[0]
        assert actual == category(status, stage, accuracy)


def test_filters_intersect_on_same_head_and_chip(env):
    store, ingestor, principal = env
    result = upload(ingestor, principal, body([
        item('visual-a-mlir', kernel=100, c=case(fingerprint='1'*64)),
        item('v-a-triton', backend='triton-npu', status='compile_fail', fail_stage='mlir', c=case(fingerprint='1'*64)),
        item('visual-b-mlir', c=case(SEED_B, fingerprint='2'*64), kernel=200, npu_arch='950'),
        item('v-b-triton', backend='triton-npu', c=case(SEED_B, fingerprint='2'*64), kernel=100, npu_arch='950'),
    ]))
    assert result["accepted"] == 4
    with store.write() as conn:
        conn.execute("UPDATE executions SET function_status='compile_fail',fail_stage='mlir' WHERE exec_uuid='v-a-triton'")
        from autofusex.server.aggregates import rebuild
        rebuild(conn, store.setting(conn, "perf.tie_threshold"))
    assert len(queries.case_page(store, chips='a2a3')['groups']) == 1
    assert len(queries.case_page(store, chips='a5')['groups']) == 1
    assert not queries.case_page(store, chips='none')['groups']
    assert not queries.case_page(store, results='none')['groups']
    assert not queries.case_page(store, q='backend:af-mlir', results='mlir')['groups']
    assert not queries.case_page(store, q='stage:mlir', selected_backends='af-mlir')['groups']
    assert not queries.case_page(store, q='arch:950', chips='a2a3')['groups']
    assert queries.case_page(store, q='status:function_fail', results='mlir', chips='a2a3')['groups'][0]['case']['case_key'] == SEED_A
    paged = queries.case_page(store, page=2, limit=1, chips='a2a3')
    assert paged['total'] == paged['total_pages'] == paged['page'] == 1
    assert paged['groups'][0]['case']['case_key'] == SEED_A
    empty = queries.case_page(store, page=10, results='none')
    assert empty == dict(groups=[], page=1, page_size=50, total=0, total_pages=0)

    stats = queries.backend_stats(store, 'triton-npu', results='mlir', chips='a2a3')['groups']
    assert stats[0]['results'] == {'mlir': 1}
    assert not queries.perf_pair(store, 'af-mlir', 'triton-npu', chips='a2a3')['groups']
    perf = queries.perf_pair(store, 'af-mlir', 'triton-npu', q='status:pass', chips='a5')['groups']
    assert perf[0]['total'] == 1
    assert perf[0]['slower'] == 1
    assert not queries.perf_pair(store, 'af-mlir', 'triton-npu', results='mlir', chips='a5')['groups']


def test_invalid_visual_filters_are_rejected():
    for kwargs in ({'results': 'invalid'}, {'chips': "x' OR 1"}, {'selected_backends': 'a,,b'}):
        with pytest.raises(search.SearchError):
            search.visual_query(**kwargs)


def test_visual_filters_reach_http_endpoints(env):
    from fastapi.testclient import TestClient
    from autofusex.server.server import create_app
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('visual-http-001', kernel=100)]))
    client = TestClient(create_app(store), base_url="http://127.0.0.1")
    for endpoint in ('/api/cases', '/api/overview/pipeline', '/api/overview/matrix?backends=af-mlir', '/api/overview/perf?x=af-mlir&y=triton-npu'):
        response = client.get(endpoint, params={'results': 'none', 'chips': 'a2a3', 'backends': 'af-mlir', 'x': 'af-mlir', 'y': 'triton-npu'})
        assert response.status_code == 200, response.text
    data = client.get('/api/overview/pipeline', params={'selected_backends': 'none', 'chips': 'a2a3'}).json()['data']
    assert data['lanes'] == []
    response = client.get('/api/cases', params={'q': 'status:pass', 'results': 'pass', 'chips': 'a2a3'})
    assert response.json()['data']['groups'][0]['case']['case_key'] == SEED_A
    response = client.get('/api/cases', params={'page': 1, 'limit': 1, 'results': 'pass', 'chips': 'a2a3'})
    assert response.status_code == 200
    assert response.json()['data']['total_pages'] == 1
    assert client.get('/api/cases', params={'page': 0}).status_code == 400


@pytest.mark.parametrize('endpoint,params', [
    ('/api/cases', {}),
    ('/api/overview/pipeline', {}),
    ('/api/overview/backend', {'backend': 'af-mlir'}),
    ('/api/overview/matrix', {'backends': 'af-mlir'}),
    ('/api/overview/perf', {'x': 'af-mlir', 'y': 'triton-npu'}),
])
def test_all_result_categories_reach_http_endpoints(env, endpoint, params):
    from fastapi.testclient import TestClient
    from autofusex.server.server import create_app
    from autofusex.server.result_categories import RESULT_KEYS

    store, ingestor, principal = env
    upload(ingestor, principal, body([
        item('all-results-mlir', kernel=100),
        item('all-results-triton', backend='triton-npu', kernel=200),
    ]))
    client = TestClient(create_app(store), base_url='http://127.0.0.1')
    results = ','.join(sorted(RESULT_KEYS))
    assert len(results) > 128
    expected = client.get(endpoint, params=params)
    response = client.get(endpoint, params={**params, 'results': results})
    assert expected.status_code == response.status_code == 200, response.text
    actual_data, expected_data = response.json()['data'], expected.json()['data']
    # The pipeline echoes whether a query was composed, even for an all-results filter.
    if endpoint == '/api/overview/pipeline':
        actual_data.pop('q')
        expected_data.pop('q')
    assert actual_data == expected_data
    for invalid in (results + ',invalid', results + '\n', 'x' * 4097):
        response = client.get(endpoint, params={**params, 'results': invalid})
        assert response.status_code == 400, response.text



def test_visual_overview_reads_aggregates_without_scanning_executions(env):
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('aggregate-fast-001', kernel=100)]))
    conn = store.reader()
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        result = queries.backend_stats(store, 'af-mlir', chips='a2a3', results='pass', q='status:pass -stage:mlir')
    finally:
        conn.set_trace_callback(None)
    assert result['groups'][0]['total'] == 1
    assert any('stats.stat_buckets' in sql for sql in statements)
    assert not any('JOIN executions' in sql for sql in statements)


def test_empty_results_selection_does_not_scan_heads(env):
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('empty-filter-scan', kernel=100)]))
    conn = store.reader()
    # A regressed category expression fails rather than silently scanning a small fixture.
    def authorize(action, table, column, database, trigger):
        if action == sqlite3.SQLITE_READ and table == 'executions' and column in {'function_status', 'accuracy_status', 'fail_stage'}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    conn.set_authorizer(authorize)
    try:
        assert queries.case_page(store, results='none', keys_only=True) == {'case_keys': []}
        assert queries.case_page(store, results='none', page=1)['total'] == 0
    finally:
        conn.set_authorizer(None)
    query = search.visual_query(results='none,pass')
    assert query.head_sql()[1] == ['pass']
    with pytest.raises(search.SearchError):
        search.visual_query(results='none', chips="bad'")
