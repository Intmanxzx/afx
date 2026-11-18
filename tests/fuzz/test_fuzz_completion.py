"""Only complete configured backend sets contribute to public results."""
import json
import pytest
from test_fuzz_shape_cases import record
from autofusex.fuzz.client.storage import ResultRepository
from autofusex.fuzz.client.storage.queries import ResultQueries
from autofusex.fuzz.server.reporting.statistics import statistics


@pytest.mark.parametrize('order', [('cpu','af-mlir','eager-npu'), ('af-mlir','eager-npu','cpu')])
def test_all_backends_required_in_any_arrival_order(tmp_path, order):
    repo = ResultRepository(tmp_path/'results')
    from autofusex.fuzz.server.reporting.incremental import IncrementalStatistics
    inc = IncrementalStatistics(repo.path)
    for i, backend in enumerate(order):
        value = record(backend, ['pass','error','pass'])
        value['execution']['expected_backends'] = json.dumps(list(order))
        repo.save(value)
        q = ResultQueries(repo.path)
        expected = 3 if i == 2 else 0
        assert q.case_summary()['total'] == expected
        assert len(q.case_page()['items']) == expected
        stats = statistics(repo.path)
        assert sum(r['count'] for r in stats['overall']) == expected
        assert inc.refresh() == stats
        assert stats['attempts'] == (3 if i == 2 else 0)
        if i < 2:
            assert stats['backend'] == stats['stages'] == stats['performance'] == []
    # Retries replace a result, not add a case; terminal errors remain visible.
    repo.save(value)
    assert q.case_summary()['total'] == 3


def test_legacy_cpu_only_hidden_but_terminal_target_unknown_visible(tmp_path):
    repo = ResultRepository(tmp_path/'results')
    repo.save(record('cpu', ['pass']))
    q = ResultQueries(repo.path)
    assert q.case_summary()['total'] == 0
    value = record('af-mlir', ['error'])
    value['shape_results'][0]['error_summary'] = 'TimeoutError: timed out'
    repo.save(value)
    assert q.case_summary()['by_status'] == {'unknown': 1}
    assert len(q.case_page()['items']) == 1


def test_manifest_normalization_sorts_expected_backends(tmp_path):
    from autofusex.fuzz.client.storage.records import normalize
    value = record('cpu', ['pass'])
    value['execution']['expected_backends'] = '["cpu","af-mlir"]'
    normalized = normalize(value, {})
    assert json.loads(normalized['contexts']['execution']['expected_backends']) == ['af-mlir','cpu']


def test_refresh_includes_later_pages():
    from pathlib import Path
    import shutil, subprocess
    node = shutil.which('node')
    if not node: pytest.skip('node unavailable')
    from importlib.resources import files
    source = files('autofusex.fuzz.server.view').joinpath('static/app.js').read_text()
    body = source[source.index('function refreshVisible('):source.index('function cell(')]
    subprocess.run([node, '-e', "const assert=require('node:assert/strict');let page=2,calls=[];const activePage=()=> 'records';const load=(...args)=>calls.push(args);const loadCharts=()=>{};" + body + "refreshVisible(true);assert.deepEqual(calls,[[true,true]]);"], check=True)
