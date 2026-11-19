import json
import pytest
from autofusex.fuzz.server.reporting.failure_origin import classify
from autofusex.fuzz.server.reporting.statistics import statistics
from autofusex.fuzz.client.storage import ResultRepository
from autofusex.fuzz.client.storage.queries import ResultQueries
from autofusex.fuzz.server.reporting.incremental import IncrementalStatistics
from test_fuzz_shape_cases import record


@pytest.mark.parametrize('summary,local,shared,expected', [
    ('InductorError: UnsupportedIndexExpr: unknown SymPy node Pow', '',
     'AF_MLIR_STRICT gate=none shortfall=no graph=autofused_add', 'gen_failed'),
    ('InductorError: UnsupportedIndexExpr: unknown SymPy node Pow', '',
     'AF_MLIR_STRICT gate=lowering shortfall=yes', 'gen_failed'),
    ('InductorError: wrapper', 'torch._dynamo.exc.Unsupported: data dependent op', '', 'gen_failed'),
    ('InductorError: wrapper', 'torch._inductor.exc.LoweringException: aten.foo', '', 'gen_failed'),
    ('ConstraintViolationError: shape guard', '', '', 'gen_failed'),
    ('InductorError: RuntimeError: Failed to execute graph code:from autofuse.pyautofuse import ascir',
     'Codegen generate kernel failed or abort', '', 'function_failed'),
    ('InductorError: wrapper', '', 'AF_MLIR_STRICT gate=lowering shortfall=yes', 'function_failed'),
    ('InductorError: wrapper', '', 'AF_MLIR_STRICT gate=none shortfall=no', 'unknown'),
    ('InductorError: wrapper', 'import autofuse.pyautofuse\nfuser.codegen(graph)', '', 'unknown'),
    ('InductorError: RuntimeError: Failed to execute graph code:from autofuse.pyautofuse import ascir', '', '', 'unknown'),
    ('InductorError: wrapper', 'File "/env/torch/_inductor/ir.py", line 20\nraise NotImplementedError("new error")', '', 'gen_failed'),
    ('InductorError: wrapper', 'import autofuse.pyautofuse\nCheck dtype failed for Cast', '', 'function_failed'),
    ('TimeoutError: timed out', '', 'AF_MLIR_STRICT gate=lowering shortfall=yes', 'environment_failed'),
    ('ValueError: device busy: another managed test owns this physical NPU', '', '', 'environment_failed'),
    ('RuntimeError: out of memory', '', '', 'environment_failed'),
    ('unrecognized failure', '', '', 'unknown'),
])
def test_attribution_requires_cause_not_wrapper_or_success_marker(summary, local, shared, expected):
    diag = {'shapes': {'0': {'error': local}}, 'error': shared}
    assert classify('af-mlir', 'error', 'first_call_combined', summary, diag, 0) == expected


def test_shared_other_shape_failure_is_not_current_shape_proof():
    diag = {'shapes': {'0': {'error': 'truncated'}, '1': {'error': 'Codegen generate kernel failed'}},
            'error': 'AF_MLIR_STRICT gate=lowering shortfall=yes'}
    assert classify('af-mlir', 'error', 'first_call_combined', '', diag, 0) == 'unknown'
    assert classify('af-mlir', 'error', 'first_call_combined', '', diag, 1) == 'function_failed'
    assert classify('triton-npu', 'error', 'first_call_combined', '', diag, 1) == 'unknown'


def test_every_surface_reclassifies_historical_rows_without_mutating_attempts(tmp_path):
    repo = ResultRepository(tmp_path/'db')
    row = record('af-mlir', ['error', 'error', 'error'])
    for i, summary in enumerate(['InductorError: UnsupportedIndexExpr: unknown SymPy node Pow',
                                 'Codegen generate kernel failed or abort', 'TimeoutError: timed out']):
        row['shape_results'][i].update(error_stage='first_call_combined', error_summary=summary)
    # Successful AF activity in this run must not override a Torch failure.
    row['execution']['diagnostics'] = json.dumps({'version': 1, 'error':
        'AF_MLIR_STRICT gate=none shortfall=no', 'shapes': {}})
    identity = repo.save(row)
    queries = ResultQueries(repo.path)
    before = queries.detail(identity)
    expected = {'gen_failed': 1, 'error': 1, 'environment_failed': 1}
    assert queries.case_summary()['by_status'] == expected
    page = queries.case_page()['items']
    assert [i['backends']['af-mlir'][0]['function_status'] for i in page] == [
        'gen_failed', 'function_failed', 'environment_failed']
    for i, category in enumerate(['gen_failed', 'function_failed', 'environment_failed']):
        assert queries.detail(identity, shape_index=i)['function_status'] == category
    for search in ['gen_failed', "WHERE status = 'gen_failed'", '构图Failed',
                   '功能Failed', "WHERE function_status = 'Failed'"]:
        assert queries.case_summary(q=search)['total'] == 1
        assert len(queries.case_page(q=search)['items']) == 1
        assert sum(r['count'] for r in statistics(repo.path, q=search)['overall']) == 1
    inc = IncrementalStatistics(repo.path)
    assert {r['status']: r['count'] for r in inc.refresh()['overall']} == expected
    assert queries.detail(identity) == before


def test_old_cached_classification_is_invalidated(tmp_path):
    from autofusex.fuzz.server.reporting.failure_origin import VERSION
    inc = IncrementalStatistics(tmp_path/'db')
    row = record('af-mlir', [])
    ResultRepository(inc.database).save(row)
    from autofusex.fuzz.server.reporting.incremental import VERSION as CACHE_VERSION
    assert CACHE_VERSION >= 6  # Global seed contributions supersede provenance groups.
    assert json.loads(inc._identity())[:2] == [CACHE_VERSION, VERSION]


def test_stale_incremental_totals_rebuild_with_new_classification(tmp_path):
    import sqlite3
    repo = ResultRepository(tmp_path/'db')
    row = record('af-mlir', ['error'])
    row['shape_results'][0].update(error_summary='UnsupportedIndexExpr: unknown SymPy node Pow')
    repo.save(row)
    inc = IncrementalStatistics(repo.path)
    assert inc.refresh()['overall'][0]['status'] == 'gen_failed'
    with sqlite3.connect(inc.path) as c:
        c.execute("UPDATE meta SET source='old-classification-v1'")
        c.execute("UPDATE totals SET n=999 WHERE kind='overall'")
    result = inc.refresh()
    assert result['overall'] == [{'label': '全部', 'status': 'gen_failed', 'count': 1}]
    assert inc.last_groups == 1


def test_detail_does_not_narrow_evidence_before_classification(tmp_path):
    repo = ResultRepository(tmp_path/'db')
    row = record('af-mlir', ['error', 'error'])
    row['execution']['diagnostics'] = json.dumps({
        'version': 1, 'error': 'AF_MLIR_STRICT gate=lowering shortfall=yes',
        'shapes': {'0': {'error': 'truncated'},
                   '1': {'error': 'Codegen generate kernel failed'}}})
    identity = repo.save(row)
    queries = ResultQueries(repo.path)
    rows = queries.case_page()['items']
    assert [r['backends']['af-mlir'][0]['function_status'] for r in rows] == [
        'unknown', 'function_failed']
    assert queries.detail(identity, shape_index=0)['function_status'] == 'unknown'
    assert queries.detail(identity, shape_index=1)['function_status'] == 'function_failed'


def test_local_torch_traceback_outranks_shared_af_failure():
    diag = {'shapes': {'0': {'error':
        'File "/env/torch/_inductor/ir.py", line 20\nraise NotImplementedError("new error")'}},
        'error': 'AF_MLIR_STRICT gate=lowering shortfall=yes'}
    assert classify('af-mlir', 'error', 'first_call_combined', '', diag, 0) == 'gen_failed'


@pytest.mark.parametrize('gate,shortfall,expected', [
    ('lowering', 'yes', 'mlir_failed'),
    ('lowering', 'no', 'mlir_failed'),
    ('candidate-preparation', 'yes', 'unconfirmed'),
    ('artifact-assembly', 'no', 'mlir_failed'),
    ('exception', 'unknown', 'mlir_failed'),
    ('none', 'no', 'unconfirmed'),
    ('cube-fusion', 'no', 'unconfirmed'),
])
def test_gate_inside_mlir_is_failure_evidence(gate, shortfall, expected):
    from autofusex.fuzz.server.reporting.failure_origin import mlir_failure_evidence
    diag = {'shapes': {'0': {'error':
        f'Codegen generate kernel failed\nAF_MLIR_STRICT gate={gate} shortfall={shortfall}'}}}
    result = mlir_failure_evidence('af-mlir', 'error', '', '', diag, 0)
    assert result['path'] == expected


def test_mlir_shared_log_bounds_af_failures_to_mlir():
    from autofusex.fuzz.server.reporting.failure_origin import failure_path
    diag = {'error': 'MLIR Codegen Result: kFailed',
            'shapes': {'0': {'error': 'Codegen generate kernel failed'},
                       '1': {'error': 'Codegen generate kernel failed'}}}
    assert failure_path('af-mlir', 'error', '', '', diag, 0) == 'mlir_failed'
    assert failure_path('af-mlir', 'error', '', '', diag, 1) == 'mlir_failed'
    assert failure_path('af-mlir', 'error', '', '', diag, -1) == 'mlir_failed'
    diag['shapes'].pop('1')
    assert failure_path('af-mlir', 'error', '', '', diag, 0, shape_count=1) == 'mlir_failed'
    assert failure_path('af-mlir', 'error', '', 'UnsupportedIndexExpr', diag, 0) == 'not_applicable'
    assert failure_path('af-mlir', 'error', '',
        'autofuse.pyautofuse Check dtype failed', diag, 0) == 'unconfirmed'


def test_mlir_failure_path_consistent_across_surfaces(tmp_path):
    repo = ResultRepository(tmp_path/'db')
    row = record('af-mlir', ['error', 'error', 'error'])
    row['execution']['diagnostics'] = json.dumps({'version': 1, 'shapes': {
        '0': {'error': 'MLIR Codegen Result: kFailed'},
        '1': {'error': 'Codegen generate kernel failed'},
        '2': {'error': 'UnsupportedIndexExpr'}}})
    identity = repo.save(row)
    queries = ResultQueries(repo.path)
    paths = ['mlir_failed', 'unconfirmed', 'not_applicable']
    assert [r['failure_path'] for r in queries.case_page()['items']] == paths
    for i, path in enumerate(paths):
        assert queries.detail(identity, shape_index=i)['mlir_failure']['path'] == path
    expected = [{'label': 'mlir_failed', 'status': 'error', 'count': 1},
                {'label': 'unconfirmed', 'status': 'error', 'count': 1}]
    assert queries.case_summary()['mlir_failed'] == 1
    chart=statistics(repo.path)
    assert {r['status']:r['count'] for r in chart['overall']}=={'mlir_failed':1,'error':1,'gen_failed':1}
    for dimension in ['backend','category','dtype','dimension','elements']:
        assert any(r['status']=='mlir_failed' for r in chart[dimension]),dimension
    assert statistics(repo.path)['failure_path'] == expected
    assert IncrementalStatistics(repo.path).refresh()['failure_path'] == expected
    for path in paths:
        query = f"WHERE failure_path = '{path}'"
        assert queries.case_summary(q=query)['total'] == 1
        assert len(queries.case_page(q=query)['items']) == 1


def test_run_evidence_does_not_require_single_shape():
    from autofusex.fuzz.server.reporting.failure_origin import failure_path
    diag = {'error': 'MLIR Codegen Result: kFailed',
            'shapes': {'0': {'error': 'Codegen generate kernel failed'}}}
    assert failure_path('af-mlir', 'error', '', '', diag, 0) == 'mlir_failed'
    assert failure_path('af-mlir', 'error', '', '', diag, 0, shape_count=3) == 'mlir_failed'
    assert failure_path('af-mlir', 'error', '', '', diag, 0, shape_count=1) == 'mlir_failed'


@pytest.mark.parametrize('status,summary,expected', [
    ('ok', 'RunMlirCodegen failed', 'not_applicable'),
    ('fail', 'RunMlirCodegen failed', 'not_applicable'),
    ('error', 'Codegen generate kernel failed', 'unconfirmed'),
    ('error', 'RunMlirCodegen failed: bad lowering', 'mlir_failed'),
    ('error', 'no plan with an MLIR-supported template', 'unconfirmed'),
])
def test_mlir_path_is_only_a_function_failure_subcategory(status, summary, expected):
    from autofusex.fuzz.server.reporting.failure_origin import failure_path
    assert failure_path('af-mlir', status, '', summary) == expected


def test_mlir_evidence_http_and_retry_cache(tmp_path):
    import threading
    from urllib.request import urlopen
    from urllib.parse import urlencode
    from autofusex.fuzz.server.view.server import create_server
    repo = ResultRepository(tmp_path/'db')
    row = record('af-mlir', ['error'])
    row['execution']['diagnostics'] = json.dumps({'version': 1, 'error':
        'MLIR Codegen Result: kFailed', 'shapes': {}})
    identity = repo.save(row)
    inc = IncrementalStatistics(repo.path)
    assert inc.refresh()['failure_path'] == [{'label': 'mlir_failed', 'status': 'error', 'count': 1}]
    with create_server(repo.path) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with urlopen(server.viewer_url+'api/attempts/'+identity+'?shape=0') as r:
                evidence = json.load(r)['mlir_failure']
                assert evidence == {'path': 'mlir_failed', 'scope': 'run',
                    'evidence': 'MLIR Codegen Result: kFailed'}
            query = urlencode({'q': "WHERE failure_path = 'mlir_failed'"})
            with urlopen(server.viewer_url+'api/summary?'+query) as r:
                assert json.load(r)['total'] == 1
        finally:
            server.shutdown()
            thread.join()
    row['execution']['diagnostics'] = json.dumps({'version': 1, 'error':
        'Codegen generate kernel failed', 'shapes': {}})
    repo.save(row)
    assert inc.refresh()['failure_path'] == [{'label': 'unconfirmed', 'status': 'error', 'count': 1}]


@pytest.mark.parametrize('value,expected', [
    ('kFailed', 'mlir_failed'), ('1', 'mlir_failed'),
    ('mlir::bridge::MlirCodegenStatus::kFailed', 'mlir_failed'),
    ('kSucceeded', 'unconfirmed'), ('2', 'unconfirmed'),
    ('kNotSupported', 'unconfirmed'), ('0', 'unconfirmed'),
    ('Unknown', 'unconfirmed'),
])
@pytest.mark.parametrize('separator', [':', '：'])
def test_explicit_codegen_result(value, expected, separator):
    from autofusex.fuzz.server.reporting.failure_origin import mlir_failure_evidence
    line = f'[DEBUG] GE(123) codegen.cpp: MLIR  Codegen Result{separator} {value}'
    diag = {'shapes': {'0': {'error':
        'Codegen generate kernel failed\n' + line}}}
    evidence = mlir_failure_evidence('af-mlir', 'error', '', '', diag, 0, 1)
    assert evidence['path'] == expected
    assert evidence['evidence'] == line


def test_successful_shared_verdict_does_not_erase_local_failure():
    from autofusex.fuzz.server.reporting.failure_origin import failure_path
    diag = {'error': '[DEBUG] MLIR Codegen Result：kSucceeded',
            'shapes': {'0': {'error': 'RunMlirCodegen failed'}}}
    assert failure_path('af-mlir', 'error', '', '', diag, 0, 1) == 'mlir_failed'
    # A successful codegen call cannot erase an observed MLIR failure.
    assert failure_path('af-mlir', 'error', '', '', diag, 0, 3) == 'mlir_failed'


def test_debug_result_survives_error_log_tail():
    import os
    from autofusex.fuzz.client.core_logs import capture_tail
    with capture_tail() as tail:
        os.write(2, '[DEBUG] MLIR  Codegen Result：kFailed\n'.encode())
        os.write(2, b'error: later error\n' * 5000)
    result = tail()
    assert 'MLIR  Codegen Result：kFailed' in result
    assert len(result) <= 1600


def test_explicit_failed_result_needs_no_legacy_error_text():
    from autofusex.fuzz.server.reporting.failure_origin import failure_path
    assert failure_path('af-mlir', 'error', '', '[DEBUG] MLIR Codegen Result：kFailed') == 'mlir_failed'
    assert failure_path('af-mlir', 'ok', '', '[DEBUG] MLIR Codegen Result：kFailed') == 'not_applicable'


def test_multiple_codegen_calls_keep_failed_verdict():
    from autofusex.fuzz.server.reporting.failure_origin import mlir_failure_evidence
    diag = {'shapes': {'0': {'error':
        'MLIR Codegen Result：kFailed\nMLIR Codegen Result：kSucceeded'}}}
    evidence = mlir_failure_evidence('af-mlir', 'error', '', '', diag, 0, 1)
    assert evidence['path'] == 'mlir_failed'
    assert evidence['evidence'] == 'MLIR Codegen Result：kFailed'


@pytest.mark.parametrize('shared', [
    "error: 'arith.index_cast' op result #0 must be signless-integer-like or memref of signless-integer, but got 'ui8'",
    'AF_MLIR_STRICT gate=lowering shortfall=no',
    'RunMlirCodegen failed',
])
def test_mlir_run_failure_bounds_multi_shape_case_across_surfaces(tmp_path, shared):
    # F0SQR0HF7D3QEYXA8AE8RJL9MQV: three failed Shapes, shared MLIR stderr.
    repo = ResultRepository(tmp_path/'db')
    row = record('af-mlir', ['error', 'error', 'error'])
    for shape in row['shape_results']:
        shape.update(error_stage='first_call_combined',
                     error_summary='Codegen generate kernel failed or abort')
    row['execution']['diagnostics'] = json.dumps({
        'version': 1, 'error': shared, 'shapes': {}})
    identity = repo.save(row)
    queries = ResultQueries(repo.path)
    assert [r['failure_path'] for r in queries.case_page()['items']] == ['mlir_failed'] * 3
    for index in range(3):
        assert queries.detail(identity, shape_index=index)['mlir_failure'] == {
            'path': 'mlir_failed', 'scope': 'run', 'evidence': shared}
    assert queries.case_summary(q="WHERE failure_path = 'mlir_failed'")['total'] == 3
    expected = [{'label': 'mlir_failed', 'status': 'error', 'count': 3}]
    assert statistics(repo.path)['failure_path'] == expected
    assert IncrementalStatistics(repo.path).refresh()['failure_path'] == expected


def test_native_mlir_verifier_error_is_sufficient_without_codegen_wrapper():
    from autofusex.fuzz.server.reporting.failure_origin import failure_path
    assert failure_path('af-mlir', 'error', '',
        "error: 'arith.index_cast' op result #0 must be signless-integer-like, but got 'ui8'") == 'mlir_failed'
