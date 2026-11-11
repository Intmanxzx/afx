import json
import pytest
from autofusedebugger.fuzz.reporting.failure_origin import classify
from autofusedebugger.fuzz.reporting.statistics import statistics
from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.storage.queries import ResultQueries
from autofusedebugger.fuzz.reporting.incremental import IncrementalStatistics
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
    ('TimeoutError: timed out', '', 'AF_MLIR_STRICT gate=lowering shortfall=yes', 'unknown'),
    ('ValueError: device busy: another managed test owns this physical NPU', '', '', 'unknown'),
    ('RuntimeError: out of memory', '', '', 'unknown'),
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
    expected = {'gen_failed': 1, 'error': 1, 'unknown': 1}
    assert queries.case_summary()['by_status'] == expected
    page = queries.case_page()['items']
    assert [i['backends']['af-mlir'][0]['function_status'] for i in page] == [
        'gen_failed', 'function_failed', 'unknown']
    for i, category in enumerate(['gen_failed', 'function_failed', 'unknown']):
        assert queries.detail(identity, shape_index=i)['function_status'] == category
    for search in ['gen_failed', "WHERE status = 'gen_failed'", '构图Failed',
                   '功能Failed', "WHERE function_status = 'Failed'", 'unknown']:
        assert queries.case_summary(q=search)['total'] == 1
        assert len(queries.case_page(q=search)['items']) == 1
        assert sum(r['count'] for r in statistics(repo.path, q=search)['overall']) == 1
    inc = IncrementalStatistics(repo.path)
    assert {r['status']: r['count'] for r in inc.refresh()['overall']} == expected
    assert queries.detail(identity) == before


def test_old_cached_classification_is_invalidated(tmp_path):
    from autofusedebugger.fuzz.reporting.failure_origin import VERSION
    inc = IncrementalStatistics(tmp_path/'db')
    row = record('af-mlir', [])
    ResultRepository(inc.database).save(row)
    assert json.loads(inc._identity())[:2] == [2, VERSION]


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
