import json

import pytest

from autofusex.server.failure_attribution import VERSION, classify
from autofusex.server.reclassification import refresh, rollback
from autofusex.server import queries
from test_platform import env, body, item, upload


ERROR = ('UnsupportedBackend: af-mlir runtime verification failed: '
         'loaded codegen was built without the MLIR bridge')
MARKER = 'AFX_MODULE_V1={"stage":"schedule"}\n'


@pytest.mark.parametrize('status', ['compile_fail', 'runtime_fail', 'unsupported'])
@pytest.mark.parametrize('source', ['summary', 'diagnostics', 'shape_results', 'execution_error'])
def test_mlir_runtime_environment_outranks_schedule(status, source):
    detail = None
    summary = MARKER + ERROR
    if source == 'diagnostics':
        summary = 'Schedule failed'
        detail = json.dumps({'contexts': {'execution': {'diagnostics': json.dumps({
            'shapes': {'0': {'error': MARKER + ERROR}}})}}}).encode()
    elif source == 'shape_results':
        summary = 'Schedule failed'
        detail = {'shape_results': [{'shape_index': 0, 'correctness': 'error',
                                     'error_summary': MARKER + ERROR}]}
    elif source == 'execution_error':
        summary = None
        detail = {'error_stage': 'capability', 'error_summary': ERROR, 'shape_results': []}
    assert classify(backend='af-mlir', declared_status=status, declared_stage='schedule',
                    summary=summary, detail=detail) == (
                        'runtime_fail', 'environment', 'infrastructure', VERSION)


def test_unrelated_or_mixed_errors_do_not_prove_environment():
    assert classify(backend='af-mlir', declared_status='compile_fail',
                    summary='Schedule failed', log=ERROR)[:2] == ('compile_fail', 'schedule')
    assert classify(backend='af-legacy', declared_status='compile_fail',
                    summary=MARKER + ERROR)[:2] == ('compile_fail', 'schedule')
    detail = {'shape_results': [
        {'shape_index': 0, 'correctness': 'error', 'error_summary': ERROR},
        {'shape_index': 1, 'correctness': 'error', 'error_summary': 'Schedule failed'},
    ]}
    assert classify(backend='af-mlir', declared_status='compile_fail',
                    summary='Schedule failed', detail=detail)[:2] == ('compile_fail', 'schedule')
    assert classify(backend='af-mlir', declared_status='pass', summary=ERROR)[:2] == ('pass', None)
    assert classify(backend='af-mlir', declared_status='unsupported',
                    summary='UnsupportedBackend: unsupported operator')[:2] == ('unsupported', None)


def test_ingest_and_refresh_environment_bucket(env):
    store, ingestor, principal = env
    upload(ingestor, principal, body([item(
        'mlir-env-0001', status='compile_fail', fail_stage='schedule', log_summary=MARKER + ERROR)]))
    assert queries.backend_stats(store, 'af-mlir')['groups'][0]['results'] == {'environment': 1}
    with store.write() as conn:
        conn.execute("UPDATE executions SET function_status='compile_fail',fail_stage='schedule',"
                     "failure_owner='autofuse',classifier_version=?", (VERSION - 1,))
    changes = refresh(store, 'mlir-env-preview')
    assert len(changes) == 1
    assert changes[0]['after'] == ['runtime_fail', 'environment', 'infrastructure', VERSION]
    assert store.reader().execute('SELECT fail_stage FROM executions').fetchone()[0] == 'schedule'
    assert len(refresh(store, 'mlir-env-apply', apply=True)) == 1
    assert queries.backend_stats(store, 'af-mlir')['groups'][0]['results'] == {'environment': 1}
    assert refresh(store, 'mlir-env-repeat', apply=True) == []
    assert rollback(store, 'mlir-env-apply') == 1
    assert queries.backend_stats(store, 'af-mlir')['groups'][0]['results'] == {'schedule': 1}
