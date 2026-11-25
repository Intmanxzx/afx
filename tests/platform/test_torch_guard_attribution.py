import json

import pytest

from autofusex.client.platform_transport import function_status
from autofusex.server.failure_attribution import classify
from autofusex.server.result_categories import category
from test_platform import env, body, item, upload


GUARD = ("AssertionError: Guard failed on the same frame it was created. "
         "This is a bug - please create an issue.Guard fail reason: 0/0: "
         "torch.sym_float((1.0 / (torch.sym_float(functions[7].__closure__[0]."
         "cell_contents['p'])))) != 0.5  # (_refs/__init__.py:1326 in pow)")
MARKER = 'AFX_MODULE_V1={"stage":"schedule"}\n'


@pytest.mark.parametrize('backend', ['af-mlir', 'af-legacy', 'triton-npu'])
@pytest.mark.parametrize('source', ['summary', 'diagnostics', 'shape_results'])
def test_guard_failure_outranks_schedule(backend, source):
    summary = MARKER + GUARD
    detail = None
    if source == 'diagnostics':
        summary = 'Schedule failed'
        detail = {'contexts': {'execution': {'diagnostics': json.dumps({
            'shapes': {'0': {'error': MARKER + GUARD}}})}}}
    elif source == 'shape_results':
        summary = 'Schedule failed'
        detail = {'shape_results': [{'shape_index': 0, 'correctness': 'error',
                                     'error_summary': MARKER + GUARD}]}
    result = classify(backend=backend, declared_status='compile_fail', declared_stage='schedule',
                      summary=summary, detail=detail)
    assert result[:3] == ('gen_fail', 'inductor', 'torch')
    assert category(*result[:2], None) == 'gen_fail'


def test_client_uses_guard_error_before_module_or_native_schedule():
    diagnostics = {'shapes': {'0': {'error': MARKER + GUARD}}}
    assert function_status('af-mlir', 'error', 'first_call_combined', 'Schedule failed',
                           diagnostics, 0)[:2] == ('gen_fail', 'inductor')
    assert function_status('af-mlir', 'error', 'first_call_combined', GUARD, None, -1,
                           extra='[ERROR] GE(1,p):t [schedule_utils.cpp:3]1 x: failed')[:2] == (
                               'gen_fail', 'inductor')


def test_shared_guard_and_generic_assertions_do_not_override_schedule():
    for summary, log in [(MARKER + 'AssertionError: invalid tiling', GUARD),
                         ('Schedule failed', GUARD)]:
        assert classify(backend='af-mlir', declared_status='compile_fail',
                        summary=summary, log=log)[:2] == ('compile_fail', 'schedule')
    detail = {'shape_results': [
        {'shape_index': 0, 'correctness': 'error', 'error_summary': GUARD},
        {'shape_index': 1, 'correctness': 'error', 'error_summary': 'Schedule failed'}]}
    assert classify(backend='af-mlir', declared_status='compile_fail',
                    summary='Schedule failed', detail=detail)[:2] == ('compile_fail', 'schedule')


def test_refresh_corrects_historical_schedule_guard(env):
    from autofusex.server.failure_attribution import VERSION
    from autofusex.server.reclassification import refresh
    from autofusex.server import queries

    store, ingestor, principal = env
    upload(ingestor, principal, body([item(
        'guard-env-0001', status='compile_fail', fail_stage='schedule',
        log_summary='Schedule failed', detail={'shape_results': [
            {'shape_index': 0, 'correctness': 'error', 'error_summary': GUARD}]})]))
    with store.write() as conn:
        conn.execute("UPDATE executions SET function_status='compile_fail',fail_stage='schedule',"
                     "failure_owner='autofuse',classifier_version=?", (VERSION - 1,))
    changes = refresh(store, 'guard-refresh', apply=True)
    assert len(changes) == 1
    assert changes[0]['after'] == ['gen_fail', 'inductor', 'torch', VERSION]
    assert queries.backend_stats(store, 'af-mlir')['groups'][0]['results'] == {'gen_fail': 1}
