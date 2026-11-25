import copy
import json

import pytest

from autofusex.server.native_runtime_attribution import native_runtime_failure
from autofusex.server.failure_attribution import classify
from test_platform import env, http, body, item, upload


def sync_aclnn_failure(detail):
    return native_runtime_failure(detail, 'af-mlir')


def evidence():
    error = ('RuntimeError: npuSynchronizeDevice: NPU function error: AclrtSynchronizeDeviceWithTimeout, error code is 507018\n'
             'EE9999[PID: 123] 2026-11-25-02:35:43.217.342 (EE9999): rtDeviceSynchronizeWithTimeout execution failed, reason=aicpu exception')
    return {'error_stage': 'worker', 'shape_results': [],
            'contexts': {'execution': {'diagnostics': json.dumps({'error': error, 'shapes': {}})}},
            'stage_timeline': [{'state': 'failed', 'worker_pid': 123, 'started': 0, 'finished': 2_000_000_000,
                                'lease': {'device_uuid': 'npu-0'}}],
            'plog_errors': ['[ERROR] OP(123,python):2026-11-25-02:35:42.599.496 [aclnn.cpp:79][NNOP] errno[161002] OpName:[aclnnReduceNansum_5] Tensor self not implemented for DT_DOUBLE, should be in dtype support list [DT_FLOAT,].']}


def test_sync_failure_overrides_legacy_timeout():
    d = evidence()
    assert sync_aclnn_failure(json.dumps(d).encode()) == ('runtime_fail', 'aclnn', 'npu_runtime')
    for status in ('timeout', 'runtime_fail', 'compile_fail'):
        assert classify(backend='af-mlir', declared_status=status, detail=d)[:3] == ('runtime_fail', 'aclnn', 'npu_runtime')
    assert classify(backend='af-mlir', declared_status='pass', detail=d)[0] == 'pass'
    assert classify(backend='af-mlir', declared_status='unsupported', detail=d)[0] == 'unsupported'


@pytest.mark.parametrize('change', ['wrong_pid','info','zero_errno','no_operator','shape_error','no_frame','multiple_failed','no_device','no_worker'])
def test_requires_same_failed_worker_native_failure(change):
    d = evidence()
    if change == 'wrong_pid': d['plog_errors'][0] = d['plog_errors'][0].replace('OP(123,', 'OP(999,')
    elif change == 'info': d['plog_errors'][0] = d['plog_errors'][0].replace('[ERROR]', '[INFO]')
    elif change == 'zero_errno': d['plog_errors'][0] = d['plog_errors'][0].replace('errno[161002]', 'errno[0]')
    elif change == 'no_operator': d['plog_errors'][0] = d['plog_errors'][0].replace('OpName:[aclnnReduceNansum_5]', 'OpName:[other]')
    elif change == 'shape_error': d['shape_results'] = [{'shape_index': 0, 'correctness': 'error', 'error_summary': 'unknown'}]
    elif change == 'no_frame': d['stage_timeline'] = []
    elif change == 'multiple_failed': d['stage_timeline'].append(copy.deepcopy(d['stage_timeline'][0]))
    elif change == 'no_device': d['stage_timeline'][0]['lease'] = {}
    elif change == 'no_worker': d['error_stage'] = 'first_call_combined'
    assert sync_aclnn_failure(d) is None
    assert classify(backend='af-mlir', declared_status='timeout', detail=d)[0] == 'timeout'


def test_scoped_sync_refresh_and_rollback(env):
    from autofusex.server.reclassification import aclnn_plan, apply_aclnn, rollback
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('sync-aclnn', status='timeout', detail=evidence())]))
    with store.write() as c:
        c.execute("UPDATE executions SET function_status='timeout',fail_stage='unknown',failure_owner='infrastructure',classifier_version=2")
    changes = aclnn_plan(store.reader())
    assert len(changes) == 1
    assert apply_aclnn(store, 'sync-aclnn-test', changes) == 1
    assert aclnn_plan(store.reader()) == []
    assert rollback(store, 'sync-aclnn-test') == 1
    assert tuple(store.reader().execute('SELECT function_status,fail_stage FROM executions').fetchone()) == ('timeout','unknown')


def test_new_upload_timeout_corrected_by_server(http):
    code, result = http('POST', '/api/client/executions', body([item('sync-http', status='timeout', detail=evidence())]))
    assert code == 200 and result['data']['accepted'] == 1
    code, result = http('GET', '/api/overview/matrix?backends=af-mlir')
    assert code == 200 and result['data']['cells']['af-mlir']['all']['groups'][0]['results'] == {'aclnn':1}


def aicpu_evidence():
    d = evidence()
    d['contexts']['execution']['diagnostics'] = {'error': 'RuntimeError: NPU execution failed', 'shapes': {}}
    d['plog_errors'] = ['[ERROR] RUNTIME(123,python): ProcLogicCqReport:Task run failed, sqe_type=1(aicpu)',
                        '[ERROR] RUNTIME(123,python): msg:aicpu execute failed.']
    return d


def test_aicpu_is_distinct_and_aclnn_takes_priority():
    from autofusex.server.result_categories import category
    d = aicpu_evidence()
    assert native_runtime_failure(d, 'af-mlir') == ('runtime_fail', 'aicpu', 'npu_runtime')
    assert classify(backend='af-mlir', declared_status='timeout', detail=d)[:2] == ('runtime_fail', 'aicpu')
    assert category('runtime_fail','aicpu',None) == 'aicpu'
    d['plog_errors'] += evidence()['plog_errors']
    assert native_runtime_failure(d, 'af-mlir')[1] == 'aclnn'
    # The generic wrapper and synchronization API wording are irrelevant.
    d['contexts']['execution']['diagnostics'] = {'error': 'other wrapper', 'shapes': {}}
    assert native_runtime_failure(d, 'af-mlir')[1] == 'aclnn'


def test_aicpu_requires_error_evidence_and_matching_scope():
    d = aicpu_evidence()
    d['plog_errors'] = [d['plog_errors'][0]]
    assert native_runtime_failure(d, 'af-mlir') is None
    d = aicpu_evidence()
    d['plog_errors'] = [line.replace('123,','999,') for line in d['plog_errors']]
    assert native_runtime_failure(d, 'af-mlir') is None
    d = aicpu_evidence()
    d['plog_errors'] = [line.replace('[ERROR]','[INFO]') for line in d['plog_errors']]
    assert native_runtime_failure(d, 'af-mlir') is None
    d = {'execution': {'diagnostics': {'shapes': {'0': {'error': 'RuntimeError: aicpu execute failed'},
                                               '1': {'error': 'RuntimeError: unknown'}}}}}
    assert native_runtime_failure(d, 'af-mlir') is None
    d['execution']['diagnostics']['shapes']['1']['error'] = 'RuntimeError: AI CPU kernel task execution failed'
    assert native_runtime_failure(d, 'af-mlir')[1] == 'aicpu'


def test_aicpu_refresh_rollback_and_http(http, env):
    from autofusex.server.reclassification import native_runtime_plan, apply_native_runtime, rollback
    from autofusex.server import queries
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('aicpu-scoped', status='timeout', detail=aicpu_evidence())]))
    with store.write() as c:
        c.execute("UPDATE executions SET function_status='timeout',fail_stage='unknown',classifier_version=2")
    changes = native_runtime_plan(store.reader())
    assert len(changes) == 1 and changes[0]['after'][1] == 'aicpu'
    assert apply_native_runtime(store, 'aicpu-test', changes) == 1
    assert queries.backend_stats(store,'af-mlir')['groups'][0]['results'] == {'aicpu':1}
    assert native_runtime_plan(store.reader()) == []
    assert rollback(store, 'aicpu-test') == 1
    code, result = http('POST','/api/client/executions',body([item('aicpu-http',status='timeout',detail=aicpu_evidence())],key='aicpu-http'))
    assert code == 200 and result['data']['accepted'] == 1
    code, result = http('GET','/api/overview/matrix?backends=af-mlir')
    assert result['data']['cells']['af-mlir']['all']['groups'][0]['results'] == {'aicpu':1}


def test_aicpu_legend_hidden_and_searchable():
    import shutil, subprocess
    from pathlib import Path
    text = Path('autofusex/server/static/result-filter.js').read_text().replace('export ', '')
    script = text[text.index('const RESULT_GROUPS ='):text.index('const RESULT_COLORS =')]
    node = shutil.which('node')
    if not node: pytest.skip('node unavailable')
    subprocess.run([node,'-e',script+";require('assert').strict.equal(DEFAULT_RESULTS.includes('aicpu'),false);require('assert').strict(RESULT_ITEMS.some(x=>x[0]==='aicpu'));"],check=True)
    assert '"aicpu"' in Path('autofusex/server/static/core.js').read_text()
