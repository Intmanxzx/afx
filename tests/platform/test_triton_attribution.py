import json

import pytest

from autofusex.server.failure_attribution import classify
from autofusex.server.triton_attribution import compile_evidence, triton_failure
from autofusex.server import queries
from test_platform import env, http, body, item, upload


EXCEPTION = 'triton.compiler.errors.MLIRCompilationError: '
DTYPE = "operator 'exp2' does not support dtype 'float64' (operand #1); supported dtypes on Atlas A2 products/Atlas A3 products: float32. Please cast the operand to a supported dtype via tl.cast before calling 'exp2'."
NATIVE = 'loc("/tmp/kernel.mlir":1:1): error: Failed to compile BiShengLIR for device\n[ERROR] Failed to run BiShengIR pipeline\n///---[ERROR][Triton][END]---'
INDUCTOR = 'AssertionError: n=copy_, n.args[0]=mul_2, placeholders={arg1_1, arg2_1, arg0_1}, graph=graph():'


def detail(*errors):
    return {'contexts': {'execution': {'diagnostics': json.dumps({
        'shapes': {str(i): {'error': e} for i, e in enumerate(errors)}})}}}


@pytest.mark.parametrize('text,family', [
    (EXCEPTION, 'compiler_exception'),
    ('triton.compiler.errors.CompilationError: at 8:11:\n'+DTYPE, 'compiler_exception'),
    ('File "/venv/triton/backends/ascend/dtype_guard.py", line 524\n'+DTYPE, 'dtype_guard'),
    (NATIVE, 'native_compiler'),
    (INDUCTOR, 'inductor_assertion'),
    ('File "/venv/triton/compiler/compiler.py", line 347\n[ERROR] Failed to run BiShengHIR pipeline', 'native_compiler'),
])
def test_compile_evidence(text, family):
    assert compile_evidence(text) == family
    assert triton_failure('triton-npu', detail=detail(text)) == ('compile_fail', 'triton_ascend', 'triton')
    assert classify(backend='triton-npu', declared_status='runtime_fail', declared_stage='unknown_retest',
                    detail=detail(text))[:3] == ('compile_fail', 'triton_ascend', 'triton')


@pytest.mark.parametrize('text', [
    DTYPE,
    '[ERROR] Failed to run BiShengIR pipeline',
    'File "/venv/triton/compiler/compiler.py", line 347',
    'MLIR kernel.mlir hivm.hir.vbrc',
    'generated raise CompilationError("triton.compiler.errors.CompilationError:")',
    'RuntimeError: Failed to execute graph code: triton.compiler.errors.CompilationError:',
    EXCEPTION+'\nRuntimeError: call aclnnMedian failed',
    EXCEPTION+'\nAFX_MODULE_V1={"stage":"schedule"}',
    EXCEPTION+'\nRuntimeError: out of memory',
    EXCEPTION+'\nPermissionError: permission denied',
    EXCEPTION+'\nValueError: materialized Torch graph differs from declared NNSmith output signature',
])
def test_insufficient_or_conflicting_evidence(text):
    assert compile_evidence(text) is None


def test_shape_scope_and_backend_protection():
    assert triton_failure('af-mlir', detail=detail(EXCEPTION)) is None
    assert triton_failure('af-legacy', detail=detail(NATIVE)) is None
    assert triton_failure('triton-npu', detail=detail(EXCEPTION, NATIVE))
    d = detail(EXCEPTION, 'InductorError: subprocess failed')
    assert triton_failure('triton-npu', EXCEPTION, d) is None
    d = detail(EXCEPTION)
    d['shape_results'] = [{'shape_index': 1, 'correctness': 'error', 'error_summary': ''}]
    assert triton_failure('triton-npu', detail=d) is None
    shared = {'execution': {'diagnostics': {'error': EXCEPTION}}}
    assert triton_failure('triton-npu', detail=shared) is None
    assert classify(backend='triton-npu', declared_status='runtime_fail', log=EXCEPTION,
                    detail=detail('RuntimeError: unknown'))[:2] == ('runtime_fail', 'unknown_retest')
    assert classify(backend='triton-npu', declared_status='pass', detail=detail(EXCEPTION))[0] == 'pass'
    assert classify(backend='triton-npu', declared_status='timeout', detail=detail(EXCEPTION))[0] == 'timeout'
    assert classify(backend='triton-npu', declared_status='unsupported', detail=detail(EXCEPTION))[0] == 'unsupported'
    assert triton_failure('triton-npu', detail=detail(INDUCTOR)) == ('gen_fail', 'inductor', 'torch')


def test_scoped_plan_stale_checks_rollback_and_retest_exclusion(env):
    from autofusex.server.reclassification import triton_plan, apply_triton, rollback
    from autofusex.server.historical_retest import reason
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('triton-scoped', backend='triton-npu', status='runtime_fail', detail=detail(EXCEPTION))]))
    with store.write() as c:
        c.execute("UPDATE executions SET function_status='runtime_fail',fail_stage='unknown_retest',failure_owner='unknown',classifier_version=6")
    changes = triton_plan(store.reader())
    assert len(changes) == 1 and changes[0]['is_head']
    assert apply_triton(store, 'triton-test', changes) == 1
    assert queries.backend_stats(store, 'triton-npu')['groups'][0]['results'] == {'triton_ascend': 1}
    assert triton_plan(store.reader()) == []
    assert rollback(store, 'triton-test') == 1
    stale = triton_plan(store.reader())
    with store.write() as c:
        c.execute("UPDATE executions SET fail_stage='aclnn'")
    with pytest.raises(ValueError, match='record changed'):
        apply_triton(store, 'stale-triton', stale)
    assert not store.reader().execute("SELECT 1 FROM classification_changes WHERE batch_id='stale-triton'").fetchone()
    row = {'backend': 'triton-npu', 'capture_version': 1, 'function_status': 'runtime_fail', 'fail_stage': 'unknown', 'log_summary': None}
    assert reason(row, detail(NATIVE)) is None


def test_upload_and_matrix(http):
    code, result = http('POST', '/api/client/executions', body([
        item('triton-http', backend='triton-npu', status='runtime_fail', detail=detail(EXCEPTION+'\n'+DTYPE))]))
    assert code == 200 and result['data']['accepted'] == 1
    code, result = http('GET', '/api/overview/matrix?backends=triton-npu')
    assert code == 200
    assert result['data']['cells']['triton-npu']['all']['groups'][0]['results'] == {'triton_ascend': 1}
