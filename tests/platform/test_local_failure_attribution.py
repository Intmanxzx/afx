import copy
import json

import pytest

from autofusex.server.failure_attribution import VERSION, classify
from autofusex.server.local_failure_attribution import local_failure
from autofusex.server.reclassification import local_failure_plan, apply_local_failure, rollback
from autofusex.server.store import put_content
from test_platform import env, http, body, item, upload


def frame(path, function):
    return f'  File "/venv/{path}", line 123, in {function}\n'


ERRORS = [
    (frame('torch/_inductor/fx_passes/post_grad.py', 'remove_noop_ops') +
     "TypeError: int_noop() got an unexpected keyword argument 'decimals'", 'gen_fail', 'inductor'),
    (frame('torch/_inductor/fx_passes/post_grad.py', 'same_meta') +
     "AttributeError: 'SymInt' object has no attribute 'size'", 'gen_fail', 'inductor'),
    (frame('torch/_functorch/_aot_autograd/functional_utils.py', 'assert_functional_graph') +
     'AssertionError: aot_autograd expected to have an entirely functional graph, but found aten.bitwise_and_',
     'gen_fail', 'inductor'),
    (frame('torch_npu/_inductor/ascendc/lowering/index_expr_lowering.py', '_validate_expression') +
     'torch._inductor.exc.InductorError: UnsupportedIndexExpr: Unsupported index_expr (torch.float32): ToFloat(ks1)',
     'gen_fail', 'inductor'),
    (frame('torch/_inductor/ir.py', 'process_kernel') +
     "InductorError: LoweringException: RuntimeError: aten::pow() Expected a value of type 'Tensor' "
     "for argument 'exponent' but instead found type 'SymInt'.", 'gen_fail', 'inductor'),
    (frame('torch_npu/_inductor/ascendc/common/utils.py', 'validate_lib') +
     'torch._inductor.exc.InductorError: PermissionError: /tmp/kernel.so must not be writable by group or others, but got 0o775',
     'runtime_fail', 'environment'),
    (frame('torch_npu/_inductor/ascendc/compiler/_kernel.py', 'load') +
     'RuntimeError: NPU kernel /tmp/autofused_add_abc/kernel init failed', 'runtime_fail', 'fused_kernel'),
    (frame('autofuse/ascendc_compile.py', 'run_compile_command') +
     'autofuse.ascendc_compile.CompileError: Device compile failed with code 1', 'compile_fail', 'af_common'),
]

ATT_GE = ('[ERROR] GE(1155915,python):2026-11-25-23:49:16.821.942 '
          '[./autofuse/att/gen_model_info/api_perf_register/v1/ascir_api_perf_v1.cpp:443]'
          '1155915 BrcbToOneBlockPerf: ErrorNo: 4294967295(failed) Data type [bfloat16] unsatisfied.')
VECTOR_CORE = '[Error]: The vector core execution is abnormal.'
MLIR_KERNEL = ('AF_MLIR_STRICT gate=none shortfall=no graph=autofused_add_abc\n'
               "RuntimeError: Failed to build ascend kernel, trigger bug by following command: 'cmd'")


def detail(*errors, serialized=True):
    diagnostics = {'shapes': {str(i): {'error': error} for i, error in enumerate(errors)}}
    return {'contexts': {'execution': {'diagnostics': json.dumps(diagnostics) if serialized else diagnostics}}}


@pytest.mark.parametrize('error,status,stage', ERRORS)
@pytest.mark.parametrize('serialized', [True, False])
def test_approved_rules_use_each_shape_trace(error, status, stage, serialized):
    d = detail(error, serialized=serialized)
    result = local_failure(d, 'af-mlir')
    assert result[:2] == (status, stage)
    assert classify(backend='af-mlir', declared_status='runtime_fail', declared_stage='unknown_retest',
                    detail=json.dumps(d)) == (*result, VERSION)


@pytest.mark.parametrize('error,status,stage', ERRORS)
def test_shared_logs_missing_frames_and_conflicts_are_not_evidence(error, status, stage):
    assert local_failure(detail(error.split('\n')[-1]), 'af-mlir') is None
    assert local_failure(detail(error.split('\n')[0]), 'af-mlir') is None
    assert local_failure({'execution': {'diagnostics': {'error': error}}}, 'af-mlir') is None
    assert local_failure(detail(error, 'truncated failure'), 'af-mlir') is None
    for conflict in ['AFX_MODULE_V1={"stage":"schedule"}', 'RuntimeError: call aclnnMedian failed',
                     'RuntimeError: out of memory', 'RuntimeError: aicpu execute failed',
                     'MLIR Codegen Result: kFailed']:
        assert local_failure(detail(error + '\n' + conflict), 'af-mlir') is None
    d = detail(error)
    d['shape_results'] = [{'shape_index': 1, 'correctness': 'error', 'error_summary': ''}]
    assert local_failure(d, 'af-mlir') is None
    assert classify(backend='af-mlir', declared_status='runtime_fail', log=error,
                    detail=detail('unresolved'))[:2] == ('runtime_fail', 'unknown_retest')


def test_module_consistency_backend_and_status_scope():
    assert local_failure(detail(ERRORS[0][0], ERRORS[1][0]), 'triton-npu')[:2] == ('gen_fail', 'inductor')
    assert local_failure(detail(ERRORS[0][0], ERRORS[5][0]), 'af-legacy') is None
    for error, _, _ in ERRORS[-2:]:
        assert local_failure(detail(error), 'triton-npu') is None
    for status in ['pass', 'unsupported', 'timeout']:
        assert classify(backend='af-mlir', declared_status=status, detail=detail(ERRORS[0][0]))[0] == status
    execution = ERRORS[6][0].replace('init failed', 'execution failed(-1)').replace('in load', 'in __call__')
    assert local_failure(detail(ERRORS[6][0], execution), 'af-mlir')[1] == 'fused_kernel'
    host = ERRORS[7][0].replace('Device compile failed with code 1',
                              'Host compile failed for /tmp/a.cpp: Host compile failed with code 1')
    assert local_failure(detail(host), 'af-legacy')[1] == 'af_common'


def test_source_literals_do_not_create_traceback_evidence():
    error = ERRORS[0][0]
    assert local_failure(detail(repr(error)), 'af-mlir') is None
    assert local_failure(detail(error.replace('TypeError:', 'raise TypeError(')), 'af-mlir') is None
    assert local_failure(detail(error.replace('in remove_noop_ops', 'in unrelated')), 'af-mlir') is None


@pytest.mark.parametrize('error,status,stage,backend', [
    (ATT_GE, 'compile_fail', 'af_common', 'af-legacy'),
    (VECTOR_CORE, 'runtime_fail', 'fused_kernel', 'af-mlir'),
    (MLIR_KERNEL, 'runtime_fail', 'fused_kernel', 'af-mlir'),
])
def test_af_native_shape_signatures(error, status, stage, backend):
    assert local_failure(detail(error), backend)[:2] == (status, stage)


def test_scoped_plan_apply_rollback_and_evidence_change(env):
    from autofusex.server import queries
    from autofusex.server.historical_retest import reason

    store, ingestor, principal = env
    d = detail(ERRORS[0][0])
    upload(ingestor, principal, body([item('local-rule', status='runtime_fail', detail=d)]))
    with store.write() as c:
        c.execute("UPDATE executions SET function_status='runtime_fail',fail_stage='unknown_retest',"
                  "failure_owner='unknown',classifier_version=10")
    changes = local_failure_plan(store.reader())
    assert len(changes) == 1 and changes[0]['is_head']
    original = dict(store.reader().execute('SELECT * FROM executions').fetchone())
    assert apply_local_failure(store, 'local-test', changes) == 1
    corrected = dict(store.reader().execute('SELECT * FROM executions').fetchone())
    derived = {'function_status', 'fail_stage', 'failure_owner', 'classifier_version'}
    assert {k: v for k, v in original.items() if k not in derived} == {
        k: v for k, v in corrected.items() if k not in derived}
    assert queries.backend_stats(store, 'af-mlir')['groups'][0]['results'] == {'gen_fail': 1}
    assert local_failure_plan(store.reader()) == []
    assert rollback(store, 'local-test') == 1
    row = store.reader().execute('SELECT * FROM executions').fetchone()
    assert reason(dict(row, fail_stage='unknown'), d) is None

    stale = local_failure_plan(store.reader())
    with store.write() as c:
        c.execute("UPDATE executions SET detail_content_id=?",
                  (put_content(c, json.dumps(detail('unresolved')).encode()),))
    with pytest.raises(ValueError, match='evidence changed'):
        apply_local_failure(store, 'changed-evidence', stale)
    assert not store.reader().execute("SELECT 1 FROM classification_changes WHERE batch_id='changed-evidence'").fetchone()


def test_scoped_manifest_rejects_changed_stage_and_tampered_scope(env):
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('local-scope', status='runtime_fail', detail=detail(ERRORS[0][0]))]))
    with store.write() as c:
        c.execute("UPDATE executions SET fail_stage='unknown_retest'")
    changes = local_failure_plan(store.reader())
    with store.write() as c:
        c.execute("UPDATE executions SET fail_stage='schedule'")
    with pytest.raises(ValueError, match='record changed'):
        apply_local_failure(store, 'stale-local', changes)
    tampered = copy.deepcopy(changes)
    tampered[0]['before'][1] = 'schedule'
    with pytest.raises(ValueError, match='outside local failure correction scope'):
        apply_local_failure(store, 'tampered-local', tampered)


def test_upload_and_overview_use_server_rules(http):
    code, response = http('POST', '/api/client/executions', body([
        item('local-http', status='runtime_fail', detail=detail(ERRORS[6][0]))]))
    assert code == 200 and response['data']['accepted'] == 1
    code, response = http('GET', '/api/overview/matrix?backends=af-mlir')
    assert code == 200
    assert response['data']['cells']['af-mlir']['all']['groups'][0]['results'] == {'fused_kernel': 1}


def test_preview_tool_opens_existing_databases_without_store(env, tmp_path, monkeypatch, capsys):
    import sys
    from tools.classification_local_failure import main
    from autofusex.server.store import Store

    store, ingestor, principal = env
    upload(ingestor, principal, body([item('preview-local', status='runtime_fail', detail=detail(ERRORS[0][0]))]))
    with store.write() as c:
        c.execute("UPDATE executions SET fail_stage='unknown_retest'")
    def reject_store(*args, **kwargs):
        raise AssertionError('preview must not initialize or migrate Store')
    monkeypatch.setattr(Store, '__init__', reject_store)
    manifest = tmp_path / 'manifest.json'
    monkeypatch.setattr(sys, 'argv', ['classification_local_failure.py', str(store.dir),
                                    '--batch', 'preview', '--offline', '--preview', str(manifest)])
    main()
    assert json.loads(capsys.readouterr().out)['current_heads'] == 1
    assert json.loads(manifest.read_text())['changes'][0]['after'] == ['gen_fail', 'inductor', 'torch', VERSION]
    assert store.reader().execute('SELECT fail_stage FROM executions').fetchone()[0] == 'unknown_retest'
