import copy
import json

import pytest

from autofusex.server.failure_attribution import VERSION, classify
from autofusex.server.mlir_attribution import mlir_failure
from autofusex.server.reclassification import mlir_plan, apply_mlir, rollback
from autofusex.server.store import put_content
from test_platform import env, http, body, item, upload


ERROR = ("error: InvalidSchedule: 'afir.load' result escapes its schedule scope but its reduction "
         "semantics cannot be analyzed: input and destination do not agree on one non-empty axis list; "
         "only an accumulating reduce may escape, so either this schedule annotation is broken or "
         "the value needs tensor lifetime lowering first")


def detail(*errors):
    return {'execution': {'diagnostics': json.dumps({'shapes': {
        str(i): {'error': e} for i, e in enumerate(errors)}})}}


@pytest.mark.parametrize('backend', ['af-mlir', 'af-legacy'])
@pytest.mark.parametrize('error', [ERROR, "error: 'afir.minimum' op has no supported Ascend implementation",
    "loc(unknown): error: 'afir.reduce_sum' op result shape must match the inferred reduce shape",
    "test.mlir:3:5: error: 'afir.load' op InvalidDescriptor: invalid rank"])
def test_native_diagnostic_formats(backend, error):
    d = detail('RuntimeError: EOFError:\nWorker log:\n' + error)
    assert mlir_failure(backend, detail=d) == ('compile_fail', 'mlir', 'autofuse')
    assert classify(backend=backend, declared_status='runtime_fail', declared_stage='unknown_retest',
                    detail=d) == ('compile_fail', 'mlir', 'autofuse', VERSION)


def test_worker_death_before_shapes_and_historical_triage():
    from autofusex.server.historical_retest import reason
    d = detail()
    d.update(error_stage='worker', error_summary='RuntimeError: EOFError:\nWorker log:\n' + ERROR)
    assert mlir_failure('af-legacy', 'RuntimeError: EOFError:', d)[1] == 'mlir'
    assert reason(dict(capture_version=1, function_status='runtime_fail', fail_stage='unknown',
                       backend='af-legacy', log_summary='RuntimeError: EOFError:'), d) is None
    del d['error_summary']
    d['execution']['diagnostics'] = {'error': ERROR, 'shapes': {}}
    assert mlir_failure('af-legacy', detail=d)[1] == 'mlir'
    del d['error_stage']
    assert mlir_failure('af-legacy', detail=d) is None


@pytest.mark.parametrize('text', ["'afir.load'", 'InvalidSchedule', 'error: InvalidSchedule: bad axes',
    '%0 = "afir.load"(%arg0)', 'note: ' + ERROR, 'print("' + ERROR + '")', repr(ERROR),
    'error: other compiler mentions afir.load', "error: 'other.load' op failed"])
def test_keywords_ir_and_source_strings_are_not_proof(text):
    assert mlir_failure('af-mlir', detail=detail(text)) is None


def test_scope_conflicts_shared_logs_and_incomplete_shapes():
    assert mlir_failure('triton-npu', detail=detail(ERROR)) is None
    assert mlir_failure('cpu', ERROR) is None
    assert mlir_failure('af-mlir', ERROR, detail('truncated')) is None
    assert mlir_failure('af-mlir', detail=detail(ERROR, 'truncated')) is None
    d = detail(ERROR)
    d['shape_results'] = [{'shape_index': 1, 'correctness': 'error', 'error_summary': ''}]
    assert mlir_failure('af-mlir', detail=d) is None
    d = detail('truncated')
    d['execution']['diagnostics'] = {'error': ERROR, 'shapes': {'0': {'error': 'truncated'}}}
    assert mlir_failure('af-mlir', detail=d) is None
    for conflict in ['AFX_MODULE_V1={"stage":"schedule"}', 'RuntimeError: call aclnnSign failed',
                     'RuntimeError: out of memory', 'RuntimeError: aicpu execute failed']:
        assert mlir_failure('af-mlir', detail=detail(ERROR + '\n' + conflict)) is None
    assert mlir_failure('af-mlir', detail=detail('AFX_MODULE_V1={"stage":"codegen"}\n' + ERROR))[1] == 'mlir'
    for status in ['pass', 'unsupported', 'timeout']:
        assert classify(backend='af-mlir', declared_status=status, detail=detail(ERROR))[0] == status


def test_scoped_migration_is_reversible_and_preserves_facts(env):
    from autofusex.server import queries
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('afir-migrate', status='runtime_fail', detail=detail(ERROR))]))
    with store.write() as c:
        c.execute("UPDATE executions SET function_status='runtime_fail',fail_stage='unknown_retest',"
                  "failure_owner='unknown',classifier_version=11")
    changes = mlir_plan(store.reader())
    assert len(changes) == 1 and changes[0]['is_head']
    before = dict(store.reader().execute('SELECT * FROM executions').fetchone())
    assert apply_mlir(store, 'afir-test', changes) == 1
    after = dict(store.reader().execute('SELECT * FROM executions').fetchone())
    fields = {'function_status', 'fail_stage', 'failure_owner', 'classifier_version'}
    assert {k: v for k, v in before.items() if k not in fields} == {
        k: v for k, v in after.items() if k not in fields}
    assert queries.backend_stats(store, 'af-mlir')['groups'][0]['results'] == {'mlir': 1}
    assert mlir_plan(store.reader()) == []
    assert rollback(store, 'afir-test') == 1
    with store.write() as c:
        c.execute('UPDATE executions SET detail_content_id=?',
                  (put_content(c, json.dumps(detail('truncated')).encode()),))
    with pytest.raises(ValueError, match='evidence changed'):
        apply_mlir(store, 'afir-stale', changes)


def test_manifest_scope_is_rechecked(env):
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('afir-scope', status='runtime_fail', detail=detail(ERROR))]))
    with store.write() as c:
        c.execute("UPDATE executions SET fail_stage='unknown_retest'")
    changes = mlir_plan(store.reader())
    with store.write() as c:
        c.execute("UPDATE executions SET fail_stage='schedule'")
    tampered = copy.deepcopy(changes)
    tampered[0]['before'][1] = 'schedule'
    with pytest.raises(ValueError, match='outside MLIR correction scope'):
        apply_mlir(store, 'afir-scope', tampered)


def test_worker_upload_and_overview(http):
    d = detail()
    d.update(error_stage='worker', error_summary='RuntimeError: EOFError:\nWorker log:\n' + ERROR)
    code, response = http('POST', '/api/client/executions', body([
        item('afir-http', backend='af-legacy', status='runtime_fail', detail=d)]))
    assert code == 200 and response['data']['accepted'] == 1
    code, response = http('GET', '/api/overview/matrix?backends=af-legacy')
    assert code == 200
    assert response['data']['cells']['af-legacy']['all']['groups'][0]['results'] == {'mlir': 1}
