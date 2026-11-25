import json
import pytest

from test_platform import env, body, item, upload, case, SEED_B
from autofusex.server.historical_retest import reason, plan, apply, STAGE
from autofusex.server.reclassification import rollback
from autofusex.server.result_categories import category
from autofusex.server import queries


def record(**kw):
    return dict(capture_version=1, function_status='compile_fail', fail_stage='af_common',
                backend='af-mlir', log_summary='', **kw)


def diagnostic(errors, shared=''):
    return {'contexts': {'execution': {'diagnostics': json.dumps({
        'shapes': {str(i): {'error': e} for i, e in enumerate(errors)}, 'error': shared})}}}


def test_scope_and_shared_log_do_not_prove_module():
    row = record()
    d = diagnostic(['Check dtype failed for graph_hint/cast'], 'MLIR Codegen Result：kFailed')
    assert reason(row, d) == 'legacy_shape_module_unresolved'
    assert reason(dict(row, capture_version=2), d) is None
    assert reason(dict(row, function_status='pass'), d) is None
    assert reason(dict(row, function_status='unsupported'), d) is None
    assert reason(dict(row, function_status='timeout'), d) is None


@pytest.mark.parametrize('error', [
    'RuntimeError: call aclnnNanMedian failed, error code is 161002',
    'AFX_MODULE_V1={"stage":"schedule"}\nerror',
    'File "/venv/torch/_inductor/fx_passes/post_grad.py", line 1130\nTypeError: int_noop()',
    'MLIR Codegen Result：kFailed',
    'Schedule failed',
])
def test_known_local_evidence_is_not_marked(error):
    assert reason(record(), diagnostic([error])) is None


def test_multi_shape_unresolved_is_not_hidden_by_known_shape():
    assert reason(record(), diagnostic(['call aclnnSign failed', 'truncated unknown failure']))
    assert reason(record(), diagnostic(['fuser.schedule(x)\nfuser.codegen(x)']))


def test_preflight_and_triton_excluded():
    assert reason(record(), diagnostic([], 'loaded codegen was built without the MLIR bridge')) is None
    assert reason(dict(record(), backend='triton-npu'), diagnostic(['Failed to run BiShengIR pipeline'])) is None


def test_marker_transaction_filter_retest_and_rollback(env):
    store, ingestor, principal = env
    upload(ingestor, principal, body([
        item('old-unresolved', status='compile_fail', fail_stage='af_common',
             detail=diagnostic(['Check dtype failed for graph_hint/cast'])),
        item('old-known', status='runtime_fail', log_summary='call aclnnSign failed', c=case(SEED_B)),
    ]))
    changes = plan(store.reader())
    assert len(changes) == 1 and changes[0]['exec_uuid'] == 'old-unresolved'
    assert changes[0]['is_head']
    assert apply(store, 'triage', changes) == 1
    results = queries.backend_stats(store, 'af-mlir')['groups'][0]
    assert results['results'] == {'unknown_retest': 1, 'aclnn': 1}
    assert sum(results['results'].values()) == results['total']
    assert len(queries.case_page(store, q='stage:unknown_retest')['groups']) == 1
    assert category('gen_fail', STAGE, None) == STAGE
    assert category('pass', STAGE, None) == 'unchecked'
    assert plan(store.reader()) == []
    with pytest.raises(ValueError, match='batch already exists'):
        apply(store, 'triage', changes)
    # A new execution becomes the head; the historical marker remains auditable.
    upload(ingestor, principal, body([item('new-retest', created=2000)], key='retest-req', run='retest-run', kind='retest'))
    assert queries.case_page(store, q='stage:unknown_retest')['groups'] == []
    assert rollback(store, 'triage') == 1
    assert store.reader().execute("SELECT fail_stage FROM executions WHERE exec_uuid='old-unresolved'").fetchone()[0] == 'af_common'


def test_stale_manifest_rejected_without_partial_updates(env):
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('stale-unknown', status='runtime_fail')]))
    changes = plan(store.reader())
    with store.write() as conn:
        conn.execute("UPDATE executions SET fail_stage='aclnn'")
    with pytest.raises(ValueError, match='record changed'):
        apply(store, 'stale', changes)
    assert store.reader().execute('SELECT count(*) FROM classification_changes').fetchone()[0] == 0


def test_codegen_marker_without_branch_still_requires_retest():
    assert reason(record(), diagnostic(['AFX_MODULE_V1={"stage":"codegen"}\nunknown branch failed']))


def test_graph_identity_required_for_historical_mlir_gate():
    local = 'fuser.codegen(scheduled_autofused_add_abc)\nCodegen generate kernel failed or abort'
    # scheduled_ prefix is not part of the graph identity extracted from source.
    assert reason(record(), diagnostic([local], 'AF_MLIR_STRICT gate=lowering shortfall=no graph=autofused_add_abc')) is None
    assert reason(record(), diagnostic([local], 'AF_MLIR_STRICT gate=lowering shortfall=yes graph=autofused_other_def'))
    assert reason(record(), diagnostic(['Check dtype failed\n'+local.replace('Codegen generate kernel failed or abort','')],
                                      'AF_MLIR_STRICT gate=lowering shortfall=yes graph=autofused_add_abc'))


def test_known_torch_causes_without_complete_traceback():
    assert reason(record(), diagnostic(['UnsupportedIndexExpr: unknown SymPy node'])) is None
    assert reason(record(), diagnostic(['ConstraintViolationError: constraints violated'])) is None


def test_proven_summary_survives_truncated_local_trace():
    row = dict(record(), log_summary='InductorError: LoweringException: bad tensor')
    assert reason(row, diagnostic(['truncated intermediate frames'])) is None
    assert reason(row, diagnostic(['Check dtype failed for graph_hint/cast']))
    assert reason(dict(record(), log_summary='InductorError: PermissionError: build directory'), diagnostic(['truncated'])) is None


def test_retest_legend_default_hidden_and_label():
    import shutil, subprocess
    from pathlib import Path
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    text = Path('autofusex/server/static/page-overview.js').read_text()
    constants = text[text.index('const RESULT_GROUPS'):text.index('const RESULT_COLORS')]
    script = constants + "\nif(DEFAULT_RESULTS.includes('unknown_retest')) throw Error('must start hidden');\n" + \
        "if(!RESULT_ITEMS.some(x=>x[0]==='unknown_retest' && x[2]==='未知·待复测')) throw Error('label missing');"
    subprocess.run([node, '-e', script], check=True, capture_output=True, text=True)
