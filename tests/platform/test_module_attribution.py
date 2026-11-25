import json
import pytest
from autofusex.execution.module_attribution import boundary_failure, classify_module
from autofusex.execution.failure_details import excerpt
from autofusex.server.failure_attribution import classify
from autofusex.server.result_categories import category
from test_platform import env, http, body, item, upload
from autofusex.server import queries

@pytest.mark.parametrize('line,stage', [('fuser.schedule(None)','schedule'),('fuser.codegen(None)','codegen'),('fuser = Autofuser(None)','af_fronted')])
def test_generated_frame_survives_exception_wrapper(line,stage):
    code=compile(line,'<string>','exec')
    try: exec(code,{})
    except Exception as exc:
        fact=boundary_failure(exc,{code:line.splitlines()})
        assert fact['stage']==stage
        exc._afx_module_failure=fact
        try: raise RuntimeError('wrapper '+'x'*5000) from exc
        except Exception as wrapped:
            text=excerpt(wrapped)
    assert len(text)<=1600
    assert classify_module(text,'af-mlir')[1] == ('unknown' if stage=='codegen' else stage)

def test_source_mentions_are_not_execution_proof():
    assert classify_module('Failed to execute graph code: fuser.schedule(x); fuser.codegen(x)','af-mlir') is None

def test_nested_detail_and_aclnn():
    detail={'contexts':{'execution':{'diagnostics':json.dumps({'shapes':{'0':{'error':'AFX_MODULE_V1={"stage":"schedule"}\nfailed'}}})}}}
    assert classify(backend='af-mlir',declared_status='runtime_fail',detail=json.dumps(detail).encode())[:2]==('compile_fail','schedule')
    assert classify(backend='af-mlir',declared_status='runtime_fail',summary='aclnnSign failed')[:2]==('runtime_fail','aclnn')

def test_codegen_requires_branch_evidence():
    marker='AFX_MODULE_V1={"stage":"codegen"}\n'
    assert classify_module(marker+'MLIR Codegen Result：kFailed','af-mlir')[1]=='unknown'
    assert classify_module(marker+'MLIR Codegen Result：kNotSupported','af-mlir')[1]=='unknown'
    assert classify_module('AFX_MODULE_V1={"stage":"mlir"}','af-mlir')[1]=='mlir'
    assert classify_module(marker,'af-legacy')[1]=='unknown'

def test_ingest_and_exclusive_overview(env):
    store,ingestor,principal=env
    assert upload(ingestor,principal,body([item('new-asc-0001',status='ascgraph_fail',fail_stage='af_fronted',
        log_summary='AFX_MODULE_V1={"stage":"af_fronted"}\nTypeError: bad graph')]))['results'][0]['status']=='accepted'
    result=queries.backend_stats(store,'af-mlir')['groups'][0]
    assert result['results']=={'ascgraph_fail':1}
    assert sum(result['results'].values())==result['total']
    assert category('pass',None,None)=='unchecked'
    assert category('runtime_fail','aclnn',None)=='aclnn'

def test_shared_aclnn_log_does_not_override_schedule():
    assert classify(backend='af-mlir',declared_status='compile_fail',declared_stage='schedule',
                    summary='Schedule failed',log='another shape: aclnnSign failed')[:2]==('compile_fail','schedule')

def test_native_mlir_log_capture_preserves_error():
    from autofusex.execution.core_logs import _Tail
    tail=_Tail();tail.feed(b'[ERROR] MLIR codegen: failed lowering\n'+b'noise\n'*1000)
    assert 'MLIR codegen:' in tail.finish()

def test_http_client_classification_roundtrip(http):
    from autofusex.client.platform_transport import function_status
    diagnostic={'shapes':{'0':{'error':'AFX_MODULE_V1={"stage":"af_fronted"}\nTypeError: bad graph'}}}
    status,stage,_=function_status('af-mlir','error','first_call_combined','InductorError',diagnostic,0)
    payload=body([item('http-module-0001',status=status,fail_stage=stage,
                       detail={'execution':{'diagnostics':json.dumps(diagnostic)}})])
    code,result=http('POST','/api/client/executions',payload)
    assert code==200 and result['data']['accepted']==1
    code,result=http('GET','/api/overview/matrix?backends=af-mlir')
    assert code==200
    group=result['data']['cells']['af-mlir']['all']['groups'][0]
    assert group['results']=={'ascgraph_fail':1} and group['total']==1

def test_overview_uses_exclusive_results_and_selected_denominator():
    import shutil,subprocess
    from pathlib import Path
    node=shutil.which('node')
    if not node:pytest.skip('node unavailable')
    text=Path('autofusex/server/static/page-overview.js').read_text()
    start=text.index('function stageMetric(');end=text.index('function donutChart(',start)
    script='const assert=require("assert").strict;\nconst pct=(a,b)=>b?(100*a/b).toFixed(1)+"%":"—";\n'+text[start:end]+'''
const group={total:10,results:{pass:3,aclnn:4,compile_unknown:2,unchecked:1},stage_fail:{first_call_combined:4,unknown:2}};
assert.equal(selectedTotal(group,new Set(Object.keys(group.results))),10);
assert.equal(stageMetric(group,'compile_unknown'),2);
assert.equal(selectedTotal(group,new Set(['pass'])),3);
assert.equal(selectedPassRate(group,new Set(Object.keys(group.results))),'30.0%');
assert.equal(selectedPassRate(group,new Set(['pass','unchecked'])),'75.0%');
assert.equal(selectedPassRate(group,new Set(['pass'])),'100.0%');
assert.equal(selectedPassRate(group,new Set(['aclnn'])),'0.0%');
assert.equal(selectedPassRate(group,new Set()),'—');
assert.equal(selectedPassRate(group,new Set(['nnsmith'])),'—');
assert.equal(group.total,10);
'''
    subprocess.run([node,'-e',script],check=True,capture_output=True,text=True)
    assert 'pct(stageMetric(g,"pass"), g.total)' not in text


def test_terminal_shape_aclnn_outranks_native_summary_and_shared_mlir():
    local = 'RuntimeError: median: NPU function error: call aclnnMedian failed, error code is 161002'
    detail = {'contexts': {'execution': {'diagnostics': json.dumps({
        'shapes': {'0': {'error': local}},
        'error': "error: 'afir.minimum' op has no supported Ascend implementation"})}},
        'shape_results': [{'shape_index': 0, 'correctness': 'error', 'error_summary': local}]}
    for backend in ('af-legacy', 'af-mlir', 'triton-npu'):
        result = classify(backend=backend, declared_status='runtime_fail', declared_stage='unknown',
                          summary='[ERROR] OP CheckParams OpName:[aclnnMedian_5] BF16 unsupported', detail=detail)
        assert result[:3] == ('runtime_fail', 'aclnn', 'npu_runtime')
    assert classify(backend='af-mlir', declared_status='compile_fail', declared_stage='mlir',
                    detail=json.dumps(detail).encode())[:2] == ('runtime_fail', 'aclnn')
    assert classify(backend='af-mlir', declared_status='runtime_fail', declared_stage='unknown_retest',
                    detail=detail)[:2] == ('runtime_fail', 'aclnn')


def test_local_aclnn_requires_all_failing_shapes_and_ignores_shared_log():
    from autofusex.server.failure_attribution import local_aclnn_failure
    d = {'execution': {'diagnostics': {'shapes': {'0': {'error': 'call aclnnSign failed'},
                                                '1': {'error': 'AFX_MODULE_V1={"stage":"schedule"}\nfailed'}},
                                    'error': 'call aclnnMedian failed'}}}
    assert local_aclnn_failure(d, 'af-mlir') is None
    d['execution']['diagnostics']['shapes'] = {}
    assert local_aclnn_failure(d, 'af-mlir') is None
    d['shape_results'] = [{'shape_index': 0, 'correctness': 'error', 'error_summary': 'call aclnnSign failed'},
                          {'shape_index': 1, 'correctness': 'error', 'error_summary': ''}]
    assert local_aclnn_failure(d, 'af-mlir') is None
    d['shape_results'][1]['error_summary'] = 'call aclnnMedian failed'
    assert local_aclnn_failure(d, 'af-mlir') == ('runtime_fail', 'aclnn', 'npu_runtime')


def test_local_boundary_marker_beats_aclnn_text_in_same_shape():
    from autofusex.server.failure_attribution import local_aclnn_failure
    d = {'execution': {'diagnostics': {'shapes': {'0': {'error':
        'AFX_MODULE_V1={"stage":"schedule"}\ncall aclnnSign failed'}}}}}
    assert local_aclnn_failure(d, 'af-mlir') is None


def test_scoped_aclnn_refresh_and_rollback(env):
    from autofusex.server.reclassification import aclnn_plan, apply_aclnn, rollback
    store, ingestor, principal = env
    detail = {'execution': {'diagnostics': {'shapes': {'0': {'error': 'call aclnnMedian failed'}}}}}
    upload(ingestor, principal, body([item('aclnn-scoped-1', status='runtime_fail', detail=detail)]))
    with store.write() as c:
        c.execute("UPDATE executions SET fail_stage='unknown',failure_owner='unknown',classifier_version=2")
    plan = aclnn_plan(store.reader())
    assert len(plan) == 1 and plan[0]['is_head']
    assert apply_aclnn(store, 'aclnn-test', plan) == 1
    assert queries.backend_stats(store, 'af-mlir')['groups'][0]['results'] == {'aclnn': 1}
    assert aclnn_plan(store.reader()) == []
    assert rollback(store, 'aclnn-test') == 1
    assert store.reader().execute('SELECT fail_stage FROM executions').fetchone()[0] == 'unknown'


def test_http_aclnn_shape_roundtrip(http):
    from autofusex.server.failure_attribution import VERSION
    detail = {'contexts': {'execution': {'diagnostics': json.dumps({
        'shapes': {'0': {'error': 'NPU function error: call aclnnMedian failed, error code is 161002'}}})}}}
    payload = body([item('http-aclnn-shape', status='runtime_fail', fail_stage='unknown',
                         log_summary='[ERROR] OP CheckParams OpName:[aclnnMedian_5]', detail=detail)])
    code, result = http('POST', '/api/client/executions', payload)
    assert code == 200 and result['data']['accepted'] == 1
    code, result = http('GET', '/api/overview/matrix?backends=af-mlir')
    assert code == 200 and result['data']['cells']['af-mlir']['all']['groups'][0]['results'] == {'aclnn': 1}
    code, result = http('GET', '/api/client/protocol')
    assert code == 200 and result['data']['classifier_version'] == VERSION


def pyascir_detail(cause='Check dtype failed for graph_hint/cast Cast; input_dtypes: [DT_INT64], output_dtypes: [DT_FLOAT]'):
    line = '[ERROR] GE(123,python):2026-11-25 [./autofuse/compiler/py_module/pyascir_types.cpp:234]123 DoInference: ErrorNo: 4294967295(failed)Assert '+cause+' failed'
    d = {'plog_errors': [line], 'contexts': {'execution': {'diagnostics': json.dumps({
        'shapes': {'0': {'error': 'generated fuser.schedule(x); fuser.codegen(x)\n'+cause}},
        'error': "error: 'afir.minimum' op has no supported Ascend implementation"})}},
        'shape_results': [{'shape_index': 0, 'correctness': 'error', 'error_summary': cause}]}
    return line, d


def test_pyascir_native_inference_proves_frontend():
    line, d = pyascir_detail()
    for backend in ('af-mlir', 'af-legacy'):
        result = classify(backend=backend, declared_status='compile_fail', declared_stage='unknown_retest',
                          summary=line, detail=json.dumps(d).encode())
        assert result[:3] == ('ascgraph_fail', 'af_fronted', 'autofuse')
    from autofusex.server.failure_attribution import frontend_failure
    assert frontend_failure('af-mlir', line) == ('ascgraph_fail', 'af_fronted', 'autofuse')
    assert frontend_failure('triton-npu', line, d) is None


def test_pyascir_filename_alone_or_unrelated_shape_not_proof():
    from autofusex.server.failure_attribution import frontend_failure
    line, d = pyascir_detail()
    assert frontend_failure('af-mlir', line.replace('[ERROR]', '[INFO]')) is None
    assert frontend_failure('af-mlir', line.replace('pyascir_types.cpp', 'schedule.cpp')) is None
    assert frontend_failure('af-mlir', 'generated source mentions pyascir_types.cpp') is None
    diag = json.loads(d['contexts']['execution']['diagnostics'])
    diag['shapes']['1'] = {'error': 'call aclnnMedian failed'}
    d['contexts']['execution']['diagnostics'] = diag
    assert frontend_failure('af-mlir', line, d) is None
    diag['shapes']['1'] = {'error': 'AFX_MODULE_V1={"stage":"schedule"}\n'+d['shape_results'][0]['error_summary']}
    assert frontend_failure('af-mlir', line, d) is None


def test_frontend_scoped_refresh_removes_retest_marker_and_rolls_back(env):
    from autofusex.server.reclassification import frontend_plan, apply_frontend, rollback
    store, ingestor, principal = env
    line, d = pyascir_detail()
    upload(ingestor, principal, body([item('pyascir-scoped', status='compile_fail', detail=d, log_summary=line)]))
    with store.write() as c:
        c.execute("UPDATE executions SET function_status='compile_fail',fail_stage='unknown_retest',classifier_version=2")
    changes = frontend_plan(store.reader())
    assert len(changes) == 1
    assert apply_frontend(store, 'pyascir-test', changes) == 1
    assert queries.backend_stats(store, 'af-mlir')['groups'][0]['results'] == {'ascgraph_fail': 1}
    assert frontend_plan(store.reader()) == []
    assert rollback(store, 'pyascir-test') == 1
    assert store.reader().execute('SELECT fail_stage FROM executions').fetchone()[0] == 'unknown_retest'


def test_pyascir_frontend_does_not_use_hintgraph_keyword():
    from autofusex.server.failure_attribution import frontend_failure
    error = 'TypeError: Argument must be a HintGraph object'
    d = {'execution': {'diagnostics': {'shapes': {'0': {'error': error}}}}}
    assert frontend_failure('af-mlir', detail=d) is None
    hint_log = '[ERROR] GE [other.cpp:12] Init: ErrorNo: 4294967295(failed)Assert Argument must be a HintGraph object failed'
    assert frontend_failure('af-mlir', summary=hint_log, detail=d) is None
    d['execution']['diagnostics']['shapes']['0']['error'] = 'TypeError: pyascir graph argument is invalid'
    assert frontend_failure('af-mlir', detail=d) == ('ascgraph_fail', 'af_fronted', 'autofuse')
    line = '[ERROR] GE [./autofuse/compiler/py_module/pyascir_common_utils.cpp:128]12 Init: ErrorNo: 4294967295(failed)Assert graph copy invalid failed'
    d['execution']['diagnostics']['shapes']['0']['error'] = 'RuntimeError: graph copy invalid'
    assert frontend_failure('af-legacy', summary=line, detail=d) == ('ascgraph_fail', 'af_fronted', 'autofuse')
    d['execution']['diagnostics']['shapes']['0']['error'] = 'RuntimeError: Failed to execute graph code: graph = ascir.HintGraph(x)'
    assert frontend_failure('af-mlir', detail=d) is None
    assert frontend_failure('af-mlir', detail={'execution': {'diagnostics': {'error': error}}}) is None


def test_nnsmith_signature_is_generation_failure_with_local_evidence():
    message = 'ValueError: materialized Torch graph differs from declared NNSmith output signature'
    d = {'contexts': {'execution': {'diagnostics': {'shapes': {
        str(i): {'error': 'Traceback (most recent call last):\n'+message+'\n'} for i in range(3)}}}}}
    for backend in ('triton-npu', 'af-mlir', 'af-legacy'):
        assert classify(backend=backend, declared_status='runtime_fail', declared_stage='unknown', detail=d)[:3] == ('gen_fail', 'nnsmith', 'nnsmith')
    assert classify(backend='triton-npu', declared_status='compile_fail', declared_stage='unknown_retest', detail=d)[:2] == ('gen_fail', 'nnsmith')
    assert category('gen_fail', 'nnsmith', None) == 'nnsmith'
    from autofusex.server.failure_attribution import nnsmith_failure
    assert nnsmith_failure(message) == ('gen_fail', 'nnsmith', 'nnsmith')
    assert nnsmith_failure(detail={'execution': {'diagnostics': {'error': message}}}) is None
    d['contexts']['execution']['diagnostics']['shapes']['2']['error'] = 'call aclnnSign failed'
    assert nnsmith_failure(message, d) is None
    d['contexts']['execution']['diagnostics']['shapes'] = {'0': {'error': 'generated raise ValueError("materialized Torch graph differs from declared NNSmith output signature")'}}
    assert nnsmith_failure(detail=d) is None


def test_nnsmith_scoped_refresh_and_rollback(env):
    from autofusex.server.reclassification import nnsmith_plan, apply_nnsmith, rollback
    store, ingestor, principal = env
    message = 'ValueError: materialized Torch graph differs from declared NNSmith output signature'
    d = {'shape_results': [{'shape_index': 0, 'correctness': 'error', 'error_summary': message}]}
    upload(ingestor, principal, body([item('nnsmith-scoped', backend='triton-npu', status='runtime_fail', detail=d)]))
    with store.write() as c:
        c.execute("UPDATE executions SET function_status='runtime_fail',fail_stage='unknown',classifier_version=3")
    changes = nnsmith_plan(store.reader())
    assert len(changes) == 1 and apply_nnsmith(store, 'nnsmith-test', changes) == 1
    assert queries.backend_stats(store, 'triton-npu')['groups'][0]['results'] == {'nnsmith': 1}
    assert nnsmith_plan(store.reader()) == []
    assert rollback(store, 'nnsmith-test') == 1
    assert store.reader().execute('SELECT fail_stage FROM executions').fetchone()[0] == 'unknown'
