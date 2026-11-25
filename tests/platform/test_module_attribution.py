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
    assert upload(ingestor,principal,body([item('new-asc-0001',status='ascgraph_fail',fail_stage='af_fronted')]))['results'][0]['status']=='accepted'
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

def test_overview_uses_exclusive_results_and_stable_denominator():
    import shutil,subprocess
    from pathlib import Path
    node=shutil.which('node')
    if not node:pytest.skip('node unavailable')
    text=Path('autofusex/server/static/page-overview.js').read_text()
    start=text.index('function stageMetric(');end=text.index('function donutChart(',start)
    script='const assert=require("assert").strict;\n'+text[start:end]+'''
const group={total:10,results:{pass:3,aclnn:4,compile_unknown:2,unchecked:1},stage_fail:{first_call_combined:4,unknown:2}};
assert.equal(selectedTotal(group,new Set(Object.keys(group.results))),10);
assert.equal(stageMetric(group,'compile_unknown'),2);
assert.equal(selectedTotal(group,new Set(['pass'])),3);
assert.equal(group.total,10);
'''
    subprocess.run([node,'-e',script],check=True,capture_output=True,text=True)
    assert 'pct(stageMetric(g,"pass"), g.total)' in text
