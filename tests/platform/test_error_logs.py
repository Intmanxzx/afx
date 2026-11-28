import json

from autofusex.execution.ascir import construction_source, saved_construction_source, validate_sources
from autofusex.server.error_logs import presentation, summary, merged_log
from autofusex.server import queries
from test_platform import env, body, item, upload


SOURCE = '''from autofuse.pyautofuse import ascir
from autofuse.pyautofuse import Autofuser, AutofuserOptions
"""SchedulerNode comments, not ASCIR"""
graph_hint = ascir.HintGraph('g')
a = ascir.ops.Data('a', graph_hint)
a.y.dtype = ascir.dtypes.float32
fuser = Autofuser(AutofuserOptions(graph_type=1))
scheduled = fuser.schedule(graph_hint)
tiling, host, device = fuser.codegen(scheduled)
'''


def test_construction_is_complete_source_not_traceback_or_scheduling():
    text = construction_source(SOURCE)
    assert 'ascir.HintGraph' in text and 'a.y.dtype' in text
    assert 'SchedulerNode' not in text and 'fuser.schedule' not in text and 'fuser.codegen' not in text
    assert construction_source('RuntimeError: bad graph') == ''
    assert saved_construction_source('RuntimeError: Failed to execute graph code:' + SOURCE + ' Optimize fail') == text
    assert saved_construction_source('RuntimeError: Failed to execute graph code:' + SOURCE[:100]) == ''
    rows = [{'shape_index': 1, 'graph': 'g', 'source': text}]
    assert json.loads(validate_sources(json.dumps(rows))) == rows


def test_source_survives_record_and_shape_view_without_dump():
    from autofusex.execution.storage.records import normalize
    from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
    seed = str(SamplingConfig.profile('small').seed(0))
    rows = [{'shape_index': 1, 'graph': 'g', 'source': construction_source(SOURCE)}]
    result = {'seed': seed, 'workload_id': 'a'*64, 'oracle_id': 'cpu', 'backend': 'af-mlir',
              'status': 'error', 'execution': {'dump': False, 'ascir_sources': json.dumps(rows)}}
    normalized = normalize(result, {})
    assert json.loads(normalized['contexts']['execution']['ascir_sources']) == rows


def test_python_triton_native_and_plog_all_visible_with_scope():
    detail = {'contexts': {'execution': {'diagnostics': json.dumps({'version': 1, 'shapes': {
        '0': {'error': 'ValueError: materialized Torch graph differs from declared NNSmith output signature'},
        '1': {'error': 'triton.compiler.errors.MLIRCompilationError: unsupported dtype'}},
        'error': '[ERROR] Failed to run BiShengIR pipeline'})}},
        'plog_errors': ['[ERROR] GE native failed']}
    view = presentation(detail, 'RuntimeError: saved error')
    logs = '\n'.join(r['text'] for r in view['error_logs'])
    for text in ('NNSmith', 'MLIRCompilationError', 'BiShengIR', 'GE native', 'saved error'):
        assert text in logs
    assert summary(detail).startswith('ValueError:')
    assert view['ascir_graphs'] == []
    assert summary(log='InductorError: wrapper\nValueError: actual failure') == 'ValueError: actual failure'


def test_summary_prefers_concrete_compiler_error_over_qualified_inductor_wrapper():
    error = ('torch._inductor.exc.InductorError: RuntimeError: Failed to build ascend kernel\n'
             '/home/developer/Ascend/include/kernel_impl.h:34:5: error: static assertion failed due to requirement SupportType<__bf16>() '
             'Failed to check dtype in Adds')
    detail = {'contexts': {'execution': {'diagnostics': {'shapes': {'0': {'error': error}}}}}}
    assert summary(detail).startswith('error: static assertion failed')


def test_execution_api_exposes_sources_independent_of_error_logs(env):
    store, ingestor, principal = env
    rows = [{'shape_index': 0, 'graph': 'g', 'source': construction_source(SOURCE)}]
    detail = {'contexts': {'execution': {'ascir_sources': json.dumps(rows)}},
              'shape_views': {'0': {'shape_log': 'RuntimeError: graph failure'}},
              'plog_errors': ['[ERROR] native failure']}
    upload(ingestor, principal, body([item('logs-graph-api', status='compile_fail', detail=detail,
        log_summary='RuntimeError: graph failure')]))
    out = queries.execution_detail(store, 'logs-graph-api')
    view = out['diagnostics']
    assert view['ascir_graphs'] == [{'shape_index': 0, 'source': rows[0]['source']}]
    assert 'RuntimeError' not in view['ascir_graphs'][0]['source']
    assert any('graph failure' in r['text'] for r in view['error_logs'])
    assert out['log_summary'] == 'RuntimeError: graph failure'


def test_merged_log_prefers_traceback_and_adds_only_missing_evidence():
    trace = 'Traceback (most recent call last):\n  File "probe.py", line 1\n    run()\nValueError: actual failure'
    entries = [{'name': '执行错误', 'text': 'ValueError: actual failure'},
               {'name': '运行日志', 'text': trace},
               {'name': '保存日志', 'text': 'ValueError: actual failure\n\n' + trace + '\n\n[ERROR] native detail'},
               {'name': 'CANN plog', 'text': '[ERROR] native detail'}]
    log = merged_log(entries)
    assert log.startswith(trace)
    assert log.count('ValueError: actual failure') == 1
    assert log.count('Traceback (most recent call last):') == 1
    assert log.count('[ERROR] native detail') == 1
    assert '保存日志' not in log and '执行错误' not in log and '运行日志' not in log


def test_merged_log_preserves_independent_shapes_with_identical_errors():
    detail = {'contexts': {'execution': {'diagnostics': {'shapes': {
        '0': {'error': 'ValueError: same cause'}, '1': {'error': 'ValueError: same cause'}}}}}}
    log = presentation(detail)['error_log']
    assert 'Shape 0\nValueError: same cause' in log
    assert 'Shape 1\nValueError: same cause' in log
    assert presentation({}, None)['error_log'] == ''


def test_pages_render_generic_errors_and_only_selected_shape_graph():
    from pathlib import Path
    import shutil
    import subprocess
    import pytest
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    case = Path('autofusex/server/static/page-case.js').read_text()
    discovery = Path('autofusex/server/static/page-discovery.js').read_text()
    script = r'''
const assert=require('node:assert/strict');
class E {
 constructor(tag,attrs,children){this.tag=tag;this.attrs=attrs||{};this.hidden=!!this.attrs.hidden;this.children=children.flat(Infinity).filter(v=>v!=null)}
 append(...children){this.children.push(...children.flat(Infinity).filter(v=>v!=null))}
}
const el=(tag,attrs,...children)=>new E(tag,attrs,children);
const text=n=>Array.isArray(n)?n.map(text).join('\n'):n instanceof E?n.children.map(text).join('\n'):String(n);
const statusChip=()=>'',label=String,archLabel=String,time=String,ns=String,secs=String,stageName=String;
''' + case[case.index('function renderExecution('):case.index('// DAG layout')] + '\n' + discovery[
        discovery.index('export function execValue('):discovery.index('function pivotTable(')].replace('export ', '') + r'''
const d={backend:'af-mlir',env:{},shapes:[],detail:{diagnostic_view:{shape_log:'WRONG ERROR AS GRAPH'}},
 diagnostics:{error_log:'ValueError: python failure\nMLIRCompilationError: bad dtype\n[ERROR] native failed',
 ascir_graphs:[{shape_index:0,source:'GRAPH_ZERO'},{shape_index:1,source:'GRAPH_ONE'}]}};
const rendered=renderExecution(d,1);
// replaceChildren(...rendered) requires top-level DOM nodes, not a nested array.
assert(rendered.every(n=>n instanceof E));
assert.equal(rendered.filter(n=>n.tag==='details').length,0);
assert.equal(rendered.filter(n=>n.tag==='pre' && text(n).includes('python failure')).length,1);
const log=text(rendered);
assert(log.includes('错误日志') && log.includes('ValueError: python failure') && log.includes('MLIRCompilationError: bad dtype'));
assert(!log.includes('CANN plog 错误') && !log.includes('WRONG ERROR AS GRAPH'));
assert(log.includes('GRAPH_ONE') && !log.includes('GRAPH_ZERO'));
assert(text(execValue({backend:'cpu',function_status:'gen_fail',log_summary:'ValueError: cpu error'},'log_summary')).includes('cpu error'));
assert(text(execValue({function_status:'runtime_fail'},'log_summary')).includes('未采集到错误日志'));
assert.equal(execValue({function_status:'pass',accuracy_status:'fail'},'log_summary'),'');
assert.equal(execValue({function_status:'pass',accuracy_status:'fail',log_summary:'accuracy mismatch'},'log_summary'),'');
d.diagnostics.error_log='';
assert(renderExecution(d,1).every(n=>n instanceof E));
assert(text(renderExecution(d,1)).includes('未保存错误日志'));
// Issue views must retain the compact layout when accuracy tables change.
const accuracy={...d,function_status:'pass',accuracy_status:'fail',shapes:[{shape_index:0,kernel_ns:123,e2e_ns:456}]};
const accurate=renderExecution(accuracy,0);
assert(text(accurate).includes('有限值误差') && text(accurate).includes('非有限值统计'));
assert(!text(accurate).includes('有限值误差与非有限值异常分开统计。'));
assert(!text(accurate).includes('错误日志') && !text(accurate).includes('ASCIR 表达的图结构'));
assert(!text(accurate).includes('性能耗时'));
const extra=accurate.find(n=>n.attrs.class==='execution-detail-extra');
const toggle=accurate.find(n=>n.attrs.class==='execution-detail-toggle').children[0];
assert(extra.hidden && text(extra).includes('运行环境') && text(extra).includes('用例执行耗时'));
assert.equal(text(toggle),'查看详细执行信息');
toggle.attrs.onclick();
assert(!extra.hidden && toggle.hidden);
assert(!text(renderExecution({...accuracy,function_status:'compile_fail'},0)).includes('精度对比'));
for(const record of [{...accuracy,issue_dimension:'performance'},{...accuracy,issue:{dimension:'performance'}}]) {
 const performance=renderExecution(record,0);
 assert(text(performance).includes('性能耗时'));
 assert(!text(performance).includes('精度对比') && !text(performance).includes('错误日志'));
}
const requested=renderExecution({...accuracy,issue_dimension:'accuracy'},0,'function');
assert(text(requested).includes('错误日志') && text(requested).includes('GRAPH_ZERO'));
assert(!text(requested).includes('精度对比'));
'''
    subprocess.run([node, '-e', script], check=True, capture_output=True, text=True)
