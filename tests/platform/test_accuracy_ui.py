from pathlib import Path
import shutil
import subprocess
import pytest


def test_execution_shows_comparison_table_without_summary():
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    source = Path('autofusex/server/static/page-case.js').read_text()
    source = source[source.index('function renderExecution('):source.index('// DAG layout')]
    harness = '''
const assert = require('node:assert/strict');
const el = (tag, attrs, ...children) => ({tag, attrs, children:children.flat(Infinity)});
const text = n => Array.isArray(n) ? n.map(text).join(' ') : n && typeof n === 'object' ? text(n.children) : String(n ?? '');
const label = x => x, statusChip = () => '', time = () => '', archLabel = x => x, secs = () => '', ns = x => x;
'''
    checks = '''
const d = {backend:'af-mlir', accuracy_status:'pass', accuracy_ref:'eager-npu', env:{}, shapes:[], detail:{accuracy_evidence:{comparisons:[
 {reference:'cpu',passed:false,shapes:[{shape_index:0,passed:false,metrics:{mismatches:1,elements:100,mismatch_ratio:.01,max_abs:2}}]},
 {reference:'eager-npu',passed:true,shapes:[{shape_index:0,passed:true,metrics:{mismatches:0,elements:100}}]}
]}}};
let rendered = text(renderExecution(d, 0));
for (const expected of ['af-mlir 与 CPU','af-mlir 与 EagerNPU','1.0000%']) assert(rendered.includes(expected), expected);
d.accuracy_status='fail'; d.accuracy_ref='cpu';
d.detail.accuracy_evidence.comparisons.pop(); d.detail.baseline_status='unavailable';
assert(!text(renderExecution(d,0)).includes('精度判定：'));
d.backend='eager-npu'; d.function_status='pass';
d.detail.accuracy_evidence.comparisons[0].shapes[0].metrics.outputs=[{output_index:0,mismatches:1,elements:100,rmse:0.25,nan_mismatches:1}];
rendered=text(renderExecution(d,0));
for (const expected of ['eager-npu 与 CPU','输出 0','RMSE','NaN 失败元素','不一致']) assert(rendered.includes(expected), expected);
assert(!rendered.includes('EagerNPU 复核：'));
d.function_status='runtime_fail';
assert(!text(renderExecution(d,0)).includes('最终标杆：'));
d.detail.accuracy_evidence.comparisons[0].shapes[0].metrics.outputs[0].spatial_pattern='scattered';
d.detail.accuracy_evidence.comparisons[0].shapes[0].metrics.outputs[0].tolerance_ratio_max=20;
const tree=renderExecution(d,0);
assert(text(tree).includes('零散分布'));
assert(!text(tree).includes('Kernel'));
assert(JSON.stringify(tree).includes('accuracy-medium'));
assert(JSON.stringify(tree).includes('accuracy-high'));


'''
    subprocess.run([node, '-e', harness + source + checks], check=True, capture_output=True, text=True)
