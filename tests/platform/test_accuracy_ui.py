from pathlib import Path
import shutil
import subprocess
import pytest


def test_execution_shows_both_comparisons_and_unavailable_baseline():
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    source = Path('autofusex/server/static/page-case.js').read_text()
    source = source[source.index('function renderExecution('):source.index('// DAG layout')]
    harness = '''
const assert = require('node:assert/strict');
const el = (tag, attrs, ...children) => ({tag, children:children.flat(Infinity)});
const text = n => Array.isArray(n) ? n.map(text).join(' ') : n && typeof n === 'object' ? text(n.children) : String(n ?? '');
const label = x => x, statusChip = () => '', time = () => '', archLabel = x => x, secs = () => '', ns = x => x;
'''
    checks = '''
const d = {backend:'af-mlir', accuracy_status:'pass', accuracy_ref:'eager-npu', env:{}, shapes:[], detail:{accuracy_evidence:{comparisons:[
 {reference:'cpu',passed:false,shapes:[{shape_index:0,passed:false,metrics:{mismatches:1,elements:100,mismatch_ratio:.01,max_abs:2}}]},
 {reference:'eager-npu',passed:true,shapes:[]}
]}}};
let rendered = text(renderExecution(d, 0));
for (const expected of ['最终标杆：EagerNPU','CPU 首比：失败','EagerNPU 复核：通过','1.0000%']) assert(rendered.includes(expected), expected);
d.accuracy_status='fail'; d.accuracy_ref='cpu';
d.detail.accuracy_evidence.comparisons.pop(); d.detail.baseline_status='unavailable';
assert(text(renderExecution(d,0)).includes('EagerNPU 复核：执行不可用'));
'''
    subprocess.run([node, '-e', harness + source + checks], check=True, capture_output=True, text=True)
