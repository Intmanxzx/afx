from pathlib import Path
import shutil
import subprocess
import pytest

def test_execution_shows_split_accuracy_tables():
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    source = Path('autofusex/server/static/page-case.js').read_text()
    source = source[source.index('function renderExecution('):source.index('// DAG layout')]
    harness = '''
const el = (tag, attrs, ...children) => ({tag, attrs, children:children.flat(Infinity), append(...nodes){this.children.push(...nodes.flat(Infinity));}});
const text = n => Array.isArray(n) ? n.map(text).join(' ') : n && typeof n === 'object' ? text(n.children) : String(n ?? '');
const label=x=>x,statusChip=()=>'',time=()=>'',archLabel=x=>x,secs=()=>'',ns=x=>x;
const d={backend:'af-mlir',function_status:'pass',accuracy_status:'fail',accuracy_ref:'eager-npu',env:{},shapes:[],detail:{accuracy_evidence:{comparisons:[{reference:'eager-npu',passed:false,shapes:[{shape_index:0,passed:false,metrics:{mismatches:1,elements:10,outputs:[{output_index:0,mismatches:1,elements:10,nan_mismatches:1,nonfinite:1}]}}]}]}}};
const rendered=text(renderExecution(d,0));
    for (const value of ['统计总览','统计细分','非有限错误元素','非有限错误原因','NaN数量（被测/标杆）','+Inf数量（被测/标杆）','−Inf数量（被测/标杆）','错误形态']) if (!rendered.includes(value)) throw new Error(value);
const outputs = [
 {output_index:0,elements:1,mismatches:1,finite_pairs:0,finite_mismatches:0,nonfinite:1,nan_mismatches:1,inf_mismatches:0,nonfinite_evidence:{actual:{nan:1,pos_inf:0,neg_inf:0},reference:{nan:1,pos_inf:0,neg_inf:0}}},
 {output_index:1,elements:449,mismatches:0,finite_pairs:389,finite_mismatches:0,nonfinite:60,nan_mismatches:0,inf_mismatches:0,nonfinite_evidence:{actual:{nan:0,pos_inf:60,neg_inf:0},reference:{nan:0,pos_inf:60,neg_inf:0}}}
];
d.detail.accuracy_evidence.comparisons[0].shapes[0].metrics = {elements:450,mismatches:1,nonfinite:61,outputs};
const find = (n, tag) => n && typeof n === 'object' ? [...(n.tag === tag ? [n] : []), ...(n.children || []).flatMap(c => find(c,tag))] : [];
const table = renderExecution(d,0).find(n => n.tag === 'section' && text(n.children[0]) === '统计细分');
const rows = find(table,'tbody')[0].children;
const values = row => row.children.map(text);
const assert = require('assert');
assert.deepStrictEqual(values(rows[0]).slice(4), ['0/389','1/61','NaN 错误 1','1 / 1','60 / 60','0 / 0']);
assert.deepStrictEqual(values(rows[1]).slice(4), ['0/0','1/1','NaN 错误 1','1 / 1','0 / 0','0 / 0']);
assert.strictEqual(rows[1].children[5].attrs.class, 'accuracy-high');
assert.strictEqual(rows[1].children[7].attrs.class, '');
assert.strictEqual(values(rows[2])[5], '0/60');
assert.strictEqual(rows[2].children[5].attrs.class, 'accuracy-ok');
d.detail.accuracy_evidence.comparisons[0].shapes[0].metrics = {elements:1,mismatches:1};
const unknown = renderExecution(d,0).find(n => n.tag === 'section' && text(n.children[0]) === '统计细分');
assert.deepStrictEqual(values(find(unknown,'tbody')[0].children[0]).slice(4), ['—/—','—/—','—','—','—','—']);
'''
    subprocess.run([node,'-e',harness+source],check=True,capture_output=True,text=True)
