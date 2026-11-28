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
    for (const value of ['统计总览','统计细分','NaN数量（被测/标杆）','+Inf数量（被测/标杆）','−Inf数量（被测/标杆）','错误形态']) if (!rendered.includes(value)) throw new Error(value);
'''
    subprocess.run([node,'-e',harness+source],check=True,capture_output=True,text=True)
