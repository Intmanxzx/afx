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
const el = (tag, attrs, ...children) => ({tag, attrs, children:children.flat(Infinity)});
const text = n => Array.isArray(n) ? n.map(text).join(' ') : n && typeof n === 'object' ? text(n.children) : String(n ?? '');
const label=x=>x,statusChip=()=>'',time=()=>'',archLabel=x=>x,secs=()=>'',ns=x=>x;
const d={backend:'af-mlir',accuracy_status:'fail',accuracy_ref:'eager-npu',env:{},shapes:[],detail:{accuracy_evidence:{comparisons:[{reference:'eager-npu',passed:false,shapes:[{shape_index:0,passed:false,metrics:{mismatches:1,elements:10,outputs:[{output_index:0,mismatches:1,elements:10,nan_mismatches:1,nonfinite:1}]}}]}]}}};
const rendered=text(renderExecution(d,0));
for (const value of ['有限值误差','非有限值统计','NaN 失败元素','空间形态']) if (!rendered.includes(value)) throw new Error(value);
'''
    subprocess.run([node,'-e',harness+source],check=True,capture_output=True,text=True)
