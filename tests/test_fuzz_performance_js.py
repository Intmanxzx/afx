"""Execute the performance renderer against a minimal DOM, with many windows."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_performance_scope_grouping_paging_and_refresh():
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    source = Path('autofusedebugger/fuzz/view/static/app.js').read_text()
    helpers = source[source.index('const efficiencyView='):source.index('function renderStatisticsTables(')]
    script = r'''
const assert=require('node:assert/strict');
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.value='';this.open=false;}
 append(...children){this.children.push(...children);}
 replaceChildren(...children){this.children=children;}
 setAttribute(key,value){this[key]=value;}
 get lastChild(){return this.children.at(-1);}
}
const document={createElement:tag=>new Element(tag)}, host=new Element('div');
const $=()=>host;
const seconds=x=>x==null?'':(x/1e9).toFixed(1)+' s';
const cell=(row,value,tag='td')=>{const e=new Element(tag);e.textContent=value;row.append(e);return e;};
const all=(node,tag)=>[...(node.tag===tag?[node]:[]),...node.children.flatMap(c=>all(c,tag))];
const text=node=>[node.textContent,...node.children.map(text)].join(' ');
''' + helpers + r'''
const row=(window_id,env,status='ok')=>({window_id,backend:'af-mlir',environment_id:env,mode:'static',temperature:'cold',status,success:status==='ok'?2:0,failed:status==='ok'?0:1,completed:status==='ok'?2:1,case_errors:0,window_seconds:60,cpu_pool_utilization:.5,npu_slot_utilization:{device: .8},processing_p50_ns:1e9,processing_p95_ns:2e9,e2e_p50_ns:3e9,e2e_p95_ns:4e9,software:{torch:'test'},unique_shapes:2,cold:2,warm:0});
const rows=[];
for(let w=0;w<100;w++)for(let e=0;e<23;e++)rows.push(row('window-'+w,'env-'+e),row('window-'+w,'env-'+e,'error'));
const data={tool_efficiency:{windows:rows,processing:[]}};
const groups=efficiencyGroups(rows.slice(0,4));
assert.equal(groups.length,2);assert.equal(groups[0].success,2);assert.equal(groups[0].completed,3);
renderEfficiency(data);
assert.equal(all(host,'select')[0].value,'');
assert.equal(all(host,'select')[0].children.length,101);
assert.equal(all(host,'table')[0].children.length,11); // bounded overview, not 4600 rows
assert.ok(text(host).includes('见分组明细'));
assert.ok(!text(host).includes('"torch"')); // lazy environment details
let buttons=all(host,'button');buttons[1].onclick();
assert.equal(efficiencyView.page,1);
renderEfficiency(data);assert.equal(efficiencyView.page,1); // background refresh preserves page
let select=all(host,'select')[0];select.value='window-8';select.onchange();
assert.equal(efficiencyView.page,0);assert.ok(text(host).includes('当前窗口：window-8'));
renderEfficiency(data);assert.equal(all(host,'select')[0].value,'window-8');
let input=all(host,'input')[0];input.value='env-22';input.oninput();
assert.equal(all(host,'table')[0].children.length,2);
let details=all(host,'details')[0];details.open=true;details.ontoggle();
assert.ok(text(host).includes('"torch"'));assert.ok(text(host).includes('不能直接平均'));
renderEfficiency(data);assert.equal(all(host,'input')[0].value,'env-22');assert.equal(all(host,'details')[0].open,true);
input=all(host,'input')[0];input.value='missing';input.oninput();assert.ok(text(host).includes('没有匹配'));
efficiencyView.query='';efficiencyView.windowId='';efficiencyView.details=false;
renderEfficiency({tool_efficiency:{windows:[row('new-window','env-new'),...rows]}});
assert.ok(text(host).includes('当前窗口：new-window')); // latest tracks new data
renderEfficiency({});assert.ok(text(host).includes('暂无已完成'));
efficiencyView.windowId='gone';renderEfficiency(data);
assert.equal(efficiencyView.windowId,'');assert.ok(text(host).includes('已切换'));
'''
    subprocess.run([node, '-e', script], check=True, capture_output=True, text=True)
