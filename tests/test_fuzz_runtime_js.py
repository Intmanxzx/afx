"""Fleet rendering stays bounded for multi-user/multi-machine inventories."""
from pathlib import Path
import shutil
import subprocess
import pytest


def test_fleet_rendering_filters_and_paging():
    node=shutil.which('node')
    if not node:pytest.skip('node unavailable')
    source=Path('autofusedebugger/fuzz/view/static/runtime.js').read_text()
    source=source[:source.index("window.addEventListener('hashchange'")]
    script=r'''
const assert=require('node:assert/strict');
class E {constructor(tag){this.tag=tag;this.children=[];this.style={};this.textContent='';}append(...xs){this.children.push(...xs);}replaceChildren(...xs){this.children=xs;}setAttribute(k,v){this[k]=v;}}
const document={createElement:t=>new E(t)};
const all=(e,t)=>[...(e.tag===t?[e]:[]),...e.children.flatMap(x=>all(x,t))];
''' + source + r'''
const machine=i=>({user_name:'user-'+i,user_id:'id-'+i,machine_id:'machine-'+i,name:'host-'+i,status:'online',cards:[{logical:'npu:0',model:'910',uuid:'uuid-'+i,status:'busy'}],runs:[{backend:['af-mlir'],state:'running',stages:{first_call_combined:1},age_seconds:1}],case_errors:0,record_failed:0,completed_per_min:3,success_per_min:2,active_cases:1,queued:2,age_seconds:1});
const data={users:[{id:'a',name:'Alice'},{id:'b',name:'Bob'}],machine_options:[],models:['910'],backends:['af-mlir'],versions:['v1'],summary:{active_users:2,online_machines:23,registered_machines:23,busy_cards:23,online_cards:46,completed_per_min:69,success_per_min:46,active_cases:23,queued:46},machines:Array.from({length:23},(_,i)=>machine(i)),trend:[{time:100,completed:4,success:2}]};
const host=new E('div');renderFleet(host,data);
assert.equal(all(host,'select').length,6);assert.equal(all(host,'table')[0].children.length,11);
all(host,'button')[1].onclick();assert.equal(fleetState.page,1);
all(host,'button')[1].onclick();assert.equal(all(host,'table')[0].children.length,4);
assert.equal(all(host,'button')[1].disabled,true);
renderFleet(host,{...data,machines:[]});assert.equal(fleetState.page,0);
'''
    subprocess.run([node,'-e',script],check=True,capture_output=True,text=True)
