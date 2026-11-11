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
class E {constructor(tag){this.tag=tag;this.children=[];this.style={};this.textContent='';this.open=false;}append(...xs){this.children.push(...xs);}replaceChildren(...xs){this.children=xs;}setAttribute(k,v){this[k]=v;}}
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
renderFleet(host,data);
let details=all(host,'details');
const detached=details[0];details[0].open=true;details.at(-1).open=true;
// Refresh before the browser's queued toggle event: both must remain expanded.
renderFleet(host,{...data,machines:[...data.machines].reverse()});
assert.equal(all(host,'details').at(-1).open,true);
fleetState.page=2;renderFleet(host,{...data,machines:[...data.machines].reverse()});
assert.equal(all(host,'details')[2].open,true); // original machine, not row index
// A late event from the removed node cannot overwrite current state.
detached.open=false;detached.ontoggle();
renderFleet(host,{...data,machines:[...data.machines].reverse()});
assert.equal(all(host,'details')[2].open,true);
all(host,'details')[2].open=false;all(host,'details')[2].ontoggle();
renderFleet(host,{...data,machines:[...data.machines].reverse()});
assert.equal(all(host,'details')[2].open,false);
// Same machine under different users, and different pages, are independent.
fleetState.page=0;
const shared={...data,machines:[machine(0),{...machine(0),user_id:'other-user'}]};
renderFleet(host,shared);all(host,'details')[0].open=true;
renderFleet(host,shared);
assert.equal(all(host,'details')[0].open,true);assert.equal(all(host,'details')[1].open,false);
const otherHost=new E('div');renderFleet(otherHost,shared,true);
assert.equal(all(otherHost,'details')[0].open,false);assert.equal(all(otherHost,'details').at(-1).open,false);
// Filter out and restore a row; error/recovery also preserves disclosure choices.
renderFleet(host,{...data,machines:[]});renderFleet(host,shared);
assert.equal(all(host,'details')[0].open,true);
rememberFleetDisclosures(host);host.replaceChildren();renderFleet(host,shared);
assert.equal(all(host,'details')[0].open,true);assert.equal(all(host,'details').at(-1).open,true);
all(host,'details').at(-1).open=false;renderFleet(host,shared);
assert.equal(all(host,'details').at(-1).open,false);

'''
    subprocess.run([node,'-e',script],check=True,capture_output=True,text=True)


def test_fleet_refresh_only_on_efficiency_page():
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    source = Path('autofusedebugger/fuzz/view/static/runtime.js').read_text()
    source = source[:source.index("window.addEventListener('hashchange'")]
    script = r'''
const assert = require('node:assert/strict');
const location = {hash: ''};
const host = {};
const document = {getElementById(id) {
 assert.equal(id, 'fleet-efficiency');return host;
}};
let requests = 0, renders = 0;
const fetch = async () => {requests++;return {ok:true,json:async()=>({})}};
''' + source + r'''
renderFleet = (target, data, efficiency) => {
 assert.equal(target, host);assert.equal(efficiency, true);renders++;
};
(async () => {
 for(const hash of ['', '#overview', '#records', '#kernel']) {
  location.hash = hash;await refreshFleet(true);
 }
 assert.equal(requests, 0);assert.equal(renders, 0);
 for(const hash of ['#efficiency', '#performance']) {
  location.hash = hash;await refreshFleet(true);
 }
 assert.equal(requests, 2);assert.equal(renders, 2);
 location.hash = '#overview';await refreshFleet();
 assert.equal(requests, 2);
})().catch(error => {console.error(error);process.exitCode = 1});
'''
    subprocess.run([node, '-e', script], check=True, capture_output=True, text=True)
    html = Path('autofusedebugger/fuzz/view/static/index.html').read_text()
    assert 'id="fleet-overview"' not in html
    assert 'id="fleet-efficiency"' in html
    assert 'id="chartOverall"' in html
