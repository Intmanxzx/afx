"""Fleet rendering stays bounded for multi-user/multi-machine inventories."""
from pathlib import Path
import shutil
import subprocess
import pytest


def test_fleet_rendering_filters_and_paging():
    node=shutil.which('node')
    if not node:pytest.skip('node unavailable')
    source=Path('autofusex/fuzz/view/static/runtime.js').read_text()
    source=source[:source.index("window.addEventListener('hashchange'")]
    script=r'''
const assert=require('node:assert/strict');
class E {constructor(tag){this.tag=tag;this.children=[];this.style={};this.textContent='';this.open=false;}append(...xs){this.children.push(...xs);}replaceChildren(...xs){this.children=xs;}setAttribute(k,v){this[k]=v;}}
const document={createElement:t=>new E(t)};
const all=(e,t)=>[...(e.tag===t?[e]:[]),...e.children.flatMap(x=>all(x,t))];
''' + source + r'''
const machine=i=>({user_name:'user-'+i,user_id:'id-'+i,machine_id:'machine-'+i,name:'host-'+i,status:'online',cards:[{logical:'npu:0',model:'910',uuid:'uuid-'+i,status:'busy'}],runs:[{backend:['af-mlir'],state:'running',stages:{first_call_combined:1},age_seconds:1}],case_errors:0,record_failed:0,completed_per_min:3,success_per_min:2,active_cases:1,queued:2,age_seconds:1});
const data={users:[{id:'a',name:'Alice'},{id:'b',name:'Bob'}],machine_options:[],models:['910'],backends:['af-mlir'],versions:['v1'],summary:{total_users:5,online_users:3,active_users:2,online_machines:23,registered_machines:23,busy_cards:23,online_cards:46,known_cards:46,completed_per_min:69,success_per_min:46,active_cases:23,queued:46},machines:Array.from({length:23},(_,i)=>machine(i)),trend:[{time:100,completed:4,success:2}]};
const host=new E('div');renderFleet(host,data);
assert.ok(all(host,'th').some(e=>e.textContent==='完成数/成功数 / min'));
assert.ok(all(host,'span').some(e=>e.textContent==='完成数 / min'));
assert.ok(all(host,'span').some(e=>e.textContent==='成功数 / min'));
const userCard=all(host,'span').find(e=>e.textContent==='总用户数 / 在线用户数 / 活跃用户数');
assert.ok(userCard);assert.ok(all(host,'strong').some(e=>e.textContent==='5 / 3 / 2'));
assert.ok(!all(host,'span').some(e=>['活跃用户','在线 / 已登记机器'].includes(e.textContent)));
const named={...machine(0),display_name:'NiuYH-npu0'};
renderFleet(host,{...data,machines:[named],machine_options:[{id:named.machine_id,name:named.display_name}]});
assert.equal(all(host,'summary')[0].textContent,'user-0');
assert.equal(all(host,'h4')[0].textContent,'NiuYH-npu0');
assert.ok(!all(host,'p').some(p=>p.title==='machine-0'));
const choice=all(host,'select')[1].children[1];
assert.equal(choice.textContent,'NiuYH-npu0');assert.equal(choice.value,'machine-0');
renderFleet(host,data);
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
// One user row aggregates multiple machines; distinct user IDs never merge.
const one=machine(0),two={...machine(1),user_id:one.user_id,user_name:one.user_name};
two.cards=[...two.cards,{logical:'npu:1',uuid:'second-card',status:'unknown'}];
two.runs=[{backend:['af-mlir'],state:'unknown',age_seconds:40000,stages:{}},
 {backend:['af-mlir'],state:'idle',age_seconds:200,stages:{}},...two.runs];
const grouped={...data,machines:[one,two]};
renderFleet(host,grouped);
const table=all(host,'table')[0];assert.equal(table.children.length,2);
assert.equal(table.children[1].children[1].textContent,'2 / 2');
assert.equal(table.children[1].children[2].textContent,3);
assert.equal(table.children[1].children[3].textContent,'6.00 / 4.00');
assert.equal(table.children[1].children[4].textContent,'2 / 4');
assert.equal(all(host,'h4').length,2);
const texts=e=>[e.textContent,e.title||'',...e.children.flatMap(texts)].join(' ');
for(const hidden of ['主机名','机器 ID','host-0','machine-0','uuid-0','second-card','执行槽','历史 / 过期会话','已结束','结束状态未确认','写库失败累计'])assert.ok(!texts(host).includes(hidden),hidden);
assert.ok(texts(host).includes('npu:0 · 910'));
assert.ok(texts(host).includes('编译与首次执行'));
assert.ok(texts(host).includes('本时间段 Case 错误'));
renderFleet(host,{...data,machines:[{...one,runs:[{backend:['af-mlir'],state:'unknown',age_seconds:40000,stages:{}}]}]});
assert.ok(texts(host).includes('暂无最近的运行信息。'));
assert.ok(!texts(host).includes('结束状态未确认'));
assert.equal(fleetUsers([one,{...two,user_id:'other',user_name:one.user_name}]).length,2);
assert.equal(fleetAge(40000),'11 小时 6 分前');


'''
    subprocess.run([node,'-e',script],check=True,capture_output=True,text=True)


def test_fleet_refresh_only_on_efficiency_page():
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    source = Path('autofusex/fuzz/view/static/runtime.js').read_text()
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
    html = Path('autofusex/fuzz/view/static/index.html').read_text()
    assert 'id="fleet-overview"' not in html
    assert 'id="fleet-efficiency"' in html
    assert 'id="chartOverall"' in html
