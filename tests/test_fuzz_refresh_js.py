"""Exercise actual request/cache/routing helpers without a browser dependency."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_refresh_routing_etag_and_inflight_dedup():
    node=shutil.which('node')
    if not node:pytest.skip('node unavailable')
    source=Path('autofusedebugger/fuzz/view/static/app.js').read_text()
    helpers=source[source.index('const responseCache='):source.index('function cell(')]
    script=r'''
const assert=require('node:assert/strict');
const elements=new Map();
const $=id=>{if(!elements.has(id))elements.set(id,{textContent:'',after(){}});return elements.get(id)};
const document={createElement(){return {setAttribute(){}}}};
const beijing=x=>String(x);
let location={hash:'#overview'},page=0;
let calls=[],http=[],resolve;
function load(silent,list){calls.push(['load',silent,list])}
function loadCharts(){calls.push(['charts'])}
let fetch=async(path,options)=>{http.push({path,options});return new Promise(r=>{resolve=r})};
function response(status,etag='v1'){
 return {status,ok:status===200,headers:new Map([['ETag',etag],['X-Cache-State','ready']]),json:async()=>({total:3})};
}
''' + helpers + r'''
(async()=>{
 refreshVisible(true);assert.deepEqual(calls,[['load',true,false],['charts']]);
 calls=[];location.hash='#records';refreshVisible(true);assert.deepEqual(calls,[['load',true,true]]);
 calls=[];page=1;refreshVisible(true);assert.deepEqual(calls,[['load',true,true]]);calls=[];
 page=0;location.hash='#attempt=abc';refreshVisible(true);assert.deepEqual(calls,[]);
 const first=api('api/summary?'),second=api('api/summary?');
 assert.equal(http.length,1);resolve(response(200));
 assert.deepEqual(await first,{total:3});await second;
 const third=api('api/summary?');assert.equal(http[1].options.headers['If-None-Match'],'v1');
 resolve(response(304));assert.deepEqual(await third,{total:3});
 const older=api('api/cases?q=old');const oldResolve=resolve;
 const newer=api('api/cases?q=new');assert.equal(http[2].options.signal.aborted,true);
 resolve(response(200));oldResolve(response(200));await older;await newer;
})().catch(error=>{console.error(error);process.exitCode=1});
'''
    subprocess.run([node,'-e',script],check=True,capture_output=True,text=True)
    assert '},15000)' in source
    assert '30000' not in source
