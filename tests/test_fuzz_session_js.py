"""Login lifecycle without leaking upload credentials into persistent storage."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_browser_session_lifecycle():
    node = shutil.which('node')
    if not node:
        pytest.skip('node unavailable')
    source = Path('autofusex/fuzz/view/static/session.js').read_text()
    script = r'''
const assert=require('assert').strict;
class E {
 constructor(tag){this.tag=tag;this.children=[];this.events={};this.value='';this.textContent='';this.disabled=false;}
 append(...xs){this.children.push(...xs)} setAttribute(k,v){this[k]=v}
 addEventListener(k,f){this.events[k]=f} focus(){} showModal(){this.open=true}
 close(){this.open=false;this.events.close && this.events.close()}
}
const host=new E('span'),body=new E('body'),events=[],classes=new Map();
body.classList={toggle:(key,value)=>classes.set(key,value)};
const document={body,hidden:false,getElementById:()=>host,createElement:t=>new E(t),
 addEventListener(){},dispatchEvent:e=>events.push(e)};
const window={},CustomEvent=class{constructor(type,init){this.type=type;this.detail=init.detail}};
const calls=[];let answer={ok:false,status:401,json:async()=>({error:'not signed in'})};
const fetch=async(path,options)=>{calls.push({path,options});return answer};
''' + source + r'''
(async()=>{
 await new Promise(resolve=>setImmediate(resolve));
 const [label,button]=host.children,dialog=body.children[0],form=dialog.children[0];
 const [,input,error,actions]=form.children,[cancel,submit]=actions.children;
 assert.equal(input['aria-label'],'Token');
 assert.equal(cancel.textContent,'取消');assert.equal(submit.textContent,'登录');
 assert.equal(button.textContent,'登录');assert.equal(window.afSession.authenticated(),false);
 await button.events.click();assert.equal(dialog.open,true);
 input.value='my-private-long-token';
 answer={ok:true,status:200,json:async()=>({user_id:'bob',display_name:'Bob',csrf:'csrf-secret'})};
 await form.events.submit({preventDefault(){}});
 assert.equal(input.value,'');assert.equal(dialog.open,false);
 assert.equal(label.textContent,'Bob');assert.equal(button.textContent,'退出');
 assert.equal(classes.get('authenticated'),true);
 assert.equal(JSON.parse(calls[calls.length-1].options.body).token,'my-private-long-token');
 assert.equal(calls[calls.length-1].options.credentials,'same-origin');
 assert.equal(events[events.length-1].detail.display_name,'Bob');assert.equal(events[events.length-1].detail.csrf,undefined);
 answer={ok:true,status:200,json:async()=>({signed_out:true})};
 await button.events.click();
 assert.equal(calls[calls.length-1].options.method,'DELETE');
 assert.equal(calls[calls.length-1].options.headers['X-CSRF-Token'],'csrf-secret');
 assert.equal(window.afSession.authenticated(),false);assert.equal(classes.get('authenticated'),false);
 answer={ok:false,status:401,json:async()=>({error:'invalid'})};
 await button.events.click();input.value='bad-token';
 await form.events.submit({preventDefault(){}});
 assert.equal(error.textContent,'invalid');assert.equal(input.value,'');assert.equal(submit.disabled,false);
 cancel.events.click();assert.equal(dialog.open,false);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    subprocess.run([node, '-e', script], check=True, capture_output=True, text=True)
    assert 'localStorage' not in source and 'sessionStorage' not in source
