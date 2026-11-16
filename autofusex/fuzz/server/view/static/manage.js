(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let profile = null, devices = [], busy = false;
  async function api(route, payload) {
    const response = await fetch('/api/manage/' + route, {method:payload === undefined ? 'GET':'POST',
      credentials:'same-origin', cache:'no-store', headers:payload === undefined ? {} : {'Content-Type':'application/json','X-CSRF-Token':profile?.csrf || ''},
      body:payload === undefined ? undefined : JSON.stringify(payload)});
    const value = await response.json(); if (!response.ok) throw new Error(value.error || '请求失败'); return value;
  }
  async function session() { const r=await fetch('/api/session',{cache:'no-store'}); profile=r.ok?await r.json():null; }
  function selected() {const d=devices.find(d=>d.id===$('devices').value);if(!d)throw new Error('请先选择设备');return d;}
  async function refresh() {
    if (busy || !profile) return; busy=true;
    try {
      const previous=$('devices').value; devices=(await api('devices')).items;
      $('devices').replaceChildren(...devices.map(d=>{const o=document.createElement('option');o.value=d.id;o.textContent=d.name+(Date.now()/1000-d.seen<40?' · 在线':' · 离线');return o;}));
      if(devices.some(d=>d.id===previous))$('devices').value=previous;
      if(devices.length){$('device-state').textContent=JSON.stringify(selected(),null,2);const ops=(await api('operations',{device_id:selected().id})).items;
        $('operations').replaceChildren(...ops.map(o=>{const el=document.createElement('pre');el.className='operation';el.textContent=JSON.stringify(o,null,2);return el;}));}
    } catch(e){$('notice').textContent=e.message;} finally{busy=false;}
  }
  async function guard(action){try{await action();}catch(e){$('notice').textContent=e.message;}}
  document.addEventListener('af-session-changed',()=>guard(async()=>{await session();$('notice').textContent=profile?'已登录；请选择设备':'请先登录';await refresh();}));
  $('pair').addEventListener('submit',event=>{event.preventDefault();guard(async()=>{await api('approve',{code:$('code').value.trim().toUpperCase()});$('code').value='';$('notice').textContent='绑定已批准，等待客户端领取凭据';await refresh();});});
  $('refresh').onclick=()=>guard(refresh);$('devices').onchange=()=>guard(refresh);
  $('validate').onclick=()=>guard(async()=>{$('preview').textContent=JSON.stringify(await api('validate',{yaml:$('spec').value}),null,2);});
  $('submit').onclick=()=>guard(async()=>{
    const kind=$('kind').value;let payload;
    if(['environment.plan','task.validate','task.submit','config'].includes(kind))payload=await api('validate',{yaml:$('spec').value});
    else payload=JSON.parse($('spec').value);
    if(['environment.apply','build.submit','task.submit','task.action'].includes(kind)&&!confirm(`确认对 ${selected().name} 执行 ${kind}？`))return;
    const result=kind==='config'?await api('config',{device_id:selected().id,base_revision:selected().revision,config:payload}):
      await api('submit',{device_id:selected().id,kind,payload,request_key:crypto.randomUUID()});
    $('preview').textContent=JSON.stringify(result,null,2);$('notice').textContent='已提交，实际结果请查看操作进度';await refresh();
  });
  let schemas={}, formReader=null, templates=[];
  // Render schema-owned fields; strings always enter DOM through textContent/value.
  function field(schema,value,name,required=true) {
    const box=document.createElement('div');box.className='schema-field';
    if(schema.oneOf){const select=document.createElement('select'),child=document.createElement('div');
      schema.oneOf.forEach((branch,i)=>{const o=document.createElement('option');o.value=i;o.textContent=branch.properties?.source?.enum?.[0]||branch.properties?.mode?.enum?.[0]||String(i+1);select.append(o);});
      let selected=schema.oneOf.findIndex(b=>Object.entries(b.properties||{}).every(([k,s])=>!s.enum||value?.[k]===undefined||s.enum.includes(value[k])));select.value=Math.max(0,selected);
      let nested;function update(){nested=field(schema.oneOf[Number(select.value)],value,name,true);child.replaceChildren(nested.node);}select.onchange=()=>{value={};update();};update();box.append(select,child);return {node:box,read:()=>nested.read()};}
    const label=document.createElement('label');label.textContent=name;box.append(label);
    let enabled=null;if(!required){enabled=document.createElement('input');enabled.type='checkbox';enabled.checked=value!==undefined;label.prepend(enabled);}
    let read;
    if(schema.type==='object'){const entries=Object.entries(schema.properties).map(([key,s])=>{const child=field(s,value?.[key],key,(schema.required||[]).includes(key));box.append(child.node);return [key,child];});read=()=>Object.fromEntries(entries.map(([k,f])=>[k,f.read()]).filter(([,v])=>v!==undefined));}
    else if(schema.enum){const input=document.createElement('select');schema.enum.forEach(v=>{const o=document.createElement('option');o.value=JSON.stringify(v);o.textContent=String(v);input.append(o);});if(value!==undefined)input.value=JSON.stringify(value);box.append(input);read=()=>JSON.parse(input.value);}
    else if(schema.type==='boolean'){const input=document.createElement('input');input.type='checkbox';input.checked=Boolean(value);box.append(input);read=()=>input.checked;}
    else if(schema.type==='array'){const input=document.createElement('textarea');input.rows=3;input.value=JSON.stringify(value||[],null,2);input.setAttribute('aria-label',name+' JSON 数组');box.append(input);read=()=>JSON.parse(input.value);}
    else {const input=document.createElement('input');input.type=schema.type==='integer'?'number':'text';input.value=value??'';if(schema.minimum!==undefined)input.min=schema.minimum;if(schema.maximum!==undefined)input.max=schema.maximum;box.append(input);read=()=>schema.type==='integer'?Number(input.value):input.value;}
    return {node:box,read:()=>enabled&&!enabled.checked?undefined:read()};
  }
  function renderForm(value){if(!schemas[value.kind])throw new Error('不支持的配置类型');const form=field(schemas[value.kind],value,value.kind);formReader=form.read;$('schema-form').replaceChildren(form.node);}
  $('new-task').onclick=()=>guard(async()=>{const value=await api('validate',{config:{schema_version:1,kind:'fuzz_task',environment:'dev',workload:{mode:'generate'}}});renderForm(value);$('kind').value='task.validate';});
  $('new-environment').onclick=()=>renderForm({schema_version:1,kind:'environment',name:'dev',python:'3.11',compatibility_profile:'locally-approved',cann:{source:'download',install_root:'~/Ascend',packages:[]},packages:{torch:'',torch_npu:'',nnsmith:''}});
  $('new-policy').onclick=()=>guard(async()=>{renderForm(await api('validate',{config:{schema_version:1,kind:'client_policy'}}));$('kind').value='config';});
  $('form-to-editor').onclick=()=>guard(async()=>{if(!formReader)throw new Error('请先打开表单');const value=await api('validate',{config:formReader()});$('spec').value=JSON.stringify(value,null,2);});
  $('editor-to-form').onclick=()=>guard(async()=>{renderForm(await api('validate',{yaml:$('spec').value}));});
  async function loadTemplates(){templates=(await api('templates')).items;$('templates').replaceChildren(new Option('选择模板',''),...templates.map(t=>new Option(t.name+' · r'+t.revision,t.name)));}
  $('templates').onchange=()=>{const t=templates.find(t=>t.name===$('templates').value);if(t){$('spec').value=JSON.stringify(t.spec,null,2);$('template-name').value=t.name;renderForm(t.spec);}};
  $('save-template').onclick=()=>guard(async()=>{const name=$('template-name').value.trim();const old=templates.find(t=>t.name===name);await api('template',{name,base_revision:old?.revision||0,spec:await api('validate',{yaml:$('spec').value})});await loadTemplates();});
  $('export-config').onclick=()=>guard(async()=>{const value=await api('validate',{yaml:$('spec').value});const exported=await api('export',{config:value});const url=URL.createObjectURL(new Blob([exported.yaml],{type:'application/yaml'}));const a=document.createElement('a');a.href=url;a.download=value.kind+'.yaml';a.click();URL.revokeObjectURL(url);});
  $('history').onclick=()=>guard(async()=>{$('history-view').textContent=JSON.stringify(await api('history',{device_id:selected().id}),null,2);});
  $('audit').onclick=()=>guard(async()=>{$('history-view').textContent=JSON.stringify(await api('audit'),null,2);});
  $('revoke').onclick=()=>guard(async()=>{if(confirm('撤销后该设备将无法远程控制和上传。确认？')){await api('revoke',{device_id:selected().id});await refresh();}});
  document.addEventListener('af-session-changed',()=>guard(async()=>{await session();if(profile){schemas=await api('schemas');await loadTemplates();}}));
  guard(async()=>{await session();if(profile){schemas=await api('schemas');await loadTemplates();}await refresh();});setInterval(()=>{if(!document.hidden)refresh();},5000);
})();
