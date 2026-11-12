/* Retest controls are only injected by the central service, never offline view. */
(() => {
  'use strict';
  if (!window.afSession) return;
  const el = (tag, text) => {const n=document.createElement(tag);if(text)n.textContent=text;return n;};
  const api = (path, method='GET', body) => window.afSession.request('api/retest/'+path,
    {method,...(body===undefined?{}:{body:JSON.stringify(body)})});
  const selected=new Map(); let rows=[], pending=null, polling=false, batchBefore=null;
  const bar=el('div');bar.className='retest-toolbar';bar.hidden=true;
  const count=el('span','已选 0 条'), all=el('button','选择当前页'),clear=el('button','清空选择'),submit=el('button','批量复测'),mine=el('button','我的复测');
  for(const b of [all,clear,submit,mine])b.type='button';
  bar.append(all,clear,count,submit,mine);
  document.getElementById('records-page').prepend(bar);
  const message=el('p');message.setAttribute('role','status');bar.append(message);
  const dialog=el('dialog'),form=el('form'),title=el('h2','提交复测'),description=el('p');
  const clients=el('select'),repetitions=el('input'),send=el('button','确认提交'),cancel=el('button','取消'),error=el('p');
  clients.setAttribute('aria-label','自己的执行客户端');
  repetitions.type='number';repetitions.min='1';repetitions.max='10';repetitions.value='1';
  repetitions.setAttribute('aria-label','重复次数');send.type='submit';cancel.type='button';
  error.setAttribute('role','alert');
  form.append(title,description,el('p','执行 CPU 参考与 AF-MLIR 完整固定 Shape 序列；按结果首次入库顺序更新当前记录。'),clients,
    el('label','重复次数（1–10）'),repetitions,send,cancel,error);dialog.append(form);document.body.append(dialog);
  let jobSignature='';
  const jobs=el('dialog'),jobList=el('div'),jobError=el('p'),jobClose=el('button','关闭');jobClose.type='button';
  jobs.append(el('h2','我的复测'),jobError,jobList,jobClose);document.body.append(jobs);
  jobClose.addEventListener('click',()=>jobs.close());
  const reasonNames={published:'结果已验证',late_result:'租约已过期，结果仍按入库顺序展示',unverified:'本次未验证',reproduction_mismatch:'用例复现一致性异常',cancelled_or_interrupted:'已取消或中断'};
  const stateNames={queued:'待领取',leased:'待运行',running:'执行中',completed:'已完成',cancel_requested:'取消中',cancelled:'已取消',interrupted:'执行中断'};
  const target = item => {
    const a=item.backends['af-mlir']?.[0] || Object.entries(item.backends).find(([b,a])=>b!=='cpu'&&a.length)?.[1][0];
    return a ? {attempt_id:a.original_attempt_id||a.id,shape_index:item.shape_index} : null;
  };
  const key=t=>t.attempt_id+':'+t.shape_index;
  function update(){
    bar.hidden=!window.afSession.authenticated();
    const visible=new Set(rows.map(target).filter(Boolean).map(key));
    const outside=[...selected.keys()].filter(k=>!visible.has(k)).length;
    count.textContent=`已选 ${selected.size} 条${outside?'（其中 '+outside+' 条不在当前页）':''}`;
    submit.disabled=selected.size===0;
  }
  function decorate(){
    document.querySelectorAll('.retest-row-control,.retest-provenance').forEach(n=>n.remove());
    const nodes=[...document.getElementById('rows').children].filter(n=>!n.classList.contains('history-footer'));
    rows.forEach((item,i)=>{
      const cell=nodes[i]?.querySelector('[data-column="seed"]');if(!cell)return;
      if(!window.afSession.authenticated())return;
      const t=target(item);if(!t)return;
      const wrapper=el('span');wrapper.className='retest-row-control';
      const box=el('input');box.type='checkbox';box.checked=selected.has(key(t));box.setAttribute('aria-label',`选择 ${item.seed_short} S${item.shape_index}`);
      box.addEventListener('change',()=>{if(box.checked)selected.set(key(t),t);else selected.delete(key(t));update();});
      const button=el('button','复测');button.type='button';button.addEventListener('click',()=>open([t]));
      wrapper.append(box,button);cell.prepend(wrapper);
    });update();
  }
  all.addEventListener('click',()=>{rows.filter(r=>!r.historical).map(target).filter(Boolean).forEach(t=>selected.set(key(t),t));decorate();});
  clear.addEventListener('click',()=>{selected.clear();decorate();});
  submit.addEventListener('click',()=>open([...selected.values()]));
  cancel.addEventListener('click',()=>dialog.close());
  async function open(targets){
    if(!window.afSession.authenticated())return;
    pending={targets,request_key:crypto.randomUUID()};
    clients.replaceChildren();error.textContent='';send.disabled=true;repetitions.value='1';
    description.textContent=`选择 ${targets.length} 条记录；同 seed、后端与配置合并执行。实际任务数由服务器校验后返回。`;
    dialog.showModal();
    try{
      const answer=await api('clients');
      for(const c of answer.items){const option=el('option',c.name+(c.online?'':'（离线）'));option.value=c.id;option.disabled=!c.online;clients.append(option);}
      const available=[...clients.options].find(o=>!o.disabled);
      if(available){clients.value=available.value;await preview();}else error.textContent='没有可接单客户端，请升级并运行自己机器上的客户端后台服务。';
    }catch(e){error.textContent=e.message;}
  }
  let previewGeneration=0;
  async function preview(){
    if(!pending||!clients.value)return;
    const generation=++previewGeneration;send.disabled=true;error.textContent='';
    try{const answer=await api('preview','POST',{...pending,client_id:clients.value,repetitions:Number(repetitions.value)});
      if(generation!==previewGeneration)return;
      description.textContent=`已选择 ${answer.selected_count} 条 Shape 记录，合并为 ${answer.task_count} 个执行任务（含重复次数）。`;send.disabled=false;
    }catch(e){if(generation===previewGeneration)error.textContent=e.message;}
  }
  form.addEventListener('submit',async event=>{
    event.preventDefault();if(!pending)return;send.disabled=true;error.textContent='';
    // Keep the same complete payload on a network retry. Editing choices starts
    // a distinct request instead of reusing an idempotency key with new content.
    const payload={...pending,client_id:clients.value,repetitions:Number(repetitions.value)};
    try{const answer=await api('batches','POST',payload);message.textContent=`已提交${answer.task_count===undefined?'':' '+answer.task_count+' 个任务'}，等待客户端领取。`;selected.clear();dialog.close();decorate();await showJobs();}
    catch(e){error.textContent=e.message;}finally{send.disabled=false;}
  });
  for(const n of [clients,repetitions])n.addEventListener('change',()=>{if(pending){pending.request_key=crypto.randomUUID();preview();}});
  async function showJobs(){batchBefore=null;jobSignature='';if(!jobs.open)jobs.showModal();await refreshJobs();}
  mine.addEventListener('click',showJobs);
  async function refreshJobs(){
    if(polling||!jobs.open||!window.afSession.authenticated())return;polling=true;jobError.textContent='';
    try{
      const answer=await api('batches'+(batchBefore?'?before='+batchBefore:''));const signature=JSON.stringify(answer);if(signature===jobSignature)return;jobSignature=signature;jobList.replaceChildren();
      if(!answer.items.length)jobList.append(el('p','暂无复测任务'));
      for(const batch of answer.items){
        const item=el('article'),label=el('p',new Date(batch.created*1000).toLocaleString()+' · '+Object.entries(batch.states).map(([s,n])=>(stateNames[s]||s)+' '+n).join(' / '));
        const details=el('button','查看任务'),stop=el('button','取消未完成任务');details.type=stop.type='button';
        stop.disabled=!['queued','leased','running'].some(s=>batch.states[s]);
        const content=el('div');
        async function taskPage(after=0){
          try{
            const page=await api('batches/'+batch.id+'?after='+after);content.replaceChildren();
            for(const task of page.items){
              content.append(el('p',`${task.spec.backend} · seed ${task.spec.seed} · 第 ${task.repetition+1} 次 · ${stateNames[task.state]||task.state}`));
              if(task.attempt_id){const link=el('a','查看执行结果');link.href='#attempt='+encodeURIComponent(task.attempt_id);link.addEventListener('click',()=>jobs.close());content.append(link);}
            }
            if(after){const first=el('button','任务首页');first.type='button';first.onclick=()=>taskPage();content.append(first);}
            if(page.next_cursor){const more=el('button','下一页任务');more.type='button';more.onclick=()=>taskPage(page.next_cursor);content.append(more);}
          }catch(e){jobError.textContent=e.message;}
        }
        details.addEventListener('click',()=>taskPage());
        stop.addEventListener('click',async()=>{stop.disabled=true;try{await api('batches/'+batch.id+'/cancel','POST',{});await refreshJobs();}catch(e){jobError.textContent=e.message;}});
        item.append(label,details,stop,content);jobList.append(item);
      }
      if(batchBefore){const first=el('button','最新批次');first.type='button';first.onclick=()=>{batchBefore=null;refreshJobs();};jobList.append(first);}
      if(answer.next_cursor){const next=el('button','更早批次');next.type='button';next.onclick=()=>{batchBefore=answer.next_cursor;refreshJobs();};jobList.append(next);}
    }catch(e){jobError.textContent=e.message;}finally{polling=false;}
  }
  let historyGeneration=0;
  document.addEventListener('af-attempt-rendered',async event=>{
    const generation=++historyGeneration;
    let section=document.getElementById('retest-history');
    if(!section){section=el('section');section.id='retest-history';document.getElementById('detail').append(section);}
    section.replaceChildren(el('h3','复测记录'),el('p','加载中…'));
    try{const answer=await api('history/'+event.detail.id);if(generation!==historyGeneration)return;
      section.replaceChildren(el('h3','复测记录'));
      if(!answer.items.length)section.append(el('p','暂无复测记录'));
      for(const run of answer.items){const article=el('article');article.append(el('p',run.executor+' · '+(stateNames[run.state]||run.state)+(run.reason?' · '+(reasonNames[run.reason]||run.reason):'')));
        article.append(el('p','环境：'+Object.entries(run.environment||{}).filter(([,v])=>v).map(([k,v])=>k+' '+v).join(' · ')));
        for(const late of run.late_results||[]){const link=el('a','查看迟到执行结果');link.href='#attempt='+encodeURIComponent(late.attempt_id);article.append(link);}
        for(const shape of run.shapes){const before=run.baseline?.find(b=>b.shape_index===shape.shape_index)?.correctness||'未执行';const p=el('p',`S${shape.shape_index}：${before} → ${shape.correctness} · 原因与解决方案：${shape.summary_id?'已关联总结 '+shape.summary_id+'，内容暂不可用':'暂无分析总结'}`);article.append(p);}
        if(run.attempt_id){const link=el('a','查看本次执行');link.href='#attempt='+encodeURIComponent(run.attempt_id);article.append(link);}section.append(article);}
    }catch(e){if(generation===historyGeneration)section.replaceChildren(el('h3','复测记录'),el('p',e.message));}
  });
  document.addEventListener('af-records-rendered',event=>{rows=event.detail;decorate();});
  document.addEventListener('af-session-changed',()=>{if(!window.afSession.authenticated()){selected.clear();dialog.close();jobs.close();}decorate();});
  setInterval(()=>{if(!document.hidden)refreshJobs();},5000);
  update();
})();
