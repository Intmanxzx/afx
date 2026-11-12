'use strict';
// Live fleet state is independent of result/search caching and immutable history.
const fleetState={user_id:'',machine_id:'',model:'',backend:'',version:'',minutes:'5',page:0};
let fleetRequest=0,fleetBusy=false;
function fleetElement(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function fleetCell(row,text,tag='td'){row.append(fleetElement(tag,text));}
function fleetTable(headers){const t=fleetElement('table'),tr=fleetElement('tr');headers.forEach(h=>fleetCell(tr,h,'th'));t.append(tr);return t;}
function fleetWrap(t){const w=fleetElement('div',undefined,'table-scroll');w.append(t);return w;}
// Scope disclosure state to the page and stable user/machine identity, not row order.
const fleetDisclosures=new WeakMap();
function fleetDisclosureState(host){
 if(!fleetDisclosures.has(host))fleetDisclosures.set(host,{open:new Map(),nodes:new Map()});
 return fleetDisclosures.get(host);
}
function rememberFleetDisclosures(host){
 const state=fleetDisclosureState(host);
 // Read the DOM synchronously: a native toggle event may still be queued.
 for(const [key,node] of state.nodes)state.open.set(key,node.open);
 state.nodes.clear();
}
function fleetDisclosure(host,key,cls){
 const state=fleetDisclosureState(host),node=fleetElement('details',undefined,cls);
 node.open=state.open.get(key)===true;state.nodes.set(key,node);
 node.ontoggle=()=>{
  // Detached details may emit a delayed toggle after a refresh.
  if(state.nodes.get(key)===node)state.open.set(key,node.open);
 };
 return node;
}
function fleetControls(host,data){
 const bar=fleetElement('div',undefined,'fleet-filters');
 const choices={user_id:[['','全部用户'],...data.users.map(u=>[u.id,u.name])],machine_id:[['','全部机器'],...data.machine_options.map(m=>[m.id,m.name])],model:[['','全部卡型号'],...data.models.map(x=>[x,x])],backend:[['','全部后端'],...data.backends.map(x=>[x,x])],version:[['','全部版本'],...data.versions.map(x=>[x,x])],minutes:[['1','最近 1 分钟'],['5','最近 5 分钟'],['15','最近 15 分钟'],['60','最近 1 小时']]};
 const names={user_id:'用户',machine_id:'机器',model:'卡型号',backend:'后端',version:'软件版本',minutes:'统计范围'};
 for(const [key,options] of Object.entries(choices)){const label=fleetElement('label',names[key]),select=fleetElement('select');select.setAttribute('aria-label','运行'+names[key]);for(const [value,text] of options){const option=fleetElement('option',text);option.value=value;select.append(option);}select.value=fleetState[key];select.onchange=()=>{fleetState[key]=select.value;if(key==='user_id')fleetState.machine_id='';fleetState.page=0;refreshFleet(true);};label.append(select);bar.append(label);}host.append(bar);
}
function fleetCards(host,s){
 const cards=fleetElement('div',undefined,'efficiency-cards');
 const entries=[['活跃用户',s.active_users],['在线 / 已登记机器',s.online_machines+' / '+s.registered_machines],['占用 / 新鲜观测 NPU',s.busy_cards+' / '+s.online_cards],['完成 Shape 执行/min',s.completed_per_min.toFixed(2)],['成功 Shape 执行/min',s.success_per_min.toFixed(2)],['运行中 Case / 排队任务',s.active_cases+' / '+s.queued]];
 for(const [label,value] of entries){const c=fleetElement('div');c.append(fleetElement('span',label),fleetElement('strong',value));cards.append(c);}host.append(cards);
}
function fleetTrend(host,data){
 const box=fleetDisclosure(host,'trend','efficiency-details');box.append(fleetElement('summary','最近 1 小时吞吐趋势（每分钟完成 / 成功）'));
 const bars=fleetElement('div',undefined,'fleet-trend');const maximum=Math.max(1,...data.trend.map(b=>b.completed));
 for(const bucket of data.trend){const bar=fleetElement('div');const time=new Date(bucket.time*1000).toLocaleTimeString();bar.title=time+' · 完成 '+bucket.completed+' / 成功 '+bucket.success;bar.setAttribute('aria-label',bar.title);const total=fleetElement('span',undefined,'fleet-total'),success=fleetElement('span',undefined,'fleet-success');total.style.height=(100*bucket.completed/maximum)+'%';success.style.height=(100*bucket.success/maximum)+'%';bar.append(total,success);bars.append(bar);}box.append(bars,fleetElement('p','浅绿：完成；深绿：成功。只包含连续收到的心跳计数增量；断网区间不补记为实时产出。','muted'));host.append(box);
}
function renderFleet(host,data,efficiency=false){
 rememberFleetDisclosures(host);
 host.replaceChildren();host.append(fleetElement('h2',efficiency?'工具效率 · 多用户运行概览':'实时运行状态'));
 fleetControls(host,data);fleetCards(host,data.summary);
 host.append(fleetElement('p','每 10 秒心跳；超过 30 秒延迟，60 秒失联。卡占用是调度执行槽，不是 AICore 使用率。完成量按后端 Shape 执行计数，含重试，不等于去重用例覆盖数。','muted'));
 host.append(fleetElement('p','机器在线由上传器心跳判断；卡状态由执行器新鲜观测判断。后端、卡型号、版本筛选匹配运行会话，吞吐为匹配会话整体产出，不是单卡基准。旧客户端和失联执行器不推断为在线执行。','muted'));
 const n=data.machines.length,pages=Math.max(1,Math.ceil(n/10));fleetState.page=Math.min(fleetState.page,pages-1);
 const t=fleetTable(['用户 / 机器','在线状态','占用 / 观测 / 已知卡','完成 / 成功 Shape执行/min','运行 Case / 排队任务','最后心跳']);
 for(const m of data.machines.slice(fleetState.page*10,(fleetState.page+1)*10)){
  const tr=fleetElement('tr'),name=fleetElement('td'),details=fleetDisclosure(host,JSON.stringify(['machine',m.user_id,m.machine_id]));details.append(fleetElement('summary',m.display_name||m.user_name+' / '+m.name));
  details.append(fleetElement('p','用户：'+m.user_name+' · 主机名：'+m.name,'muted'));
  details.append(fleetElement('p','机器 ID：'+m.machine_id,'muted'));
  for(const d of m.cards)details.append(fleetElement('p',`${d.logical} · ${d.model} · ${{busy:'执行槽占用',idle:'执行槽空闲',unknown:'状态未知'}[d.status]} · ${d.uuid}`));
  for(const run of m.runs)details.append(fleetElement('p',`${run.backend.join(', ')} · ${{running:'运行中',idle:'已结束',error:'异常结束',unknown:'执行器状态未知'}[run.state]} · 阶段 ${Object.entries(run.stages).map(([k,v])=>k+': '+v).join(' / ')||'—'} · 观测 ${run.age_seconds}s 前`));
  details.append(fleetElement('p','本时间段 Case 错误：'+m.case_errors+'；保留运行会话写库失败累计：'+m.record_failed));name.append(details);tr.append(name);
  for(const v of [{online:'在线',delayed:'心跳延迟',offline:'失联'}[m.status],m.cards.filter(c=>c.status==='busy').length+' / '+m.cards.filter(c=>c.status!=='unknown').length+' / '+m.cards.length,m.completed_per_min.toFixed(2)+' / '+m.success_per_min.toFixed(2),m.active_cases+' / '+m.queued,m.age_seconds+' s 前'])fleetCell(tr,v);t.append(tr);
 }
 host.append(fleetWrap(t));if(!n)host.append(fleetElement('p','暂无符合条件的机器心跳。旧结果不会被当成在线设备。','muted'));
 const pager=fleetElement('div',undefined,'efficiency-toolbar'),prev=fleetElement('button','上一页'),next=fleetElement('button','下一页');prev.disabled=fleetState.page===0;next.disabled=fleetState.page>=pages-1;prev.onclick=()=>{fleetState.page--;renderFleet(host,data,efficiency);};next.onclick=()=>{fleetState.page++;renderFleet(host,data,efficiency);};pager.append(fleetElement('span',n+' 个用户/机器条目 · 第 '+(fleetState.page+1)+' / '+pages+' 页'),prev,next);host.append(pager);fleetTrend(host,data);
 if(efficiency)host.append(fleetElement('p','实时卡/机器筛选作用于心跳观测。下方历史分位数是独立的结果统计范围，不与实时吞吐混算。','muted'));
}
async function refreshFleet(force=false){
 const page=location.hash.slice(1)||'overview';if(!['efficiency','performance'].includes(page))return;
 if(fleetBusy&&!force)return;const seq=++fleetRequest;fleetBusy=true;
 const host=document.getElementById('fleet-efficiency');
 const p=new URLSearchParams();for(const [k,v] of Object.entries(fleetState))if(k!=='page'&&v)p.set(k,v);
 try{const response=await fetch('api/runtime?'+p,{cache:'no-store',credentials:'omit'});if(!response.ok)throw new Error(response.status===404?'当前查看器不提供实时心跳；离线结果不代表设备在线。':'实时状态暂不可用，请检查中心端。');const data=await response.json();if(seq===fleetRequest)renderFleet(host,data,true);}
 catch(error){if(seq===fleetRequest){rememberFleetDisclosures(host);host.replaceChildren(fleetElement('p',error.message,'muted'));}}
 finally{if(seq===fleetRequest)fleetBusy=false;}
}
window.addEventListener('hashchange',()=>refreshFleet(true));
setInterval(()=>refreshFleet(),10000);refreshFleet();
