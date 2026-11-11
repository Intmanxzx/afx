'use strict';
const $ = id => document.getElementById(id);
const form = $('filters');
function beijing(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}).format(date);
}
const caseMetadata = new Map();
let logLength=100;try{logLength=Number(localStorage.getItem('af-fuzz.log-length'))||100;}catch{}
logLength=Math.max(40,Math.min(500,logLength));
const labels = {ok:'执行成功',fail:'精度失败',error:'异常',unsupported:'不支持'};
const initial = new URLSearchParams(location.search);
form.elements.q.value=initial.get('q')||initial.get('user_id')||initial.get('seed')||'';
let applied=parameters();
let cursors=[0],page=0,next=null,generation=0,detailGeneration=0;
let pageSize=10;
try{const size=Number(localStorage.getItem('af-fuzz.page-size'));if([10,25,50,100,200].includes(size))pageSize=size;}catch{}
$('page-size').value=String(pageSize);
function parameters(){const p=new URLSearchParams();const q=form.elements.q.value.trim();if(q)p.set('q',q);return p;}
const responseCache=new Map(), pendingRequests=new Map();
const cacheNotice=document.createElement('span');cacheNotice.id='cache-status';
cacheNotice.setAttribute('role','status');$('refresh-time').after(cacheNotice);
const staleRoutes=new Set();
function updateCacheStatus(response,route){
 if(response.headers.get('X-Cache-State')==='stale')staleRoutes.add(route);else staleRoutes.delete(route);
 cacheNotice.textContent=staleRoutes.size?' · 缓存更新失败，暂时显示上次成功的数据':'';
 const generated=Number(response.headers.get('X-Data-Generated-At'));
 if(generated)$('refresh-time').textContent='数据生成于 '+beijing(new Date(generated*1000));
 else if(response.ok)$('refresh-time').textContent='更新于 '+beijing(new Date());
}
async function api(path) {
 const route=path.split('?')[0], old=pendingRequests.get(route);
 if(old?.path===path)return old.promise;
 // Only replace list/chart reads; independent detail requests remain untouched.
 const managed=['api/summary','api/cases','api/statistics'].includes(route);
 if(managed&&old)old.controller.abort();
 const controller=new AbortController(), cached=responseCache.get(path);
 const promise=(async()=>{
  const headers=cached?{'If-None-Match':cached.etag}:{};
  const response=await fetch(path,{cache:'no-store',credentials:'omit',headers,signal:controller.signal});
  if(managed)updateCacheStatus(response,route);
  if(response.status===304&&cached)return cached.value;
  const value=await response.json();
  if(!response.ok)throw new Error(value.error||'读取失败');
  const etag=response.headers.get('ETag');
  if(etag){
   if(responseCache.size>=8&&!responseCache.has(path))responseCache.delete(responseCache.keys().next().value);
   responseCache.set(path,{etag,value});
  }
  return value;
 })();
 if(managed)pendingRequests.set(route,{path,promise,controller});
 try{return await promise;}finally{if(pendingRequests.get(route)?.promise===promise)pendingRequests.delete(route);}
}
function activePage(){return location.hash.startsWith('#attempt=')?'detail':(location.hash.slice(1)||'overview');}
function refreshVisible(silent=false){
 const name=activePage();
 if(name==='detail'||(silent&&name==='records'&&page>0))return;
 load(silent,name==='records');
 if(name!=='records')loadCharts();
}
function cell(row, value, tag="td") {
  const td = document.createElement(tag); td.textContent = value; row.append(td); return td;
}
async function details(identity, shapeIndex=null) {
  hideFieldTooltip();
  const request = ++detailGeneration;
  $('detail').hidden = false; $('detail-message').textContent = '加载中…';
  $('detail-content').textContent = ''; $('command').value = ''; $('copy').disabled = true;
  for(const id of ['case-meta','ascir-detail','failure-details','timing-detail','provenance'])$(id).replaceChildren();
  try {
    const result = await api('api/attempts/' + encodeURIComponent(identity)+(shapeIndex===null?'':'?shape='+shapeIndex));
    if(request !== detailGeneration) return;
    $('command').value = result.replay_command;
    renderCaseInfo(result);
    renderProvenance(result);
    renderFailureDetails(result);
    result.created_at = beijing(result.created_at);
    $('detail-content').textContent = '';
    $('detail-content').parentElement.hidden=true;
    $('detail-message').textContent = ''; $('copy').disabled = false;
    document.querySelectorAll('.page').forEach(el=>el.hidden=true);
    $('page-title').textContent='用例详情 · '+result.seed_short;
    $('detail').scrollIntoView({behavior:'smooth',block:'start'});
  } catch(error) {if(request === detailGeneration) $('detail-message').textContent = error.message;}
}
const extraColumns=[['cann','CANN'],['torch','Torch'],['torch_npu','Torch NPU'],['autofuse_commit','Autofuse commit'],['autofuse_binary','Autofuse binary'],['triton','Triton'],['python','Python'],['package_version','Debugger'],['machine','架构'],['platform','系统'],['source_id','来源 ID'],['batch_id','批次']];
const preferenceKey='af-fuzz.record-columns.v3';
let columnPreferences={order:[],visibility:{}},currentColumns=[],lastData='',lastRows=[];
try{const saved=JSON.parse(localStorage.getItem(preferenceKey));if(saved&&Array.isArray(saved.order)&&saved.visibility&&typeof saved.visibility==='object')columnPreferences=saved;}catch{}
function orderedColumns(){return [...columnPreferences.order.map(key=>currentColumns.find(c=>c.key===key)).filter(Boolean),...currentColumns.filter(c=>!columnPreferences.order.includes(c.key))];}
function isVisible(c){return columnPreferences.visibility[c.key]??(c.defaultVisible&&(c.populated||c.key==='af-mlir:total'));}
function saveColumns(){try{localStorage.setItem(preferenceKey,JSON.stringify(columnPreferences));}catch{}applyColumns();renderColumnPicker();}
function configureColumns(definitions){currentColumns=definitions;applyColumns();if(!$('column-picker').open)renderColumnPicker();}
function applyColumns(){
 const visible=orderedColumns().filter(isVisible),table=$('rows').closest('table');table.querySelector('colgroup')?.remove();
 const cols=document.createElement('colgroup'),header=document.createElement('tr');
 for(const c of visible){const col=document.createElement('col');col.style.width=c.width+'px';cols.append(col);const th=document.createElement('th');th.textContent=c.label;th.dataset.column=c.key;header.append(th);}
 table.prepend(cols);table.style.minWidth=visible.reduce((sum,c)=>sum+c.width,0)+'px';$('case-columns').replaceChildren(header);
 for(const row of $('rows').children){const cells=new Map([...row.children].map(td=>[td.dataset.column,td]));for(const c of orderedColumns()){const td=cells.get(c.key);if(td){td.hidden=!isVisible(c);row.append(td);}}row.hidden=!visible.length;}
}
function renderColumnPicker(){
 const options=$('column-options');options.replaceChildren();const columns=orderedColumns();
 columns.forEach((c,index)=>{
 const row=document.createElement('div');row.className='column-option';const label=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.checked=isVisible(c);box.dataset.column=c.key;
 box.addEventListener('change',()=>{columnPreferences.visibility[c.key]=box.checked;saveColumns();});label.append(box,document.createTextNode(c.label));row.append(label);
 for(const [offset,text] of [[-1,'上移'],[1,'下移']]){const button=document.createElement('button');button.type='button';button.className='secondary';button.textContent=offset<0?'↑':'↓';button.setAttribute('aria-label',c.label+text);button.disabled=index+offset<0||index+offset>=columns.length;button.addEventListener('click',()=>{const order=columns.map(c=>c.key);[order[index],order[index+offset]]=[order[index+offset],order[index]];columnPreferences.order=order;saveColumns();});row.append(button);}options.append(row);
 });
}
$('column-picker').addEventListener('toggle',()=>{if($('column-picker').open)renderColumnPicker();});
document.addEventListener('click',event=>{if(!$('column-picker').contains(event.target))$('column-picker').open=false;});
document.addEventListener('keydown',event=>{if(event.key==='Escape')$('column-picker').open=false;});
$('all-columns').addEventListener('click',()=>{for(const c of currentColumns)columnPreferences.visibility[c.key]=true;saveColumns();});
$('no-columns').addEventListener('click',()=>{for(const c of currentColumns)columnPreferences.visibility[c.key]=false;saveColumns();});
$('reset-columns').addEventListener('click',()=>{columnPreferences={order:[],visibility:{}};saveColumns();});
function unknown(){const span=document.createElement('span');span.className='unknown';span.textContent='Unknown';return span;}
function seconds(ns){return ns==null?null:(ns/1e9).toFixed(1)+' s';}
function renderRecords(items){
 const backends=[...new Set(items.flatMap(item=>Object.keys(item.backends)))].sort((a,b)=>(a==='cpu'?-1:b==='cpu'?1:a.localeCompare(b)));
 const definitions=[{key:'seed',label:'Seed / Shape',width:190,defaultVisible:true},{key:'ops',label:'关键 OP',width:220,defaultVisible:true},{key:'inputs',label:'输入 Shape / Dtype',width:230,defaultVisible:true},{key:'mode',label:'静态 / 动态',width:100,defaultVisible:true},{key:'user_name',label:'用户',width:120,defaultVisible:true}];
 // Existing column preferences keep their order; place the newly split column
 // beside Seed on the first visit, rather than at the far end of the table.
 if(columnPreferences.order.length&&!columnPreferences.order.includes('ops'))columnPreferences.order.splice(Math.max(0,columnPreferences.order.indexOf('seed')+1),0,'ops');
 for(const b of backends){if(b!=='cpu')definitions.push({key:b+':function',label:b+' 功能',width:130,defaultVisible:true},{key:b+':accuracy',label:b+' 精度',width:130,defaultVisible:true},{key:b+':kernel',label:b+' 性能',width:150,defaultVisible:false});definitions.push({key:b+':total',label:b+' 处理耗时（s）',width:150,defaultVisible:b==='af-mlir'},{key:b+':e2e',label:b+' 端到端延迟（s）',width:150,defaultVisible:false},{key:b+':logs',label:b+' 运行日志摘要',width:300,defaultVisible:true});}
 definitions.push({key:'time',label:'时间',width:180,defaultVisible:true},...extraColumns.map(([key,label])=>({key,label,width:210,defaultVisible:false})));
 const fragment=document.createDocumentFragment();
 for(const item of items){const tr=document.createElement('tr');
 for(const c of definitions){const td=cell(tr,'');td.dataset.column=c.key;let populated=false;
 if(c.key==='seed'){const strong=document.createElement('strong');strong.className='case-seed';strong.textContent=item.seed_short;td.append(strong);if(item.shape_mode==='dynamic'&&item.shape_index>=0){const tag=document.createElement('small');tag.className='shape-tag';tag.textContent='S'+item.shape_index;td.append(tag);}populated=true;}
 else if(c.key==='ops'){const ops=graphText(item.metadata);if(ops){const text=document.createElement('span');text.className='case-ops';text.textContent=ops;text.title=ops;td.append(text);populated=true;}}
 else if(c.key==='inputs'){td.textContent=tensorText(item.sample?.input_shapes,item.metadata.weights,item.sample?.input_dtypes||item.metadata.input_dtypes);populated=!!td.textContent;}
 else if(c.key==='mode'){td.textContent=item.shape_mode==='static'?'静态':'动态';populated=true;}
 else if(c.key==='user_name'){if(item.user_name&&item.user_name!=='设备未记录'){const button=document.createElement('button');button.className='text-link user-filter';button.textContent=item.user_name;button.addEventListener('click',()=>{form.elements.q.value=item.user_name;form.requestSubmit();});td.append(button);populated=true;}}
 else if(c.key==='time'){td.textContent=beijing(item.created_at);populated=true;}
 else if(c.key.includes(':')){const [b,kind]=c.key.split(':'),attempts=item.backends[b]||[],latest=attempts[0];
 if(latest){let text=null,state='';const log=b==='af-mlir'?(latest.run_log||''):latest.core_error;if(kind==='accuracy'){state=latest.correctness_status==='pass'?'pass':latest.correctness_status==='fail'?'accuracy-failed':'';text=state==='pass'?'Pass':state==='accuracy-failed'?'精度Failed':null;}
 if(kind==='function'){const category=latest.function_status||(['error','unsupported'].includes(latest.status)?(latest.error_stage==='generation'?'gen_failed':'function_failed'):['ok','fail'].includes(latest.status)?'pass':'');state=category==='gen_failed'?'graph-failed':category==='function_failed'?'failed':category==='pass'?'pass':'';text=state==='pass'?'Pass':state==='graph-failed'?'构图Failed':state==='failed'?'功能Failed':null;}
 if(kind==='logs'&&(log||['error','unsupported','fail'].includes(latest.status)))text=shortLog(log||(b==='af-mlir'?'未保存运行日志 · 查看详情':latest.status==='fail'?'精度不一致 · 查看误差':'错误摘要未采集 · 查看详情'));
 if(kind==='kernel'&&latest.measurement_status==='error'){text='性能Failed';state='failed';}
 if(kind==='kernel'&&latest.measurement_status!=='error'&&latest.kernel_ns.length)text=latest.kernel_ns.map(m=>'S'+m.shape_index+' '+(m.duration_ns/1e6).toFixed(3)+' ms').join(' · ');
 if(kind==='total')text=seconds(latest.processing_ns)||'未采集';
 if(kind==='e2e')text=seconds(latest.end_to_end_ns)||'未采集';
 if(text!==null){const button=document.createElement('button');button.className='text-link outcome-button '+state;button.textContent=text;if(kind==='total')button.title='当前 Shape 处理开始至要求的评估结束的墙钟跨度；不含共享准备，不等于 Kernel 耗时';if(kind==='logs'){button.title=log||text;button.classList.add('error-preview');}button.addEventListener('click',()=>{location.hash='attempt='+encodeURIComponent(latest.id)+'&shape='+item.shape_index;});td.append(button);populated=true;}
 if(attempts.length>1&&kind==='total'){const history=document.createElement('details'),summary=document.createElement('summary');history.className='history';summary.textContent=attempts.length+' 次运行';history.append(summary);for(const attempt of attempts){const button=document.createElement('button');button.className='text-link';button.textContent=beijing(attempt.created_at)+' · '+(seconds(attempt.total_ns)||'Unknown');button.addEventListener('click',()=>{location.hash='attempt='+encodeURIComponent(attempt.id)+'&shape='+item.shape_index;});history.append(button);}td.append(history);}
 caseMetadata.set(latest.id,item.metadata);}}
 else {const value=item.versions[c.key]??item[c.key];if(value&&value!=='historical-unrecorded'){td.textContent=value;populated=true;}}
 if(!populated)td.append(unknown());if(populated)c.populated=true;
 }fragment.append(tr);}
 $('rows').replaceChildren(fragment);configureColumns(definitions);
}
async function load(silent=false,includeList=true){
 const request=++generation,list=new URLSearchParams(applied);list.set('limit',String(pageSize));list.set('after',String(cursors[page]));
 if(!silent)$('message').textContent='加载中…';
 // A whole-dataset count must not hold the first page hostage. Keep failures
 // visible without replacing a valid list with an empty state.
 if(!silent)for(const id of ['total','ok','fail','errors','gen-errors'])$(id).textContent='…';
 const summaryTask=api('api/summary?'+applied).then(summary=>{
  if(request!==generation)return;
  $('total').textContent=summary.total.toLocaleString();$('ok').textContent=(summary.by_status.ok||0).toLocaleString();$('fail').textContent=(summary.by_status.fail||0).toLocaleString();$('gen-errors').textContent=(summary.by_status.gen_failed||0).toLocaleString();$('errors').textContent=((summary.by_status.error||0)+(summary.by_status.unsupported||0)).toLocaleString();
  return null;
 }).catch(error=>{if(request===generation)for(const id of ['total','ok','fail','errors','gen-errors'])$(id).textContent='—';return error;});
 try{if(includeList){const result=await api('api/cases?'+list);if(request!==generation)return;
 const signature=JSON.stringify(result.items);if(signature!==lastData){if(silent&&($('column-picker').open||$('rows').querySelector('details[open]')))return;lastData=signature;lastRows=result.items;renderRecords(result.items);}
 next=result.next_cursor;$('next').disabled=next===null;$('previous').disabled=page===0;$('page-label').textContent=`第 ${page+1} 页 · ${result.items.length} 条`;$('message').textContent=result.items.length?'':'没有符合条件的结果。';
 }else if(request===generation){$('message').textContent='';}
 }catch(error){if(request===generation&&error.name!=='AbortError')$('message').textContent=error.message;}
 const summaryError=await summaryTask;
 if(summaryError&&summaryError.name!=='AbortError'&&request===generation)$('message').textContent+=' 汇总加载失败：'+summaryError.message;
}
form.addEventListener('submit',event=>{event.preventDefault();applied=parameters();history.replaceState(null,'',location.pathname+(applied.size?'?'+applied:'')+location.hash);page=0;cursors=[0];detailGeneration++;$('detail').hidden=true;refreshVisible();});
$('reset-filters').addEventListener('click',()=>{form.reset();form.requestSubmit();});
$('page-size').addEventListener('change',()=>{pageSize=Number($('page-size').value);try{localStorage.setItem('af-fuzz.page-size',String(pageSize));}catch{}page=0;cursors=[0];load();});
$('next').addEventListener('click',()=>{if(next!==null){cursors[++page]=next;load();}});$('previous').addEventListener('click',()=>{if(page>0){page--;load();}});
$('copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('command').value);$('detail-message').textContent='已复制';}catch{$('command').select();}});
const logControl=document.createElement('label');logControl.className='log-length';logControl.textContent='日志摘要长度 ';const logInput=document.createElement('input');logInput.type='number';logInput.min='40';logInput.max='500';logInput.value=String(logLength);function applyLogLength(){
 const next=Math.max(40,Math.min(500,Math.trunc(Number(logInput.value))||100));
 logInput.value=String(next);logLength=next;
 try{localStorage.setItem('af-fuzz.log-length',String(logLength));}catch{}
 renderRecords(lastRows);
}
logInput.addEventListener('change',applyLogLength);
logInput.addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();applyLogLength();}});
logControl.append(logInput);$('column-picker').before(logControl);
setInterval(()=>{if(!document.hidden)refreshVisible(true);},15000);
document.addEventListener('visibilitychange',()=>{if(!document.hidden)refreshVisible(true);});

// Compare explicit attempt IDs from a bounded same-seed list, never join by name.
$('compare').addEventListener('click', async () => {
  const seed = $('command').value.split(' ').at(-1);
  $('comparison').hidden = false;
  $('comparison-content').textContent = '';
  $('comparison-message').textContent = '加载同 seed 结果…';
  try {
    const result = await api('api/results?limit=200&seed=' + encodeURIComponent(seed));
    const select = document.createElement('select');
    for (const item of result.items) {
      const option = document.createElement('option'); option.value = item.id;
      option.textContent = `${item.backend_display || item.backend} · ${beijing(item.created_at)} · ${item.id.slice(0,8)}`;
      select.append(option);
    }
    const left = select.cloneNode(true), right = select;
    left.setAttribute('aria-label','左侧结果'); right.setAttribute('aria-label','右侧结果');
    if (right.options.length > 1) right.selectedIndex = 1;
    const button = document.createElement('button'); button.textContent = '比较';
    button.disabled = result.items.length < 2;
    button.addEventListener('click', async () => {
      try {
        const value = await api('api/compare?left=' + encodeURIComponent(left.value) + '&right=' + encodeURIComponent(right.value));
        $('comparison-content').textContent = JSON.stringify(value,null,2);
      } catch (error) { $('comparison-content').textContent = error.message; }
    });
    $('comparison-message').replaceChildren(left,document.createTextNode(' ↔ '),right,document.createTextNode(' '),button);
    if(result.next_cursor !== null) $('comparison-message').append('仅显示最近/前200条，请用 CLI 指定 attempt ID 查看更多。');
    if(result.items.length < 2) $('comparison-content').textContent = '至少需要两条同 seed 结果。';
    $('comparison').scrollIntoView({behavior:'smooth',block:'start'});
  } catch(error) { $('comparison-message').textContent = error.message; }
});

let chartGeneration=0,lastChartData="";
async function loadCharts(){
  const current=++chartGeneration;
  try{
    const result=await api('api/statistics?'+applied);
    if(current!==chartGeneration)return;
    const signature=JSON.stringify(result);
    if(signature!==lastChartData){lastChartData=signature;updateStatistics(result);renderStatisticsTables(result);}
    $('chart-message').textContent='';
  }catch(error){if(current===chartGeneration&&error.name!=='AbortError')$('chart-message').textContent=error.message;}
}
function navigate(){
  const fragment=location.hash.slice(1)||'overview';
  const attempt=new URLSearchParams(fragment).get('attempt');
  if(attempt){const shape=new URLSearchParams(fragment).get('shape');details(attempt,shape===null?null:Number(shape));return;}
  const route=fragment==='performance'?'efficiency':fragment;
  const name=['overview','records','efficiency','kernel'].includes(route)?route:'overview';
  for(const key of ['overview','records','efficiency','kernel'])$(key+'-page').hidden=key!==name;
  $('detail').hidden=true;$('comparison').hidden=true;
  $('page-title').textContent={overview:'总览',records:'功能&精度',efficiency:'工具效率',kernel:'Kernel性能',backends:'后端'}[name];
  document.querySelectorAll('nav a').forEach(el=>{el.classList.toggle('active',el.dataset.page===name);if(el.dataset.page===name)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
}
window.addEventListener('hashchange',()=>{hideFieldTooltip();navigate();refreshVisible();});
$('close-detail').addEventListener('click',()=>{location.hash='records';navigate();});


function duration(ns){return ns==null?'Unknown':(ns/1e6).toFixed(3)+' ms';}
function versionText(v){return ['cann','torch','torch_npu','autofuse_commit'].map(k=>({cann:'CANN',torch:'Torch',torch_npu:'Torch NPU',autofuse_commit:'Autofuse'}[k])+': '+(v[k]||(k==='autofuse_commit'&&v.autofuse_binary?'binary '+v.autofuse_binary.slice(0,12):'未采集'))).join(' ｜ ');}
function graphText(meta={}){return (meta.op_types||[]).join(' · ');}
function tensorText(shapes={},weights=[],types={}){
 const short={float16:'fp16',float32:'fp32',float64:'fp64',bfloat16:'bf16'};
 const entries=[...Object.entries(shapes).map(([n,s])=>({shape:s,dtype:types[n]})),...weights];
 return entries.map((t,i)=>'x'+(i+1)+'['+[...t.shape,short[t.dtype]||t.dtype||'Unknown'].join(',')+']').join(', ');
}
function shortLog(text){return text.length>logLength?text.slice(0,logLength)+'...':text;}
function renderCaseInfo(result){
 const box=$('case-meta');box.replaceChildren();
 const heading=document.createElement('h3');heading.textContent='用例节点 DAG';box.append(heading);
 const note=document.createElement('p');note.className='muted';
 note.textContent=result.seed_short+' · '+(result.backend_display||result.backend)+' · '+(result.contexts.execution.static_only?'静态':'动态');if(!result.contexts.execution.static_only&&result.shape_index>=0){const tag=document.createElement('small');tag.className='shape-tag';tag.textContent='S'+result.shape_index;note.append(tag);}box.append(note);
 renderCaseDag(box,result);
}
function renderProvenance(result){
 const box=$('provenance');box.replaceChildren();const heading=document.createElement('h3');heading.textContent='运行环境';box.append(heading);
 const table=document.createElement('table'),env=result.contexts.environment;
 const labels={device_name:'用户',cann:'CANN',torch:'Torch',torch_npu:'Torch NPU',autofuse_commit:'Autofuse commit',autofuse_binary:'Autofuse binary',triton:'Triton',python:'Python',package_version:'Debugger',machine:'架构',platform:'系统'};
 for(const [key,label] of Object.entries(labels)){if(env[key]){const row=document.createElement('tr');cell(row,label);cell(row,String(env[key]));table.append(row);}}box.append(table);
 const telemetry=result.contexts.execution.telemetry?JSON.parse(result.contexts.execution.telemetry):{stages:{},shapes:{}};
 const host=$('timing-detail');host.replaceChildren();
 const headingTime=document.createElement('h3');headingTime.textContent='用例执行耗时';host.append(headingTime);
 const note=document.createElement('p');note.className='muted';note.textContent='本次运行 '+result.shape_count+' 个 Shape 的计时。执行总计包含能力检查、准备和这些 Shape；生成耗时单独记录。下列指标不能逐项相加。';host.append(note);
 const tableTime=document.createElement('table'),head=document.createElement('tr');cell(head,'阶段','th');cell(head,'耗时','th');tableTime.append(head);
 for(const [key,value] of Object.entries(telemetry.stages||{})){const row=document.createElement('tr');cell(row,stageName(key));cell(row,(value/1e9).toFixed(1)+' s');tableTime.append(row);}
 for(const [index,metrics] of Object.entries(telemetry.shapes||{})){for(const [key,value] of Object.entries(metrics)){const row=document.createElement('tr');cell(row,'S'+index+' '+stageName(key));cell(row,(value/1e9).toFixed(1)+' s');tableTime.append(row);}}
 if(tableTime.children.length>1)host.append(tableTime);else{const empty=document.createElement('p');empty.textContent='未采集本次运行计时';host.append(empty);}

 const timeline=document.createElement('table'),th=document.createElement('tr');for(const title of ['泳道 / 阶段','状态','相对接纳时间（s）','依赖等待（s）','资源等待（s）','运行（s）','worker / 冷暖'])cell(th,title,'th');timeline.append(th);
 for(const f of result.stage_timeline||[]){const row=document.createElement('tr');const lane=f.lease?.device_uuid?'NPU '+f.lease.device_uuid:['generate','reference'].includes(f.kind)?'CPU 共享':f.kind==='compare'?'CPU':'记录';const begin=f.started??f.dispatched,end=f.finished;for(const value of [stageLabel(f),f.state,begin==null?'未采集':seconds(begin-f.admitted),f.ready==null?'未采集':seconds(f.ready-f.admitted),f.ready==null||f.dispatched==null?'未采集':seconds(f.dispatched-f.ready),begin==null||end==null?'未采集':seconds(end-begin),(f.worker_id||'未采集')+' / '+(f.temperature||'未采集')])cell(row,value);timeline.append(row);}
 if((result.stage_timeline||[]).length){host.append(timeline);const lanes=document.createElement('div');lanes.className='stage-lanes';const fs=result.stage_timeline.filter(f=>f.started!=null&&f.finished!=null);const origin=Math.min(...fs.map(f=>f.admitted));const extent=Math.max(1,...fs.map(f=>f.finished-origin));for(const f of fs){const lane=document.createElement('div');lane.className='stage-lane';const label=document.createElement('span');label.textContent=stageLabel(f);const track=document.createElement('div');track.className='stage-track';const add=(a,b,cls)=>{if(b<=a)return;const bar=document.createElement('span');bar.className=cls;bar.style.left=(100*(a-origin)/extent)+'%';bar.style.width=(100*(b-a)/extent)+'%';bar.title=stageName(f.kind)+' '+seconds(b-a);track.append(bar);};add(f.admitted,f.ready??f.started,'stage-dependency');add(f.ready??f.started,f.dispatched??f.started,'stage-resource');add(f.started,f.finished,f.state==='failed'?'stage-failed':'stage-running');lane.append(label,track);lanes.append(lane);}const legend=document.createElement('p');legend.className='muted';legend.textContent='灰：依赖等待 · 黄：资源等待 · 绿：运行 · 红：失败；共享生成/参考不摊分为单 Shape。';lanes.append(legend);host.append(lanes);}

 const life=document.createElement('p');life.className='muted';life.title=(result.lifecycle||[]).map(e=>e.event+' '+e.wall_time).join(' → ');life.textContent='生命周期（UTC，悬停查看日期）：'+((result.lifecycle||[]).map(e=>e.event+' '+lifecycleTime(e.wall_time)).join(' → ')||'未采集')+'。端到端延迟：'+(seconds(result.end_to_end_ns)||'未采集');host.append(life);
}

// Keep the selected scope across background refreshes, without pinning "latest".
const efficiencyView={windowId:'',query:'',page:0,details:false,processing:false};
const efficiencyPageSize=10;
function efficiencyGroups(rows){
 const groups=new Map();
 for(const r of rows){
  const key=JSON.stringify([r.backend,r.environment_id,r.mode]);
  if(!groups.has(key))groups.set(key,{backend:r.backend,environment_id:r.environment_id,mode:r.mode,rows:[],success:0,failed:0,case_errors:0,completed:0});
  const g=groups.get(key);g.rows.push(r);
  for(const name of ['success','failed','case_errors','completed'])g[name]+=Number(r[name]||0);
 }
 return [...groups.values()];
}
function renderEfficiency(result){
 const host=$('efficiency-table');host.replaceChildren();
 const element=(tag,text,cls)=>{const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el;};
 const table=(headers)=>{const t=element('table'),head=element('tr');for(const h of headers)cell(head,h,'th');t.append(head);return t;};
 const wrap=t=>{const box=element('div',undefined,'table-scroll');box.append(t);return box;};
 const windows=new Map();
 for(const row of result.tool_efficiency?.windows||[]){if(!windows.has(row.window_id))windows.set(row.window_id,[]);windows.get(row.window_id).push(row);}
 const ids=[...windows.keys()];
 const toolbar=element('div',undefined,'efficiency-toolbar');
 const label=element('label','运行窗口 '),select=element('select');select.setAttribute('aria-label','运行窗口');
 const latest=element('option','最近一次已完成运行');latest.value='';select.append(latest);
 ids.forEach((id,i)=>{const option=element('option',`${i===0?'最新':'历史 '+i} · ${id}`);option.value=id;select.append(option);});
 if(efficiencyView.windowId&&!windows.has(efficiencyView.windowId)){
  efficiencyView.windowId='';efficiencyView.page=0;
  host.append(element('p','原选窗口已不在当前结果中，已切换到最近一次已完成运行。','muted'));
 }
 select.value=efficiencyView.windowId;select.disabled=!ids.length;label.append(select);toolbar.append(label);
 const filter=element('input');filter.type='search';filter.placeholder='筛选后端 / 环境 / static / dynamic';filter.setAttribute('aria-label','工具效率维度筛选');filter.value=efficiencyView.query;toolbar.append(filter);host.append(toolbar);
 const body=element('div');host.append(body);
 const render=()=>{
  body.replaceChildren();
  const id=efficiencyView.windowId||ids[0],rows=windows.get(id)||[];
  if(!rows.length){body.append(element('p','暂无已完成运行窗口的效率数据。','muted'));return;}
  const window=rows[0],duration=Number(window.window_seconds);
  const scope=element('p',`当前窗口：${id} · 时长 ${Number.isFinite(duration)?duration.toFixed(1)+' s':'未采集'} · 可选 ${ids.length} 个窗口（最多最近 100 个）`,'muted efficiency-scope');body.append(scope);
  body.append(element('p','吞吐按完整窗口时长计算；下表只合并同后端、同环境、同 Shape 模式的计数。资源占用属于整个窗口，不随下方筛选变化。','muted'));
  const cards=element('div',undefined,'efficiency-cards');
  const card=(title,value)=>{const box=element('div');box.append(element('span',title),element('strong',value));cards.append(box);};
  card('CPU 资源池使用率',window.cpu_pool_utilization==null?'未采集':(100*window.cpu_pool_utilization).toFixed(1)+'%');
  const devices=Object.entries(window.npu_slot_utilization||{});
  devices.forEach(([uuid,value],i)=>{card('NPU '+(i+1)+' 执行槽使用率',(100*value).toFixed(1)+'%');cards.lastChild.title=uuid;});
  card('窗口取消 / 写库失败',(window.cancelled_attempts??'未采集')+' / '+(window.record_failed_attempts??'未采集'));body.append(cards);
  body.append(element('p','CPU 为 token-seconds / 容量，NPU 为执行槽占用 / 设备时间，不等于实测 CPU / AICore 使用率；窗口包含填充和排空。','muted'));
  const tokens=efficiencyView.query.toLowerCase().split(/\s+/).filter(Boolean);
  const groups=efficiencyGroups(rows).filter(g=>tokens.every(t=>[g.backend,g.environment_id,g.mode,...g.rows.map(r=>r.hardware||'')].join(' ').toLowerCase().includes(t)));
  const pages=Math.max(1,Math.ceil(groups.length/efficiencyPageSize));efficiencyView.page=Math.min(efficiencyView.page,pages-1);
  const visible=groups.slice(efficiencyView.page*efficiencyPageSize,(efficiencyView.page+1)*efficiencyPageSize);
  const summary=table(['后端 / Shape 模式','环境','成功 / 失败 Shape / Case 错误','成功 Shape/min','已完成 Shape/min（含失败）','处理 P50 / P95','端到端 P50 / P95']);
  const rate=n=>duration>0?(n*60/duration).toFixed(2):'未采集';
  const latency=(g,key)=>g.rows.length===1?(seconds(g.rows[0][key+'_p50_ns'])||'未采集')+' / '+(seconds(g.rows[0][key+'_p95_ns'])||'未采集'):'见分组明细';
  for(const g of visible){const tr=element('tr');for(const value of [g.backend+' / '+g.mode,(g.environment_id||'未知').slice(0,12),g.success+' / '+g.failed+' / '+g.case_errors,rate(g.success),rate(g.completed),latency(g,'processing'),latency(g,'e2e')])cell(tr,value);tr.children[1].title=g.environment_id;summary.append(tr);}
  body.append(wrap(summary));
  if(!groups.length)body.append(element('p','没有匹配的后端或环境。','muted'));
  const pager=element('div',undefined,'efficiency-toolbar');
  const previous=element('button','上一页'),next=element('button','下一页');previous.disabled=efficiencyView.page===0;next.disabled=efficiencyView.page>=pages-1;
  previous.onclick=()=>{efficiencyView.page--;render();};next.onclick=()=>{efficiencyView.page++;render();};
  pager.append(element('span',`${groups.length} 个对比分组 · 第 ${efficiencyView.page+1} / ${pages} 页 · 每页 ${efficiencyPageSize} 组`),previous,next);body.append(pager);
  const details=element('details',undefined,'efficiency-details');details.open=efficiencyView.details;
  details.append(element('summary','诊断明细：当前页分组的冷 / 热、状态、延迟与环境'));
  const detailBody=element('div');details.append(detailBody);let populated=false;
  const populate=()=>{
   if(populated)return;populated=true;
   detailBody.append(element('p','不同冷 / 热和状态分组的 P50/P95 不能直接平均，以下保留各组原始统计；唯一 Shape 数也不跨组相加。','muted'));
   for(const g of visible){
    detailBody.append(element('h4',g.backend+' / '+g.mode+' · '+g.environment_id.slice(0,12)));
    const environment=element('details');environment.append(element('summary','完整环境与设备信息'));
    environment.append(element('pre',JSON.stringify({environment_id:g.environment_id,hardware:g.rows[0].hardware,software:g.rows[0].software,devices:Object.fromEntries(devices)},null,2)));detailBody.append(environment);
    const t=table(['冷 / 热 · 状态','成功 / 失败 / Case 错误','处理 P50 / P95（s）','端到端 P50 / P95（s）','唯一 Shape / 重试','摊销秒/Shape','冷 / 热调用数']);
    for(const r of g.rows){const tr=element('tr');for(const value of [r.temperature+' / '+r.status,r.success+' / '+r.failed+' / '+r.case_errors,(seconds(r.processing_p50_ns)||'未采集')+' / '+(seconds(r.processing_p95_ns)||'未采集'),(seconds(r.e2e_p50_ns)||'未采集')+' / '+(seconds(r.e2e_p95_ns)||'未采集'),r.unique_shapes+' / '+(r.retry_attempts??'未采集'),r.amortized_seconds_per_shape==null?'未采集':r.amortized_seconds_per_shape.toFixed(2),r.cold+' / '+r.warm])cell(tr,value);t.append(tr);}detailBody.append(wrap(t));
   }
  };
  if(details.open)populate();details.ontoggle=()=>{efficiencyView.details=details.open;if(details.open)populate();};body.append(details);
 };
 select.onchange=()=>{efficiencyView.windowId=select.value;efficiencyView.page=0;render();};
 filter.oninput=()=>{efficiencyView.query=filter.value;efficiencyView.page=0;render();};render();
 const processing=element('details',undefined,'efficiency-details');processing.open=efficiencyView.processing;
 processing.append(element('summary','全部匹配记录的处理耗时（非当前窗口）'));
 const t=table(['后端','处理耗时样本数','平均处理耗时（s）','最大处理耗时（s）']);
 for(const r of result.tool_efficiency?.processing||[]){const tr=element('tr');for(const v of [r.backend,r.samples,seconds(r.mean_ns),seconds(r.max_ns)])cell(tr,v);t.append(tr);}processing.append(wrap(t));processing.ontoggle=()=>{efficiencyView.processing=processing.open;};host.append(processing);
}

function renderStatisticsTables(result){
 renderEfficiency(result);
 const t=document.createElement('table'),head=document.createElement('tr');for(const label of ['后端','指标','样本','均值','最大'])cell(head,label,'th');t.append(head);
 for(const r of result.precise_metrics||[]){const row=document.createElement('tr');for(const value of [r.backend,r.stage==='compile'?'编译耗时':'Kernel 耗时',r.samples,duration(r.mean_ns),duration(r.max_ns)])cell(row,value);t.append(row);}
 $('performance-table').replaceChildren(t);$('metric-empty').hidden=!!result.precise_metrics?.length;
}

function stageName(stage){return ({generate:'共享生成',reference:'CPU 期望结果',compare:'CPU 结果校验',record:'持久化',shape_processing:'Shape 处理耗时（墙钟跨度）',generation:'生成用例',capability:'后端能力检查',prepare:'准备执行器（不代表编译完成）',input_binding:'输入绑定',first_call_combined:'首调(惰性编译/执行/同步)',correctness:'精度校验',execution_total:'执行总计',compile:'独立编译',kernel_mean:'Kernel 平均耗时',performance:'性能测量'}[stage]||stage||'Unknown');}
function renderFailureDetails(result){
 const view=result.diagnostic_view||{};
 function section(id,title,content){const box=$(id);box.replaceChildren();const heading=document.createElement('h3'),pre=document.createElement('pre');heading.textContent=title;pre.textContent=content;box.append(heading,pre);return box;}
 section('ascir-detail','ASCIR 表达的图结构',view.shape_log||'未保存当前 Shape 的 ASCIR / 图结构文本');
 // Full saved run log already contains its extracted shared lines; do not repeat them.
 const log=view.run_log||view.shared_log||view.shared_lines?.join('\n')||view.summary||'未保存报错日志';
 const box=section('failure-details','报错日志摘要',log);

}

// Pure metadata layout: no graph execution or model/runtime dependencies.
function caseDag(result){
 const meta=result.metadata||{},graph=meta.graph||{},sample=result.sample||meta.shape_variants?.[0]||{};
 if(!graph.nodes?.length)return {nodes:[],edges:[]};
 const nodes=[],byName=new Map(),edges=[];
 function add(name,label,tensor={},kind='op',inputs=[],attrs={}){
  const node={name,label,...tensor,kind,inputs,attrs,level:0};nodes.push(node);byName.set(name,node);return node;
 }
 const shapes=sample.input_shapes||meta.input_shapes||{},types=sample.input_dtypes||meta.input_dtypes||{};
 for(const [name,shape] of Object.entries(shapes))add(name,'Input',{shape,dtype:types[name]},'input');
 for(const weight of meta.weights||[])add(weight.name,'Weight',weight,'weight');
 for(const node of graph.nodes)add(node.output,node.op,{},'op',node.inputs||[],node.attrs||{});
 for(const node of [...nodes])for(const name of node.inputs)if(!byName.has(name))add(name,'未记录输入',{},'input');
 const pending=new Set(nodes),resolved=new Set();
 while(pending.size){
  let progress=false;
  for(const node of pending){
   if(node.inputs.some(name=>!resolved.has(name)))continue;
   const inputs=node.inputs.map(name=>byName.get(name));
   node.level=inputs.length?Math.max(...inputs.map(n=>n.level))+1:0;
   if(node.kind==='op'){
    const first=inputs[0],shape=first?.shape;
    node.dtype=inputs.length&&inputs.every(n=>n.dtype===first.dtype)?first.dtype:undefined;
    if(shape){
     if(['relu','neg'].includes(node.label))node.shape=[...shape];
     else if(node.label==='transpose'){
      const {dim0,dim1}=node.attrs,rank=shape.length;
      if(Number.isInteger(dim0)&&Number.isInteger(dim1)&&dim0>=-rank&&dim0<rank&&dim1>=-rank&&dim1<rank){const a=(dim0+rank)%rank,b=(dim1+rank)%rank;node.shape=[...shape];[node.shape[a],node.shape[b]]=[node.shape[b],node.shape[a]];}
     }else if(['sum','mean','amin','amax'].includes(node.label)&&Array.isArray(node.attrs.reduce_dim)){
      const dims=node.attrs.reduce_dim.map(d=>d<0?d+shape.length:d);
      if(dims.every(d=>Number.isInteger(d)&&d>=0&&d<shape.length))node.shape=shape.filter((_,i)=>!dims.includes(i));
     }else if(['add','mul'].includes(node.label)&&inputs.every(n=>Array.isArray(n.shape))){
      const rank=Math.max(...inputs.map(n=>n.shape.length)),output=[];let valid=true;
      for(let i=0;i<rank;i++){const dims=inputs.map(n=>n.shape[n.shape.length-rank+i]??1),nonUnit=[...new Set(dims.filter(d=>d!==1))];if(nonUnit.length>1)valid=false;output.push(nonUnit[0]??1);}if(valid)node.shape=output;
     }
    }
    node.inferred=!!node.shape;
   }
   // Captured outputs take precedence over inference, including scalar [].
   const outputShape=(sample.output_shapes||meta.output_shapes||{})[node.name];
   if(outputShape){node.shape=outputShape;node.dtype=(sample.output_dtypes||meta.output_dtypes||{})[node.name]||node.dtype;node.inferred=false;}
   node.isOutput=(graph.outputs||[]).includes(node.name);
   resolved.add(node.name);pending.delete(node);progress=true;
  }
  if(!progress)return {nodes:[],edges:[],error:'已保存节点依赖存在环，无法绘制 DAG'};
 }
 for(const node of nodes)for(const name of node.inputs)edges.push({from:byName.get(name),to:node});
 return {nodes,edges};
}
function renderCaseDag(host,result){
 const graph=caseDag(result);
 if(!graph.nodes.length){const empty=document.createElement('p');empty.className='muted';empty.textContent=graph.error||'未保存节点依赖，无法绘制 DAG';host.append(empty);return;}
 const ns='http://www.w3.org/2000/svg';
 function svg(tag,attrs={},text){const el=document.createElementNS(ns,tag);for(const [key,value] of Object.entries(attrs))el.setAttribute(key,value);if(text!==undefined)el.textContent=text;return el;}
 const levels=[];for(const node of graph.nodes)(levels[node.level]??=[]).push(node);
 const width=Math.max(640,...levels.map(nodes=>nodes.length*270+32)),height=levels.length*150+32;
 const canvas=svg('svg',{viewBox:`0 0 ${width} ${height}`,width,height,role:'img','aria-label':'用例节点 DAG，箭头表示张量依赖'});
 const defs=svg('defs'),marker=svg('marker',{id:'dag-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'});marker.append(svg('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#829b92'}));defs.append(marker);canvas.append(defs);
 for(const layer of levels)layer.forEach((node,i)=>{node.x=(width-layer.length*270)/2+i*270+10;node.y=node.level*150+16;});
 for(const {from,to} of graph.edges){const x=from.x+125,y=from.y+100,endX=to.x+125,endY=to.y;const path=svg('path',{d:`M ${x} ${y} C ${x} ${y+25}, ${endX} ${endY-25}, ${endX} ${endY}`,class:'dag-edge','marker-end':'url(#dag-arrow)'});path.append(svg('title',{},from.name+' → '+to.name));canvas.append(path);}
 for(const node of graph.nodes){
  const tensor=(node.shape?'['+node.shape.join(', ')+']':'Shape 未采集')+' · '+(node.dtype||'Dtype 未采集')+(node.inferred?' *':'');
  const title=node.label+' · '+node.name+(node.isOutput?' · Output':'');
  const tooltip=title+'\n'+tensor+(Object.keys(node.attrs).length?'\n'+JSON.stringify(node.attrs):'');
  const group=svg('g',{transform:`translate(${node.x},${node.y})`,class:'dag-node '+node.kind,tabindex:0,'aria-label':tooltip,'data-tooltip':tooltip});group.append(svg('title',{},tooltip),svg('rect',{width:250,height:100,rx:10}));
  function label(y,text,cls){const el=svg('text',{x:12,y,class:cls},text.length>32?text.slice(0,31)+'…':text);group.append(el);}
  label(26,title,'dag-op');label(50,node.name+(node.isOutput?' → Output':''),'dag-name');label(76,tensor,'dag-tensor');canvas.append(group);
 }
 const scroll=document.createElement('div');scroll.className='dag-scroll';scroll.append(canvas);host.append(scroll);
}

// Use a real wrapping tooltip instead of browser titles (which may truncate long logs).
const fieldTooltip=document.createElement('div');fieldTooltip.id='field-tooltip';fieldTooltip.className='field-tooltip';fieldTooltip.setAttribute('role','tooltip');fieldTooltip.hidden=true;document.body.append(fieldTooltip);
let tooltipTarget=null;
function hideFieldTooltip(){if(tooltipTarget)tooltipTarget.removeAttribute('aria-describedby');tooltipTarget=null;fieldTooltip.hidden=true;}
function showFieldTooltip(event){
 const target=event.target.closest?.('[data-tooltip], [title], td, th, input[readonly]');
 if(!target||target.closest('#field-tooltip'))return;
 const content=target.dataset.tooltip||target.getAttribute('title')||target.value||target.textContent;
 if(!content?.trim())return;
 hideFieldTooltip();tooltipTarget=target;
 // Preserve explicit full values, not the shortened preview text.
 if(target.hasAttribute('title')){target.dataset.tooltip=target.getAttribute('title');target.removeAttribute('title');}
 fieldTooltip.textContent=content;fieldTooltip.hidden=false;target.setAttribute('aria-describedby','field-tooltip');
 const bounds=target.getBoundingClientRect(),tip=fieldTooltip.getBoundingClientRect();
 const x=event.clientX??bounds.left,y=event.clientY??bounds.bottom;
 fieldTooltip.style.left=Math.max(8,Math.min(x,innerWidth-tip.width-8))+'px';
 fieldTooltip.style.top=Math.max(8,Math.min(y+14,innerHeight-tip.height-8))+'px';
}
document.addEventListener('pointerover',showFieldTooltip);
document.addEventListener('focusin',showFieldTooltip);
document.addEventListener('pointerout',event=>{if(tooltipTarget&&!tooltipTarget.contains(event.relatedTarget)&&!fieldTooltip.contains(event.relatedTarget))hideFieldTooltip();});
fieldTooltip.addEventListener('pointerleave',hideFieldTooltip);
document.addEventListener('focusout',hideFieldTooltip);
document.addEventListener('keydown',event=>{if(event.key==='Escape')hideFieldTooltip();});
window.addEventListener('scroll',event=>{if(event.target!==fieldTooltip)hideFieldTooltip();},true);
window.addEventListener('resize',hideFieldTooltip);

navigate();refreshVisible();

function lifecycleTime(value){
 const date=new Date(value);
 return Number.isNaN(date.getTime())?'未采集':date.toISOString().slice(11,23);
}
function stageLabel(f){
 if(f.kind==='reference'||f.kind==='compare')return stageName(f.kind);
 if(f.kind==='first_call_combined')return (f.lease?.device_uuid?'NPU':'CPU')+stageName(f.kind);
 return stageName(f.kind);
}
