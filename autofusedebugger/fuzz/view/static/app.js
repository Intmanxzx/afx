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
let pageSize=50;
try{const size=Number(localStorage.getItem('af-fuzz.page-size'));if([10,25,50,100,200].includes(size))pageSize=size;}catch{}
$('page-size').value=String(pageSize);
function parameters(){const p=new URLSearchParams();const q=form.elements.q.value.trim();if(q)p.set('q',q);return p;}
async function api(path) {
  const response = await fetch(path, {cache:'no-store',credentials:'omit'});
  const value = await response.json();
  if(!response.ok) throw new Error(value.error || '读取失败');
  return value;
}
function cell(row, value, tag="td") {
  const td = document.createElement(tag); td.textContent = value; row.append(td); return td;
}
async function details(identity, shapeIndex=null) {
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
    $('page-title').textContent='用例详情 · '+result.seed_short+' / S'+(shapeIndex??'—');
    $('detail').scrollIntoView({behavior:'smooth',block:'start'});
  } catch(error) {if(request === detailGeneration) $('detail-message').textContent = error.message;}
}
const extraColumns=[['cann','CANN'],['torch','Torch'],['torch_npu','Torch NPU'],['autofuse_commit','Autofuse commit'],['autofuse_binary','Autofuse binary'],['triton','Triton'],['python','Python'],['package_version','Debugger'],['machine','架构'],['platform','系统'],['source_id','来源 ID'],['batch_id','批次']];
const preferenceKey='af-fuzz.record-columns.v3';
let columnPreferences={order:[],visibility:{}},currentColumns=[],lastData='',lastRows=[];
try{const saved=JSON.parse(localStorage.getItem(preferenceKey));if(saved&&Array.isArray(saved.order)&&saved.visibility&&typeof saved.visibility==='object')columnPreferences=saved;}catch{}
function orderedColumns(){return [...columnPreferences.order.map(key=>currentColumns.find(c=>c.key===key)).filter(Boolean),...currentColumns.filter(c=>!columnPreferences.order.includes(c.key))];}
function isVisible(c){return columnPreferences.visibility[c.key]??(c.defaultVisible&&c.populated);}
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
function unknown(){const span=document.createElement('span');span.className='unknown';span.textContent='Unkown';return span;}
function seconds(ns){return ns==null?null:(ns/1e9).toFixed(1)+' s';}
function renderRecords(items){
 const backends=[...new Set(items.flatMap(item=>Object.keys(item.backends)))].sort((a,b)=>(a==='cpu'?-1:b==='cpu'?1:a.localeCompare(b)));
 const definitions=[{key:'seed',label:'Seed / Shape',width:190,defaultVisible:true},{key:'ops',label:'关键 OP',width:220,defaultVisible:true},{key:'inputs',label:'输入 Shape / Dtype',width:230,defaultVisible:true},{key:'mode',label:'静态 / 动态',width:100,defaultVisible:true},{key:'user_name',label:'用户',width:120,defaultVisible:true}];
 // Existing column preferences keep their order; place the newly split column
 // beside Seed on the first visit, rather than at the far end of the table.
 if(columnPreferences.order.length&&!columnPreferences.order.includes('ops'))columnPreferences.order.splice(Math.max(0,columnPreferences.order.indexOf('seed')+1),0,'ops');
 for(const b of backends){if(b!=='cpu')definitions.push({key:b+':function',label:b+' 功能',width:130,defaultVisible:true},{key:b+':accuracy',label:b+' 精度',width:130,defaultVisible:true},{key:b+':kernel',label:b+' 性能',width:150,defaultVisible:false});definitions.push({key:b+':total',label:b+' 总耗时',width:150,defaultVisible:b==='af-mlir'},{key:b+':logs',label:b+' 运行日志摘要',width:300,defaultVisible:true});}
 definitions.push({key:'time',label:'时间',width:180,defaultVisible:true},...extraColumns.map(([key,label])=>({key,label,width:210,defaultVisible:false})));
 const fragment=document.createDocumentFragment();
 for(const item of items){const tr=document.createElement('tr');
 for(const c of definitions){const td=cell(tr,'');td.dataset.column=c.key;let populated=false;
 if(c.key==='seed'){const strong=document.createElement('strong');strong.className='case-seed';strong.textContent=item.seed_short+' / S'+(item.shape_index<0?'—':item.shape_index);td.append(strong);populated=true;}
 else if(c.key==='ops'){const ops=graphText(item.metadata);if(ops){const text=document.createElement('span');text.className='case-ops';text.textContent=ops;text.title=ops;td.append(text);populated=true;}}
 else if(c.key==='inputs'){td.textContent=tensorText(item.sample?.input_shapes,item.metadata.weights,item.sample?.input_dtypes||item.metadata.input_dtypes);populated=!!td.textContent;}
 else if(c.key==='mode'){td.textContent=item.shape_mode==='static'?'静态':'动态';populated=true;}
 else if(c.key==='user_name'){if(item.user_name&&item.user_name!=='设备未记录'){const button=document.createElement('button');button.className='text-link user-filter';button.textContent=item.user_name;button.addEventListener('click',()=>{form.elements.q.value=item.user_name;form.requestSubmit();});td.append(button);populated=true;}}
 else if(c.key==='time'){td.textContent=beijing(item.created_at);populated=true;}
 else if(c.key.includes(':')){const [b,kind]=c.key.split(':'),attempts=item.backends[b]||[],latest=attempts[0];
 if(latest){let text=null,state='';const log=b==='af-mlir'?(latest.run_log||''):latest.core_error;if(kind==='accuracy'){state=latest.correctness_status==='pass'?'pass':latest.correctness_status==='fail'?'accuracy-failed':'';text=state==='pass'?'Pass':state==='accuracy-failed'?'精度Failed':null;}
 if(kind==='function'){state=['error','unsupported'].includes(latest.status)?'failed':['ok','fail'].includes(latest.status)?'pass':'';text=state==='pass'?'Pass':state==='failed'?'功能Failed':null;}
 if(kind==='logs'&&(log||['error','unsupported','fail'].includes(latest.status)))text=shortLog(log||(b==='af-mlir'?'未保存运行日志 · 查看详情':latest.status==='fail'?'精度不一致 · 查看误差':'错误摘要未采集 · 查看详情'));
 if(kind==='kernel'&&latest.measurement_status==='error'){text='性能Failed';state='failed';}
 if(kind==='kernel'&&latest.measurement_status!=='error'&&latest.kernel_ns.length)text=latest.kernel_ns.map(m=>'S'+m.shape_index+' '+(m.duration_ns/1e6).toFixed(3)+' ms').join(' · ');
 if(kind==='total')text=seconds(latest.total_ns);
 if(text!==null){const button=document.createElement('button');button.className='text-link outcome-button '+state;button.textContent=text;if(kind==='total')button.title='当前 Shape：输入绑定 + 首次调用（含惰性编译）+ 精度校验；不等于 Kernel 耗时';if(kind==='logs'){button.title=log||text;button.classList.add('error-preview');}button.addEventListener('click',()=>{location.hash='attempt='+encodeURIComponent(latest.id)+'&shape='+item.shape_index;});td.append(button);populated=true;}
 if(attempts.length>1&&kind==='total'){const history=document.createElement('details'),summary=document.createElement('summary');history.className='history';summary.textContent=attempts.length+' 次运行';history.append(summary);for(const attempt of attempts){const button=document.createElement('button');button.className='text-link';button.textContent=beijing(attempt.created_at)+' · '+(seconds(attempt.total_ns)||'Unkown');button.addEventListener('click',()=>{location.hash='attempt='+encodeURIComponent(attempt.id)+'&shape='+item.shape_index;});history.append(button);}td.append(history);}
 caseMetadata.set(latest.id,item.metadata);}}
 else {const value=item.versions[c.key]??item[c.key];if(value&&value!=='historical-unrecorded'){td.textContent=value;populated=true;}}
 if(!populated)td.append(unknown());if(populated)c.populated=true;
 }fragment.append(tr);}
 $('rows').replaceChildren(fragment);configureColumns(definitions);
}
async function load(silent=false){
 const request=++generation,list=new URLSearchParams(applied);list.set('limit',String(pageSize));list.set('after',String(cursors[page]));
 if(!silent)$('message').textContent='加载中…';
 try{const [summary,result]=await Promise.all([api('api/summary?'+applied),api('api/cases?'+list)]);if(request!==generation)return;
 $('refresh-time').textContent='更新于 '+beijing(new Date());
 $('total').textContent=summary.total.toLocaleString();$('ok').textContent=(summary.by_status.ok||0).toLocaleString();$('fail').textContent=(summary.by_status.fail||0).toLocaleString();$('errors').textContent=((summary.by_status.error||0)+(summary.by_status.unsupported||0)).toLocaleString();
 const signature=JSON.stringify(result.items);if(signature!==lastData){if(silent&&($('column-picker').open||$('rows').querySelector('details[open]')))return;lastData=signature;lastRows=result.items;renderRecords(result.items);}
 next=result.next_cursor;$('next').disabled=next===null;$('previous').disabled=page===0;$('page-label').textContent=`第 ${page+1} 页 · ${result.items.length} 条`;$('message').textContent=result.items.length?'':'没有符合条件的结果。';
 }catch(error){if(request===generation)$('message').textContent=error.message;}
}
form.addEventListener('submit',event=>{event.preventDefault();applied=parameters();history.replaceState(null,'',location.pathname+(applied.size?'?'+applied:'')+location.hash);page=0;cursors=[0];detailGeneration++;$('detail').hidden=true;load();loadCharts();});
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
load();setInterval(()=>{if(!document.hidden){load(true);loadCharts();}},30000);

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
  }catch(error){if(current===chartGeneration)$('chart-message').textContent=error.message;}
}
function navigate(){
  const fragment=location.hash.slice(1)||'overview';
  const attempt=new URLSearchParams(fragment).get('attempt');
  if(attempt){const shape=new URLSearchParams(fragment).get('shape');details(attempt,shape===null?null:Number(shape));return;}
  const name=['overview','records','performance'].includes(fragment)?fragment:'overview';
  for(const key of ['overview','records','performance'])$(key+'-page').hidden=key!==name;
  $('detail').hidden=true;$('comparison').hidden=true;
  $('page-title').textContent={overview:'总览',records:'功能&精度',performance:'性能分析',backends:'后端'}[name];
  document.querySelectorAll('nav a').forEach(el=>{el.classList.toggle('active',el.dataset.page===name);if(el.dataset.page===name)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
}
window.addEventListener('hashchange',navigate);
$('close-detail').addEventListener('click',()=>{location.hash='records';navigate();});
loadCharts();navigate();

function duration(ns){return ns==null?'Unkown':(ns/1e6).toFixed(3)+' ms';}
function versionText(v){return ['cann','torch','torch_npu','autofuse_commit'].map(k=>({cann:'CANN',torch:'Torch',torch_npu:'Torch NPU',autofuse_commit:'Autofuse'}[k])+': '+(v[k]||(k==='autofuse_commit'&&v.autofuse_binary?'binary '+v.autofuse_binary.slice(0,12):'未采集'))).join(' ｜ ');}
function graphText(meta={}){return (meta.op_types||[]).join(' · ');}
function tensorText(shapes={},weights=[],types={}){
 const short={float16:'fp16',float32:'fp32',float64:'fp64',bfloat16:'bf16'};
 const entries=[...Object.entries(shapes).map(([n,s])=>({shape:s,dtype:types[n]})),...weights];
 return entries.map((t,i)=>'x'+(i+1)+'['+[...t.shape,short[t.dtype]||t.dtype||'Unkown'].join(',')+']').join(', ');
}
function shortLog(text){return text.length>logLength?text.slice(0,logLength)+'...':text;}
function renderCaseInfo(result){
 const box=$('case-meta');box.replaceChildren();
 const heading=document.createElement('h3');heading.textContent='用例节点 DAG';box.append(heading);
 const note=document.createElement('p');note.className='muted';
 note.textContent=result.seed_short+' / S'+(result.shape_index??'—')+' · '+(result.backend_display||result.backend)+' · '+(result.contexts.execution.static_only?'静态':'动态')+'；箭头表示张量依赖，节点标注输出 Shape / Dtype（中间张量推导值以 * 标记）。';box.append(note);
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
 const note=document.createElement('p');note.className='muted';note.textContent='本次运行全部 Shape 的计时。执行总计包含能力检查、准备和全部 Shape；生成耗时单独记录。下列指标不能逐项相加。';host.append(note);
 const tableTime=document.createElement('table'),head=document.createElement('tr');cell(head,'阶段','th');cell(head,'耗时','th');tableTime.append(head);
 for(const [key,value] of Object.entries(telemetry.stages||{})){const row=document.createElement('tr');cell(row,stageName(key));cell(row,duration(value));tableTime.append(row);}
 if(tableTime.children.length>1)host.append(tableTime);else{const empty=document.createElement('p');empty.textContent='未采集本次运行计时';host.append(empty);}

}

function renderStatisticsTables(result){
 const t=document.createElement('table'),head=document.createElement('tr');for(const label of ['后端','指标','样本','均值','最大'])cell(head,label,'th');t.append(head);
 for(const r of result.precise_metrics||[]){const row=document.createElement('tr');for(const value of [r.backend,r.stage==='compile'?'编译耗时':'Kernel 耗时',r.samples,duration(r.mean_ns),duration(r.max_ns)])cell(row,value);t.append(row);}
 $('performance-table').replaceChildren(t);$('metric-empty').hidden=!!result.precise_metrics?.length;
}

function stageName(stage){return ({generation:'生成用例',capability:'后端能力检查',prepare:'准备执行器（不代表编译完成）',input_binding:'输入绑定',first_call_combined:'首次调用（惰性编译 / 执行 / 同步）',correctness:'精度校验',execution_total:'执行总计',compile:'独立编译',kernel_mean:'Kernel 平均耗时',performance:'性能测量'}[stage]||stage||'Unkown');}
function renderFailureDetails(result){
 const view=result.diagnostic_view||{};
 function section(id,title,content){const box=$(id);box.replaceChildren();const heading=document.createElement('h3'),pre=document.createElement('pre');heading.textContent=title;pre.textContent=content;box.append(heading,pre);return box;}
 section('ascir-detail','ASCIR 表达的图结构',view.shape_log||'未保存当前 Shape 的 ASCIR / 图结构文本');
 // Full saved run log already contains its extracted shared lines; do not repeat them.
 const log=view.run_log||view.shared_log||view.shared_lines?.join('\n')||view.summary||'未保存报错日志';
 const box=section('failure-details','报错日志摘要',log);
 if(view.shared_explanation){const note=document.createElement('p');note.className='muted';note.textContent=view.shared_explanation;box.append(note);}
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
