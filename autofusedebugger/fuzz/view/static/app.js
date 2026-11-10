'use strict';
const $ = id => document.getElementById(id);
const form = $('filters');
function beijing(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}).format(date);
}
const caseMetadata = new Map();
const labels = {ok:'通过',fail:'精度失败',error:'异常',unsupported:'不支持'};
const initial = new URLSearchParams(location.search);
for (const name of ['seed','status']) if(initial.has(name)) form.elements[name].value = initial.get(name);
let applied = parameters();
let cursors = [0], page = 0, next = null, generation = 0, detailGeneration = 0;
function parameters() {
  const p = new URLSearchParams();
  const seed = form.elements.seed.value.trim();
  const status = form.elements.status.value;
  if(seed) p.set('seed',seed);
  if(status) p.set('status',status);
  return p;
}
async function api(path) {
  const response = await fetch(path, {cache:'no-store',credentials:'omit'});
  const value = await response.json();
  if(!response.ok) throw new Error(value.error || '读取失败');
  return value;
}
function cell(row, value) {
  const td = document.createElement('td'); td.textContent = value; row.append(td); return td;
}
async function details(identity) {
  const request = ++detailGeneration;
  $('detail').hidden = false; $('detail-message').textContent = '加载中…';
  $('detail-content').textContent = ''; $('command').value = ''; $('copy').disabled = true;
  try {
    const result = await api('api/attempts/' + encodeURIComponent(identity));
    if(request !== detailGeneration) return;
    $('command').value = result.replay_command;
    const meta=result.metadata||caseMetadata.get(identity);
    $('case-meta').textContent=meta ? '算子：'+meta.op_types.join(' · ')+' ｜ 输入：'+JSON.stringify(meta.input_shapes)+' ｜ 输出：'+JSON.stringify(meta.output_shapes)+' ｜ 输入dtype：'+JSON.stringify(meta.input_dtypes)+' ｜ 输出dtype：'+JSON.stringify(meta.output_dtypes)+' ｜ Shape样本：'+JSON.stringify(meta.shape_variants) : '';
    renderProvenance(result);
    result.created_at = beijing(result.created_at);
    $('detail-content').textContent = JSON.stringify(result,null,2);
    const table = document.createElement('table');
    const header = document.createElement('tr');
    for(const name of ['Shape','正确性','性能','首次调用含编译(ms)','测量均值(ms)','错误阶段','错误摘要']) cell(header,name);
    table.append(header);
    for(const shape of result.shape_results) {
      const row = document.createElement('tr');
      for(const value of [shape.shape_index,shape.correctness,shape.performance,shape.first_call_combined_ns == null ? '—' : (shape.first_call_combined_ns/1e6).toFixed(3),shape.mean_wall_ns == null ? '—' : (shape.mean_wall_ns/1e6).toFixed(3),shape.error_stage || '—',shape.error_summary || '—']) cell(row,value);
      table.append(row);
    }
    $('shape-detail').replaceChildren(table);
    $('detail-message').textContent = ''; $('copy').disabled = false;
    document.querySelectorAll('.page').forEach(el=>el.hidden=true);
    $('page-title').textContent='运行详情';
    $('detail').scrollIntoView({behavior:'smooth',block:'start'});
  } catch(error) {if(request === detailGeneration) $('detail-message').textContent = error.message;}
}
const extraColumns=[['cann','CANN'],['torch','Torch'],['torch_npu','Torch NPU'],['autofuse_commit','Autofuse commit'],['autofuse_binary','Autofuse binary'],['triton','Triton'],['python','Python'],['package_version','Debugger'],['machine','架构'],['platform','系统'],['user_id','用户 ID'],['source_id','来源 ID'],['batch_id','批次']];
const preferenceKey='af-fuzz.record-columns.v1';
let columnPreferences={order:[],hidden:extraColumns.map(([key])=>key)}, currentColumns=[];
try{const saved=JSON.parse(localStorage.getItem(preferenceKey));if(saved && Array.isArray(saved.order)&&saved.order.every(x=>typeof x==='string')&&Array.isArray(saved.hidden)&&saved.hidden.every(x=>typeof x==='string'))columnPreferences=saved;}catch{}
function saveColumns(){try{localStorage.setItem(preferenceKey,JSON.stringify(columnPreferences));}catch{}applyColumns();renderColumnPicker();}
function configureColumns(definitions){
 currentColumns=definitions;
 for(const [key] of definitions)if(!columnPreferences.order.includes(key))columnPreferences.order.push(key);
 applyColumns();renderColumnPicker();
}
function orderedColumns(){return columnPreferences.order.map(key=>currentColumns.find(c=>c[0]===key)).filter(Boolean);}
function applyColumns(){
 const visible=orderedColumns().filter(([key])=>!columnPreferences.hidden.includes(key));
 if(!visible.length)return;
 const table=$('rows').closest('table');table.querySelector('colgroup')?.remove();
 const cols=document.createElement('colgroup'),header=document.createElement('tr');
 for(const [key,label,width] of visible){const col=document.createElement('col');col.style.width=width+'px';cols.append(col);const th=document.createElement('th');th.textContent=label;th.dataset.column=key;header.append(th);}
 table.prepend(cols);table.style.minWidth=visible.reduce((sum,c)=>sum+c[2],0)+'px';$('case-columns').replaceChildren(header);
 for(const row of $('rows').children){const cells=new Map([...row.children].map(td=>[td.dataset.column,td]));for(const [key] of orderedColumns()){const td=cells.get(key);if(td){td.hidden=columnPreferences.hidden.includes(key);row.append(td);}}}
}
function renderColumnPicker(){
 const options=$('column-options');options.replaceChildren();const columns=orderedColumns();
 columns.forEach(([key,label],index)=>{
   const row=document.createElement('div');row.className='column-option';const control=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.checked=!columnPreferences.hidden.includes(key);box.dataset.column=key;
   box.disabled=box.checked&&columns.filter(([k])=>!columnPreferences.hidden.includes(k)).length===1;
   box.addEventListener('change',()=>{columnPreferences.hidden=columnPreferences.hidden.filter(k=>k!==key);if(!box.checked)columnPreferences.hidden.push(key);saveColumns();});control.append(box,document.createTextNode(label));row.append(control);
   for(const [offset,text] of [[-1,'上移'],[1,'下移']]){const button=document.createElement('button');button.type='button';button.className='secondary';button.textContent=offset<0?'↑':'↓';button.setAttribute('aria-label',label+text);button.disabled=index+offset<0||index+offset>=columns.length;button.addEventListener('click',()=>{const other=columns[index+offset][0],a=columnPreferences.order.indexOf(key),b=columnPreferences.order.indexOf(other);[columnPreferences.order[a],columnPreferences.order[b]]=[columnPreferences.order[b],columnPreferences.order[a]];saveColumns();});row.append(button);}options.append(row);
 });
}
$('reset-columns').addEventListener('click',()=>{columnPreferences={order:currentColumns.map(c=>c[0]),hidden:extraColumns.map(c=>c[0])};saveColumns();});
async function load() {

  const request = ++generation;
  $('message').textContent = '加载中…'; $('next').disabled = true; $('previous').disabled = true;
  const p = applied, list = new URLSearchParams(p);
  list.set('limit','50'); list.set('after',String(cursors[page]));
  try {
    const [summary,result] = await Promise.all([api('api/summary?' + p),api('api/cases?' + list)]);
    if(request !== generation) return;
    $('refresh-time').textContent = '更新于 ' + beijing(new Date());
    $('total').textContent = summary.total.toLocaleString();
    $('ok').textContent = (summary.by_status.ok || 0).toLocaleString();
    $('fail').textContent = (summary.by_status.fail || 0).toLocaleString();
    $('errors').textContent = ((summary.by_status.error || 0)+(summary.by_status.unsupported || 0)).toLocaleString();
    $('rows').replaceChildren();
    const present=new Set(result.items.flatMap(item=>Object.keys(item.backends)));
    const order=['cpu','npu-eager','triton-npu','af-legacy','af-mlir'];
    const backends=[...order.filter(b=>present.has(b)),...Array.from(present).filter(b=>!order.includes(b)).sort()];
    const table=$('rows').closest('table');
    table.querySelector('colgroup')?.remove();
    const columns=document.createElement('colgroup');
    for(const width of [220,260,...backends.map(()=>190),180]){const col=document.createElement('col');col.style.width=width+'px';columns.append(col);}
    table.prepend(columns);table.style.minWidth=(660+190*backends.length)+'px';
    const header=document.createElement('tr');
    for(const label of ['Seed / 测试配置','设备',...backends,'时间']){const th=document.createElement('th');th.textContent=label;header.append(th);}
    $('case-columns').replaceChildren(header);
    for(const item of result.items) {
      const tr=document.createElement('tr');
      const seedCell=cell(tr,'');const seed=document.createElement('strong');seed.className='case-seed';seed.textContent=item.seed_short;seedCell.append(seed);
      const meta=document.createElement('small');meta.className='case-ops';meta.textContent=(item.metadata.op_types||[]).join(' · ')+' ｜ '+(item.metadata.n_nodes??'—')+' 节点';seedCell.append(meta);
      const mode=document.createElement('span');mode.className='config-label';mode.textContent=(item.execution?.static_only?'静态':'动态')+' · '+(item.evaluation?.performance?'正确性 + 性能':'正确性');seedCell.append(mode);
      const device=cell(tr,'');const name=document.createElement('strong');name.textContent=item.device_name||'设备未记录';device.append(name);
      for(const backend of backends){
        const td=cell(tr,'');const attempts=item.backends[backend]||[];
        if(!attempts.length){const empty=document.createElement('span');empty.className='not-run';empty.textContent='未运行';td.append(empty);continue;}
        const latest=attempts[0];const button=document.createElement('button');button.className='backend-result secondary';
        const state=document.createElement('span');state.className='state '+(Object.hasOwn(labels,latest.status)?latest.status:'');state.textContent=labels[latest.status]||latest.status;
        const time=document.createElement('span');time.className='result-duration';time.textContent=duration(latest.total_ns);
        const scope=document.createElement('span');scope.className='result-scope';scope.textContent=latest.duration_scope==='partial_recorded_only'?'已记录阶段耗时':'生成 + 执行总耗时';
        button.append(state,time,scope);button.setAttribute('aria-label',backend+' '+state.textContent+'，查看详情');
        button.addEventListener('click',()=>details(latest.id));td.append(button);
        const target=document.createElement('small');target.textContent=latest.device||'设备序号未采集';td.append(target);
        if(attempts.length>1){const history=document.createElement('details');history.className='history';const title=document.createElement('summary');title.textContent=attempts.length+' 次运行';history.append(title);
          for(const attempt of attempts){const link=document.createElement('button');link.className='history-result secondary';link.textContent=beijing(attempt.created_at)+' '+labels[attempt.status];link.addEventListener('click',()=>details(attempt.id));history.append(link);}td.append(history);}
        caseMetadata.set(latest.id,item.metadata);
      }
      const date=cell(tr,'');date.className='case-date';date.textContent=beijing(item.created_at);
      const keys=['seed','device',...backends.map(b=>'backend:'+b),'time'];
      for(const [key,label] of extraColumns){cell(tr,item.versions[key]??item[key]??'未采集');keys.push(key);}
      [...tr.children].forEach((td,i)=>td.dataset.column=keys[i]);$('rows').append(tr);
    }
    const definitions=[['seed','Seed / 测试配置',220],['device','设备',180],...backends.map(b=>['backend:'+b,b,190]),['time','时间',180],...extraColumns.map(([key,label])=>[key,label,210])];
    configureColumns(definitions);
    next = result.next_cursor;
    $('next').disabled = next === null; $('previous').disabled = page === 0;
    $('page-label').textContent = `第 ${page+1} 页 · 本页 ${result.items.length} 条 · 每页最多 50 条`;
    $('message').textContent = result.items.length ? '' : '没有符合条件的结果。';
  } catch(error) {if(request === generation) {$('message').textContent = error.message; $('rows').replaceChildren();
    for(const id of ['total','ok','fail','errors']) $(id).textContent = '—';
    $('page-label').textContent = '读取失败';}}
}
form.addEventListener('submit',event=>{event.preventDefault();applied=parameters();history.replaceState(null,'',location.pathname+(applied.size?'?'+applied:'')+location.hash);page=0;cursors=[0];detailGeneration++;$('detail').hidden=true;load();loadCharts();});
$('reset-filters').addEventListener('click',()=>{form.reset();form.requestSubmit();});
$('next').addEventListener('click',()=>{if(next!==null){cursors[++page]=next;load();}});
$('previous').addEventListener('click',()=>{if(page>0){page--;load();}});
$('copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('command').value);$('detail-message').textContent='已复制';}catch{ $('command').select();$('detail-message').textContent='请复制已选中的命令';}});
load();

setInterval(() => { if (!document.hidden) { load(); loadCharts(); } }, 30000);

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

let chartGeneration=0;
async function loadCharts(){
  const current=++chartGeneration;
  try{
    const result=await api('api/statistics?'+applied);
    if(current!==chartGeneration)return;
    updateStatistics(result);
    renderStatisticsTables(result);
    $('chart-message').textContent='';
  }catch(error){if(current===chartGeneration)$('chart-message').textContent=error.message;}
}
function navigate(){
  const fragment=location.hash.slice(1)||'overview';
  const attempt=new URLSearchParams(fragment).get('attempt');
  if(attempt){details(attempt);return;}
  const name=['overview','records','performance','backends'].includes(fragment)?fragment:'overview';
  for(const key of ['overview','records','performance','backends'])$(key+'-page').hidden=key!==name;
  $('detail').hidden=true;$('comparison').hidden=true;
  $('page-title').textContent={overview:'总览',records:'测试记录',performance:'性能与耗时',backends:'后端'}[name];
  document.querySelectorAll('nav a').forEach(el=>{el.classList.toggle('active',el.dataset.page===name);if(el.dataset.page===name)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
}
window.addEventListener('hashchange',navigate);
$('close-detail').addEventListener('click',()=>{location.hash='records';navigate();});
loadCharts();navigate();

function duration(ns){return ns==null?'未采集':(ns/1e6).toFixed(3)+' ms';}
function versionText(v){return ['cann','torch','torch_npu','autofuse_commit'].map(k=>({cann:'CANN',torch:'Torch',torch_npu:'Torch NPU',autofuse_commit:'Autofuse'}[k])+': '+(v[k]||(k==='autofuse_commit'&&v.autofuse_binary?'binary '+v.autofuse_binary.slice(0,12):'未采集'))).join(' ｜ ');}
function renderProvenance(result){
  const env=result.contexts.environment;
  $('provenance').textContent=(result.provenance?.device_name||env.device_name||'设备未记录')+' ｜ 用户 '+(result.provenance?.user_id||env.user_id||'未采集')+' ｜ '+versionText(env);
  const names={generation:'生成用例',capability:'能力检查',prepare:'后端准备（非编译完成）',input_binding:'输入绑定',first_call_combined:'首次调用（编译+执行）',correctness:'正确性校验',warmup:'预热',measurement:'性能测量',execution_total:'执行总计'};
  const telemetry=result.contexts.execution.telemetry?JSON.parse(result.contexts.execution.telemetry):{stages:{prepare:result.prepare_ns},shapes:Object.fromEntries(result.shape_results.map(s=>[s.shape_index,{first_call_combined:s.first_call_combined_ns}]))};
  const table=document.createElement('table');table.className='metric-table';const head=document.createElement('tr');for(const title of ['范围',...Object.values(names)])cell(head,title);table.append(head);
  for(const [scope,metrics] of [['整体',telemetry.stages||{}],...Object.entries(telemetry.shapes||{}).map(([k,v])=>['Shape '+k,v])]){const row=document.createElement('tr');cell(row,scope);for(const key of Object.keys(names))cell(row,duration(metrics[key]));table.append(row);}
  $('timing-detail').replaceChildren(table);
}

function renderStatisticsTables(result){
 function table(target,headers,rows){const t=document.createElement('table');t.className='metric-table';const h=document.createElement('tr');headers.forEach(v=>cell(h,v));t.append(h);for(const values of rows){const r=document.createElement('tr');values.forEach(v=>cell(r,v));t.append(r);}$(target).replaceChildren(t);}
 table('performance-table',['后端','Shape模式','正确性','性能状态','口径','Shape数','均值','最小','最大'],result.performance.map(r=>[r.backend,r.mode,r.correctness,r.performance,r.metric||'未测量',r.samples,duration(r.mean_ns),duration(r.min_ns),duration(r.max_ns)]));
 table('stage-table',['后端','阶段','样本数','累计','均值','最大'],result.stages.map(r=>[r.backend,r.stage,r.samples,duration(r.total_ns),duration(r.mean_ns),duration(r.max_ns)]));
}
