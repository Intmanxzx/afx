'use strict';
const $ = id => document.getElementById(id);
const form = $('filters');
function beijing(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}).format(date);
}
const caseMetadata = new Map();
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
    renderFailureDetails(result);
    result.created_at = beijing(result.created_at);
    $('detail-content').textContent = JSON.stringify(result,null,2);
    const table = document.createElement('table');
    const header = document.createElement('tr');
    for(const name of ['Shape','正确性','性能','首次调用含编译(ms)','测量均值(ms)','错误阶段','错误摘要']) cell(header,name);
    table.append(header);
    for(const shape of result.shape_results) {
      const row = document.createElement('tr');
      for(const value of [shape.shape_index,({pass:'Pass',fail:'Failed',error:'Failed'}[shape.correctness]||'Unkown'),(shape.performance==='error'?'Failed':shape.performance==='measured'?'已记录（见口径）':'Unkown'),shape.first_call_combined_ns == null ? '—' : (shape.first_call_combined_ns/1e6).toFixed(3),shape.mean_wall_ns == null ? '—' : (shape.mean_wall_ns/1e6).toFixed(3),shape.error_stage || '—',shape.error_summary || '—']) cell(row,value);
      table.append(row);
    }
    $('shape-detail').replaceChildren(table);
    $('detail-message').textContent = ''; $('copy').disabled = false;
    document.querySelectorAll('.page').forEach(el=>el.hidden=true);
    $('page-title').textContent='运行详情';
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
 const definitions=[{key:'seed',label:'Seed / 用例',width:220,defaultVisible:true},{key:'user_name',label:'用户',width:140,defaultVisible:true}];
 for(const b of backends){if(b!=='cpu')definitions.push({key:b+':accuracy',label:b+' 精度',width:130,defaultVisible:true},{key:b+':kernel',label:b+' 性能',width:150,defaultVisible:true});definitions.push({key:b+':total',label:b+' 总耗时',width:150,defaultVisible:true});}
 definitions.push({key:'time',label:'时间',width:180,defaultVisible:true},...extraColumns.map(([key,label])=>({key,label,width:210,defaultVisible:false})));
 const fragment=document.createDocumentFragment();
 for(const item of items){const tr=document.createElement('tr');
 for(const c of definitions){const td=cell(tr,'');td.dataset.column=c.key;let populated=false;
 if(c.key==='seed'){const strong=document.createElement('strong');strong.className='case-seed';strong.textContent=item.seed_short;const meta=document.createElement('small');meta.textContent=(item.metadata.op_types||[]).join(' · ');td.append(strong,meta);populated=true;}
 else if(c.key==='user_name'){if(item.user_name&&item.user_name!=='设备未记录'){const button=document.createElement('button');button.className='text-link user-filter';button.textContent=item.user_name;button.addEventListener('click',()=>{form.elements.q.value=item.user_name;form.requestSubmit();});td.append(button);populated=true;}}
 else if(c.key==='time'){td.textContent=beijing(item.created_at);populated=true;}
 else if(c.key.includes(':')){const [b,kind]=c.key.split(':'),attempts=item.backends[b]||[],latest=attempts[0];
 if(latest){let text=null,state='';if(kind==='accuracy'){state=latest.correctness_status==='pass'?'pass':['fail','error'].includes(latest.correctness_status)?'failed':'';text=state==='pass'?'Pass':state==='failed'?'Failed':null;}
 if(kind==='kernel'&&latest.kernel_ns.length)text=latest.kernel_ns.map(m=>'S'+m.shape_index+' '+(m.duration_ns/1e6).toFixed(3)+' ms').join(' · ');
 if(kind==='total')text=seconds(latest.total_ns);
 if(text!==null){const button=document.createElement('button');button.className='text-link outcome-button '+state;button.textContent=text;button.addEventListener('click',()=>details(latest.id));td.append(button);populated=true;}
 if(attempts.length>1&&kind==='total'){const history=document.createElement('details'),summary=document.createElement('summary');history.className='history';summary.textContent=attempts.length+' 次运行';history.append(summary);for(const attempt of attempts){const button=document.createElement('button');button.className='text-link';button.textContent=beijing(attempt.created_at)+' · '+(seconds(attempt.total_ns)||'Unkown');button.addEventListener('click',()=>details(attempt.id));history.append(button);}td.append(history);}
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

function duration(ns){return ns==null?'Unkown':(ns/1e6).toFixed(3)+' ms';}
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
 const t=document.createElement('table'),head=document.createElement('tr');for(const label of ['后端','指标','样本','均值','最大'])cell(head,label);t.append(head);
 for(const r of result.precise_metrics||[]){const row=document.createElement('tr');for(const value of [r.backend,r.stage==='compile'?'编译耗时':'Kernel 耗时',r.samples,duration(r.mean_ns),duration(r.max_ns)])cell(row,value);t.append(row);}
 $('performance-table').replaceChildren(t);$('metric-empty').hidden=!!result.precise_metrics?.length;
}

function renderFailureDetails(result){
 let box=$('failure-details');if(!box){box=document.createElement('section');box.id='failure-details';$('shape-detail').after(box);}box.replaceChildren();
 const raw=result.contexts.execution.diagnostics;if(!raw)return;
 const data=JSON.parse(raw);
 const title=document.createElement('h3');title.textContent='核心日志与精度误差';box.append(title);
 if(data.error){const log=document.createElement('pre');log.textContent=data.error;box.append(log);}
 for(const [index,metrics] of Object.entries(data.shapes||{})){
  const heading=document.createElement('h4');heading.textContent='Shape '+index+' · 判定标准：'+(metrics.basis||'cpu');box.append(heading);
  const table=document.createElement('table');for(const [key,label] of [['cpu_pass','CPU一致'],['secondary_used','已运行 eager-npu'],['secondary_pass','eager-npu一致'],['max_abs','最大绝对误差'],['mean_abs','平均绝对误差'],['max_rel','最大相对误差'],['mismatches','不一致元素'],['elements','总元素'],['nonfinite','非有限元素']]){if(metrics[key]===undefined)continue;const row=document.createElement('tr');cell(row,label);cell(row,String(metrics[key]));table.append(row);}box.append(table);
  if(metrics.error){const log=document.createElement('pre');log.textContent=metrics.error;box.append(log);}
 }
}
