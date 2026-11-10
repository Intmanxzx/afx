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
    const meta=caseMetadata.get(identity);
    $('case-meta').textContent=meta ? '算子：'+meta.op_types.join(' · ')+' ｜ 输入：'+JSON.stringify(meta.input_shapes)+' ｜ 输出：'+JSON.stringify(meta.output_shapes) : '';
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
async function load() {

  const request = ++generation;
  $('message').textContent = '加载中…'; $('next').disabled = true; $('previous').disabled = true;
  const p = applied, list = new URLSearchParams(p);
  list.set('limit','50'); list.set('after',String(cursors[page]));
  try {
    const [summary,result] = await Promise.all([api('api/summary?' + p),api('api/results?' + list)]);
    if(request !== generation) return;
    $('refresh-time').textContent = '更新于 ' + beijing(new Date());
    $('total').textContent = summary.total.toLocaleString();
    $('ok').textContent = (summary.by_status.ok || 0).toLocaleString();
    $('fail').textContent = (summary.by_status.fail || 0).toLocaleString();
    $('errors').textContent = ((summary.by_status.error || 0)+(summary.by_status.unsupported || 0)).toLocaleString();
    $('rows').replaceChildren();
    for(const item of result.items) {
      const tr = document.createElement('tr'); cell(tr,item.seed_short || item.seed); cell(tr,(item.backend_display || item.backend) + (item.display_name ? " · " + item.display_name : ""));
      const state = document.createElement('span'); state.className = 'state ' + (Object.hasOwn(labels,item.status)?item.status:'');
      state.textContent = labels[item.status] || item.status; cell(tr,'').append(state);
      cell(tr,beijing(item.created_at)); const button = document.createElement('button'); button.textContent = '查看';
      button.addEventListener('click',()=>details(item.id)); cell(tr,'').append(button); $('rows').append(tr);
    }
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
    const result=await api('api/report?'+applied);
    if(current!==chartGeneration)return;
    for(const item of result.cases)caseMetadata.set(item.attempt_id,item);
    updateCharts(result.cases);
    $('chart-message').textContent='';
  }catch(error){if(current===chartGeneration)$('chart-message').textContent=error.message;}
}
function navigate(){
  const fragment=location.hash.slice(1)||'overview';
  const attempt=new URLSearchParams(fragment).get('attempt');
  if(attempt){details(attempt);return;}
  const name=['overview','records','backends'].includes(fragment)?fragment:'overview';
  for(const key of ['overview','records','backends'])$(key+'-page').hidden=key!==name;
  $('detail').hidden=true;$('comparison').hidden=true;
  $('page-title').textContent={overview:'总览',records:'实验记录',backends:'后端'}[name];
  document.querySelectorAll('nav a').forEach(el=>{el.classList.toggle('active',el.dataset.page===name);if(el.dataset.page===name)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
}
window.addEventListener('hashchange',navigate);
$('close-detail').addEventListener('click',()=>{location.hash='records';navigate();});
loadCharts();navigate();
