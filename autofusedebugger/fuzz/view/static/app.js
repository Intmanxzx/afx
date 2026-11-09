'use strict';
const $ = id => document.getElementById(id);
const form = $('filters');
const labels = {ok:'通过',fail:'精度失败',error:'异常',unsupported:'不支持'};
let applied = new URLSearchParams();
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
    $('detail-content').textContent = JSON.stringify(result,null,2);
    $('detail-message').textContent = ''; $('copy').disabled = false;
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
    $('total').textContent = summary.total.toLocaleString();
    $('ok').textContent = (summary.by_status.ok || 0).toLocaleString();
    $('fail').textContent = (summary.by_status.fail || 0).toLocaleString();
    $('errors').textContent = ((summary.by_status.error || 0)+(summary.by_status.unsupported || 0)).toLocaleString();
    $('rows').replaceChildren();
    for(const item of result.items) {
      const tr = document.createElement('tr'); cell(tr,item.seed); cell(tr,item.backend);
      const state = document.createElement('span'); state.className = 'state ' + (Object.hasOwn(labels,item.status)?item.status:'');
      state.textContent = labels[item.status] || item.status; cell(tr,'').append(state);
      cell(tr,item.created_at); const button = document.createElement('button'); button.textContent = '查看';
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
form.addEventListener('submit',event=>{event.preventDefault();applied=parameters();page=0;cursors=[0];detailGeneration++;$('detail').hidden=true;load();});
$('reset-filters').addEventListener('click',()=>{form.reset();form.requestSubmit();});
$('next').addEventListener('click',()=>{if(next!==null){cursors[++page]=next;load();}});
$('previous').addEventListener('click',()=>{if(page>0){page--;load();}});
$('copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('command').value);$('detail-message').textContent='已复制';}catch{ $('command').select();$('detail-message').textContent='请复制已选中的命令';}});
load();
