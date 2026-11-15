/* Collaborative issue classes. Loaded by the central service; offline viewer stays read-only. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  if (!$('issue-list')) return;
  const labels = {function:'功能问题', accuracy:'精度问题', performance:'性能问题'};
  let dimension='function', groups=[], clients=[];
  const api = async (path, method='GET', body) => {
    if (window.afSession) return window.afSession.request(path, {method, ...(body === undefined ? {} : {body:JSON.stringify(body)})});
    if (method !== 'GET') throw new Error('离线查看器不支持领取或复测');
    const response = await fetch(path, {cache:'no-store'}); const value = await response.json();
    if (!response.ok) throw new Error(value.error || '读取失败'); return value;
  };
  const text = (value, fallback='—') => value === null || value === undefined || value === '' ? fallback : String(value);
  function status(g) {
    if (g.status === 'claimed') return `${text(g.owner_name)}解决中`;
    if (g.status === 'retesting') return `${text(g.owner_name)}复测中`;
    return '待领取';
  }
  function card(g) {
    const article=document.createElement('article'); article.className='issue-card';
    const head=document.createElement('div'); head.className='issue-card-head';
    const title=document.createElement('h3'); title.textContent=g.title; const count=document.createElement('strong'); count.textContent=`${g.count} 个用例`;
    head.append(title,count); article.append(head);
    const meta=document.createElement('p'); meta.className='issue-meta'; meta.textContent=`${labels[g.dimension]} · ${status(g)}`; article.append(meta);
    const samples=document.createElement('details'); const summary=document.createElement('summary'); summary.textContent='查看示例用例'; samples.append(summary);
    for(const item of g.examples||[]){const line=document.createElement('p');line.textContent=`${text(item.seed)} · S${item.shape_index} · ${text(item.message)}`;samples.append(line);} article.append(samples);
    const actions=document.createElement('div'); actions.className='issue-actions';
    if (!window.afSession?.authenticated?.()) { const note=document.createElement('span'); note.className='muted'; note.textContent='登录后可领取'; actions.append(note); }
    else if (g.status==='open') { const button=document.createElement('button');button.type='button';button.textContent='领取问题';button.onclick=async()=>{button.disabled=true;try{await api('api/issues/claim','POST',{issue_key:g.issue_key});await load();}catch(e){$('issues-message').textContent=e.message;button.disabled=false;}};actions.append(button); }
    else if (g.owner_id === window.afSession.userId?.()) { const release=document.createElement('button');release.type='button';release.className='secondary';release.textContent='释放';release.onclick=async()=>{try{await api('api/issues/release','POST',{issue_key:g.issue_key});await load();}catch(e){$('issues-message').textContent=e.message;}};actions.append(release); }
    if (g.status==='claimed' && g.owner_id === window.afSession.userId?.()) {
      const retest=document.createElement('button');retest.type='button';retest.textContent='一键复测全部用例';retest.onclick=()=>openRetest(g,retest);actions.append(retest);
    }
    article.append(actions); return article;
  }
  async function openRetest(g, button) {
    try {
      if (!clients.length) clients=(await api('api/retest/clients')).items;
      if (!clients.length) throw new Error('没有在线的复测客户端');
      const online=clients.find(c=>c.online && c.backends.includes('af-mlir')) || clients.find(c=>c.online);
      if (!online) throw new Error('没有支持 af-mlir 的在线复测客户端');
      const repetitions=Number(prompt('复测次数（1-10）','1')||1); if(!Number.isInteger(repetitions)||repetitions<1||repetitions>10)throw new Error('复测次数必须为 1-10');
      button.disabled=true; await api('api/issues/retest','POST',{issue_key:g.issue_key,client_id:online.id,repetitions,request_key:crypto.randomUUID()}); await load();
    } catch(e) { $('issues-message').textContent=e.message; button.disabled=false; }
  }
  async function load() {
    $('issues-message').textContent='加载中…'; $('issue-list').replaceChildren();
    try { const value=await api('api/issues?dimension='+encodeURIComponent(dimension)); groups=value.items||[]; $('issues-message').textContent=groups.length?`共 ${groups.length} 类，按用例数降序排列`:'当前没有待处理的问题'; $('issue-list').append(...groups.map(card)); }
    catch(e) { groups=[]; $('issues-message').textContent=e.message; }
  }
  document.querySelectorAll('#issue-tabs button').forEach(button=>button.addEventListener('click',()=>{dimension=button.dataset.dimension;document.querySelectorAll('#issue-tabs button').forEach(b=>b.classList.toggle('active',b===button));load();}));
  document.addEventListener('af-session-changed',load); window.addEventListener('hashchange',()=>{if(location.hash==='#issues')load();});
  window.addEventListener('load',()=>{if(location.hash==='#issues')load();});
})();
