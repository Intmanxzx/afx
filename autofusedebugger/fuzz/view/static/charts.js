'use strict';
let CASES=[];
const escapeChart = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const OP_CATEGORIES={Elementwise:['add','relu','neg','mul'],Reduce:['sum'],Transpose:['transpose']};
const _rootStyle = getComputedStyle(document.documentElement);
const STATUS_COLORS = {
  unknown: '#9aa6a0',
  pass: _rootStyle.getPropertyValue('--viz-pass').trim(),
  fail: _rootStyle.getPropertyValue('--viz-fail').trim(),
  error: _rootStyle.getPropertyValue('--viz-gen-error').trim(),
  gen_error: _rootStyle.getPropertyValue('--viz-gen-error').trim(),
};
const VIZ = {
  surfaceGap: _rootStyle.getPropertyValue('--viz-surface-gap').trim(),
  grid: _rootStyle.getPropertyValue('--viz-grid').trim(),
  axis: _rootStyle.getPropertyValue('--viz-axis').trim(),
  textPrimary: _rootStyle.getPropertyValue('--viz-text-primary').trim(),
  textSecondary: _rootStyle.getPropertyValue('--viz-text-secondary').trim(),
  textMuted: _rootStyle.getPropertyValue('--viz-text-muted').trim(),
};
const FONT = 'system-ui, -apple-system, sans-serif';
const STATUS_KEYS = ['pass', 'fail', 'gen_error', 'unknown'];
const STATUS_LABELS = {pass:'Pass',fail:'精度Failed',gen_error:'功能Failed',unknown:'Unknown'};

// ---- 悬停浮层（替代原生 <title>）：跟随鼠标，内容用 textContent 写入，
// 系列名/分类名来自用户数据，不能走 innerHTML 拼接。 ----
const _tooltip = document.querySelector('#vizTooltip');
function showTooltip(evt, title, entries) {
  while (_tooltip.firstChild) _tooltip.removeChild(_tooltip.firstChild);
  const titleEl = document.createElement('div');
  titleEl.textContent = title;
  titleEl.style.cssText = 'font-weight:600;margin-bottom:2px';
  _tooltip.appendChild(titleEl);
  for (const [key, val] of entries) {
    const row = document.createElement('div');
    const keyEl = document.createElement('span');
    keyEl.className = 'tt-key';
    keyEl.textContent = key + ' ';
    const valEl = document.createElement('span');
    valEl.className = 'tt-val';
    valEl.textContent = String(val);
    row.appendChild(keyEl);
    row.appendChild(valEl);
    _tooltip.appendChild(row);
  }
  _tooltip.classList.add('show');
  moveTooltip(evt);
}
function moveTooltip(evt) {
  const pad = 14;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  const rect = _tooltip.getBoundingClientRect();
  if (x + rect.width > window.innerWidth - 8) x = evt.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = evt.clientY - rect.height - pad;
  _tooltip.style.left = x + 'px';
  _tooltip.style.top = y + 'px';
}
function hideTooltip() {
  _tooltip.classList.remove('show');
}

// 给 SVG 图形元素挂 tooltip：content() 返回 [标题, [[键, 值], ...]]，
// 悬停/键盘聚焦都显示同一份内容，移开/失焦隐藏。focus 没有鼠标坐标，
// 就把浮层放到元素自身的屏幕位置旁。
function attachTooltip(el, content) {
  const place = (evt) => {
    if (evt.clientX !== undefined) {
      moveTooltip(evt);
    } else {
      const r = el.getBoundingClientRect();
      moveTooltip({ clientX: r.left + r.width / 2, clientY: r.top });
    }
  };
  const bound = (evt) => { showTooltip(evt, ...content()); place(evt); };
  el.addEventListener('pointerenter', bound);
  el.addEventListener('pointermove', moveTooltip);
  el.addEventListener('pointerleave', hideTooltip);
  el.setAttribute('tabindex', '0');
  el.addEventListener('focus', bound);
  el.addEventListener('blur', hideTooltip);
}

// 单个 case 的最大单维度值 / 最大元素数：汇总它所有 input/output tensor 的 shape。
function caseMaxDim(c) {
  let maxDim = 0, maxElem = 0;
  for (const shapes of [c.input_shapes, c.output_shapes]) {
    for (const dims of Object.values(shapes || {})) {
      if (!dims.length) continue;
      maxDim = Math.max(maxDim, ...dims);
      maxElem = Math.max(maxElem, dims.reduce((a, b) => a * b, 1));
    }
  }
  return { maxDim, maxElem };
}

const DIM_BUCKETS = [
  { label: '\u22648', test: v => v <= 8 },
  { label: '\u226464', test: v => v <= 64 },
  { label: '\u2264512', test: v => v <= 512 },
  { label: '\u22644096', test: v => v <= 4096 },
  { label: '\u226416384', test: v => v <= 16384 },
  { label: '\u226465536', test: v => v <= 65536 },
  { label: '\u2264131072', test: v => v <= 131072 },
  { label: '>131072', test: () => true },
];
const ELEM_BUCKETS = [
  { label: '\u22641K', test: v => v <= 1e3 },
  { label: '\u226416K', test: v => v <= 16e3 },
  { label: '\u226464K', test: v => v <= 64e3 },
  { label: '\u2264256K', test: v => v <= 256e3 },
  { label: '\u22641M', test: v => v <= 1e6 },
  { label: '\u226416M', test: v => v <= 16e6 },
  { label: '\u226464M', test: v => v <= 64e6 },
  { label: '>64M', test: () => true },
];

function bucketOf(buckets, value) {
  for (const b of buckets) {
    if (b.test(value)) return b.label;
  }
  return buckets[buckets.length - 1].label;
}

// 按 key 分组统计 pass/fail/gen_error 计数，group(c) 可以返回单个字符串或字符串数组
// （比如一个 case 同时含 reduce+elewise 两类 op，会同时计入两个分类桶）。
function groupCounts(cases, groupFn, order, statusKeys = STATUS_KEYS) {
  const counts = new Map();
  for (const c of cases) {
    let keys = groupFn(c);
    if (!Array.isArray(keys)) keys = [keys];
    for (const k of new Set(keys)) {
      if (!counts.has(k)) counts.set(k, Object.fromEntries(statusKeys.map(s => [s, 0])));
      counts.get(k)[c.status] = (counts.get(k)[c.status] || 0) + 1;
    }
  }
  const keys = order ? order.filter(k => counts.has(k)) : [...counts.keys()];
  return keys.map(k => ({ key: k, ...counts.get(k) }));
}

// ---- 环形图：中心放 pass rate 大数字 + 总数，各段占比靠图例（不标在线上）。
// 段间留 2px 白缝（用 stroke 里的 --viz-surface-gap 画），环宽 24。 ----
function drawDonut(svgId, legendId, segments) {
  const svg = document.querySelector('#' + svgId);
  const W = 380, H = 200, cx = 190, cy = 98, r = 72, ringW = 24;
  const total = segments.reduce((s, seg) => s + seg.value, 0);
  // Legends carry labels only; numbers belong in tooltips.
  const legend = document.querySelector('#' + legendId);
  const legendEls = segments.map(seg => {
    return `<span><i style="background:${seg.color}"></i>${STATUS_LABELS[seg.label] || escapeChart(seg.label)}</span>`;
  });
  legend.innerHTML = legendEls.join('');
  if (total === 0) {
    svg.innerHTML = `<text x="${cx}" y="${cy}" text-anchor="middle" fill="${VIZ.textMuted}" font-family="${FONT}" font-size="13">无数据</text>`;
    return;
  }
  const C = 2 * Math.PI * r;
  let offset = 0;
  const parts = [];
  // 从 12 点方向顺时针：先整体 rotate(-90)，再逐段用 dasharray/dashoffset。
  for (const seg of segments.filter(s => s.value > 0)) {
    const frac = seg.value / total;
    const len = frac * C;
    // 段间 ~2px 白缝：dasharray 的单位就是路径长度（px），直接减 2。
    const dash = Math.max(len - 2, 0.5);
    parts.push(`<circle class="seg" cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${seg.color}" stroke-width="${ringW}" stroke-dasharray="${dash} ${C - dash}" stroke-dashoffset="${-offset}" data-key="${seg.label}"></circle>`);
    offset += len;
  }
  const passRate = (() => {
    const passSeg = segments.find(s => s.label === 'pass');
    return total && passSeg ? (passSeg.value / total * 100).toFixed(1) : '0.0';
  })();
  const passVal = segments.find(s => s.label === 'pass');
  svg.innerHTML = `
    <g transform="rotate(-90 ${cx} ${cy})">${parts.join('')}</g>
    <text x="${cx}" y="${cy - 3}" text-anchor="middle" fill="${VIZ.textPrimary}" font-family="${FONT}" font-size="27" font-weight="650">${passRate}%</text>
    <text x="${cx}" y="${cy + 16}" text-anchor="middle" fill="${VIZ.textMuted}" font-family="${FONT}" font-size="11">pass rate · ${passVal ? passVal.value : 0}/${total}</text>
  `;
  // 每段挂 tooltip（value 从闭包里的 segments 取，聚焦/悬停同款内容）。
  svg.querySelectorAll('.seg').forEach(el => {
    const seg = segments.find(s => s.label === el.dataset.key);
    attachTooltip(el, () => [STATUS_LABELS[seg.label] || seg.label, [['数量', seg.value], ['占比', (total ? seg.value / total * 100 : 0).toFixed(1) + '%']]]);
  });
}

// ---- 竖直堆叠柱状图：细柱（≤24px）、顶端 4px 圆角、段间 2px 白缝、
// 极淡实线网格 + 基线；柱顶标总数，段内数值仅在放得下时显示。 ----
function drawStackedBars(svgId, legendId, rows, statusKeys = STATUS_KEYS) {
  const svg = document.querySelector('#' + svgId);
  const W = 380, H = 260;
  const leftPad = 34, rightPad = 10, topPad = 18, bottomPad = 42;
  const chartH = H - topPad - bottomPad;
  const plotW = W - leftPad - rightPad;
  const legend = document.querySelector('#' + legendId);
  legend.innerHTML = statusKeys.map(s =>
    `<span><i style="background:${STATUS_COLORS[s]}"></i>${STATUS_LABELS[s]}</span>`
  ).join('');
  if (rows.length === 0) {
    svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="${VIZ.textMuted}" font-family="${FONT}" font-size="13">无数据</text>`;
    return;
  }
  const maxTotal = Math.max(...rows.map(r => statusKeys.reduce((n, s) => n + (r[s] || 0), 0)), 1);
  const slot = plotW / rows.length;
  const barW = Math.min(slot * 0.52, 24);
  const gap = slot - barW;

  // y 轴：整洁刻度（nice step：1/2/5×10^k），顶部留 headroom 让柱顶总数不顶格。
  const niceStep = (x) => {
    const pow = Math.pow(10, Math.floor(Math.log10(x)));
    for (const m of [1, 2, 5, 10]) {
      if (m * pow >= x) return m * pow;
    }
    return 10 * pow;
  };
  const yMax = Math.ceil(maxTotal / niceStep(maxTotal / 4)) * niceStep(maxTotal / 4);
  const step = niceStep(yMax / 4);
  const ticks = [];
  for (let t = 0; t <= yMax; t += step) ticks.push(t);
  const gridLines = ticks.map(t => {
    const y = topPad + chartH - (t / yMax) * chartH;
    const isBase = t === 0;
    return `
      <line x1="${leftPad}" y1="${y}" x2="${W - rightPad}" y2="${y}" stroke="${isBase ? VIZ.axis : VIZ.grid}" stroke-width="1"></line>
      <text x="${leftPad - 8}" y="${y + 4}" text-anchor="end" fill="${VIZ.textMuted}" font-family="${FONT}" font-size="11">${t}</text>
    `;
  }).join('');

  const parts = [];
  rows.forEach((r, i) => {
    const x = leftPad + i * (barW + gap) + (slot - barW) / 2;
    const total = statusKeys.reduce((n, s) => n + (r[s] || 0), 0);
    const topY = topPad + chartH - (total / yMax) * chartH;
    const segs = statusKeys
      .map(s => ({ v: r[s], color: STATUS_COLORS[s], label: s }))
      .filter(s => s.v > 0);
    let yBottom = topPad + chartH;
    for (const [j, s] of segs.entries()) {
      const h = (s.v / yMax) * chartH;
      // 段间 2px 白缝：从下往上堆，每段顶部让出 2px（最后一段/顶段除外），
      // 顶段的数据端由 4px 圆角自然收边。高度不够时至少画 1px 色带。
      const isTop = j === segs.length - 1;
      const segH = Math.max(h - (isTop ? 0 : 2), 1);
      const segY = yBottom - segH;
      let shape;
      if (isTop && segH > 4) {
        // path: 顶边两角 4px 圆角。
        shape = `M ${x} ${segY + 4} a 4 4 0 0 1 4 -4 h ${barW - 8} a 4 4 0 0 1 4 4 v ${segH - 4} h ${-barW} Z`;
      } else {
        shape = `M ${x} ${segY} h ${barW} v ${segH} h ${-barW} Z`;
      }
      parts.push(`<path class="seg" d="${shape}" fill="${s.color}" data-key="${escapeChart(r.key)}|${s.label}"></path>`);
      // 段内数值：仅段高放得下一行 12px 字才显示（白字，深浅底都可读）。
      if (segH >= 16 && barW >= 18) {
        parts.push(`<text x="${x + barW / 2}" y="${segY + segH / 2 + 4}" text-anchor="middle" fill="#fff" font-family="${FONT}" font-size="11.5" font-weight="600" pointer-events="none">${s.v}</text>`);
      }
      yBottom = yBottom - h - 2;  // 下一段起点：本段理论顶再留 2px 缝
      if (yBottom < topPad) yBottom = topPad;
    }
    // 柱顶总数（深色加粗）。
    parts.push(`<text x="${x + barW / 2}" y="${topY - 6}" text-anchor="middle" fill="${VIZ.textPrimary}" font-family="${FONT}" font-size="12.5" font-weight="600" pointer-events="none">${total}</text>`);
    // x 轴分类标签（超 9 字符截断）。
    const label = r.key.length > 9 ? r.key.slice(0, 8) + '\u2026' : r.key;
    parts.push(`<text x="${x + barW / 2}" y="${topPad + chartH + 20}" text-anchor="middle" fill="${VIZ.textSecondary}" font-family="${FONT}" font-size="12" pointer-events="none">${escapeChart(label)}</text>`);
  });
  svg.innerHTML = gridLines + parts.join('');
  // 每段挂 tooltip：分类名做标题，pass/fail/gen_error 计数 + 总数。
  svg.querySelectorAll('.seg').forEach(el => {
    const [catKey, statusKey] = el.dataset.key.split('|');
    attachTooltip(el, () => {
      const row = rows.find(r => r.key === catKey);
      const entries = statusKeys.filter(s => row[s] > 0).map(s => [STATUS_LABELS[s], row[s]]);
      entries.push(['总计', statusKeys.reduce((n, s) => n + (row[s] || 0), 0)]);
      return [catKey, entries];
    });
  });
}

function renderStats() {
  drawStackedBars('chartBackend','legendBackend',groupCounts(CASES,c=>c.backend));
  // 1. 总体 pass/fail/gen_error 环形图（中心 pass rate）
  const overall = STATUS_KEYS.map(s => ({
    label: s, value: CASES.filter(c => c.status === s).length, color: STATUS_COLORS[s],
  }));
  drawDonut('chartOverall', 'legendOverall', overall);

  // 2. 按算子分类：一个 case 可能同时属于多个分类（比如同时含 reduce+elewise 的图）
  const categoryNames = Object.keys(OP_CATEGORIES);
  const categoryRows = groupCounts(CASES, c => {
    const ops = c.op_types || [];
    const hit = categoryNames.filter(name => ops.some(op => OP_CATEGORIES[name].includes(op)));
    return hit.length ? hit : ['(未分类)'];
  }, [...categoryNames, '(未分类)']);
  drawStackedBars('chartCategory', 'legendCategory', categoryRows);

  // 3/4. 按 shape 档位：单 tensor 最大维度值 / 最大元素数
  const dimRows = groupCounts(CASES, c => bucketOf(DIM_BUCKETS, caseMaxDim(c).maxDim), DIM_BUCKETS.map(b => b.label));
  drawStackedBars('chartShapeDim', 'legendShapeDim', dimRows);
  const elemRows = groupCounts(CASES, c => bucketOf(ELEM_BUCKETS, caseMaxDim(c).maxElem), ELEM_BUCKETS.map(b => b.label));
  drawStackedBars('chartShapeElem', 'legendShapeElem', elemRows);

  // 5. 按 dtype：汇总一个 case 所有 input/output tensor 出现过的 dtype（去重），
  // 和分类一样允许一个 case 同时计入多种 dtype。
  const dtypeRows = groupCounts(CASES, c => {
    const dtypes = new Set([...Object.values(c.input_dtypes || {}), ...Object.values(c.output_dtypes || {})]);
    return dtypes.size ? [...dtypes] : ['(无)'];
  });
  dtypeRows.sort((a, b) => (b.pass + b.fail + b.gen_error) - (a.pass + a.fail + a.gen_error));
  drawStackedBars('chartDtype', 'legendDtype', dtypeRows);
}

window.updateCharts = cases => { CASES=cases; renderStats(); };

const legendHidden=new Map();let latestStatistics=null;
window.updateStatistics = result => {
 latestStatistics=result;
 const convert=(rows,id)=>{const hidden=legendHidden.get(id)||new Set(),groups=new Map();for(const r of rows){if(!groups.has(r.label))groups.set(r.label,{key:r.label,pass:0,fail:0,gen_error:0,unknown:0});const key=r.status==='ok'?'pass':r.status==='fail'?'fail':r.status==='unknown'?'unknown':'gen_error';if(!hidden.has(key))groups.get(r.label)[key]+=r.count;}return [...groups.values()];};
 const overall=convert(result.overall,'Overall')[0]||{};
 drawDonut('chartOverall','legendOverall',STATUS_KEYS.map(s=>({label:s,value:overall[s]||0,color:STATUS_COLORS[s]})));
 for(const [key,id] of [['backend','Backend'],['category','Category'],['dtype','Dtype'],['dimension','ShapeDim'],['elements','ShapeElem']])drawStackedBars('chart'+id,'legend'+id,convert(result[key],id));
 for(const [key,id] of [['overall','Overall'],['backend','Backend'],['category','Category'],['dtype','Dtype'],['dimension','ShapeDim'],['elements','ShapeElem']]){
  const legend=document.getElementById('legend'+id);legend.replaceChildren();const hidden=legendHidden.get(id)||new Set();const raw=convert(result[key],'__raw');const counts=Object.fromEntries(STATUS_KEYS.map(s=>[s,raw.reduce((n,r)=>n+r[s],0)]));const total=STATUS_KEYS.filter(s=>!hidden.has(s)).reduce((n,s)=>n+counts[s],0);
  for(const status of STATUS_KEYS){if(status==='unknown'&&!counts[status])continue;const button=document.createElement('button');button.type='button';button.dataset.status=status;button.setAttribute('aria-pressed',String(!hidden.has(status)));const swatch=document.createElement('i');swatch.style.background=STATUS_COLORS[status];button.append(swatch,document.createTextNode(STATUS_LABELS[status]));button.addEventListener('click',()=>{if(hidden.has(status))hidden.delete(status);else hidden.add(status);legendHidden.set(id,hidden);updateStatistics(latestStatistics);});legend.append(button);}
 }
};
