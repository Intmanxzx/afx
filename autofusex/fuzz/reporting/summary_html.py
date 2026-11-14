"""Lightweight cross-case summary index for an NNSmith fuzz session.

Deliberately separate from `autofusex.html.render_report_html`: that
renderer is a single-case deep-dive view (graph viz, tensor viewer, fusion
detail) meant to be opened one case at a time. This one is a sortable/
filterable table over many cases -- pass/fail counts, op types, and a link to
the failing case's bug directory for reproduction. A failing case's own deep
report (if generated) is a separate file this index links to, not something
this module renders itself.
"""

from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any


def _json_for_script(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def render_summary_html(
    summary: dict[str, Any],
    title: str = "AF Fuzz Summary",
    op_categories: dict[str, set[str]] | None = None,
) -> str:
    cases = summary.get("cases", [])
    payload = _json_for_script(cases)
    # {category_name: [op_name, ...]} -- plain lists (JSON-serializable), computed
    # client-side against each case's op_types. None/empty means the caller
    # (e.g. a standalone re-render from a saved summary.json with no NNSmith
    # env available) didn't have OP_CATEGORIES to pass; the stats tab's
    # category chart then just shows "uncategorized" for every case instead
    # of failing to render.
    categories_payload = _json_for_script(
        {name: sorted(ops) for name, ops in (op_categories or {}).items()}
    )
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    gen_errors = summary.get("gen_errors", 0)
    perf_buttons = perf_pages = perf_script = ""
    if "performance_cases" in summary:
        from .performance_tabs import render_performance_tabs
        perf_buttons, perf_pages, perf_script = render_performance_tabs(summary)
    pass_rate = f"{(passed / total * 100):.1f}" if total else "0.0"

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<style>
  :root {{
    color-scheme: light;
    --accent: #3d5a80;
    --pass: #2f9e6e; --pass-bg: #e6f5ee;
    --fail: #e2635c; --fail-bg: #fcebea;
    --gen-error: #e8a33d; --gen-error-bg: #fdf1dc;
    /* 统计图表配色（dataviz 验证器六项全 PASS：CVD ΔE 8.7 / 常态 ΔE 28 / 对比度全过）。
       表格里的状态徽章仍用上面 --pass/--fail/--gen-error 三色，两套互不影响。 */
    --viz-pass: #12805e;
    --viz-fail: #d03b3b;
    --viz-gen-error: #6250d6;
    --viz-surface-gap: #ffffff;
    --viz-grid: #eef1f5;
    --viz-axis: #e2e6ed;
    --viz-text-primary: #1a2233;
    --viz-text-secondary: #667085;
    --viz-text-muted: #98a2b3;
  }}
  body {{ margin: 0; font: 14px/1.5 system-ui, -apple-system, sans-serif; color: #16202a; background: #f5f7fa; }}
  header {{ padding: 16px 20px; background: #fff; border-bottom: 1px solid #d7dee8; }}
  h1 {{ margin: 0 0 8px; font-size: 18px; }}
  .stats {{ display: flex; gap: 16px; color: #617083; }}
  .stats strong {{ color: #16202a; }}
  .stats .fail {{ color: var(--fail); }}
  .stats .pass {{ color: var(--pass); }}
  main {{ padding: 16px 20px; }}
  .toolbar {{ margin-bottom: 10px; display: flex; gap: 8px; }}
  input, select {{ height: 30px; border: 1px solid #d7dee8; border-radius: 6px; padding: 0 8px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d7dee8; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 7px 10px; border-bottom: 1px solid #edf0f5; text-align: left; font-size: 13px; }}
  th {{ background: #fbfcfe; color: #617083; cursor: pointer; white-space: nowrap; }}
  tr:hover {{ background: #f2f5fb; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
  .badge.pass {{ color: var(--pass); background: var(--pass-bg); }}
  .badge.fail {{ color: var(--fail); background: var(--fail-bg); }}
  .badge.error, .badge.gen_error {{ color: var(--gen-error); background: var(--gen-error-bg); }}
  .ops {{ color: #617083; font-size: 12px; max-width: 360px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .shapes {{ color: #617083; font-size: 12px; max-width: 220px; white-space: pre-line; overflow-y: auto; max-height: 72px; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .tabs {{ display: flex; gap: 4px; margin-top: 10px; }}
  .tabs button {{ height: 30px; padding: 0 14px; border: 1px solid #d7dee8; background: #fff; border-radius: 6px 6px 0 0; cursor: pointer; font-size: 13px; color: #617083; }}
  .tabs button.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .tabpage {{ display: none; }}
  .tabpage.active {{ display: block; }}
  /* 所有图表卡片统一用固定列宽（不用 1fr 拉伸），保证汇总统计页签里五张图
     大小完全一致；一行放不满就按 chart-grid 这条规则自然换行/留白，不会把
     某一行的卡片拉宽拉高到跟别的行不一样。窄屏（卡片宽度放不下 360px）时
     退化成一列铺满，保证移动端不会横向溢出。 */
  .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fill, 380px); gap: 16px; justify-content: start; }}
  .chart-grid-primary {{ margin-bottom: 16px; }}
  @media (max-width: 420px) {{
    .chart-grid {{ grid-template-columns: 1fr; justify-content: stretch; }}
  }}
  .chart-card {{ display: flex; flex-direction: column; background: #fff; border: 1px solid #e7eaf0; border-radius: 12px; padding: 16px 18px 14px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04); }}
  .chart-card h3 {{ margin: 0 0 4px; font-size: 14px; font-weight: 600; color: var(--viz-text-primary); }}
  .chart-card .chart-sub {{ margin: 0 0 8px; font-size: 12px; color: var(--viz-text-muted); }}
  .chart-card svg {{ width: 100%; display: block; }}
  .chart-card .legend {{ margin-top: auto; padding-top: 10px; border-top: 1px solid #f0f2f5; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--viz-text-secondary); }}
  .legend span {{ display: inline-flex; align-items: center; gap: 5px; }}
  .legend i {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}
  /* 悬停浮层：跟随鼠标，替代浏览器原生 <title>。 */
  .viz-tooltip {{ position: fixed; pointer-events: none; z-index: 10; background: #1a2233; color: #fff; border-radius: 8px; padding: 8px 12px; font-size: 12px; line-height: 1.6; box-shadow: 0 4px 12px rgba(16, 24, 40, 0.18); opacity: 0; transition: opacity 0.1s; max-width: 260px; }}
  .viz-tooltip.show {{ opacity: 1; }}
  .viz-tooltip .tt-key {{ color: #b6bfce; margin-right: 6px; }}
  .viz-tooltip .tt-val {{ font-weight: 600; }}
  svg .seg {{ cursor: default; }}
  svg .seg:hover, svg .seg:focus {{ opacity: 0.82; outline: none; }}
</style>
</head>
<body>
<header>
  <h1>{_html.escape(title)}</h1>
  <div class="stats">
    <span>Total <strong>{total}</strong></span>
    <span class="pass">Passed <strong>{passed}</strong></span>
    <span class="fail">Failed <strong>{failed}</strong></span>
    <span>Gen errors <strong>{gen_errors}</strong></span>
    <span>Pass rate <strong>{pass_rate}%</strong></span>
  </div>
  <div class="tabs">
    <button id="tabBtnStats" class="active">汇总统计</button>
    <button id="tabBtnTable">用例列表</button>
    {perf_buttons}
  </div>
</header>
<main>
  <div id="tabStats" class="tabpage active">
    <div class="chart-grid chart-grid-primary">
      <div class="chart-card">
        <h3>总体结果</h3>
        <p class="chart-sub">{total} 个用例 · pass / fail / 生成错误占比</p>
        <svg id="chartOverall" viewBox="0 0 380 200" role="img" aria-label="总体 pass fail gen_error 环形图"></svg>
        <div class="legend" id="legendOverall"></div>
      </div>
      <div class="chart-card">
        <h3>按算子分类</h3>
        <p class="chart-sub">一个用例含多类算子时计入多类 · 悬停看细分</p>
        <svg id="chartCategory" viewBox="0 0 380 260" role="img" aria-label="按算子分类的 pass fail gen_error 堆叠柱状图"></svg>
        <div class="legend" id="legendCategory"></div>
      </div>
      <div class="chart-card">
        <h3>按 dtype</h3>
        <p class="chart-sub">用例所有输入/输出张量出现过的 dtype（去重计入）</p>
        <svg id="chartDtype" viewBox="0 0 380 260" role="img" aria-label="按 dtype 的 pass fail gen_error 堆叠柱状图"></svg>
        <div class="legend" id="legendDtype"></div>
      </div>
    </div>
    <div class="chart-grid chart-grid-secondary">
      <div class="chart-card">
        <h3>按 shape 档位：单 tensor 最大维度值</h3>
        <p class="chart-sub">取用例所有张量形状里最大的单个维度值分档</p>
        <svg id="chartShapeDim" viewBox="0 0 380 260" role="img" aria-label="按单 tensor 最大维度值分档的堆叠柱状图"></svg>
        <div class="legend" id="legendShapeDim"></div>
      </div>
      <div class="chart-card">
        <h3>按 shape 档位：单 tensor 最大元素数</h3>
        <p class="chart-sub">取张量形状各维度乘积（元素总数）的最大值分档</p>
        <svg id="chartShapeElem" viewBox="0 0 380 260" role="img" aria-label="按单 tensor 最大元素数分档的堆叠柱状图"></svg>
        <div class="legend" id="legendShapeElem"></div>
      </div>
    </div>
  </div>
  <div id="tabTable" class="tabpage">
    <div class="toolbar">
      <input id="filterOps" placeholder="按 op 名称筛选 (子串)">
      <select id="filterStatus">
        <option value="">全部状态</option>
        <option value="pass">pass</option>
        <option value="fail">fail</option>
        <option value="gen_error">gen_error</option>
      </select>
    </div>
    <table id="caseTable">
      <thead>
        <tr>
          <th data-key="index">#</th>
          <th data-key="status">状态</th>
          <th data-key="seed">Seed</th>
          <th data-key="n_nodes">节点数</th>
          <th data-key="shape_mode">Shape模式 / 尺寸组</th>
          <th data-key="n_kernels">Kernel数</th>
          <th data-key="duration_s">耗时(s)</th>
          <th data-key="symptom">Symptom</th>
          <th data-key="stage">Stage</th>
          <th data-key="op_types">算子集合</th>
          <th data-key="input_shapes">Input Shape</th>
          <th data-key="output_shapes">Output Shape</th>
          <th>复现路径</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
  {perf_pages}
</main>
<div class="viz-tooltip" id="vizTooltip"></div>
<script>
const CASES = {payload};
const OP_CATEGORIES = {categories_payload};
const tbody = document.querySelector('#caseTable tbody');
let sortKey = 'index', sortAsc = true;

document.querySelector('#tabBtnStats').addEventListener('click', () => switchTab('tabStats', 'tabBtnStats'));
document.querySelector('#tabBtnTable').addEventListener('click', () => switchTab('tabTable', 'tabBtnTable'));
function switchTab(pageId, btnId) {{
  document.querySelectorAll('.tabpage').forEach(p => p.classList.toggle('active', p.id === pageId));
  document.querySelectorAll('.tabs button').forEach(b => b.classList.toggle('active', b.id === btnId));
}}

function formatShapes(shapes, dtypes) {{
  const entries = Object.entries(shapes || {{}});
  if (entries.length === 0) return '-';
  return entries.map(([name, dims]) => {{
    const dtype = (dtypes || {{}})[name];
    const dtypePrefix = dtype ? `${{dtype}}` : '';
    return `${{name}}: ${{dtypePrefix}}[${{dims.join(', ')}}]`;
  }}).join('\\n');
}}

function render() {{
  const opsFilter = document.querySelector('#filterOps').value.toLowerCase();
  const statusFilter = document.querySelector('#filterStatus').value;
  let rows = CASES.filter(c => {{
    if (statusFilter && c.status !== statusFilter) return false;
    if (opsFilter && !(c.op_types || []).some(o => o.toLowerCase().includes(opsFilter))) return false;
    return true;
  }});
  rows.sort((a, b) => {{
    const av = a[sortKey], bv = b[sortKey];
    const cmp = (av > bv) - (av < bv);
    return sortAsc ? cmp : -cmp;
  }});
  tbody.innerHTML = rows.map(c => `
    <tr>
      <td>${{c.index}}</td>
      <td><span class="badge ${{c.status}}">${{c.status}}</span></td>
      <td>${{c.seed}}</td>
      <td>${{c.n_nodes || '-'}}</td>
      <td class="shapes">${{c.shape_mode || 'static'}}${{(c.shape_variants || []).map((v,i) => '<br>#' + i + ' ' + formatShapes(v.input_shapes,c.input_dtypes)).join('')}}</td>
      <td>${{c.n_kernels || '-'}}</td>
      <td>${{c.duration_s.toFixed(2)}}</td>
      <td>${{c.symptom || '-'}}</td>
      <td>${{c.stage || '-'}}</td>
      <td class="ops" title="${{(c.op_types || []).join(', ')}}">${{(c.op_types || []).join(', ') || '-'}}</td>
      <td class="shapes" title="${{formatShapes(c.input_shapes, c.input_dtypes)}}">${{formatShapes(c.input_shapes, c.input_dtypes)}}</td>
      <td class="shapes" title="${{formatShapes(c.output_shapes, c.output_dtypes)}}">${{formatShapes(c.output_shapes, c.output_dtypes)}}</td>
      <td>${{c.bug_dir ? `<a href="${{c.bug_dir}}/repro.py">repro.py</a>` : '-'}}</td>
    </tr>
  `).join('');
}}

document.querySelectorAll('#caseTable th[data-key]').forEach(th => {{
  th.addEventListener('click', () => {{
    const key = th.dataset.key;
    sortAsc = (sortKey === key) ? !sortAsc : true;
    sortKey = key;
    render();
  }});
}});
document.querySelector('#filterOps').addEventListener('input', render);
document.querySelector('#filterStatus').addEventListener('change', render);
render();

// ---- 汇总统计 tab：环形图 (总体 pass/fail) + 堆叠柱状图 (分类/shape档位/dtype) ----
//
// 图表配色从 CSS :root 的 --viz-* 变量读取，和 <style> 保持同一份取值来源，
// 避免 CSS/JS 两处各写一份颜色定义导致日后改色时漏改。
const _rootStyle = getComputedStyle(document.documentElement);
const STATUS_COLORS = {{
  pass: _rootStyle.getPropertyValue('--viz-pass').trim(),
  fail: _rootStyle.getPropertyValue('--viz-fail').trim(),
  error: _rootStyle.getPropertyValue('--viz-gen-error').trim(),
  gen_error: _rootStyle.getPropertyValue('--viz-gen-error').trim(),
}};
const VIZ = {{
  surfaceGap: _rootStyle.getPropertyValue('--viz-surface-gap').trim(),
  grid: _rootStyle.getPropertyValue('--viz-grid').trim(),
  axis: _rootStyle.getPropertyValue('--viz-axis').trim(),
  textPrimary: _rootStyle.getPropertyValue('--viz-text-primary').trim(),
  textSecondary: _rootStyle.getPropertyValue('--viz-text-secondary').trim(),
  textMuted: _rootStyle.getPropertyValue('--viz-text-muted').trim(),
}};
const FONT = 'system-ui, -apple-system, sans-serif';
const STATUS_KEYS = ['pass', 'fail', 'gen_error'];

// ---- 悬停浮层（替代原生 <title>）：跟随鼠标，内容用 textContent 写入，
// 系列名/分类名来自用户数据，不能走 innerHTML 拼接。 ----
const _tooltip = document.querySelector('#vizTooltip');
function showTooltip(evt, title, entries) {{
  while (_tooltip.firstChild) _tooltip.removeChild(_tooltip.firstChild);
  const titleEl = document.createElement('div');
  titleEl.textContent = title;
  titleEl.style.cssText = 'font-weight:600;margin-bottom:2px';
  _tooltip.appendChild(titleEl);
  for (const [key, val] of entries) {{
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
  }}
  _tooltip.classList.add('show');
  moveTooltip(evt);
}}
function moveTooltip(evt) {{
  const pad = 14;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  const rect = _tooltip.getBoundingClientRect();
  if (x + rect.width > window.innerWidth - 8) x = evt.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = evt.clientY - rect.height - pad;
  _tooltip.style.left = x + 'px';
  _tooltip.style.top = y + 'px';
}}
function hideTooltip() {{
  _tooltip.classList.remove('show');
}}

// 给 SVG 图形元素挂 tooltip：content() 返回 [标题, [[键, 值], ...]]，
// 悬停/键盘聚焦都显示同一份内容，移开/失焦隐藏。focus 没有鼠标坐标，
// 就把浮层放到元素自身的屏幕位置旁。
function attachTooltip(el, content) {{
  const place = (evt) => {{
    if (evt.clientX !== undefined) {{
      moveTooltip(evt);
    }} else {{
      const r = el.getBoundingClientRect();
      moveTooltip({{ clientX: r.left + r.width / 2, clientY: r.top }});
    }}
  }};
  const bound = (evt) => {{ showTooltip(evt, ...content()); place(evt); }};
  el.addEventListener('pointerenter', bound);
  el.addEventListener('pointermove', moveTooltip);
  el.addEventListener('pointerleave', hideTooltip);
  el.setAttribute('tabindex', '0');
  el.addEventListener('focus', bound);
  el.addEventListener('blur', hideTooltip);
}}

// 单个 case 的最大单维度值 / 最大元素数：汇总它所有 input/output tensor 的 shape。
function caseMaxDim(c) {{
  let maxDim = 0, maxElem = 0;
  for (const shapes of [c.input_shapes, c.output_shapes]) {{
    for (const dims of Object.values(shapes || {{}})) {{
      if (!dims.length) continue;
      maxDim = Math.max(maxDim, ...dims);
      maxElem = Math.max(maxElem, dims.reduce((a, b) => a * b, 1));
    }}
  }}
  return {{ maxDim, maxElem }};
}}

const DIM_BUCKETS = [
  {{ label: '\\u22648', test: v => v <= 8 }},
  {{ label: '\\u226464', test: v => v <= 64 }},
  {{ label: '\\u2264512', test: v => v <= 512 }},
  {{ label: '\\u22644096', test: v => v <= 4096 }},
  {{ label: '\\u226416384', test: v => v <= 16384 }},
  {{ label: '\\u226465536', test: v => v <= 65536 }},
  {{ label: '\\u2264131072', test: v => v <= 131072 }},
  {{ label: '>131072', test: () => true }},
];
const ELEM_BUCKETS = [
  {{ label: '\\u22641K', test: v => v <= 1e3 }},
  {{ label: '\\u226416K', test: v => v <= 16e3 }},
  {{ label: '\\u226464K', test: v => v <= 64e3 }},
  {{ label: '\\u2264256K', test: v => v <= 256e3 }},
  {{ label: '\\u22641M', test: v => v <= 1e6 }},
  {{ label: '\\u226416M', test: v => v <= 16e6 }},
  {{ label: '\\u226464M', test: v => v <= 64e6 }},
  {{ label: '>64M', test: () => true }},
];

function bucketOf(buckets, value) {{
  for (const b of buckets) {{
    if (b.test(value)) return b.label;
  }}
  return buckets[buckets.length - 1].label;
}}

// 按 key 分组统计 pass/fail/gen_error 计数，group(c) 可以返回单个字符串或字符串数组
// （比如一个 case 同时含 reduce+elewise 两类 op，会同时计入两个分类桶）。
function groupCounts(cases, groupFn, order, statusKeys = STATUS_KEYS) {{
  const counts = new Map();
  for (const c of cases) {{
    let keys = groupFn(c);
    if (!Array.isArray(keys)) keys = [keys];
    for (const k of new Set(keys)) {{
      if (!counts.has(k)) counts.set(k, Object.fromEntries(statusKeys.map(s => [s, 0])));
      counts.get(k)[c.status] = (counts.get(k)[c.status] || 0) + 1;
    }}
  }}
  const keys = order ? order.filter(k => counts.has(k)) : [...counts.keys()];
  return keys.map(k => ({{ key: k, ...counts.get(k) }}));
}}

// ---- 环形图：中心放 pass rate 大数字 + 总数，各段占比靠图例（不标在线上）。
// 段间留 2px 白缝（用 stroke 里的 --viz-surface-gap 画），环宽 24。 ----
function drawDonut(svgId, legendId, segments) {{
  const svg = document.querySelector('#' + svgId);
  const W = 380, H = 200, cx = 190, cy = 98, r = 72, ringW = 24;
  const total = segments.reduce((s, seg) => s + seg.value, 0);
  // 图例始终渲染（含 0 值段，占比标 0.0%），环形段只画 >0 的。
  const legend = document.querySelector('#' + legendId);
  const legendEls = segments.map(seg => {{
    const pct = total ? (seg.value / total * 100).toFixed(1) : '0.0';
    return `<span><i style="background:${{seg.color}}"></i>${{seg.label}} ${{seg.value}} · ${{pct}}%</span>`;
  }});
  legend.innerHTML = legendEls.join('');
  if (total === 0) {{
    svg.innerHTML = `<text x="${{cx}}" y="${{cy}}" text-anchor="middle" fill="${{VIZ.textMuted}}" font-family="${{FONT}}" font-size="13">无数据</text>`;
    return;
  }}
  const C = 2 * Math.PI * r;
  let offset = 0;
  const parts = [];
  // 从 12 点方向顺时针：先整体 rotate(-90)，再逐段用 dasharray/dashoffset。
  for (const seg of segments.filter(s => s.value > 0)) {{
    const frac = seg.value / total;
    const len = frac * C;
    // 段间 ~2px 白缝：dasharray 的单位就是路径长度（px），直接减 2。
    const dash = Math.max(len - 2, 0.5);
    parts.push(`<circle class="seg" cx="${{cx}}" cy="${{cy}}" r="${{r}}" fill="none" stroke="${{seg.color}}" stroke-width="${{ringW}}" stroke-dasharray="${{dash}} ${{C - dash}}" stroke-dashoffset="${{-offset}}" data-key="${{seg.label}}"></circle>`);
    offset += len;
  }}
  const passRate = (() => {{
    const passSeg = segments.find(s => s.label === 'pass');
    return total && passSeg ? (passSeg.value / total * 100).toFixed(1) : '0.0';
  }})();
  const passVal = segments.find(s => s.label === 'pass');
  svg.innerHTML = `
    <g transform="rotate(-90 ${{cx}} ${{cy}})">${{parts.join('')}}</g>
    <text x="${{cx}}" y="${{cy - 3}}" text-anchor="middle" fill="${{VIZ.textPrimary}}" font-family="${{FONT}}" font-size="27" font-weight="650">${{passRate}}%</text>
    <text x="${{cx}}" y="${{cy + 16}}" text-anchor="middle" fill="${{VIZ.textMuted}}" font-family="${{FONT}}" font-size="11">pass rate · ${{passVal ? passVal.value : 0}}/${{total}}</text>
  `;
  // 每段挂 tooltip（value 从闭包里的 segments 取，聚焦/悬停同款内容）。
  svg.querySelectorAll('.seg').forEach(el => {{
    const seg = segments.find(s => s.label === el.dataset.key);
    attachTooltip(el, () => [seg.label, [['count', seg.value], ['share', (total ? seg.value / total * 100 : 0).toFixed(1) + '%']]]);
  }});
}}

// ---- 竖直堆叠柱状图：细柱（≤24px）、顶端 4px 圆角、段间 2px 白缝、
// 极淡实线网格 + 基线；柱顶标总数，段内数值仅在放得下时显示。 ----
function drawStackedBars(svgId, legendId, rows, statusKeys = STATUS_KEYS) {{
  const svg = document.querySelector('#' + svgId);
  const W = 380, H = 260;
  const leftPad = 34, rightPad = 10, topPad = 18, bottomPad = 42;
  const chartH = H - topPad - bottomPad;
  const plotW = W - leftPad - rightPad;
  const legend = document.querySelector('#' + legendId);
  legend.innerHTML = statusKeys.map(s =>
    `<span><i style="background:${{STATUS_COLORS[s]}}"></i>${{s}}</span>`
  ).join('');
  if (rows.length === 0) {{
    svg.innerHTML = `<text x="${{W / 2}}" y="${{H / 2}}" text-anchor="middle" fill="${{VIZ.textMuted}}" font-family="${{FONT}}" font-size="13">无数据</text>`;
    return;
  }}
  const maxTotal = Math.max(...rows.map(r => statusKeys.reduce((n, s) => n + (r[s] || 0), 0)), 1);
  const slot = plotW / rows.length;
  const barW = Math.min(slot * 0.52, 24);
  const gap = slot - barW;

  // y 轴：整洁刻度（nice step：1/2/5×10^k），顶部留 headroom 让柱顶总数不顶格。
  const niceStep = (x) => {{
    const pow = Math.pow(10, Math.floor(Math.log10(x)));
    for (const m of [1, 2, 5, 10]) {{
      if (m * pow >= x) return m * pow;
    }}
    return 10 * pow;
  }};
  const yMax = Math.ceil(maxTotal / niceStep(maxTotal / 4)) * niceStep(maxTotal / 4);
  const step = niceStep(yMax / 4);
  const ticks = [];
  for (let t = 0; t <= yMax; t += step) ticks.push(t);
  const gridLines = ticks.map(t => {{
    const y = topPad + chartH - (t / yMax) * chartH;
    const isBase = t === 0;
    return `
      <line x1="${{leftPad}}" y1="${{y}}" x2="${{W - rightPad}}" y2="${{y}}" stroke="${{isBase ? VIZ.axis : VIZ.grid}}" stroke-width="1"></line>
      <text x="${{leftPad - 8}}" y="${{y + 4}}" text-anchor="end" fill="${{VIZ.textMuted}}" font-family="${{FONT}}" font-size="11">${{t}}</text>
    `;
  }}).join('');

  const parts = [];
  rows.forEach((r, i) => {{
    const x = leftPad + i * (barW + gap) + (slot - barW) / 2;
    const total = statusKeys.reduce((n, s) => n + (r[s] || 0), 0);
    const topY = topPad + chartH - (total / yMax) * chartH;
    const segs = statusKeys
      .map(s => ({{ v: r[s], color: STATUS_COLORS[s], label: s }}))
      .filter(s => s.v > 0);
    let yBottom = topPad + chartH;
    for (const [j, s] of segs.entries()) {{
      const h = (s.v / yMax) * chartH;
      // 段间 2px 白缝：从下往上堆，每段顶部让出 2px（最后一段/顶段除外），
      // 顶段的数据端由 4px 圆角自然收边。高度不够时至少画 1px 色带。
      const isTop = j === segs.length - 1;
      const segH = Math.max(h - (isTop ? 0 : 2), 1);
      const segY = yBottom - segH;
      let shape;
      if (isTop && segH > 4) {{
        // path: 顶边两角 4px 圆角。
        shape = `M ${{x}} ${{segY + 4}} a 4 4 0 0 1 4 -4 h ${{barW - 8}} a 4 4 0 0 1 4 4 v ${{segH - 4}} h ${{-barW}} Z`;
      }} else {{
        shape = `M ${{x}} ${{segY}} h ${{barW}} v ${{segH}} h ${{-barW}} Z`;
      }}
      parts.push(`<path class="seg" d="${{shape}}" fill="${{s.color}}" data-key="${{r.key}}|${{s.label}}"></path>`);
      // 段内数值：仅段高放得下一行 12px 字才显示（白字，深浅底都可读）。
      if (segH >= 16 && barW >= 18) {{
        parts.push(`<text x="${{x + barW / 2}}" y="${{segY + segH / 2 + 4}}" text-anchor="middle" fill="#fff" font-family="${{FONT}}" font-size="11.5" font-weight="600" pointer-events="none">${{s.v}}</text>`);
      }}
      yBottom = yBottom - h - 2;  // 下一段起点：本段理论顶再留 2px 缝
      if (yBottom < topPad) yBottom = topPad;
    }}
    // 柱顶总数（深色加粗）。
    parts.push(`<text x="${{x + barW / 2}}" y="${{topY - 6}}" text-anchor="middle" fill="${{VIZ.textPrimary}}" font-family="${{FONT}}" font-size="12.5" font-weight="600" pointer-events="none">${{total}}</text>`);
    // x 轴分类标签（超 9 字符截断）。
    const label = r.key.length > 9 ? r.key.slice(0, 8) + '\\u2026' : r.key;
    parts.push(`<text x="${{x + barW / 2}}" y="${{topPad + chartH + 20}}" text-anchor="middle" fill="${{VIZ.textSecondary}}" font-family="${{FONT}}" font-size="12" pointer-events="none">${{label}}</text>`);
  }});
  svg.innerHTML = gridLines + parts.join('');
  // 每段挂 tooltip：分类名做标题，pass/fail/gen_error 计数 + 总数。
  svg.querySelectorAll('.seg').forEach(el => {{
    const [catKey, statusKey] = el.dataset.key.split('|');
    attachTooltip(el, () => {{
      const row = rows.find(r => r.key === catKey);
      const entries = statusKeys.filter(s => row[s] > 0).map(s => [s, row[s]]);
      entries.push(['total', statusKeys.reduce((n, s) => n + (row[s] || 0), 0)]);
      return [catKey, entries];
    }});
  }});
}}

function renderStats() {{
  // 1. 总体 pass/fail/gen_error 环形图（中心 pass rate）
  const overall = STATUS_KEYS.map(s => ({{
    label: s, value: CASES.filter(c => c.status === s).length, color: STATUS_COLORS[s],
  }}));
  drawDonut('chartOverall', 'legendOverall', overall);

  // 2. 按算子分类：一个 case 可能同时属于多个分类（比如同时含 reduce+elewise 的图）
  const categoryNames = Object.keys(OP_CATEGORIES);
  const categoryRows = groupCounts(CASES, c => {{
    const ops = c.op_types || [];
    const hit = categoryNames.filter(name => ops.some(op => OP_CATEGORIES[name].includes(op)));
    return hit.length ? hit : ['(未分类)'];
  }}, [...categoryNames, '(未分类)']);
  drawStackedBars('chartCategory', 'legendCategory', categoryRows);

  // 3/4. 按 shape 档位：单 tensor 最大维度值 / 最大元素数
  const dimRows = groupCounts(CASES, c => bucketOf(DIM_BUCKETS, caseMaxDim(c).maxDim), DIM_BUCKETS.map(b => b.label));
  drawStackedBars('chartShapeDim', 'legendShapeDim', dimRows);
  const elemRows = groupCounts(CASES, c => bucketOf(ELEM_BUCKETS, caseMaxDim(c).maxElem), ELEM_BUCKETS.map(b => b.label));
  drawStackedBars('chartShapeElem', 'legendShapeElem', elemRows);

  // 5. 按 dtype：汇总一个 case 所有 input/output tensor 出现过的 dtype（去重），
  // 和分类一样允许一个 case 同时计入多种 dtype。
  const dtypeRows = groupCounts(CASES, c => {{
    const dtypes = new Set([...Object.values(c.input_dtypes || {{}}), ...Object.values(c.output_dtypes || {{}})]);
    return dtypes.size ? [...dtypes] : ['(无)'];
  }});
  dtypeRows.sort((a, b) => (b.pass + b.fail + b.gen_error) - (a.pass + a.fail + a.gen_error));
  drawStackedBars('chartDtype', 'legendDtype', dtypeRows);
}}

renderStats();
{perf_script}
</script>
</body>
</html>
"""


def write_summary_html(
    summary: dict[str, Any],
    out_path: str | Path,
    title: str = "AF Fuzz Summary",
    op_categories: dict[str, set[str]] | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_summary_html(summary, title=title, op_categories=op_categories),
        encoding="utf-8",
    )
