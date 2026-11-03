"""Optional performance views for the shared summary renderer (no NPU imports)."""
import json
import html


CHARTS = [
    ("Overall", "性能总体结果", "pass / fail / error · pass rate 分母包含全部用例"),
    ("Category", "按算子分类", "多类算子可重复计入 · 悬停看性能 pass/fail"),
    ("Dtype", "按 dtype", "输入/输出张量 dtype 去重计入"),
    ("ShapeDim", "按 shape：最大维度值", "与功能/精度汇总使用相同档位"),
    ("ShapeElem", "按 shape：最大元素数", "与功能/精度汇总使用相同档位"),
    ("Nodes", "按图节点数", "计算节点数 · 不含输入和常量节点"),
    ("Kernels", "按 AscendC 每步 kernel 数", "CSV Count 求和 ÷ 实际采集步数"),
]


def render_performance_tabs(summary):
    cases = summary["performance_cases"]
    payload = json.dumps(cases, ensure_ascii=False).replace("</", "<\\/")
    counts = {s: sum(c.get("performance_status") == s for c in cases) for s in ("pass", "fail", "error")}
    buttons = '<button id="tabBtnPerf">性能对比</button><button id="tabBtnPerfStats">性能对比汇总</button>'
    columns = [
        ("index", "序号"), ("performance_status", "性能状态"), ("seed", "Seed"), ("n_nodes", "节点数"),
        ("asc_kernels", "AscendC Kernel数/步"), ("tri_kernels", "Triton Kernel数/步"),
        ("asc_duration", "AscendC 执行耗时(s)"), ("tri_duration", "Triton 执行耗时(s)"),
        ("asc_time", "AscendC Kernel耗时(μs/步)"), ("tri_time", "Triton Kernel耗时(μs/步)"),
        ("ascendc_speedup", "性能加速比"), ("op_types", "算子集合"),
        ("input_shapes", "Input Shape / dtype"), ("output_shapes", "Output Shape / dtype"),
        ("shape_details", "Shape模式 / 各尺寸耗时"), ("accuracy", "精度 AscendC / Triton"), ("error", "异常详情"),
    ]
    heads = ''.join(f'<th data-perf-key="{key}">{label}</th>' for key, label in columns)
    cards = ''.join(f'''<div class="chart-card"><h3>{title}</h3><p class="chart-sub">{sub}</p>
<svg id="perfChart{key}" viewBox="0 0 380 {'200' if key == 'Overall' else '260'}" role="img" aria-label="{title}"></svg>
<div class="legend" id="perfLegend{key}"></div></div>''' for key, title, sub in CHARTS)
    note = '''<p style="color:#617083">性能 pass：AscendC kernel 耗时 ≤ Triton-Ascend；fail：AscendC 更慢；error：采集失败、耗时缺失或内容指纹不匹配。<br>
每步 kernel 耗时 = 所有算子 Total Time 求和 ÷ 采集步数；加速比 = Triton-Ascend / AscendC（≥1 为 pass）。精度状态独立展示。<br>
动态模式逐尺寸采集，汇总耗时为所有尺寸 Total Time 求和 ÷ 所有采集步数；各尺寸结果见表格。<br>
执行耗时为后端 worker 总墙钟时间，包含启动、构图、编译、校验、预热、采集和导出，不参与性能判定；“—”表示无数据。</p>'''
    stats = f'<div class="stats"><span>Total <strong>{len(cases)}</strong></span><span class="pass">性能 pass <strong>{counts["pass"]}</strong></span><span class="fail">性能 fail <strong>{counts["fail"]}</strong></span><span>error <strong>{counts["error"]}</strong></span></div>'
    state = html.escape(str(summary.get("session_status", "")))
    if state:
        limit = summary.get("requested_cases")
        limit_text = "持续运行" if limit is None else f"目标 {limit} 例"
        stats += f'<span>运行状态 <strong>{state}</strong> · {limit_text}</span>'
    pages = f'''<div id="tabPerf" class="tabpage">{stats}{note}
<div class="toolbar"><input id="perfFilterOps" placeholder="按 op 或 seed 筛选"><select id="perfFilterStatus">
<option value="">全部性能状态</option><option>pass</option><option>fail</option><option>error</option></select><span id="perfRowCount"></span></div>
<div style="overflow-x:auto"><table id="perfTable"><thead><tr>{heads}</tr></thead><tbody></tbody></table></div></div>
<div id="tabPerfStats" class="tabpage">{stats}{note}<div class="chart-grid">{cards}</div></div>'''
    script = SCRIPT.replace('__PAYLOAD__', payload)
    return buttons, pages, script


SCRIPT = r'''
const PERF_CASES = __PAYLOAD__;
const PERF_KEYS = ['pass', 'fail', 'error'];
document.querySelector('#tabBtnPerf').onclick = () => switchTab('tabPerf', 'tabBtnPerf');
document.querySelector('#tabBtnPerfStats').onclick = () => switchTab('tabPerfStats', 'tabBtnPerfStats');
const perfEscape = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const perfNumber = (value, digits=3) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
const perfRows = PERF_CASES.map(c => {
  const a = c.backends?.ascendc || {}, t = c.backends?.['triton-ascend'] || {};
  return {...c, asc_kernels:a.mean_kernel_count, tri_kernels:t.mean_kernel_count,
    asc_duration:a.duration_s, tri_duration:t.duration_s, asc_time:a.mean_kernel_us, tri_time:t.mean_kernel_us,
    shape_details:(c.shape_mode || 'static') + (c.shape_variants || []).map((v,i) =>
      '\n#' + i + ' ' + formatShapes(v.input_shapes,c.input_dtypes) + '\nAscendC / Triton μs: ' +
      perfNumber(a.shape_results?.[i]?.mean_kernel_us) + ' / ' + perfNumber(t.shape_results?.[i]?.mean_kernel_us) +
      ' · ' + (a.shape_results?.[i]?.correctness || '—') + ' / ' + (t.shape_results?.[i]?.correctness || '—')).join(''),
    accuracy:`${a.correctness || 'not_checked'} / ${t.correctness || 'not_checked'}`,
    error:c.error || Object.entries(c.backends || {}).filter(([name,b]) => b.status !== 'ok' || b.error || b.correctness_error)
      .map(([name,b]) => `${name} [${b.error_stage || b.status}]: ${b.error || b.correctness_error || b.status}`).join('\n') ||
      (c.performance_status === 'error' ? '耗时缺失/无效或 case 指纹不匹配' : '')};
});
let perfSortKey = 'index', perfSortAsc = true;
function renderPerfTable() {
  const query = document.querySelector('#perfFilterOps').value.trim().toLowerCase();
  const status = document.querySelector('#perfFilterStatus').value;
  const rows = perfRows.filter(c => (!status || c.performance_status === status) &&
    (!query || String(c.seed).includes(query) || (c.op_types || []).some(op => op.toLowerCase().includes(query))));
  rows.sort((a,b) => {
    const x = a[perfSortKey], y = b[perfSortKey];
    if (x == null) return y == null ? 0 : 1;
    if (y == null) return -1;
    const cmp = typeof x === 'number' && typeof y === 'number' ? x-y : String(x).localeCompare(String(y));
    return perfSortAsc ? cmp : -cmp;
  });
  document.querySelector('#perfRowCount').textContent = `${rows.length} / ${PERF_CASES.length} 个用例 · 点击表头排序`;
  const td = (value, css='') => `<td class="${css}" title="${perfEscape(value)}">${perfEscape(value)}</td>`;
  document.querySelector('#perfTable tbody').innerHTML = rows.map(c => '<tr>' +
    td(c.index) + `<td><span class="badge ${c.performance_status}">${c.performance_status}</span></td>` + td(c.seed) + td(c.n_nodes ?? '—') +
    td(perfNumber(c.asc_kernels,2)) + td(perfNumber(c.tri_kernels,2)) +
    td(perfNumber(c.asc_duration,2)) + td(perfNumber(c.tri_duration,2)) +
    td(perfNumber(c.asc_time)) + td(perfNumber(c.tri_time)) + td(c.ascendc_speedup == null ? '—' : perfNumber(c.ascendc_speedup) + '×') +
    td((c.op_types || []).join(', '),'ops') + td(formatShapes(c.input_shapes,c.input_dtypes),'shapes') +
    td(formatShapes(c.output_shapes,c.output_dtypes),'shapes') + td(c.shape_details,'shapes') + td(c.accuracy) + td(c.error,'shapes') + '</tr>').join('');
}
document.querySelectorAll('#perfTable th[data-perf-key]').forEach(th => th.onclick = () => {
  const key = th.dataset.perfKey;
  perfSortAsc = perfSortKey === key ? !perfSortAsc : true;
  perfSortKey = key;
  renderPerfTable();
});
document.querySelector('#perfFilterOps').oninput = renderPerfTable;
document.querySelector('#perfFilterStatus').onchange = renderPerfTable;
renderPerfTable();
function renderPerfStats() {
  const cases = PERF_CASES.map(c => ({...c,status:c.performance_status}));
  drawDonut('perfChartOverall','perfLegendOverall',PERF_KEYS.map(s => ({label:s,value:cases.filter(c => c.status === s).length,color:STATUS_COLORS[s]})));
  const draw = (name,group,order) => drawStackedBars('perfChart'+name,'perfLegend'+name,groupCounts(cases,group,order,PERF_KEYS),PERF_KEYS);
  const categories = Object.keys(OP_CATEGORIES);
  draw('Category',c => {
    const hit = categories.filter(name => (c.op_types || []).some(op => OP_CATEGORIES[name].includes(op)));
    return hit.length ? hit : ['(未分类)'];
  },[...categories,'(未分类)']);
  draw('Dtype',c => {
    const dt = [...new Set([...Object.values(c.input_dtypes || {}),...Object.values(c.output_dtypes || {})])];
    return dt.length ? dt : ['(无)'];
  });
  draw('ShapeDim',c => Object.keys(c.input_shapes || {}).length || Object.keys(c.output_shapes || {}).length ? bucketOf(DIM_BUCKETS,caseMaxDim(c).maxDim) : '(无数据)', [...DIM_BUCKETS.map(b => b.label),'(无数据)']);
  draw('ShapeElem',c => Object.keys(c.input_shapes || {}).length || Object.keys(c.output_shapes || {}).length ? bucketOf(ELEM_BUCKETS,caseMaxDim(c).maxElem) : '(无数据)', [...ELEM_BUCKETS.map(b => b.label),'(无数据)']);
  const buckets = ['≤1','≤4','≤8','≤16','≤32','>32','(无数据)'];
  const bucket = v => v == null ? '(无数据)' : v <= 1 ? '≤1' : v <= 4 ? '≤4' : v <= 8 ? '≤8' : v <= 16 ? '≤16' : v <= 32 ? '≤32' : '>32';
  draw('Nodes',c => bucket(c.n_nodes),buckets);
  draw('Kernels',c => bucket(c.backends?.ascendc?.mean_kernel_count),buckets);
}
renderPerfStats();
'''
