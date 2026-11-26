"use strict";
// 总览: pipeline/capability tabs + Kernel performance comparison.
import { el, svg, api, qs, errorBox, label, fmt, pct, DIM_LABEL, DIM_ORDER } from "./core.js";
import { select, stackBar, legend } from "./widgets.js";
import { RESULT_ITEMS, RESULT_COLORS, selectedResults, resultFilterParams } from "./result-filter.js";
import { PAGES } from "./router.js";

// Cache the complete result distribution for a search/chip cohort. Result and
// backend toggles are projections of this snapshot, rather than new SQL scans.
const overviewCache = new Map();
function overviewData(path) {
  const cached = overviewCache.get(path);
  if (cached && Date.now() - cached.at < 60000) return cached.promise;
  const entry = { at: Date.now(), promise: api(path) };
  overviewCache.set(path, entry);
  entry.promise.catch(() => { if (overviewCache.get(path) === entry) overviewCache.delete(path); });
  if (overviewCache.size > 32) overviewCache.delete(overviewCache.keys().next().value);
  return entry.promise;
}
PAGES.overview = async (root, params) => {
  if (root.forceRefresh) overviewCache.clear();
  const filters = resultFilterParams(params);
  const active = selectedResults(params);
  const arch = "";
  const cohort = { q: filters.q, chips: filters.chips };
  const pipeline = await overviewData("/api/overview/pipeline" + qs({ arch, ...cohort }));
  const selectedBackends = filters.selected_backends ? new Set(filters.selected_backends.split(",")) : null;
  const shown = pipeline.lanes.filter(l => !selectedBackends || selectedBackends.has(l.backend));
  const head = el("h2", {}, "基础能力统计分布");
  const box = el("section", { class: "panel hero" }, head); root.append(box);
  const lanes = shown.filter(l => !l.baseline).map(l => l.backend);
  const tabsDone = renderBackendTabs(box, lanes, arch, cohort, active, pipeline.lanes.filter(l => !l.baseline).map(l => l.backend));
  const perfBox = el("section", { class: "panel" }); root.append(perfBox);
  const npuBackends = shown.map(l => l.backend).filter(b => b !== "cpu");
  const perfDone = renderPerf(perfBox, npuBackends, arch, params.get("x") || (npuBackends.includes("af-mlir") ? "af-mlir" : npuBackends[0]), params.get("y") || (npuBackends.includes("triton-npu") ? "triton-npu" : npuBackends.find(b => b !== "af-mlir")), params.get("pdim") || "op_category", false, filters);
  root.settled = Promise.all([tabsDone, perfDone]);
};

const LANE_COLOR = { "af-mlir": "var(--lane-mlir)", "af-legacy": "var(--lane-legacy)", "triton-npu": "var(--lane-triton)", "eager-npu": "var(--lane-eager)" };

// Capability matrix: one column per backend, one row per dimension, so a row compares the same
// metric across backends. Scrolls horizontally when the backends do not fit (desktop and mobile).
async function renderBackendTabs(box, lanes, arch, filters, active, allLanes = lanes) {
  const body = el("div", {}, el("div", { class: "empty" }, "加载中…"));
  let matrix;
  const redraw = () => {
    if (!matrix) return;
    const dims = ["all", ...DIM_ORDER.filter(d => matrix.dims.includes(d))];
    const results = lanes.map(backend => dims.map(dim => matrix.cells[backend][dim] || { dim, groups: [] }));
    const grid = el("div", { class: "cap-grid" });
    grid.style.gridTemplateColumns = `repeat(${lanes.length}, minmax(320px, 1fr))`;
    for (const backend of lanes) grid.append(el("div", { class: "cap-head" }, el("span", { class: "pipe-v-dot", style: { background: LANE_COLOR[backend] || "var(--accent)" } }), label(backend)));
    dims.forEach((dim, row) => {
      grid.append(el("div", { class: "cap-row-label" }, dim === "all" ? "通过率" : (DIM_LABEL[dim] || dim)));
      const totals = new Map();
      for (let col = 0; col < lanes.length; col++) for (const g of (results[col][row].groups || [])) totals.set(g.value, (totals.get(g.value) || 0) + selectedTotal(g, active));
      const order = [...totals.keys()].sort(BUCKET_DIMS.includes(dim) ? bucketCompare : (a, b) => totals.get(b) - totals.get(a));
      lanes.forEach((backend, col) => {
        const result = results[col][row];
        grid.append(dim === "all" ? donutChart(result, backend, true, active) : statusChart(result, backend, true, order, active));
      });
    });
    body.replaceChildren(el("div", { class: "cap-scroll" }, grid));
  };
  box.append(body);
  if (!lanes.length) { body.replaceChildren(el("div", { class: "empty" }, "暂无数据")); return; }
  try { matrix = await overviewData("/api/overview/matrix" + qs({ backends: allLanes.join(","), arch, ...filters })); }
  catch (err) { body.replaceChildren(errorBox(err)); return; }
  redraw();
}
function stageMetric(group, key) {
  return (group.results || {})[key] || 0;
}
function selectedTotal(group, active) { return [...active].reduce((n, key) => n + stageMetric(group, key), 0); }
function selectedPassRate(group, active) {
  return pct(active.has("pass") ? stageMetric(group, "pass") : 0, selectedTotal(group, new Set([...active].filter(key => key !== "invalid"))));
}
function donutChart(result, backend, compact = false, active = new Set(RESULT_ITEMS.map(([key]) => key))) {
  if (result.error) return el("div", { class: "chart" }, errorBox(result.error));
  const g = result.groups[0] || { total: 0, pass: 0, accuracy_fail: 0, unsupported: 0, stage_fail: {} };
  const total = selectedTotal(g, active);
  const invalidCount = stageMetric(g, "invalid");
  const parts = RESULT_ITEMS.map(([key, cls,, title]) => [key, title, stageMetric(g,key), RESULT_COLORS[cls]]).filter(p => active.has(p[0]) && p[2] > 0);
  const R = 78, r = 48, C = 100, root = svg("svg", { viewBox: "0 0 200 200", width: 200, height: 200, role: "img", "aria-label": "正确性分布" });
  let angle = -Math.PI / 2;
  for (const [, name, n, color] of parts) { const sweep = 2 * Math.PI * n / (total || 1), end = angle + Math.min(sweep, 2 * Math.PI - 1e-4), large = sweep > Math.PI ? 1 : 0; const p = (rad,a) => `${C + rad*Math.cos(a)} ${C + rad*Math.sin(a)}`; root.append(svg("path", { d:`M ${p(R,angle)} A ${R} ${R} 0 ${large} 1 ${p(R,end)} L ${p(r,end)} A ${r} ${r} 0 ${large} 0 ${p(r,angle)} Z`, fill:color, "data-tooltip":`${name} ${fmt(n)}（${pct(n,total)}）` })); angle += sweep; }
  root.append(svg("text", {x:C,y:C+5,"text-anchor":"middle",class:"donut-center"}, selectedPassRate(g, active)));
  return el("div", { class: "chart chart-main-result" }, compact ? el("div", { class: "muted small" }, `显示 ${fmt(total)} / ${fmt(g.total)} 个用例`) : el("h3", {}, `${label(backend)} · 通过率（${fmt(total)} 个用例）`), el("div", { class: "muted small" }, "无效 " + fmt(invalidCount) + " 个，不计通过率"), root);
}

const BUCKET_DIMS = ["dim_bucket", "size_bucket"];
function bucketValue(valueLabel) {
  const m = /^(<=|>)\s*([\d.]+)\s*([KM]?)$/.exec(valueLabel || "");
  if (!m) return Infinity;
  const n = Number(m[2]) * ({ K: 1024, M: 1048576 }[m[3]] || 1);
  return m[1] === ">" ? n + 0.5 : n;
}
function bucketCompare(a, b) { return bucketValue(a) - bucketValue(b); }
function statusChart(result, backend, compact = false, order = null, active = new Set(RESULT_ITEMS.map(([key]) => key))) {
  if (result.error) return el("div", { class: "chart" }, errorBox(result.error));
  const limit = result.dim === "op_display" ? 24 : 14;
  let groups;
  if (order) {
    const byValue = new Map(result.groups.map(g => [g.value, g]));
    const empty = v => ({ value: v, total: 0, pass: 0, accuracy_fail: 0, unsupported: 0, stage_fail: {} });
    groups = order.slice(0, limit).map(v => byValue.get(v) || empty(v));
  } else {
    groups = result.groups.slice(0, limit);
  }
  const note = result.dim === "op_display" ? `（覆盖 ${result.groups.length} 个算子，显示用例最多的 ${groups.length} 个）` : "";
  const title = compact ? (note ? el("div", { class: "muted small" }, note.replace(/[（）]/g, "")) : null)
    : el("h3", {}, `${label(backend)} · ${DIM_LABEL[result.dim] || result.dim}${note}`);
  return el("div", { class: "chart" + (result.dim === "op_display" && !compact ? " wide" : "") }, title,
    groups.length ? groups.map(g => { const total = selectedTotal(g, active); const shown = key => active.has(key) ? stageMetric(g, key) : 0; return el("div", { class: "hbar", "data-tooltip": RESULT_ITEMS.map(([key,,,title]) => title + " " + fmt(shown(key))).join(" · ") }, el("div", { class: "label" }, g.value), stackBar(RESULT_ITEMS.map(([key,cls,,title]) => [cls, shown(key), title]), total), el("div", { class: "muted" }, selectedPassRate(g, active))); }) : el("div", { class: "empty" }, "无数据"));
}

async function renderPerf(box, backends, arch, x, y, dim, custom = false, filters = {}) {
  if (custom) box.dataset.custom = "1";  // user picked backends/dimension: background refresh must not reset it
  const opts = backends.map(b => [b, label(b)]);
  if (!x || !y || x === y) {
    box.replaceChildren(el("h2", {}, "Kernel 性能对比"), el("div", { class: "empty" }, "至少需要两个有 kernel 耗时的 NPU 后端。客户端需开启性能测量（--perf）。"));
    return;
  }
  const body = el("div", {}, el("div", { class: "empty" }, "加载中…"));
  box.replaceChildren(el("h2", {}, select(opts, x, v => renderPerf(box, backends, arch, v, y, dim, true, filters)), "vs", select(opts, y, v => renderPerf(box, backends, arch, x, v, dim, true, filters)),
    el("button", { onclick: () => renderPerf(box, backends, arch, y, x, dim, true, filters) }, "⇄ 交换"), "Kernel 性能统计分布", el("span", { class: "spacer" }),
    select(Object.entries(DIM_LABEL).filter(([k]) => k !== "all"), dim, v => renderPerf(box, backends, arch, x, y, v, true, filters))), body);
  try {
    const [all, byDim] = await Promise.all([overviewData("/api/overview/perf" + qs({ x, y, dim: "all", arch, ...filters })), overviewData("/api/overview/perf" + qs({ x, y, dim, arch, ...filters }))]);
    const g = all.groups[0];
    if (!g) { body.replaceChildren(el("div", { class: "empty" }, "两者都有 kernel 耗时的用例为 0。")); return; }
    const tiles = el("div", { class: "tiles" },
      [["可比用例", fmt(g.total)], [`${label(x)} 更快`, pct(g.faster, g.total)], ["持平", pct(g.tie, g.total)], [`${label(x)} 更慢`, pct(g.slower, g.total)],
        ["几何平均耗时比", g.geomean_ratio ? g.geomean_ratio.toFixed(3) + "x" : "—"]].map(([k, v]) => el("div", { class: "tile" }, el("div", { class: "v" }, v), el("div", { class: "k" }, k))));
    body.replaceChildren(
      el("div", { class: "muted small" }, `以 ${label(y)} 为标杆。耗时比 = ${label(x)} / ${label(y)}；≤ 1/(1+t) 记为更快（Pass），≥ 1+t 记为更慢（Failed），t = ${all.tie_threshold}。`),
      tiles,
      el("div", { class: "chart-grid" },
        el("div", { class: "chart" }, el("h3", {}, "耗时比分布（log2）"), histogram(g.histogram)),
        el("div", { class: "chart" }, el("h3", {}, `按${DIM_LABEL[dim]}`), legend([["c-faster", "更快"], ["c-tie", "持平"], ["c-slower", "更慢"]]),
          byDim.groups.slice(0, 14).map(r => el("div", { class: "hbar" }, el("div", { class: "label", title: r.value }, r.value),
            stackBar([["c-faster", r.faster, "更快"], ["c-tie", r.tie, "持平"], ["c-slower", r.slower, "更慢"]], r.total),
            el("div", { class: "muted" }, r.geomean_ratio ? r.geomean_ratio.toFixed(2) + "x" : "—"))))));
  } catch (err) { body.replaceChildren(errorBox(err)); }
}
function histogram(bins) {
  const W = 460, H = 190, pad = 28;
  if (!bins.length) return el("div", { class: "muted" }, "无数据");
  // Zoom to the data (with a margin) instead of the full +-64x range.
  const lo = Math.min(-4, Math.min(...bins.map(b => b.bin)) - 2), hi = Math.max(4, Math.max(...bins.map(b => b.bin)) + 2);
  const max = Math.max(1, ...bins.map(b => b.count)), span = hi - lo + 1, bw = (W - 2 * pad) / span;
  const xs = b => pad + (b - lo) * bw;
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "耗时比分布" });
  for (const b of bins) {
    const h = (H - 2 * pad) * b.count / max;
    const color = b.log2_low < -0.07 ? "var(--ok)" : b.log2_low >= 0.07 ? "var(--bad)" : "var(--tie)";
    root.append(svg("rect", { x: xs(b.bin) + 1, y: H - pad - h, width: Math.max(2, bw - 2), height: h, rx: 2, fill: color,
      "data-tooltip": `${(2 ** b.log2_low).toFixed(2)}x – ${(2 ** (b.log2_low + 0.25)).toFixed(2)}x：${b.count}` }));
  }
  root.append(svg("line", { x1: xs(0), x2: xs(0), y1: pad / 2, y2: H - pad, stroke: "var(--muted)", "stroke-dasharray": "3 3" }));
  for (const v of [-16, -12, -8, -4, 0, 4, 8, 12, 16]) {
    if (v < lo || v > hi) continue;
    const ratio = 2 ** (v / 4);
    root.append(svg("text", { x: xs(v), y: H - 8, "text-anchor": "middle" }, ratio >= 1 ? `${ratio}x` : `1/${1 / ratio}x`));
  }
  return root;
}
