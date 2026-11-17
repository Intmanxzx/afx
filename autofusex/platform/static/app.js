"use strict";
// AutofuseX platform UI. Plain DOM; every server value goes through textContent.

const $ = (sel, root = document) => root.querySelector(sel);
const TOKEN_KEY = "afx.token";
const COLS_KEY = "afx.discovery.columns";
const VIEW_KEY = "afx.discovery.view";
const BACKEND_LABEL = { "cpu": "CPU", "eager-npu": "EagerNPU", "af-legacy": "AF-Legacy", "af-mlir": "AF-MLIR", "triton-npu": "TritonNPU" };
const STAGE_LABEL = { schedule: "Schedule", graph_check: "图校验", mlir: "MLIR", lowering: "Lowering", codegen: "Codegen",
  tiling: "Tiling", autofuse: "AF 其他", first_call_combined: "首次调用", timeout: "超时", unknown: "未知" };
const DIM_LABEL = { all: "全部", op_category: "算子类别", dtype: "数据类型", size_bucket: "张量规模", dynamic: "静态/动态", source: "用例来源" };
const STATUS_LABEL = { open: "未领取", claimed: "修复中", verifying: "验证中", regressing: "回归中", summarizing: "归档中", resolved: "已解决", closed: "已关闭" };
let me = null;

// ---------- DOM helpers ----------
function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "style") Object.assign(node.style, v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}
function svg(tag, attrs, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, String(v));
  for (const c of children.flat()) if (c != null) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}
const fmt = n => n == null ? "—" : Number(n).toLocaleString("zh-CN");
const pct = (a, b) => b ? (100 * a / b).toFixed(1) + "%" : "—";
const label = b => BACKEND_LABEL[b] || b;
const time = ms => ms ? new Date(ms).toLocaleString("zh-CN", { hour12: false }) : "—";
const ns = v => v == null ? "—" : v >= 1e9 ? (v / 1e9).toFixed(2) + "s" : v >= 1e6 ? (v / 1e6).toFixed(1) + "ms" : (v / 1e3).toFixed(1) + "µs";
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};
function token() { try { return sessionStorage.getItem(TOKEN_KEY); } catch { return null; } }
function rkey() { return (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random()).slice(0, 64); }

// ---------- API ----------
class ApiError extends Error { constructor(status, code, message) { super(message); this.status = status; this.code = code; } }
async function api(path, { method = "GET", body, raw = false } = {}) {
  const headers = {};
  if (token()) headers["Authorization"] = "Bearer " + token();
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (raw && res.ok) return res.blob();
  let payload = null;
  try { payload = await res.json(); } catch { /* non JSON */ }
  if (res.status === 401) {
    showLogin(token() ? "登录已失效，请重新输入 token" : "该操作需要登录：请输入管理员离线发放的 token");
    throw new ApiError(401, "unauthorized", "需要登录");
  }
  if (!res.ok || !payload || !payload.ok) {
    const err = (payload && payload.error) || {};
    throw new ApiError(res.status, err.code || "http_error", err.message || res.statusText);
  }
  return payload.data;
}
const qs = obj => { const p = new URLSearchParams(); for (const [k, v] of Object.entries(obj)) if (v !== null && v !== undefined && v !== "") p.set(k, v); const s = p.toString(); return s ? "?" + s : ""; };
function errorBox(err) { return el("div", { class: "error-box" }, err instanceof ApiError ? `${err.message}（${err.status} ${err.code}）` : String(err)); }

// ---------- routing ----------
function route() {
  const raw = location.hash.replace(/^#/, "") || "overview";
  const [page, query] = raw.split("?");
  return { page, params: new URLSearchParams(query || "") };
}
function go(page, params = {}) {
  const current = route().params;
  const merged = new URLSearchParams();
  if (current.get("q") && params.q === undefined) merged.set("q", current.get("q"));
  for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined && v !== "") merged.set(k, v);
  const s = merged.toString();
  location.hash = page + (s ? "?" + s : "");
}
const PAGES = {};
async function render() {
  $("#top").classList.remove("hidden");
  const { page, params } = route();
  for (const a of document.querySelectorAll("#nav a")) a.classList.toggle("active", a.dataset.page === page);
  $("#search").value = params.get("q") || "";
  closeDrawer();
  hideActionBar();
  const main = $("#main");
  const q = params.get("q");
  const content = el("div");
  const loading = el("div", { class: "empty" }, "加载中…");
  if (q) content.append(el("div", { class: "filter-note" }, "已过滤：", el("span", { class: "mono" }, q),
    el("button", { onclick: () => go(page, { q: null }) }, "清除")));
  // Attach before rendering so canvases and widths measure real layout.
  main.replaceChildren(loading, content);
  const ticket = ++renderToken;
  try {
    await (PAGES[page] || PAGES.overview)(content, params);
  } catch (err) {
    if (!(err instanceof ApiError && err.status === 401)) content.append(errorBox(err));
  }
  if (ticket === renderToken) loading.remove();
}
let renderToken = 0;

// ---------- login ----------
function showLogin(message) {
  closeDrawer();
  hideActionBar();
  const input = el("textarea", { placeholder: "afx_…", "aria-label": "token" });
  const msg = el("div", { class: "muted small" }, message || "使用管理员离线发放的 token 登录。token 只保存在当前标签页。");
  const submit = async () => {
    const value = input.value.trim();
    if (!value.startsWith("afx_")) { msg.textContent = "token 应以 afx_ 开头"; return; }
    try { sessionStorage.setItem(TOKEN_KEY, value); } catch { msg.textContent = "浏览器禁止会话存储"; return; }
    try { await loadMe(); render(); } catch (err) { try { sessionStorage.removeItem(TOKEN_KEY); } catch { /* ignore */ } msg.textContent = "登录失败：" + err.message; }
  };
  $("#main").replaceChildren(el("div", { class: "panel login" }, el("h2", {}, "登录 AutofuseX 平台"), input, msg,
    el("div", { class: "toolbar", style: { marginTop: "10px" } }, el("button", { class: "primary", onclick: submit }, "登录"),
      el("button", { onclick: () => render() }, "返回浏览"))));
}
async function loadMe() {
  me = await api("/api/me");
  updateWho();
}
function updateWho() {
  $("#who").textContent = me ? `${me.user_id} · ${me.role === "admin" ? "管理员" : me.role}` : "访客（只读）";
  $("#logout").textContent = me ? "退出" : "登录";
}
const isAdmin = () => me && me.role === "admin";

// ---------- drawer & action bar ----------
function openDrawer(...children) {
  const d = $("#drawer");
  d.replaceChildren(el("button", { class: "close", onclick: closeDrawer }, "关闭"), ...children);
  d.classList.remove("hidden");
  d.scrollTop = 0;
}
function closeDrawer() { $("#drawer").classList.add("hidden"); }
function showActionBar(...children) { const b = $("#actionbar"); b.replaceChildren(...children); b.classList.remove("hidden"); }
function hideActionBar() { $("#actionbar").classList.add("hidden"); }
let openMenu = null;
function popMenu(anchor, items) {
  if (openMenu) openMenu.remove();
  const rect = anchor.getBoundingClientRect();
  const menu = el("div", { class: "menu", style: { top: (rect.bottom + scrollY) + "px", left: Math.max(8, rect.right + scrollX - 160) + "px" } },
    items.map(([text, fn]) => el("button", { onclick: () => { menu.remove(); openMenu = null; fn(); } }, text)));
  document.body.append(menu);
  openMenu = menu;
  setTimeout(() => document.addEventListener("click", function off(e) { if (!menu.contains(e.target)) { menu.remove(); openMenu = null; document.removeEventListener("click", off); } }), 0);
}

// ---------- shared widgets ----------
function stackBar(parts, total) {
  const track = el("div", { class: "track" });
  for (const [cls, n, title] of parts) if (n > 0) track.append(el("i", { class: cls, title: `${title} ${fmt(n)} (${pct(n, total)})`, style: { width: (100 * n / total) + "%" } }));
  return track;
}
function legend(items) { return el("div", { class: "legend" }, items.map(([cls, t]) => el("span", {}, el("i", { class: cls }), t))); }
function select(options, value, onchange, attrs = {}) {
  const s = el("select", attrs, options.map(([v, t]) => el("option", { value: v, selected: v === value }, t)));
  s.addEventListener("change", () => onchange(s.value));
  return s;
}
function statusChip(e) {
  if (e.function_status === "pass") {
    if (e.accuracy_status === "fail") return el("span", {}, el("span", { class: "chip warn" }, "精度失败"), e.accuracy_ref === "eager-npu" ? el("span", { class: "float-tag", title: "CPU 不支持，以 EagerNPU 为标杆" }, "EagerNPU") : null);
    return el("span", { class: "chip ok" }, e.accuracy_status === "pass" ? "通过" : "功能通过");
  }
  if (e.function_status === "unsupported") return el("span", { class: "chip gray" }, "不支持");
  const map = { build_fail: "构图失败", compile_fail: "编译失败", runtime_fail: "运行失败", timeout: "超时" };
  return el("span", { class: "chip bad", title: e.fail_stage || "" }, (map[e.function_status] || e.function_status) + (e.fail_stage ? " · " + (STAGE_LABEL[e.fail_stage] || e.fail_stage) : ""));
}
async function archOptions() {
  return (await api("/api/overview/pipeline")).arches;
}

// =====================================================================
// 总览
// =====================================================================
PAGES.overview = async (root, params) => {
  const q = params.get("q");
  const arches = await archOptions();
  const arch = params.get("arch") || arches[0] || "";
  const pipeline = await api("/api/overview/pipeline" + qs({ arch, q }));
  const head = el("h2", {}, "功能与精度汇总", el("span", { class: "spacer" }),
    arches.length ? select(arches.map(a => [a, a]), arch, v => go("overview", { arch: v })) : null);
  root.append(el("section", { class: "panel" }, head, renderPipeline(pipeline, q)));
  const lanes = pipeline.lanes.filter(l => !l.baseline).map(l => l.backend);
  const tabsBox = el("section", { class: "panel" });
  root.append(tabsBox);
  renderBackendTabs(tabsBox, lanes, arch, q, params.get("tab") || lanes.find(b => b === "af-mlir") || lanes[0]);
  const perfBox = el("section", { class: "panel" });
  root.append(perfBox);
  const npuBackends = pipeline.lanes.map(l => l.backend).filter(b => b !== "cpu");
  renderPerf(perfBox, npuBackends, arch, params.get("x") || (npuBackends.includes("af-mlir") ? "af-mlir" : npuBackends[0]),
    params.get("y") || (npuBackends.includes("triton-npu") ? "triton-npu" : npuBackends.find(b => b !== "af-mlir")), params.get("pdim") || "op_category");
};

function renderPipeline(data, q) {
  const lanes = data.lanes.filter(l => !l.baseline);
  if (!lanes.length) return el("div", { class: "empty" }, "暂无执行记录");
  const families = {};
  for (const lane of lanes) if (lane.family) (families[lane.family] = families[lane.family] || []).push(lane);
  const stageNames = [...new Set(lanes.flatMap(l => Object.keys(l.stage_fail)))].sort((a, b) => (b === "schedule") - (a === "schedule") || a.localeCompare(b));
  const grid = el("div", { class: "pipe" });
  const cols = 4 + stageNames.length + 3;
  grid.style.gridTemplateColumns = `minmax(110px,auto) minmax(110px,auto) repeat(${cols - 2}, minmax(96px,auto))`;
  const max = Math.max(1, data.total_cases);
  const link = extra => () => go("discovery", { q: [q, extra].filter(Boolean).join(" ") });
  const cell = (cls, title, n, rowSpan, onclick, row, col) => {
    const c = el("div", { class: "cell " + cls + (n ? "" : " zero"), onclick, title: `${title}: ${fmt(n)}` }, el("b", {}, fmt(n)), el("span", {}, title),
      el("div", { class: "bar", style: { width: Math.max(4, 100 * n / max) + "%" } }));
    c.style.gridRow = `${row} / span ${rowSpan}`;
    c.style.gridColumn = String(col);
    return c;
  };
  grid.append(cell("total", "总用例", data.total_cases, lanes.length, link(""), 1, 1));
  lanes.forEach((lane, i) => {
    const row = i + 1;
    const laneCell = el("div", { class: "lane" }, label(lane.backend));
    laneCell.style.gridRow = String(row);
    laneCell.style.gridColumn = "2";
    grid.append(laneCell);
    let col = 3;
    // shared stages are merged across a family when every member reports the same count
    const fam = lane.family ? families[lane.family] : null;
    const sharedWith = (key, value) => fam && fam.length > 1 && fam.every(m => (key(m) || 0) === value);
    const first = fam && fam[0] === lane;
    const addShared = (key, cls, title, filter) => {
      const value = key(lane) || 0;
      if (sharedWith(key, value)) { if (first) grid.append(cell(cls, `${lane.family.toUpperCase()} ${title}`, value, fam.length, link(filter), row, col)); }
      else grid.append(cell(cls, title, value, 1, link(`backend:${lane.backend} ` + filter), row, col));
      col++;
    };
    addShared(l => l.build_fail, "build", "构图失败", "status:function_fail");
    for (const stage of stageNames) addShared(l => l.stage_fail[stage], "stage", (STAGE_LABEL[stage] || stage) + "失败", `stage:${stage}`);
    const other = lane.function_fail - Object.values(lane.stage_fail).reduce((a, b) => a + b, 0);
    if (other > 0) { grid.append(cell("func", "其他功能失败", other, 1, link(`backend:${lane.backend} status:function_fail`), row, col)); }
    col++;
    grid.append(cell("acc", "精度失败", lane.accuracy_fail, 1, link(`backend:${lane.backend} status:accuracy_fail`), row, col++));
    grid.append(cell("pass", "通过", lane.pass, 1, link(`backend:${lane.backend} status:pass`), row, col++));
    if (lane.unsupported) grid.append(cell("build", "不支持", lane.unsupported, 1, link(`backend:${lane.backend} status:unsupported`), row, col));
  });
  return el("div", {}, el("div", { class: "muted small", style: { marginBottom: "8px" } }, "以 CPU 为精度标杆（CPU 不支持时使用 EagerNPU）。点击任一节点查看对应用例。"), grid);
}

async function renderBackendTabs(box, lanes, arch, q, active) {
  const tabs = el("div", { class: "tabs" });
  const body = el("div");
  box.replaceChildren(el("h2", {}, "基础能力统计分布"), tabs, body);
  const cache = {};
  const show = async backend => {
    for (const b of tabs.children) b.classList.toggle("active", b.dataset.backend === backend);
    if (!cache[backend]) {
      body.replaceChildren(el("div", { class: "empty" }, "加载中…"));
      const dims = (await api("/api/overview/dims")).filter(d => d !== "all");
      const results = await Promise.all(dims.map(dim => api("/api/overview/backend" + qs({ backend, dim, arch, q })).catch(err => ({ dim, error: err }))));
      cache[backend] = el("div", {}, legend([["c-pass", "通过"], ["c-func", "功能失败"], ["c-acc", "精度失败"], ["c-uns", "不支持"]]),
        el("div", { class: "chart-grid" }, results.map(r => statusChart(r, backend))));
    }
    body.replaceChildren(cache[backend]);
  };
  for (const backend of lanes) tabs.append(el("button", { "data-backend": backend, onclick: () => show(backend) }, `${label(backend)} 基础能力`));
  if (active) show(active);
}
function statusChart(result, backend) {
  if (result.error) return el("div", { class: "chart" }, errorBox(result.error));
  const groups = result.groups.slice(0, 14);
  return el("div", { class: "chart" }, el("h3", {}, `${label(backend)} · 按${DIM_LABEL[result.dim] || result.dim}`),
    groups.length ? groups.map(g => el("div", { class: "hbar" }, el("div", { class: "label", title: g.value }, g.value),
      stackBar([["c-pass", g.pass, "通过"], ["c-func", g.function_fail, "功能失败"], ["c-acc", g.accuracy_fail, "精度失败"], ["c-uns", g.unsupported, "不支持"]], g.total),
      el("div", { class: "muted" }, pct(g.pass, g.total)))) : el("div", { class: "empty" }, "无数据"));
}

async function renderPerf(box, backends, arch, x, y, dim) {
  const opts = backends.map(b => [b, label(b)]);
  if (!x || !y || x === y) {
    box.replaceChildren(el("h2", {}, "Kernel 性能对比"), el("div", { class: "empty" }, "至少需要两个有 kernel 耗时的 NPU 后端。客户端需开启性能测量（--perf）。"));
    return;
  }
  const body = el("div", {}, el("div", { class: "empty" }, "加载中…"));
  box.replaceChildren(el("h2", {}, select(opts, x, v => renderPerf(box, backends, arch, v, y, dim)), "vs", select(opts, y, v => renderPerf(box, backends, arch, x, v, dim)),
    el("button", { onclick: () => renderPerf(box, backends, arch, y, x, dim) }, "⇄ 交换"), "Kernel 性能统计分布", el("span", { class: "spacer" }),
    select(Object.entries(DIM_LABEL).filter(([k]) => k !== "all"), dim, v => renderPerf(box, backends, arch, x, y, v))), body);
  try {
    const [all, byDim] = await Promise.all([api("/api/overview/perf" + qs({ x, y, dim: "all", arch })), api("/api/overview/perf" + qs({ x, y, dim, arch }))]);
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
  const W = 420, H = 160, pad = 24;
  const lo = -24, hi = 24, max = Math.max(1, ...bins.map(b => b.count));
  const xs = b => pad + (b - lo) / (hi - lo + 1) * (W - 2 * pad);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "ratio histogram" });
  for (const b of bins) {
    const h = (H - 2 * pad) * b.count / max;
    const color = b.log2_low < -0.07 ? "var(--ok)" : b.log2_low >= 0.07 ? "var(--bad)" : "var(--tie)";
    root.append(svg("rect", { x: xs(b.bin), y: H - pad - h, width: Math.max(2, (W - 2 * pad) / (hi - lo + 1) - 1), height: h, fill: color },
      svg("title", {}, `2^${b.log2_low} 起：${b.count}`)));
  }
  root.append(svg("line", { x1: xs(0), x2: xs(0), y1: pad / 2, y2: H - pad, stroke: "var(--muted)", "stroke-dasharray": "3 3" }));
  for (const [v, t] of [[-16, "1/16x"], [-8, "1/4x"], [0, "1x"], [8, "4x"], [16, "16x"]]) root.append(svg("text", { x: xs(v), y: H - 6, "text-anchor": "middle" }, t));
  return root;
}

// =====================================================================
// 问题发现
// =====================================================================
const COLUMNS = [
  ["user_id", "用户"], ["backend", "后端"], ["npu_arch", "芯片"], ["status", "结果"], ["log_summary", "日志摘要"],
  ["compile_ns", "编译"], ["kernel_ns", "Kernel"], ["e2e_ns", "E2E"], ["run", "批次"], ["created_at", "时间"],
];
PAGES.discovery = async (root, params) => {
  const tabs = el("div", { class: "tabs" },
    el("button", { class: params.get("sub") !== "retests" ? "active" : "", onclick: () => go("discovery", {}) }, "执行记录"),
    el("button", { class: params.get("sub") === "retests" ? "active" : "", onclick: () => go("discovery", { sub: "retests" }) }, "复测记录"));
  root.append(tabs);
  if (params.get("sub") === "retests") return renderRetests(root);
  const q = params.get("q");
  const view = store.get(VIEW_KEY, "expanded");
  const visible = new Set(store.get(COLS_KEY, COLUMNS.map(c => c[0])));
  const panel = el("section", { class: "panel" });
  root.append(panel);
  const selected = new Set();
  const table = el("div", { class: "table-wrap" });
  const more = el("div", { class: "toolbar", style: { marginTop: "10px" } });
  const colpick = el("div", { class: "colpick hidden" }, COLUMNS.map(([k, t]) => el("label", {},
    el("input", { type: "checkbox", checked: visible.has(k), onchange: e => { e.target.checked ? visible.add(k) : visible.delete(k); store.set(COLS_KEY, [...visible]); redraw(); } }), t)));
  panel.append(el("h2", {}, "执行记录", el("span", { class: "muted small" }, "按用例分组，组内按时间倒序，同色为同一批次"), el("span", { class: "spacer" }),
    select([["expanded", "展开视图"], ["pivot", "后端透视"]], view, v => { store.set(VIEW_KEY, v); render(); }),
    el("button", { onclick: () => colpick.classList.toggle("hidden") }, "显示列")), colpick, table, more);
  let groups = [];
  let cursor = null;
  const load = async () => {
    const page = await api("/api/cases" + qs({ q, after: cursor, limit: 30 }));
    groups = groups.concat(page.groups);
    cursor = page.next_cursor;
    redraw();
    more.replaceChildren(cursor ? el("button", { onclick: load }, "加载更多") : el("span", { class: "muted small" }, groups.length ? "已全部加载" : ""));
  };
  const updateBar = () => {
    if (!selected.size) return hideActionBar();
    showActionBar(el("span", {}, `已选 ${selected.size} 个用例`),
      el("button", { onclick: () => retestDrawer([...selected], groups) }, "复测"),
      isAdmin() ? el("button", { onclick: () => pinGlobal([...selected]) }, "加入回归集") : null,
      el("button", { onclick: () => { selected.clear(); redraw(); hideActionBar(); } }, "取消"));
  };
  const redraw = () => {
    table.replaceChildren(groups.length ? (view === "pivot" ? pivotTable(groups, selected, updateBar) : expandedTable(groups, visible, selected, updateBar))
      : el("div", { class: "empty" }, "没有匹配的用例"));
  };
  await load();
};

function caseCell(group, selected, updateBar, rowSpan) {
  const c = group.case;
  const box = el("input", { type: "checkbox", checked: selected.has(c.case_key), "aria-label": "选择用例",
    onchange: e => { e.target.checked ? selected.add(c.case_key) : selected.delete(c.case_key); updateBar(); } });
  const menuBtn = el("button", { class: "icon-btn", title: "更多操作", onclick: e => popMenu(e.currentTarget, [
    ["复测", () => retestDrawer([c.case_key], [group])],
    ["复制复现命令", () => navigator.clipboard && navigator.clipboard.writeText(`afx fuzz replay ${c.case_key} --backend af-mlir`)],
    ...(isAdmin() ? [["加入回归集", () => pinGlobal([c.case_key])]] : []),
    ["查看用例详情", () => caseDrawer(group)],
  ]) }, "⋯");
  return el("td", { class: "case-cell", rowspan: rowSpan },
    el("div", {}, box, " ", el("a", { href: "#", class: "mono", onclick: e => { e.preventDefault(); caseDrawer(group); } }, c.case_key.slice(0, 14) + "…"),
      el("span", { class: "row-actions" }, el("button", { class: "icon-btn", title: "复测", onclick: () => retestDrawer([c.case_key], [group]) }, "⟳")), menuBtn),
    el("div", { class: "muted small" }, `${c.case_source} · ${c.is_dynamic ? "动态" : "静态"} · ${c.node_count ?? "?"} 节点 · ${c.dtypes || ""}`),
    el("div", { class: "muted small", title: c.op_types || "" }, (c.op_types || "").split(",").slice(0, 4).map(o => o.replace(/^aten\./, "")).join(", ")));
}
function batchClasses(executions) {
  const order = [];
  for (const e of executions) if (!order.includes(e.run_id)) order.push(e.run_id);
  return id => "bt" + (order.indexOf(id) % 6);
}
function expandedTable(groups, visible, selected, updateBar) {
  const cols = COLUMNS.filter(([k]) => visible.has(k));
  const body = el("tbody");
  for (const group of groups) {
    const rows = group.executions.length ? group.executions : [null];
    const batchOf = batchClasses(group.executions);
    rows.forEach((e, i) => {
      const tr = el("tr", { class: i === 0 ? "group-start" : "" });
      if (i === 0) tr.append(caseCell(group, selected, updateBar, rows.length));
      if (!e) { tr.append(el("td", { colspan: cols.length, class: "muted" }, "暂无执行")); body.append(tr); return; }
      const cls = batchOf(e.run_id);
      for (const [k] of cols) tr.append(el("td", { class: cls + (k === cols[0][0] ? " batch" : "") }, execValue(e, k)));
      body.append(tr);
    });
    if (group.more_executions) body.append(el("tr", {}, el("td", { colspan: cols.length + 1, class: "muted small" }, "仅显示最近 10 条，更多见用例详情")));
  }
  return el("table", { class: "grid" }, el("thead", {}, el("tr", {}, el("th", {}, "用例"), cols.map(([, t]) => el("th", {}, t)))), body);
}
function execValue(e, k) {
  switch (k) {
    case "backend": return el("span", {}, label(e.backend), e.is_head ? null : el("span", { class: "muted small", title: "非当前结果（历史/复测/验证）" }, " ·历史"), e.stale_graph ? el("span", { class: "chip gray" }, "旧图") : null);
    case "status": return statusChip(e);
    case "log_summary": return el("span", { class: "small mono", title: e.log_summary || "" }, (e.log_summary || "").slice(0, 80));
    case "compile_ns": case "kernel_ns": case "e2e_ns": return ns(e[k]);
    case "run": return el("span", {}, `#${e.run_id}`, e.run_kind !== "fuzz" ? el("span", { class: "chip info", style: { marginLeft: "4px" } }, { retest: "复测", verify: "验证", regress: "回归", import: "导入" }[e.run_kind] || e.run_kind) : null);
    case "created_at": return el("span", { class: "small" }, time(e.created_at));
    case "npu_arch": return e.npu_arch || "—";
    default: return e[k] ?? "—";
  }
}
function pivotTable(groups, selected, updateBar) {
  const backends = [...new Set(groups.flatMap(g => g.executions.map(e => e.backend)))].sort();
  const body = el("tbody");
  for (const group of groups) {
    const byRun = new Map();
    for (const e of group.executions) { const k = e.run_id; if (!byRun.has(k)) byRun.set(k, {}); byRun.get(k)[e.backend] = byRun.get(k)[e.backend] || e; }
    const runs = [...byRun.entries()];
    const batchOf = batchClasses(group.executions);
    (runs.length ? runs : [[null, {}]]).forEach(([run, cells], i) => {
      const tr = el("tr", { class: i === 0 ? "group-start" : "" });
      if (i === 0) tr.append(caseCell(group, selected, updateBar, Math.max(1, runs.length)));
      const any = Object.values(cells)[0];
      tr.append(el("td", { class: run ? batchOf(run) + " batch" : "" }, run ? `#${run}` : "—", any ? el("div", { class: "muted small" }, `${any.user_id} · ${time(any.created_at)}`) : null));
      for (const b of backends) { const e = cells[b]; tr.append(el("td", { class: run ? batchOf(run) : "" }, e ? el("div", {}, statusChip(e), el("div", { class: "muted small" }, e.kernel_ns ? "kernel " + ns(e.kernel_ns) : "")) : "—")); }
      body.append(tr);
    });
  }
  return el("table", { class: "grid" }, el("thead", {}, el("tr", {}, el("th", {}, "用例"), el("th", {}, "批次"), backends.map(b => el("th", {}, label(b))))), body);
}
async function caseDrawer(group) {
  const c = group.case;
  const execs = el("div");
  openDrawer(el("h2", {}, "用例详情"), el("div", { class: "kv" },
    [["用例 ID", el("span", { class: "mono" }, c.case_key)], ["来源", c.case_source], ["版本", c.case_revision], ["节点数", c.node_count ?? "—"], ["算子", c.op_types || "—"], ["数据类型", c.dtypes || "—"],
      ["最大维度", fmt(c.max_dim)], ["最大张量元素", fmt(c.max_numel)], ["首次出现", time(c.first_seen_at)]].flat().map(v => el("div", {}, v))), el("h3", {}, "执行记录"), execs);
  for (const e of group.executions) {
    const box = el("div", { class: "panel" }, el("div", {}, label(e.backend), " ", statusChip(e), el("span", { class: "muted small" }, ` · ${e.user_id} · ${time(e.created_at)} · 批次 #${e.run_id}`)));
    execs.append(box);
    if (e.function_status !== "pass" || e.accuracy_status === "fail") {
      api("/api/executions/" + encodeURIComponent(e.exec_uuid)).then(d => {
        box.append(el("div", { class: "kv small" }, ["CANN", d.env.cann, "后端版本", d.env.backend_version, "afx", d.env.afx, "torch", d.env.torch, "OS", d.env.os].map(v => el("div", {}, v ?? "—"))));
        if (d.log) box.append(el("pre", { class: "log" }, d.log));
        else if (d.log_missing) box.append(el("div", { class: "muted small" }, "日志不在当前数据中"));
      }).catch(err => box.append(errorBox(err)));
    }
  }
}
function retestDrawer(keys, groups) {
  const known = new Set(groups.flatMap(g => g.executions.map(e => e.backend)).filter(b => b !== "cpu"));
  const backends = [...(known.size ? known : new Set(["af-mlir"]))];
  const checks = backends.map(b => el("label", {}, el("input", { type: "checkbox", value: b, checked: b.startsWith("af-") }), " ", label(b)));
  const reps = el("input", { type: "number", min: 1, max: 5, value: 1, style: { width: "80px" } });
  const out = el("div");
  const submit = async () => {
    const chosen = checks.map(c => c.querySelector("input")).filter(i => i.checked).map(i => i.value);
    if (!chosen.length) { out.replaceChildren(errorBox("至少选择一个后端")); return; }
    try {
      const res = await api("/api/retests", { method: "POST", body: { request_key: rkey(), case_keys: keys, backends: chosen, repetitions: Number(reps.value) || 1 } });
      out.replaceChildren(el("div", { class: "chip ok" }, `已创建 ${res.tasks} 个复测任务`), res.skipped_without_arch ? el("div", { class: "muted small" }, `${res.skipped_without_arch} 个用例没有该后端的执行记录，无法确定芯片`) : null,
        el("div", { class: "toolbar" }, el("button", { onclick: () => go("discovery", { sub: "retests" }) }, "查看复测记录")));
    } catch (err) { out.replaceChildren(errorBox(err)); }
  };
  openDrawer(el("h2", {}, `复测 ${keys.length} 个用例`),
    el("div", { class: "muted small" }, "任务派发给对应芯片上在线的客户端，复测结果会作为新批次出现在原用例下，并更新当前结果。"),
    el("div", { class: "form-grid", style: { marginTop: "12px" } }, el("div", {}, "后端"), el("div", { class: "toolbar" }, checks), el("div", {}, "重复次数"), reps),
    el("div", { class: "toolbar", style: { marginTop: "12px" } }, el("button", { class: "primary", onclick: submit }, "提交复测")), out);
}
async function pinGlobal(keys) {
  const reason = prompt(`把 ${keys.length} 个用例加入全局必选回归集，原因：`);
  if (reason === null) return;
  try { const r = await api("/api/regression/pins", { method: "POST", body: { request_key: rkey(), scope: "global", case_keys: keys, reason } }); alert(`已加入 ${r.pinned} 个用例`); }
  catch (err) { alert(err.message); }
}
async function renderRetests(root) {
  const rows = await api("/api/retests");
  root.append(el("section", { class: "panel" }, el("h2", {}, "我提交的复测"),
    rows.length ? el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["批次", "提交时间", "任务", "进行中", "通过", "失败", "无效", "进度"].map(t => el("th", {}, t)))),
      el("tbody", {}, rows.map(r => el("tr", {}, el("td", { class: "mono small" }, (r.batch || "").slice(0, 12)), el("td", {}, time(r.created_at)), el("td", {}, r.tasks), el("td", {}, r.running),
        el("td", {}, r.passed), el("td", {}, r.failed), el("td", {}, r.invalid),
        el("td", { style: { minWidth: "160px" } }, el("div", { class: "hbar", style: { gridTemplateColumns: "1fr" } }, stackBar([["c-pass", r.passed, "通过"], ["c-func", r.failed, "失败"], ["c-uns", r.invalid, "无效"], ["c-acc", r.running, "进行中"]], r.tasks)))))))
      : el("div", { class: "empty" }, "还没有复测记录。在执行记录中选中用例后点击“复测”。")));
}

// =====================================================================
// 修复闭环
// =====================================================================
const STATIONS = [["claimed", "领取"], ["verifying", "验证"], ["regressing", "回归"], ["summarizing", "总结"], ["resolved", "关闭"]];
function subway(issue) {
  const index = { open: -1, claimed: 0, verifying: 1, regressing: 2, summarizing: 3, resolved: 4, closed: 4 }[issue.status];
  const lastRound = (issue.rounds || [])[issue.rounds ? issue.rounds.length - 1 : 0];
  const bounced = issue.status === "claimed" && issue.round_no > 0;
  const box = el("div", { class: "subway", title: STATUS_LABEL[issue.status] + (bounced ? `（第 ${issue.round_no} 轮未通过，退回修复）` : "") });
  STATIONS.forEach(([key, text], i) => {
    if (i) box.append(el("div", { class: "line" + (i <= index ? " done" : "") }));
    let cls = i < index ? "done" : i === index ? (issue.status === "resolved" || issue.status === "closed" ? "done" : "now") : "";
    if (bounced && i === 1) cls = "back";
    box.append(el("div", { class: "st " + cls }, el("div", { class: "dot" }), issue.status === "closed" && i === 4 ? "关闭" : text));
  });
  void lastRound;
  return box;
}
PAGES.repair = async (root, params) => {
  const dimension = params.get("dimension") || "function";
  const status = params.get("status") || "";
  const tabs = el("div", { class: "tabs" }, [["function", "功能问题"], ["accuracy", "精度问题"], ["performance", "性能问题"]].map(([k, t]) =>
    el("button", { class: k === dimension ? "active" : "", onclick: () => go("repair", { dimension: k, status }) }, t)));
  const filter = select([["", "全部状态"], ...Object.entries(STATUS_LABEL)], status, v => go("repair", { dimension, status: v }));
  const panel = el("section", { class: "panel" }, el("h2", {}, "问题列表", el("span", { class: "muted small" }, "每行是一类问题，Agent 一次领取一类"), el("span", { class: "spacer" }), filter), tabs);
  root.append(panel);
  const body = el("tbody");
  const table = el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["问题", "后端", "用例数", "状态", "领取人", "更新时间"].map(t => el("th", {}, t)))), body);
  const more = el("div", { class: "toolbar", style: { marginTop: "10px" } });
  panel.append(el("div", { class: "table-wrap" }, table), more);
  let cursor = null;
  const load = async () => {
    const page = await api("/api/issues" + qs({ dimension, status, after: cursor, limit: 50 }));
    cursor = page.next_cursor;
    for (const issue of page.issues) body.append(el("tr", {},
      el("td", {}, el("a", { href: "#", onclick: e => { e.preventDefault(); issueDrawer(issue.issue_id); } }, issue.issue_id), el("div", { class: "small", title: issue.title }, issue.title.slice(0, 110))),
      el("td", {}, label(issue.backend)), el("td", {}, fmt(issue.case_count)), el("td", {}, subway(issue)),
      el("td", { class: "small" }, issue.owner_user ? `${issue.owner_user} / ${issue.agent_id}` : "—"), el("td", { class: "small" }, time(issue.updated_at))));
    if (!body.children.length) body.append(el("tr", {}, el("td", { colspan: 6, class: "empty" }, "暂无问题")));
    more.replaceChildren(cursor ? el("button", { onclick: load }, "加载更多") : "");
  };
  await load();
};
async function issueDrawer(issueId) {
  openDrawer(el("div", { class: "empty" }, "加载中…"));
  const [issue, cases, regression] = await Promise.all([api("/api/issues/" + issueId), api(`/api/issues/${issueId}/cases`), api(`/api/issues/${issueId}/cases?role=regression`)]);
  const rounds = el("div");
  for (const r of issue.rounds) {
    const box = el("div", { class: "panel" }, el("div", {}, `第 ${r.round_no} 轮 · ${r.kind === "verify" ? "验证" : "回归"} · 版本 ${r.fix_version} · `,
      el("span", { class: "chip " + ({ passed: "ok", failed: "bad", invalid: "warn" }[r.outcome] || "info") }, { passed: "通过", failed: "失败", invalid: "无效" }[r.outcome] || "进行中"),
      el("span", { class: "muted small" }, ` 通过 ${r.pass_count} · 失败 ${r.fail_count} · 无效 ${r.invalid_count}`)));
    const tasks = el("div");
    box.append(el("button", { class: "icon-btn", onclick: async () => {
      const list = await api(`/api/issues/${issueId}/rounds/${r.round_no}/tasks`);
      tasks.replaceChildren(el("table", { class: "grid" }, el("tbody", {}, list.map(t => el("tr", {}, el("td", { class: "mono small" }, t.case_key.slice(0, 18) + "…"), el("td", {}, t.npu_arch), el("td", {}, t.result || t.status), el("td", { class: "small muted" }, t.detail || ""))))));
    } }, "查看任务"), tasks);
    rounds.append(box);
  }
  const adminBox = isAdmin() && !["resolved", "closed"].includes(issue.status) ? el("div", { class: "toolbar" },
    el("button", { class: "danger", onclick: async () => { const reason = prompt("关闭原因（不修复/误报/重复）："); if (!reason) return; try { await api(`/api/admin/issues/${issueId}/close`, { method: "POST", body: { request_key: rkey(), reason } }); issueDrawer(issueId); } catch (err) { alert(err.message); } } }, "关闭问题"),
    el("button", { onclick: async () => { const into = prompt("合并到哪个问题 ID？"); if (!into) return; try { await api(`/api/admin/issues/${issueId}/merge`, { method: "POST", body: { request_key: rkey(), into: into.trim() } }); issueDrawer(issueId); } catch (err) { alert(err.message); } } }, "合并到…")) : null;
  openDrawer(el("h2", {}, issue.issue_id, " ", el("span", { class: "chip info" }, STATUS_LABEL[issue.status])), el("div", {}, issue.title), subway(issue),
    el("div", { class: "kv", style: { marginTop: "12px" } }, [["维度", { function: "功能", accuracy: "精度", performance: "性能" }[issue.dimension]], ["后端", label(issue.backend)], ["分类", issue.category],
      ["用例数", fmt(issue.case_count)], ["芯片", issue.arches.join(", ") || "—"], ["领取", issue.owner_user ? `${issue.owner_user} / ${issue.agent_id}` : "—"],
      ["修复版本", issue.fix_version || "—"], ["代码链接", issue.fix_ref ? el("a", { href: issue.fix_ref, target: "_blank", rel: "noopener noreferrer" }, issue.fix_ref) : "—"],
      ["知识", issue.knowledge_id ? el("a", { href: "#", onclick: e => { e.preventDefault(); knowledgeDrawer(issue.knowledge_id); } }, issue.knowledge_id) : "—"],
      ["首次出现", time(issue.first_seen_at)]].flat().map(v => el("div", {}, v))),
    issue.log_summary ? el("pre", { class: "log" }, issue.log_summary) : null, adminBox,
    el("h3", {}, "验证与回归"), issue.rounds.length ? rounds : el("div", { class: "muted" }, "尚未提交验证"),
    el("h3", {}, `涉及用例（${cases.cases.length}${cases.next_cursor ? "+" : ""}）`), caseList(cases.cases),
    regression.cases.length ? el("div", {}, el("h3", {}, `回归用例（${regression.cases.length}）`), caseList(regression.cases)) : null,
    el("h3", {}, "事件"), el("ul", { class: "timeline" }, issue.events.map(e => el("li", {}, `${time(e.at)} · ${e.actor} · ${e.kind}`, e.to_status ? ` → ${STATUS_LABEL[e.to_status] || e.to_status}` : "", e.detail ? el("div", { class: "muted" }, e.detail) : null))));
}
function caseList(cases) {
  return el("table", { class: "grid" }, el("tbody", {}, cases.map(c => el("tr", {},
    el("td", {}, el("a", { href: "#", class: "mono small", onclick: e => { e.preventDefault(); go("discovery", { q: `${c.case_key.slice(0, 40)}` }); } }, c.case_key.slice(0, 22) + "…")),
    el("td", {}, c.npu_arch), el("td", {}, c.pick_reason || ""), el("td", {}, c.stale ? el("span", { class: "chip gray" }, "旧图") : ""),
    el("td", {}, c.result ? el("span", { class: "chip " + (c.result === "pass" ? "ok" : c.result === "fail" ? "bad" : "warn") }, c.result) : "")))));
}

// =====================================================================
// 知识库
// =====================================================================
const KB_KIND = { bugfix_function: "功能修复", bugfix_accuracy: "精度修复", perf_tuning: "性能调优", fusion_pattern: "融合模式", capability: "能力画像", best_practice: "最佳实践", pitfall: "易错点" };
const KB_CONF = { verified: ["已验证", "ok"], measured: ["统计得出", "info"], proposed: ["待审核", "warn"] };
PAGES.knowledge = async (root, params) => {
  const filters = { q: params.get("kq") || "", kind: params.get("kind") || "", op: params.get("op") || "", backend: params.get("backend") || "", confidence: params.get("confidence") || "" };
  const apply = () => go("knowledge", { kq: filters.q, kind: filters.kind, op: filters.op, backend: filters.backend, confidence: filters.confidence });
  const qInput = el("input", { value: filters.q, placeholder: "关键词，如 tiling、融合", onchange: e => { filters.q = e.target.value.trim(); apply(); } });
  const opInput = el("input", { value: filters.op, placeholder: "算子，如 aten.add.Tensor", onchange: e => { filters.op = e.target.value.trim(); apply(); } });
  const side = el("div", { class: "panel" }, el("h2", {}, "筛选"), el("div", { class: "form-grid", style: { gridTemplateColumns: "1fr" } }, qInput,
    select([["", "全部类型"], ...Object.entries(KB_KIND)], filters.kind, v => { filters.kind = v; apply(); }),
    select([["", "全部后端"], ...Object.entries(BACKEND_LABEL)], filters.backend, v => { filters.backend = v; apply(); }),
    select([["", "全部可信度"], ...Object.entries(KB_CONF).map(([k, [t]]) => [k, t])], filters.confidence, v => { filters.confidence = v; apply(); }), opInput),
    el("div", { class: "toolbar", style: { marginTop: "12px" } }, el("button", { onclick: exportKb }, "导出 JSONL")),
    el("div", { class: "muted small" }, "Agent 检索：GET /api/knowledge/search?q=…&op=…&backend=…"));
  const list = el("div");
  root.append(el("div", { class: "kb-layout" }, side, el("section", { class: "panel" }, el("h2", {}, "知识条目"), list)));
  const page = await api("/api/knowledge/search" + qs({ ...filters, limit: 50 }));
  list.replaceChildren(...(page.items.length ? page.items.map(k => el("div", { class: "kb-card", onclick: () => knowledgeDrawer(k.knowledge_id) },
    el("div", { class: "t" }, k.title), el("div", { class: "tags" }, el("span", { class: "chip info" }, KB_KIND[k.kind] || k.kind),
      el("span", { class: "chip " + (KB_CONF[k.confidence] || ["", "gray"])[1] }, (KB_CONF[k.confidence] || [k.confidence])[0]),
      k.status !== "published" ? el("span", { class: "chip gray" }, k.status) : null,
      ...k.backends.map(b => el("span", { class: "chip gray" }, label(b))), ...k.ops.slice(0, 4).map(o => el("span", { class: "chip gray mono" }, o)),
      k.evidence ? el("span", { class: "muted small" }, `来自 ${k.evidence.issue_id} · ${fmt(k.evidence.case_count)} 个用例`) : null)))
    : [el("div", { class: "empty" }, "没有匹配的知识。问题回归通过后，Agent 提交的总结会出现在这里。")]));
};
async function exportKb() {
  try {
    const blob = await api("/api/knowledge/export", { raw: true });
    const a = el("a", { href: URL.createObjectURL(blob), download: "afx-knowledge.jsonl" });
    document.body.append(a); a.click(); a.remove();
  } catch (err) { alert(err.message); }
}
async function knowledgeDrawer(id) {
  const k = await api("/api/knowledge/" + id);
  const adminBox = isAdmin() ? el("div", { class: "toolbar" }, ["published", "draft", "deprecated"].filter(s => s !== k.status).map(s =>
    el("button", { onclick: async () => { try { await api(`/api/admin/knowledge/${id}/status`, { method: "POST", body: { request_key: rkey(), status: s } }); knowledgeDrawer(id); } catch (err) { alert(err.message); } } },
      { published: "发布", draft: "退回草稿", deprecated: "标记失效" }[s]))) : null;
  openDrawer(el("h2", {}, k.title), el("div", { class: "tags" }, el("span", { class: "chip info" }, KB_KIND[k.kind] || k.kind), el("span", { class: "chip " + (KB_CONF[k.confidence] || ["", "gray"])[1] }, (KB_CONF[k.confidence] || [k.confidence])[0]), el("span", { class: "chip gray" }, k.status)),
    adminBox,
    el("div", { class: "kv", style: { marginTop: "12px" } }, [["知识 ID", k.knowledge_id], ["关键词", k.keywords.join(", ")], ["算子", k.ops.join(", ") || "—"], ["数据类型", k.dtypes.join(", ") || "—"],
      ["后端", k.backends.map(label).join(", ") || "—"], ["适用版本", k.version_range || "—"], ["作者", `${k.author_user}${k.agent_id ? " / " + k.agent_id : ""}`], ["更新", time(k.updated_at)]].flat().map(v => el("div", {}, v))),
    ...[["现象", k.symptom], ["根因", k.root_cause], ["解决方案", k.solution], ["适用范围", k.applicability], ["子图模式", k.graph_pattern]].filter(([, v]) => v).map(([t, v]) => el("div", {}, el("h3", {}, t), el("div", { style: { whiteSpace: "pre-wrap" } }, v))),
    k.evidence ? el("div", {}, el("h3", {}, "证据"), el("pre", { class: "log" }, JSON.stringify(k.evidence, null, 2))) : null,
    el("h3", {}, "关联"), el("div", {}, k.links.map(l => l.target_kind === "issue" ? el("a", { href: "#", onclick: e => { e.preventDefault(); issueDrawer(l.target_id); }, style: { marginRight: "10px" } }, l.target_id) : el("span", { class: "mono small", style: { marginRight: "10px" } }, l.target_id.slice(0, 16) + "…"))));
}

// =====================================================================
// 能力画像
// =====================================================================
PAGES.profile = async (root, params) => {
  const arches = await archOptions();
  const arch = params.get("arch") || arches[0] || "";
  const pipeline = await api("/api/overview/pipeline" + qs({ arch }));
  const backends = pipeline.lanes.map(l => l.backend).filter(b => b !== "cpu");
  const x = params.get("x") || (backends.includes("af-mlir") ? "af-mlir" : backends[0]);
  const y = params.get("y") || (backends.includes("triton-npu") ? "triton-npu" : backends.find(b => b !== x));
  const axes = await api("/api/profile/axes");
  const fx = params.get("fx") || (axes.includes("node_count") ? "node_count" : axes[0]);
  const fy = params.get("fy") || (axes.includes("max_numel") ? "max_numel" : axes[1] || axes[0]);
  const set = patch => go("profile", { arch, x, y, fx, fy, ...patch });
  const opts = backends.map(b => [b, label(b)]);
  const head = el("h2", {}, select(opts, x, v => set({ x: v })), "相对", select(opts, y, v => set({ y: v })), "的 Kernel 性能画像", el("span", { class: "spacer" }),
    arches.length ? select(arches.map(a => [a, a]), arch, v => set({ arch: v })) : null);
  const canvas = el("canvas", { class: "scatter", "aria-label": "性能点云" });
  const info = el("div", { class: "muted small" });
  const side = el("div");
  root.append(el("section", { class: "panel" }, head,
    el("div", { class: "toolbar" }, "横轴", select(axes.map(a => [a, a]), fx, v => set({ fx: v })), "纵轴", select(axes.map(a => [a, a]), fy, v => set({ fy: v })),
      el("span", { style: { width: "160px" } }, el("div", { class: "gradient" }), el("div", { class: "legend" }, `${label(x)} 更快`, el("span", { class: "spacer", style: { flex: 1 } }), `更慢`))),
    el("div", { class: "profile-layout" }, el("div", {}, canvas, info), side)));
  if (!x || !y || x === y) { info.textContent = "至少需要两个后端都有 kernel 耗时的用例。"; return; }
  try {
    const [pts, pat] = await Promise.all([api("/api/profile/points" + qs({ x, y, fx, fy, arch })), api("/api/profile/patterns" + qs({ x, y, arch }))]);
    drawScatter(canvas, pts, info);
    side.append(el("div", { class: "panel" }, el("h2", {}, "区分度最高的维度"), pat.importance.length ? pat.importance.map(d => el("div", { class: "hbar" }, el("div", {}, DIM_LABEL[d.dim] || d.dim),
      el("div", { class: "track" }, el("i", { class: "c-pass", style: { width: Math.min(100, d.spread_log2 * 100) + "%" } })), el("div", { class: "muted" }, d.spread_log2.toFixed(2)))) : el("div", { class: "muted" }, "数据不足")),
      opPanel(`${label(x)} 优势算子`, pat.best_ops.filter(o => o.log2 < 0)), opPanel(`${label(x)} 劣势算子`, pat.worst_ops.filter(o => o.log2 > 0)),
      isAdmin() ? el("div", { class: "toolbar" }, el("button", { onclick: async () => {
        try { const r = await api("/api/admin/profile/drafts", { method: "POST", body: { request_key: rkey(), x, y, npu_arch: arch || null } }); alert(`生成 ${r.created.length} 条知识草稿（候选 ${r.candidates}），请到知识库审核发布`); }
        catch (err) { alert(err.message); } } }, "生成画像知识草稿")) : null);
  } catch (err) { info.replaceChildren(errorBox(err)); }
};
function opPanel(title, ops) {
  return el("div", { class: "panel" }, el("h2", {}, title), ops.length ? el("table", { class: "grid" }, el("tbody", {}, ops.slice(0, 10).map(o =>
    el("tr", {}, el("td", { class: "mono small" }, o.op.replace(/^aten\./, "")), el("td", {}, o.geomean_ratio.toFixed(2) + "x"), el("td", { class: "muted" }, fmt(o.cases)))))) : el("div", { class: "muted" }, "样本不足"));
}
function drawScatter(canvas, data, info) {
  const dpr = devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const pts = data.points;
  info.textContent = `${fmt(pts.length)} 个点（可比用例 ${fmt(data.total_pairs)}，抽样步长 ${data.sample_step}）。颜色为 log2(耗时比)，点击点查看用例。`;
  if (!pts.length) return;
  const style = getComputedStyle(document.documentElement);
  const logX = Math.max(...pts.map(p => p[0])) > 1000, logY = Math.max(...pts.map(p => p[1])) > 1000;
  const tx = v => logX ? Math.log10(Math.max(1, v)) : v, ty = v => logY ? Math.log10(Math.max(1, v)) : v;
  const xs = pts.map(p => tx(p[0])), ys = pts.map(p => ty(p[1]));
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs) || 1, Math.min(...ys), Math.max(...ys) || 1];
  const pad = 36;
  const px = v => pad + (tx(v) - x0) / ((x1 - x0) || 1) * (W - 2 * pad), py = v => H - pad - (ty(v) - y0) / ((y1 - y0) || 1) * (H - 2 * pad);
  const ok = style.getPropertyValue("--ok").trim(), bad = style.getPropertyValue("--bad").trim(), tie = style.getPropertyValue("--tie").trim();
  ctx.globalAlpha = 0.65;
  for (const p of pts) {
    const r = Math.max(-2, Math.min(2, p[2]));
    ctx.fillStyle = Math.abs(r) < 0.07 ? tie : r < 0 ? ok : bad;
    ctx.globalAlpha = 0.25 + 0.55 * Math.min(1, Math.abs(r) / 2);
    ctx.fillRect(px(p[0]) - 1.5, py(p[1]) - 1.5, 3, 3);
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = style.getPropertyValue("--muted").trim();
  ctx.font = "11px sans-serif";
  ctx.fillText(`${data.fx}${logX ? " (log10)" : ""} →`, W - pad - 140, H - 10);
  ctx.save(); ctx.translate(12, pad + 120); ctx.rotate(-Math.PI / 2); ctx.fillText(`${data.fy}${logY ? " (log10)" : ""} →`, 0, 0); ctx.restore();
  canvas.onclick = async e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = null, bestD = 64;
    for (const p of pts) { const d = (px(p[0]) - mx) ** 2 + (py(p[1]) - my) ** 2; if (d < bestD) { bestD = d; best = p; } }
    if (!best) return;
    const page = await api("/api/cases" + qs({ case_id: best[3] }));
    if (page.groups[0]) caseDrawer(page.groups[0]);
  };
}

// =====================================================================
// 平台运维
// =====================================================================
PAGES.ops = async root => {
  const data = await api("/api/ops/summary");
  const mb = Object.values(data.db_bytes).reduce((a, b) => a + b, 0) / 1048576;
  root.append(el("div", { class: "tiles" }, [["在线客户端", fmt(data.online_clients)], ["近 24 小时执行", fmt(data.executions_24h)], ["待上传", fmt(data.pending_uploads)],
    ["复测队列", `${fmt(data.retest_queue.pending || 0)} 等待 / ${fmt(data.retest_queue.leased || 0)} 执行中`], ["数据库", mb.toFixed(1) + " MB"]].map(([k, v]) =>
    el("div", { class: "tile" }, el("div", { class: "v" }, v), el("div", { class: "k" }, k)))));
  const selected = new Set();
  const version = el("input", { placeholder: "目标 afx 版本，如 " + (data.latest_afx_version || "0.2.0"), style: { width: "220px" } });
  const updateBtn = el("button", { class: "primary", onclick: async () => {
    if (!selected.size || !version.value.trim()) return alert("先选择客户端并填写版本");
    try { const r = await api("/api/admin/clients/update", { method: "POST", body: { request_key: rkey(), client_ids: [...selected], version: version.value.trim() } }); alert(`已通知 ${r.requested} 个客户端，下一次心跳时更新`); render(); }
    catch (err) { alert(err.message); } } }, "一键更新");
  root.append(el("section", { class: "panel" }, el("h2", {}, "客户端", el("span", { class: "spacer" }), isAdmin() ? version : null, isAdmin() ? updateBtn : null),
    el("div", { class: "table-wrap" }, el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["", "状态", "用户", "主机", "芯片", "afx", "后端版本", "近 24h 执行", "上传失败", "待上传", "最后心跳"].map(t => el("th", {}, t)))),
      el("tbody", {}, data.clients.map(c => el("tr", {},
        el("td", {}, isAdmin() ? el("input", { type: "checkbox", onchange: e => e.target.checked ? selected.add(c.client_id) : selected.delete(c.client_id) }) : ""),
        el("td", {}, el("span", { class: "chip " + (c.online ? "ok" : "gray") }, c.online ? (c.busy ? "复测中" : "在线") : "离线")),
        el("td", {}, c.user_id), el("td", { class: "small" }, c.hostname || c.machine_id.slice(0, 12)), el("td", {}, c.npu_arch || "—"),
        el("td", {}, c.afx_version || "—", c.afx_version && data.latest_afx_version && c.afx_version !== data.latest_afx_version ? el("span", { class: "chip warn", style: { marginLeft: "4px" } }, "落后") : null,
          c.target_afx_version ? el("span", { class: "chip info", style: { marginLeft: "4px" } }, "→ " + c.target_afx_version) : null),
        el("td", { class: "small" }, Object.entries(c.backend_versions).filter(([, v]) => v).map(([k, v]) => `${label(k)} ${v}`).join(" · ") || "—"),
        el("td", {}, fmt(c.last_24h.executions)), el("td", {}, fmt(c.last_24h.upload_errors)), el("td", {}, fmt(c.pending_uploads)),
        el("td", { class: "small" }, time(c.last_heartbeat_at || c.last_seen_at)))))))));
  root.append(el("section", { class: "panel" }, el("h2", {}, "近 48 小时吞吐（执行/小时）"), hourly(data.hourly)));
  if (isAdmin()) root.append(await settingsPanel());
};
function hourly(series) {
  if (!series.length) return el("div", { class: "empty" }, "暂无心跳数据");
  const W = 900, H = 160, pad = 26, max = Math.max(1, ...series.map(s => s.executions));
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "hourly throughput" });
  const bw = (W - 2 * pad) / 48;
  const last = series[series.length - 1].hour;
  for (const s of series) {
    const i = 47 - (last - s.hour), h = (H - 2 * pad) * s.executions / max;
    root.append(svg("rect", { x: pad + i * bw, y: H - pad - h, width: Math.max(1, bw - 2), height: h, fill: "var(--accent)" }, svg("title", {}, `${new Date(s.hour * 3600000).toLocaleString("zh-CN", { hour12: false })}：${s.executions} 次执行，${s.retests} 次复测`)));
  }
  root.append(svg("text", { x: pad, y: 14 }, `峰值 ${max}/小时`));
  return root;
}
async function settingsPanel() {
  const [settings, pins] = await Promise.all([api("/api/settings"), api("/api/regression/pins")]);
  const fields = [["perf.tie_threshold", "Kernel 持平阈值 t", "number", "0.01"], ["regression.budget", "回归默认预算", "number", "1"], ["regression.max_budget", "单问题回归上限", "number", "1"],
    ["issue.perf_anchor", "性能问题锚点后端", "text"], ["issue.perf_slowdown", "性能问题阈值（倍）", "number", "0.1"], ["issue.lease_minutes", "问题租约（分钟）", "number", "1"], ["verify.max_cases", "单轮验证用例上限", "number", "1"]];
  const inputs = {};
  const grid = el("div", { class: "form-grid" });
  for (const [key, text, type, step] of fields) { inputs[key] = el("input", { type, step, value: settings[key].value }); grid.append(el("div", {}, text), inputs[key]); }
  const shares = settings["regression.shares"].value;
  const shareInputs = {};
  grid.append(el("div", {}, "回归四层占比 %"), el("div", { class: "toolbar" }, Object.entries({ neighbor: "邻域", sentinel: "哨兵", history: "历史", random: "随机" }).map(([k, t]) => {
    shareInputs[k] = el("input", { type: "number", min: 0, max: 100, value: shares[k], style: { width: "70px" } }); return el("label", {}, t, " ", shareInputs[k]); })));
  const out = el("div");
  const save = async () => {
    const changes = {}, base = {};
    for (const [key, , type] of fields) { const v = type === "number" ? Number(inputs[key].value) : inputs[key].value.trim(); if (JSON.stringify(v) !== JSON.stringify(settings[key].value)) { changes[key] = v; base[key] = settings[key].revision; } }
    const newShares = Object.fromEntries(Object.entries(shareInputs).map(([k, i]) => [k, Number(i.value)]));
    if (JSON.stringify(newShares) !== JSON.stringify(shares)) { changes["regression.shares"] = newShares; base["regression.shares"] = settings["regression.shares"].revision; }
    if (!Object.keys(changes).length) { out.replaceChildren(el("span", { class: "muted" }, "没有改动")); return; }
    try { await api("/api/admin/settings", { method: "PUT", body: { changes, base_revisions: base } }); out.replaceChildren(el("span", { class: "chip ok" }, "已保存")); }
    catch (err) { out.replaceChildren(errorBox(err)); }
  };
  return el("section", { class: "panel" }, el("h2", {}, "设置（管理员）"), grid, el("div", { class: "toolbar", style: { marginTop: "12px" } }, el("button", { class: "primary", onclick: save }, "保存"), out),
    el("h3", {}, `全局与特征必选回归用例（${pins.length}）`),
    pins.length ? el("table", { class: "grid" }, el("tbody", {}, pins.map(p => el("tr", {}, el("td", {}, p.scope === "global" ? "全局" : p.scope_value), el("td", { class: "mono small" }, p.case_key.slice(0, 22) + "…"),
      el("td", { class: "small" }, p.reason || ""), el("td", { class: "small" }, p.added_by), el("td", {}, el("button", { class: "danger", onclick: async () => {
        try { await api("/api/regression/pins/" + p.pin_id, { method: "DELETE", body: { request_key: rkey() } }); render(); } catch (err) { alert(err.message); } } }, "移除")))))) : el("div", { class: "muted" }, "在问题发现页选中用例后可加入回归集"));
}

// =====================================================================
// 整网分析（预留）
// =====================================================================
PAGES.models = async root => {
  root.append(el("section", { class: "panel" }, el("h2", {}, "整网分析"), el("div", { class: "empty" },
    "规划中：执行完整推荐类模型与 LLM，接入 autofusex 的图转储、精度对比与 HTML 报告。接口已预留（/api/models/jobs），本期返回 501。")));
};

// ---------- boot ----------
window.addEventListener("hashchange", render);
$("#search-form").addEventListener("submit", e => { e.preventDefault(); const { page } = route(); go(page === "models" ? "discovery" : page, { q: $("#search").value.trim() || null }); });
$("#logout").addEventListener("click", () => {
  if (!me) return showLogin();
  try { sessionStorage.removeItem(TOKEN_KEY); } catch { /* ignore */ }
  me = null; updateWho(); render();
});
if (matchMedia("(hover: none)").matches) document.body.classList.add("touch");
(async () => {
  updateWho();
  if (token()) { try { await loadMe(); } catch { me = null; updateWho(); } }
  render();
})();
