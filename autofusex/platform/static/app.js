"use strict";
// AutofuseX platform UI. Plain DOM; every server value goes through textContent.

const $ = (sel, root = document) => root.querySelector(sel);
const TOKEN_KEY = "afx.token";
const COLS_KEY = "afx.discovery.columns";
const VIEW_KEY = "afx.discovery.view";
const BACKEND_LABEL = { "cpu": "CPU", "eager-npu": "EagerNPU", "af-legacy": "Legacy", "af-mlir": "MLIR", "triton-npu": "TritonNPU" };
const STAGE_LABEL = { schedule: "Schedule", graph_check: "图校验", mlir: "MLIR", lowering: "Lowering", codegen: "Codegen",
  tiling: "Tiling", autofuse: "AF 其他", first_call_combined: "首次调用", timeout: "超时", unknown: "未知" };
const DIM_LABEL = { all: "全部", op_category: "算子类别", op_display: "算子覆盖", dtype: "数据类型", dim_bucket: "最大维度", size_bucket: "张量规模", dynamic: "静态/动态", source: "用例来源" };
const DIM_ORDER = ["op_display", "dtype", "dim_bucket", "size_bucket", "op_category", "dynamic", "source"];
const STATUS_LABEL = { open: "未领取", claimed: "修复中", verifying: "验证中", regressing: "回归中", summarizing: "归档中", resolved: "已解决", closed: "已关闭" };
let me = null;

// ---------- global search field catalog (mirrors autofusex/platform/search.py) ----------
const SEARCH_RESULT_LABEL = { pass: "通过", fail: "失败（功能或精度）", function_fail: "功能失败", accuracy_fail: "精度失败", unsupported: "不支持" };
const SEARCH_SOURCE_LABEL = { nnsmith: "NNSmith 随机生成", baize_tf: "Baize TF 导入", baize_torch: "Baize Torch 导入", model: "整网模型导入" };
const SEARCH_DYNAMIC_LABEL = { dynamic: "动态 Shape", static: "静态 Shape" };
let searchFieldValues = {};  // op/dtype/category/size/dim/arch/user, fetched lazily from /api/search/fields
let searchFieldValuesPromise = null;
function loadSearchFieldValues() {
  if (!searchFieldValuesPromise) searchFieldValuesPromise = api("/api/search/fields").then(v => { searchFieldValues = v; }).catch(() => {});
  return searchFieldValuesPromise;
}
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
const archLabel = a => { if (!a) return a || "—"; const s = String(a).toLowerCase(); return s.includes("910") ? "A2A3" : s.includes("950") ? "A5" : a; };
const SEARCH_FIELDS = [
  { name: "backend", label: "后端", example: "backend:af-mlir", enum: () => Object.keys(BACKEND_LABEL), text: label },
  { name: "status", label: "结果", example: "status:fail", enum: () => Object.keys(SEARCH_RESULT_LABEL), text: v => SEARCH_RESULT_LABEL[v] || v },
  { name: "stage", label: "失败阶段", example: "stage:schedule", enum: () => Object.keys(STAGE_LABEL), text: v => STAGE_LABEL[v] || v },
  { name: "arch", label: "芯片", example: "arch:910b", enum: () => searchFieldValues.arch || [], text: archLabel },
  { name: "user", label: "用户", example: "user:alice", enum: () => searchFieldValues.user || [] },
  { name: "op", label: "算子", example: "op:aten.add.Tensor", enum: () => searchFieldValues.op || [] },
  { name: "dtype", label: "数据类型", example: "dtype:bfloat16", enum: () => searchFieldValues.dtype || [] },
  { name: "category", label: "算子类别", example: "category:reduce", enum: () => searchFieldValues.category || [] },
  { name: "source", label: "用例来源", example: "source:nnsmith", enum: () => Object.keys(SEARCH_SOURCE_LABEL), text: v => SEARCH_SOURCE_LABEL[v] || v },
  { name: "dynamic", label: "动态/静态", example: "dynamic:static", enum: () => Object.keys(SEARCH_DYNAMIC_LABEL), text: v => SEARCH_DYNAMIC_LABEL[v] || v },
  { name: "size", label: "张量规模", example: "size:<=64K", enum: () => searchFieldValues.size || [] },
  { name: "dim", label: "最大维度", example: "dim:<=512", enum: () => searchFieldValues.dim || [] },
  { name: "error", label: "错误文本", example: "error:tiling", desc: "日志摘要中包含的文本，3..64 个字符" },
  { name: "case", label: "用例 ID", example: "case:E000000000000000000000000A", desc: "精确匹配一个用例 ID" },
];
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
    const expired = !!token();
    if (expired) { try { sessionStorage.removeItem(TOKEN_KEY); } catch { /* ignore */ } me = null; updateWho(); }
    showLogin(expired ? "登录已失效，请重新输入 token；也可以返回以访客身份浏览" : "该操作需要登录：请输入管理员离线发放的 token");
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
let drawerOutsideHandler = null;
function openDrawer(...children) {
  const d = $("#drawer");
  d.replaceChildren(el("button", { class: "close", onclick: closeDrawer }, "关闭"), ...children);
  d.classList.remove("hidden");
  d.scrollTop = 0;
  if (drawerOutsideHandler) document.removeEventListener("click", drawerOutsideHandler);
  // Registered next tick so the click that opened the drawer doesn't immediately close it.
  setTimeout(() => {
    drawerOutsideHandler = e => { if (!d.contains(e.target)) closeDrawer(); };
    document.addEventListener("click", drawerOutsideHandler);
  }, 0);
}
function closeDrawer() {
  $("#drawer").classList.add("hidden");
  if (drawerOutsideHandler) { document.removeEventListener("click", drawerOutsideHandler); drawerOutsideHandler = null; }
}
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
  if (e.backend === "cpu" && e.function_status === "pass") return el("span", { class: "chip gray", "data-tooltip": "CPU 结果作为精度标杆" }, "参考");
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
  const head = el("h2", {}, "用例总览", el("span", { class: "spacer" }),
    arches.length ? select(arches.map(a => [a, archLabel(a)]), arch, v => go("overview", { arch: v })) : null);
  const hero = el("section", { class: "panel hero" }, head);
  root.append(hero);
  hero.append(renderPipeline(pipeline, q, hero.clientWidth - 44));
  const lanes = pipeline.lanes.filter(l => !l.baseline).map(l => l.backend);
  const tabsBox = el("section", { class: "panel" });
  root.append(tabsBox);
  renderBackendTabs(tabsBox, lanes, arch, q);
  const perfBox = el("section", { class: "panel" });
  root.append(perfBox);
  const npuBackends = pipeline.lanes.map(l => l.backend).filter(b => b !== "cpu");
  renderPerf(perfBox, npuBackends, arch, params.get("x") || (npuBackends.includes("af-mlir") ? "af-mlir" : npuBackends[0]),
    params.get("y") || (npuBackends.includes("triton-npu") ? "triton-npu" : npuBackends.find(b => b !== "af-mlir")), params.get("pdim") || "op_category");
};

const LANE_COLOR = { "af-mlir": "var(--lane-mlir)", "af-legacy": "var(--lane-legacy)", "triton-npu": "var(--lane-triton)", "eager-npu": "var(--lane-eager)" };
const LOSS_COLOR = { build: "var(--loss-build)", stage: "var(--loss-stage)", other: "var(--loss-stage)", unsupported: "var(--loss-build)", accuracy: "var(--loss-acc)" };

// Overview module 1: each backend is a pipe whose width is its case count. Every stage that
// loses cases peels a curved branch off the bottom of the pipe; what reaches the right is "pass".
function renderPipeline(data, q, width) {
  const lanes = data.lanes.filter(l => !l.baseline);
  if (!lanes.length) return el("div", { class: "empty" }, "暂无执行记录");
  const link = extra => () => go("discovery", { q: [q, extra].filter(Boolean).join(" ") });
  // Fixed stages (as specified): build -> AF Schedule (AF family only, shared Autofuse front end
  // before codegen) -> function -> accuracy -> pass. Detailed stages stay in tooltips and the stats below.
  const SCHEDULE_STAGES = ["schedule"];  // errors from the autofuse/optimize module (Optimize / schedule)
  const scheduleOf = l => l.family === "af" ? SCHEDULE_STAGES.reduce((n, st) => n + (l.stage_fail[st] || 0), 0) : 0;
  const detail = (l, skip) => Object.entries(l.stage_fail).filter(([k]) => !skip.includes(k)).sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${STAGE_LABEL[k] || k} ${fmt(v)}`).join("、");
  const columns = [
    { key: "build", label: "构图失败", kind: "build", value: l => l.build_fail, filter: b => `backend:${b} status:function_fail`,
      note: () => "NNSmith 生成、CPU 参考或 Torch/Dynamo/Inductor 链路失败" },
    { key: "schedule", label: "AF Schedule失败", kind: "stage", shared: true, value: scheduleOf,
      filter: b => `backend:${b} stage:${SCHEDULE_STAGES.join(",")}`, note: () => "Autofuse optimize 模块（Optimize / schedule）报错，发生在 codegen 之前" },
    { key: "function", label: "功能失败", kind: "stage", value: l => Math.max(0, l.function_fail - scheduleOf(l)) + l.unsupported,
      filter: b => `backend:${b} status:function_fail -stage:${SCHEDULE_STAGES.join(",")}`,
      note: l => (detail(l, l.family === "af" ? SCHEDULE_STAGES : []) || "—") + (l.unsupported ? `；含不支持 ${fmt(l.unsupported)}` : "") },
    { key: "accuracy", label: "精度失败", kind: "accuracy", value: l => l.accuracy_fail, filter: b => `backend:${b} status:accuracy_fail`, note: () => "执行成功但与标杆比对未通过" },
  ].filter(c => c.key !== "schedule" || lanes.some(l => l.family === "af"));
  return width < 860 ? pipelineVertical(data, lanes, columns, link) : pipelineSvg(data, lanes, columns, link, width);
}

function pipelineSvg(data, lanes, columns, link, width) {
  const total = Math.max(1, data.total_cases, ...lanes.map(l => l.executed));
  // Drawn at the real container width so text keeps its size; wide screens just get more room.
  const W = Math.max(860, Math.floor(width)), compact = W < 1100;
  const maxT = 46, laneH = 150, top = 64, left = 8, srcW = compact ? 80 : 92, laneStart = compact ? 250 : 300, endPad = compact ? 196 : 210;
  const H = top + lanes.length * laneH + 10;
  const colX = i => laneStart + 60 + i * ((W - endPad - laneStart - 60) / Math.max(1, columns.length));
  const th = n => n > 0 ? Math.max(2.5, maxT * n / total) : 0;
  const pipeT = n => n > 0 ? Math.max(8, th(n)) : 0;  // keep small lanes readable as pipes
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "pipe-svg", role: "img", "aria-label": "各后端用例流失管道" });
  const defs = svg("defs");
  root.append(defs);
  // column headers
  columns.forEach((c, i) => root.append(svg("text", { x: colX(i) + 22, y: 30, class: "pipe-col", "text-anchor": "middle" }, c.label)));
  root.append(svg("text", { x: W - endPad + 115, y: 30, class: "pipe-col", "text-anchor": "middle" }, "通过"));
  // family bands (shared front end)
  const families = {};
  lanes.forEach((l, i) => { if (l.family) (families[l.family] = families[l.family] || []).push(i); });
  for (const [fam, idx] of Object.entries(families)) {
    if (idx.length < 2) continue;
    const y0 = top + Math.min(...idx) * laneH - 18, y1 = top + (Math.max(...idx) + 1) * laneH - 26;
    root.append(svg("rect", { x: laneStart - 150, y: y0, width: W - laneStart + 130, height: y1 - y0, rx: 16, class: "pipe-family" }),
      svg("text", { x: laneStart - 138, y: y1 - 12, class: "pipe-family-label" }, `${fam.toUpperCase()} 族 · 共享前端`));
  }
  // total source bar
  const srcTop = top - 6, srcH = lanes.length * laneH - 60;
  const source = svg("g", { class: "pipe-source", "data-tooltip": `总用例 ${fmt(data.total_cases)}` },
    svg("rect", { x: left, y: srcTop, width: srcW, height: srcH, rx: 14 }),
    svg("text", { x: left + srcW / 2, y: srcTop + srcH / 2 - 6, "text-anchor": "middle", class: "pipe-source-n" }, fmt(data.total_cases)),
    svg("text", { x: left + srcW / 2, y: srcTop + srcH / 2 + 16, "text-anchor": "middle", class: "pipe-source-l" }, "总用例"));
  source.addEventListener("click", link(""));
  root.append(source);
  let srcOffset = srcTop + 14;
  lanes.forEach((lane, i) => {
    const color = LANE_COLOR[lane.backend] || "var(--accent)";
    const yTop = top + i * laneH + 14;
    const t0 = pipeT(lane.executed);
    const gid = "lane-grad-" + i;
    defs.append(svg("linearGradient", { id: gid, x1: 0, x2: 1, y1: 0, y2: 0 },
      svg("stop", { offset: "0%", "stop-color": color, "stop-opacity": 0.55 }), svg("stop", { offset: "100%", "stop-color": color, "stop-opacity": 0.95 })));
    // fan-out band from the source bar into the lane
    const sx = left + srcW, sy = srcOffset, ex = laneStart, ey = yTop, bt = Math.max(2.5, Math.min(t0, (srcH - 28) / lanes.length - 6));
    srcOffset += bt + 8;
    root.append(svg("path", { d: `M ${sx} ${sy} C ${sx + 90} ${sy}, ${ex - 90} ${ey}, ${ex} ${ey} L ${ex} ${ey + t0} C ${ex - 90} ${ey + t0}, ${sx + 90} ${sy + bt}, ${sx} ${sy + bt} Z`, fill: color, class: "pipe-fan" }));
    root.append(svg("text", { x: laneStart - 12, y: yTop + 6, class: "pipe-lane", "text-anchor": "end" }, label(lane.backend)),
      svg("text", { x: laneStart - 12, y: yTop + 26, class: "pipe-lane-n", "text-anchor": "end" }, `${fmt(lane.executed)} 执行`));
    // main pipe: top edge flat, bottom edge steps up at each loss
    let remaining = lane.executed, d = `M ${laneStart} ${yTop} L ${W - endPad + 20} ${yTop}`;
    const bottoms = [];
    let x = laneStart, t = pipeT(remaining);
    const drops = [];
    columns.forEach((c, ci) => {
      const loss = c.value(lane);
      const cx = colX(ci);
      bottoms.push([x, t], [cx, t]);
      if (loss > 0) {
        drops.push({ c, ci, cx, loss, yFrom: yTop + pipeT(remaining) - th(loss), t: th(loss) });
        remaining -= loss;
        t = pipeT(remaining);
        bottoms.push([cx + 26, t]);
        x = cx + 26;
      }
    });
    bottoms.push([W - endPad + 20, t]);
    let back = "";
    for (let k = bottoms.length - 1; k >= 0; k--) back += ` L ${bottoms[k][0]} ${yTop + bottoms[k][1]}`;
    const pipe = svg("path", { d: d + back + " Z", fill: `url(#${gid})`, class: "pipe-main", "data-tooltip": `执行 ${fmt(lane.executed)}` });
    pipe.addEventListener("click", link(`backend:${lane.backend}`));
    root.append(pipe);
    // drops
    for (const drop of drops) {
      const shared = lane.family && families[lane.family] && families[lane.family].length > 1 && (drop.c.key === "build" || drop.c.shared)
        && families[lane.family].every(j => drop.c.value(lanes[j]) === drop.loss);
      const firstOfFamily = shared && families[lane.family][0] === i;
      if (shared && !firstOfFamily) continue;  // the family draws one merged drop
      const dx = drop.cx, dy = drop.yFrom + drop.t / 2, endY = yTop + maxT + 46;
      const band = svg("path", { d: `M ${dx} ${dy} C ${dx + 18} ${dy}, ${dx + 22} ${dy + 12}, ${dx + 22} ${Math.max(dy + 20, endY)}`,
        stroke: LOSS_COLOR[drop.c.kind], "stroke-width": drop.t, class: "pipe-drop", "data-tooltip": `${drop.c.label} ${fmt(drop.loss)}（${pct(drop.loss, lane.executed)}）` });
      band.addEventListener("click", link(drop.c.filter(lane.backend)));
      const pillText = `${fmt(drop.loss)}`;
      const pw = 20 + pillText.length * 9.5;
      const pill = svg("g", { class: "pipe-pill " + drop.c.kind, "data-tooltip": `${drop.c.label} ${fmt(drop.loss)}（${pct(drop.loss, lane.executed)}）`, transform: `translate(${dx + 22 - pw / 2}, ${Math.max(dy + 20, endY) + 4})` },
        svg("rect", { width: pw, height: 26, rx: 13 }), svg("text", { x: pw / 2, y: 18, "text-anchor": "middle" }, pillText));
      pill.addEventListener("click", link(drop.c.filter(lane.backend)));
      root.append(band, pill);
    }
    // pass terminal
    const pw = 170, px = W - endPad + 30, py = yTop + t / 2 - 21;
    const term = svg("g", { class: "pipe-pass", transform: `translate(${px}, ${py})`, "data-tooltip": `通过 ${fmt(lane.pass)}（${pct(lane.pass, lane.executed)}）` },
      svg("rect", { width: pw, height: 42, rx: 21 }),
      svg("text", { x: 18, y: 27, class: "pipe-pass-n" }, fmt(lane.pass)),
      svg("text", { x: pw - 16, y: 27, "text-anchor": "end", class: "pipe-pass-p" }, pct(lane.pass, lane.executed)));
    term.addEventListener("click", link(`backend:${lane.backend} status:pass`));
    root.append(term);
  });
  return el("div", { class: "pipe-wrap" }, root);
}

function pipelineVertical(data, lanes, columns, link) {
  const box = el("div", { class: "pipe-v" }, el("button", { class: "pipe-v-total", onclick: link("") }, el("b", {}, fmt(data.total_cases)), el("span", {}, "总用例")));
  for (const lane of lanes) {
    const color = LANE_COLOR[lane.backend] || "var(--accent)";
    const steps = el("div", { class: "pipe-v-steps" });
    let remaining = lane.executed;
    for (const c of columns) {
      const loss = c.value(lane);
      if (!loss) continue;
      remaining -= loss;
      steps.append(el("button", { class: "pipe-v-step " + c.kind, onclick: link(c.filter(lane.backend)), "data-tooltip": `${c.label} ${fmt(loss)}（${pct(loss, lane.executed)}）` },
        el("span", {}, c.label), el("b", {}, "−" + fmt(loss)),
        el("i", { style: { width: Math.max(4, 100 * remaining / Math.max(1, lane.executed)) + "%", background: color } })));
    }
    box.append(el("div", { class: "pipe-v-lane" }, el("div", { class: "pipe-v-head" }, el("span", { class: "pipe-v-dot", style: { background: color } }), label(lane.backend),
      el("span", { class: "muted small" }, ` ${fmt(lane.executed)} 执行`)), steps,
      el("button", { class: "pipe-v-pass", onclick: link(`backend:${lane.backend} status:pass`) }, el("span", {}, "通过"), el("b", {}, `${fmt(lane.pass)} · ${pct(lane.pass, lane.executed)}`))));
  }
  return box;
}

// Capability matrix: one column per backend, one row per dimension, so a row compares the same
// metric across backends. Scrolls horizontally when the backends do not fit (desktop and mobile).
async function renderBackendTabs(box, lanes, arch, q) {
  const body = el("div", {}, el("div", { class: "empty" }, "加载中…"));
  box.replaceChildren(el("h2", {}, "基础能力统计分布"), legend([["c-pass", "通过"], ["c-func", "功能失败"], ["c-acc", "精度失败"], ["c-uns", "不支持"]]), body);
  if (!lanes.length) { body.replaceChildren(el("div", { class: "empty" }, "暂无数据")); return; }
  let matrix;
  try { matrix = await api("/api/overview/matrix" + qs({ backends: lanes.join(","), arch, q })); }
  catch (err) { body.replaceChildren(errorBox(err)); return; }
  const dims = ["all", ...DIM_ORDER.filter(d => matrix.dims.includes(d))];
  const results = lanes.map(backend => dims.map(dim => matrix.cells[backend][dim] || { dim, groups: [] }));
  const grid = el("div", { class: "cap-grid" });
  grid.style.gridTemplateColumns = `repeat(${lanes.length}, minmax(320px, 1fr))`;
  for (const backend of lanes) grid.append(el("div", { class: "cap-head" }, el("span", { class: "pipe-v-dot", style: { background: LANE_COLOR[backend] || "var(--accent)" } }), label(backend)));
  dims.forEach((dim, row) => {
    grid.append(el("div", { class: "cap-row-label" }, dim === "all" ? "正确性分布" : (DIM_LABEL[dim] || dim)));
    // One shared row order per dimension so the same line compares the same value across backends.
    const totals = new Map();
    for (let col = 0; col < lanes.length; col++) for (const g of (results[col][row].groups || [])) totals.set(g.value, (totals.get(g.value) || 0) + g.total);
    const order = [...totals.keys()].sort(BUCKET_DIMS.includes(dim) ? bucketCompare : (a, b) => totals.get(b) - totals.get(a));
    lanes.forEach((backend, col) => {
      const result = results[col][row];
      grid.append(dim === "all" ? donutChart(result, backend, true) : statusChart(result, backend, true, order));
    });
  });
  body.replaceChildren(el("div", { class: "cap-scroll" }, grid));
}

function donutChart(result, backend, compact = false) {
  if (result.error) return el("div", { class: "chart" }, errorBox(result.error));
  const g = result.groups[0] || { total: 0, pass: 0, function_fail: 0, accuracy_fail: 0, unsupported: 0, stage_fail: {} };
  const parts = [["通过", g.pass, "var(--ok)"], ["功能失败", g.function_fail, "var(--bad)"], ["精度失败", g.accuracy_fail, "var(--warn)"], ["不支持", g.unsupported, "var(--tie)"]].filter(p => p[1] > 0);
  const R = 60, r = 38, C = 80, root = svg("svg", { viewBox: "0 0 160 160", width: 160, height: 160, role: "img", "aria-label": "正确性分布" });
  let angle = -Math.PI / 2;
  for (const [name, n, color] of parts) {
    const sweep = 2 * Math.PI * n / (g.total || 1), end = angle + Math.min(sweep, 2 * Math.PI - 1e-4), large = sweep > Math.PI ? 1 : 0;
    const p = (rad, a) => `${C + rad * Math.cos(a)} ${C + rad * Math.sin(a)}`;
    root.append(svg("path", { d: `M ${p(R, angle)} A ${R} ${R} 0 ${large} 1 ${p(R, end)} L ${p(r, end)} A ${r} ${r} 0 ${large} 0 ${p(r, angle)} Z`, fill: color, "data-tooltip": `${name} ${fmt(n)}（${pct(n, g.total)}）` }));
    angle += sweep;
  }
  root.append(svg("text", { x: C, y: C + 4, "text-anchor": "middle", class: "donut-center" }, pct(g.pass, g.total)));
  const stages = Object.entries(g.stage_fail || {}).sort((a, b) => b[1] - a[1]);
  return el("div", { class: "chart" }, compact ? el("div", { class: "muted small" }, `${fmt(g.total)} 个用例`) : el("h3", {}, `${label(backend)} · 正确性分布（${fmt(g.total)} 个用例）`),
    el("div", { class: "donut-wrap" }, root, el("div", { class: "small" }, parts.map(([name, n, color]) => el("div", {}, el("i", { class: "swatch", style: { background: color } }), `${name} ${fmt(n)}（${pct(n, g.total)}）`)),
      stages.length ? el("div", { class: "muted", style: { marginTop: "6px" } }, "功能失败阶段：" + stages.map(([k, v]) => `${STAGE_LABEL[k] || k} ${fmt(v)}`).join(" · ")) : null)));
}
const BUCKET_DIMS = ["dim_bucket", "size_bucket"];
function bucketValue(label) {
  const m = /^(<=|>)\s*([\d.]+)\s*([KM]?)$/.exec(label || "");
  if (!m) return Infinity;
  const n = Number(m[2]) * ({ K: 1024, M: 1048576 }[m[3]] || 1);
  return m[1] === ">" ? n + 0.5 : n;
}
function bucketCompare(a, b) { return bucketValue(a) - bucketValue(b); }
function statusChart(result, backend, compact = false, order = null) {
  if (result.error) return el("div", { class: "chart" }, errorBox(result.error));
  const limit = result.dim === "op_display" ? 24 : 14;
  let groups;
  if (order) {
    const byValue = new Map(result.groups.map(g => [g.value, g]));
    const empty = v => ({ value: v, total: 0, pass: 0, function_fail: 0, accuracy_fail: 0, unsupported: 0 });
    groups = order.slice(0, limit).map(v => byValue.get(v) || empty(v));
  } else {
    groups = result.groups.slice(0, limit);
  }
  const note = result.dim === "op_display" ? `（覆盖 ${result.groups.length} 个算子，显示用例最多的 ${groups.length} 个）` : "";
  const title = compact ? (note ? el("div", { class: "muted small" }, note.replace(/[（）]/g, "")) : null)
    : el("h3", {}, `${label(backend)} · ${DIM_LABEL[result.dim] || result.dim}${note}`);
  return el("div", { class: "chart" + (result.dim === "op_display" && !compact ? " wide" : "") }, title,
    groups.length ? groups.map(g => el("div", { class: "hbar", "data-tooltip": `通过 ${fmt(g.pass)}（${pct(g.pass, g.total)}）· 功能失败 ${fmt(g.function_fail)}（${pct(g.function_fail, g.total)}）· 精度失败 ${fmt(g.accuracy_fail)}（${pct(g.accuracy_fail, g.total)}）` }, el("div", { class: "label" }, g.value),
      stackBar([["c-pass", g.pass, "通过"], ["c-func", g.function_fail, "功能失败"], ["c-acc", g.accuracy_fail, "精度失败"], ["c-uns", g.unsupported, "不支持"]], g.total),
      el("div", { class: "muted" }, g.total ? pct(g.pass, g.total) : "—"))) : el("div", { class: "empty" }, "无数据"));
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

// =====================================================================
// 问题发现
// =====================================================================
const COLUMNS = [
  ["user_id", "用户"], ["client", "节点"], ["backend", "后端"], ["npu_arch", "芯片"], ["status", "结果"], ["log_summary", "日志摘要"],
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
  const visible = new Set(store.get(COLS_KEY, COLUMNS.map(c => c[0]).filter(k => k !== "client" && k !== "run")));
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
  const copyBtn = el("button", { class: "mini-btn" }, "复现命令");
  copyBtn.addEventListener("click", () => { if (!navigator.clipboard) return; navigator.clipboard.writeText(`afx fuzz replay ${c.case_key}`); copyBtn.textContent = "已复制"; setTimeout(() => { copyBtn.textContent = "复现命令"; }, 1200); });
  const menuBtn = isAdmin() ? el("button", { class: "icon-btn", title: "更多操作", onclick: e => popMenu(e.currentTarget, [
    ["加入回归集", () => pinGlobal([c.case_key])],
  ]) }, "⋯") : null;
  return el("td", { class: "case-cell", rowspan: rowSpan },
    el("div", {}, box, " ", el("a", { href: "#", class: "mono", title: c.case_key, onclick: e => { e.preventDefault(); caseDrawer(group); } }, c.case_key), menuBtn),
    el("div", { class: "muted small" }, `${c.case_source} · ${c.is_dynamic ? "动态" : "静态"} · ${c.node_count ?? "?"} 节点 · ${c.dtypes || ""}`),
    el("div", { class: "muted small", "data-tooltip": (c.op_display || "") + "\n" + (c.op_types || "") }, (c.op_display || c.op_types || "").split(",").slice(0, 6).join(" · ")),
    el("div", { class: "mini-actions" },
      el("button", { class: "mini-btn", onclick: () => retestDrawer([c.case_key], [group]) }, "复测"),
      copyBtn,
      el("button", { class: "mini-btn", onclick: () => caseDrawer(group) }, "查看详情")));
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
    case "npu_arch": return archLabel(e.npu_arch);
    case "user_id": return el("span", { "data-tooltip": e.user_id }, e.user_name || e.user_id);
    case "client": return e.client_name || "—";
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
      tr.append(el("td", { class: run ? batchOf(run) + " batch" : "" }, run ? `#${run}` : "—", any ? el("div", { class: "muted small" }, `${any.user_name || any.user_id} · ${any.client_name || ""} · ${time(any.created_at)}`) : null));
      for (const b of backends) { const e = cells[b]; tr.append(el("td", { class: run ? batchOf(run) : "" }, e ? el("div", {}, statusChip(e), el("div", { class: "muted small" }, e.kernel_ns ? "kernel " + ns(e.kernel_ns) : "")) : "—")); }
      body.append(tr);
    });
  }
  return el("table", { class: "grid" }, el("thead", {}, el("tr", {}, el("th", {}, "用例"), el("th", {}, "批次"), backends.map(b => el("th", {}, label(b))))), body);
}
function caseDrawer(group) { openCase(group.case.case_key); }
function openCase(key, exec) {
  location.hash = "case?key=" + encodeURIComponent(key) + (exec ? "&exec=" + encodeURIComponent(exec) : "");
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
// 用例详情（移植自原详情页：DAG、Shape、后端对比、运行环境、计时、阶段时间线、失败详情、ASCIR、历史）
// =====================================================================
const STAGE_NAME = { generate: "共享生成", reference: "CPU 期望结果", compare: "CPU 结果校验", record: "持久化",
  shape_processing: "Shape 处理耗时（墙钟跨度）", generation: "生成用例", capability: "后端能力检查", prepare: "准备执行器（不代表编译完成）",
  input_binding: "输入绑定", first_call_combined: "首调(惰性编译/执行/同步)", correctness: "精度校验", execution_total: "执行总计",
  compile: "独立编译", kernel_mean: "Kernel 平均耗时", performance: "性能测量" };
const stageName = k => STAGE_NAME[k] || k || "Unknown";
const secs = v => v == null ? null : (v / 1e9).toFixed(3) + " s";
function stageLabel(f) {
  if (f.kind === "first_call_combined") return (f.lease && f.lease.device_uuid ? "NPU" : "CPU") + stageName(f.kind);
  return stageName(f.kind);
}
const DTYPE_SHORT = { float16: "fp16", float32: "fp32", float64: "fp64", bfloat16: "bf16" };

PAGES.case = async (root, params) => {
  const key = params.get("key");
  if (!key) { root.append(el("div", { class: "empty" }, "缺少用例 ID")); return; }
  const [spec, history] = await Promise.all([api(`/api/cases/${encodeURIComponent(key)}/spec`), api(`/api/cases/${encodeURIComponent(key)}/history?limit=50`)]);
  const c = history.case;
  let executions = history.executions, cursor = history.next_cursor;
  const shapes = (spec.shape_variants || []).length;
  let shape = Number(params.get("shape") || (shapes ? 0 : -1));
  const command = `afx fuzz replay ${c.case_key}`;
  const copy = el("button", { onclick: () => navigator.clipboard && navigator.clipboard.writeText(command).then(() => { copy.textContent = "已复制"; }) }, "复制");
  root.append(el("button", { class: "back-btn",
    onclick: () => { if (window.history.length > 1) window.history.back(); else go("discovery"); } }, "← 返回"));
  root.append(el("section", { class: "panel" },
    el("h2", {}, "用例详情 · ", el("span", { class: "mono" }, c.case_key + (shape >= 0 && c.is_dynamic ? String(shape) : "")), el("span", { class: "spacer" }),
      el("button", { onclick: () => retestDrawer([c.case_key], [{ case: c, executions }]) }, "⟳ 复测"),
      isAdmin() ? el("button", { onclick: () => pinGlobal([c.case_key]) }, "加入回归集") : null),
    el("div", { class: "toolbar" }, el("input", { class: "mono", value: command, readonly: true, style: { flex: 1 } }), copy),
    el("div", { class: "kv" }, [["来源", c.case_source], ["形态", c.is_dynamic ? `动态（${c.shape_count} 个 Shape）` : "静态"], ["节点数", c.node_count ?? "—"],
      ["算子", (c.op_display || c.op_types || "—").split(",").join(" · ")], ["数据类型", (c.dtypes || "—").split(",").join(" · ")], ["最大维度 / 张量元素", `${fmt(c.max_dim)} / ${fmt(c.max_numel)}`],
      ["首次出现", time(c.first_seen_at)], ["执行次数", fmt(history.total)]].flat().map(v => el("div", {}, v)))));
  // Shape selector + DAG
  const dagBox = el("section", { class: "panel" });
  root.append(dagBox);
  const drawDag = () => {
    const tabs = shapes > 1 ? el("div", { class: "tabs" }, (spec.shape_variants || []).map((v, i) => el("button", { class: i === shape ? "active" : "", onclick: () => { shape = i; drawDag(); compare(); selectExec(currentExec); } },
      `S${i} · ${tensorText(v.input_shapes, spec.weights, v.input_dtypes)}`.slice(0, 60)))) : null;
    const sample = shape >= 0 ? (spec.shape_variants || [])[shape] || {} : {};
    dagBox.replaceChildren(...[el("h2", {}, "用例节点 DAG", el("span", { class: "muted small" }, "箭头表示张量依赖；* 为推断形状")), tabs].filter(Boolean),
      el("div", { class: "muted small" }, "张量：" + tensorText(sample.input_shapes || spec.input_shapes || {}, spec.weights || [], sample.input_dtypes || spec.input_dtypes || {})),
      renderCaseDag(spec, sample));
  };
  // backend comparison of the latest execution per backend
  const cmpBox = el("section", { class: "panel" });
  root.append(cmpBox);
  const compare = () => {
    const latest = {};
    for (const e of executions) if (!latest[e.backend] || (e.is_head && !latest[e.backend].is_head)) latest[e.backend] = latest[e.backend] || e;
    for (const e of executions) if (e.is_head) latest[e.backend] = e;
    const rows = Object.values(latest).sort((a, b) => a.backend.localeCompare(b.backend));
    cmpBox.replaceChildren(el("h2", {}, "后端对比", el("span", { class: "muted small" }, "每个后端取当前结果；点击行查看该次执行")),
      el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["后端", "结果", "编译", "Kernel", "E2E", "节点", "时间"].map(t => el("th", {}, t)))),
        el("tbody", {}, rows.map(e => el("tr", { style: { cursor: "pointer" }, onclick: () => selectExec(e.exec_uuid) }, el("td", {}, label(e.backend)), el("td", {}, statusChip(e)),
          el("td", {}, ns(e.compile_ns)), el("td", {}, ns(e.kernel_ns)), el("td", {}, ns(e.e2e_ns)), el("td", {}, e.client_name || "—"), el("td", { class: "small" }, time(e.created_at)))))));
  };
  const execBox = el("section", { class: "panel" });
  root.append(execBox);
  let currentExec = params.get("exec");
  const selectExec = async uuid => {
    if (!uuid) { execBox.replaceChildren(el("div", { class: "empty" }, "暂无执行记录")); return; }
    currentExec = uuid;
    execBox.replaceChildren(el("div", { class: "empty" }, "加载中…"));
    try { execBox.replaceChildren(...renderExecution(await api("/api/executions/" + encodeURIComponent(uuid)), shape)); }
    catch (err) { execBox.replaceChildren(errorBox(err)); }
  };
  // history
  const histBody = el("tbody");
  const more = el("div", { class: "toolbar", style: { marginTop: "8px" } });
  root.append(el("section", { class: "panel" }, el("h2", {}, `执行历史（${fmt(history.total)}）`, el("span", { class: "muted small" }, "同色为同一批次")),
    el("div", { class: "table-wrap" }, el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["后端", "结果", "用户", "节点", "批次", "版本", "时间"].map(t => el("th", {}, t)))), histBody)), more));
  const drawHistory = () => {
    const batchOf = batchClasses(executions);
    histBody.replaceChildren(...executions.map(e => el("tr", { class: batchOf(e.run_id), style: { cursor: "pointer" }, onclick: () => selectExec(e.exec_uuid) },
      el("td", {}, label(e.backend), e.is_head ? el("span", { class: "chip info", style: { marginLeft: "4px" } }, "当前") : null, e.stale_graph ? el("span", { class: "chip gray" }, "旧图") : null),
      el("td", {}, statusChip(e)), el("td", {}, e.user_name || e.user_id), el("td", {}, e.client_name || "—"),
      el("td", {}, `#${e.run_id}`, e.run_kind !== "fuzz" ? el("span", { class: "chip info", style: { marginLeft: "4px" } }, { retest: "复测", verify: "验证", regress: "回归", import: "导入" }[e.run_kind] || e.run_kind) : null),
      el("td", { class: "small" }, `rev ${e.case_revision}`), el("td", { class: "small" }, time(e.created_at)))));
    more.replaceChildren(cursor ? el("button", { onclick: async () => {
      const page = await api(`/api/cases/${encodeURIComponent(key)}/history?limit=50&after=${cursor}`);
      executions = executions.concat(page.executions); cursor = page.next_cursor; drawHistory();
    } }, "加载更多") : el("span", { class: "muted small" }, "已显示全部"));
  };
  drawDag(); compare(); drawHistory();
  const pick = currentExec || (executions.find(e => e.is_head && e.backend !== "cpu" && (e.function_status !== "pass" || e.accuracy_status === "fail"))
    || executions.find(e => e.backend !== "cpu") || executions[0] || {}).exec_uuid;
  selectExec(pick);
};

function tensorText(shapes = {}, weights = [], types = {}) {
  const entries = [...Object.entries(shapes || {}).map(([n, sh]) => ({ shape: sh, dtype: (types || {})[n] })), ...(weights || [])];
  return entries.map((t, i) => "x" + (i + 1) + "[" + [...(t.shape || []), DTYPE_SHORT[t.dtype] || t.dtype || "?"].join(",") + "]").join(", ") || "—";
}

function renderExecution(d, shape) {
  const detail = d.detail || {};
  const view = (detail.shape_views || {})[String(shape)] || detail.diagnostic_view || {};
  const out = [el("h2", {}, `执行详情 · ${label(d.backend)}`, " ", statusChip(d), el("span", { class: "muted small" }, ` · ${d.user_name || d.user_id} · ${d.client_name || ""} · ${time(d.created_at)}`))];
  // per-shape results
  if (d.shapes && d.shapes.length) out.push(el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["Shape", "功能", "精度", "最大绝对误差", "最大相对误差", "Kernel", "E2E"].map(t => el("th", {}, t)))),
    el("tbody", {}, d.shapes.map(x => el("tr", { class: x.shape_index === shape ? "bt0" : "" }, el("td", {}, "S" + x.shape_index), el("td", {}, x.function_status), el("td", {}, x.accuracy_status || "—"),
      el("td", {}, x.max_abs_err ?? "—"), el("td", {}, x.max_rel_err ?? "—"), el("td", {}, ns(x.kernel_ns)), el("td", {}, ns(x.e2e_ns)))))));
  // failure explanation, ASCIR, log
  const mlir = detail.mlir_failure || {};
  const explanation = mlir.path === "mlir_failed" ? `MLIR 路径失败（${mlir.scope === "shape" ? "当前 Shape" : "单 Shape 运行日志"}）：${mlir.evidence}`
    : mlir.path === "unconfirmed" ? "已确认 AF 功能失败，但现有日志不能确认是 MLIR 路径失败。" : view.explanation;
  if (d.function_status !== "pass" || d.accuracy_status === "fail") {
    out.push(el("h3", {}, "失败详情"));
    if (explanation) out.push(el("div", { class: "filter-note" }, explanation));
    if (d.log_summary) out.push(el("div", { class: "mono small" }, "摘要：" + d.log_summary));
  }
  if ((detail.plog_errors || []).length) out.push(el("h3", {}, "CANN plog 错误"), el("pre", { class: "log" }, detail.plog_errors.join("\n")));
  out.push(el("h3", {}, "ASCIR 表达的图结构"), el("pre", { class: "log" }, view.shape_log || "未保存当前 Shape 的 ASCIR / 图结构文本"));
  const runLog = view.run_log || view.shared_log || (view.shared_lines || []).join("\n") || d.log || view.summary;
  out.push(el("h3", {}, "报错日志摘要"), el("pre", { class: "log" }, runLog || "未保存报错日志"));
  // provenance
  const env = (detail.contexts || {}).environment || {};
  const envRows = [["用户", d.user_name || d.user_id], ["节点", d.client_name], ["芯片", archLabel(d.npu_arch || env.device_model)], ["CANN", d.env.cann || env.cann], ["Torch", d.env.torch || env.torch],
    ["Torch NPU", d.env.torch_npu || env.torch_npu], ["后端版本", d.env.backend_version], ["Autofuse commit", env.autofuse_commit], ["Autofuse binary", env.autofuse_binary],
    ["Triton", env.triton], ["Python", d.env.python || env.python], ["AutofuseX", d.env.afx || env.package_version], ["NNSmith", d.env.nnsmith], ["OS", d.env.os],
    ["CPU 架构", d.env.cpu_arch || env.machine], ["GCC", d.env.gcc]].filter(([, v]) => v);
  out.push(el("h3", {}, "运行环境"), el("div", { class: "kv" }, envRows.flat().map(v => el("div", { class: "mono small" }, v))));
  // timing
  let telemetry = {};
  try { telemetry = JSON.parse(((detail.contexts || {}).execution || {}).telemetry || "{}"); } catch { telemetry = {}; }
  const timeRows = [...Object.entries(telemetry.stages || {}).map(([k, v]) => [stageName(k), v]),
    ...Object.entries(telemetry.shapes || {}).flatMap(([i, m]) => Object.entries(m).map(([k, v]) => [`S${i} ${stageName(k)}`, v]))];
  out.push(el("h3", {}, "用例执行耗时"), el("div", { class: "muted small" }, `本次运行 ${detail.shape_count ?? d.shapes.length} 个 Shape 的计时。执行总计包含能力检查、准备和这些 Shape；生成耗时单独记录。下列指标不能逐项相加。`),
    timeRows.length ? el("table", { class: "grid" }, el("tbody", {}, timeRows.map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", {}, (v / 1e9).toFixed(1) + " s"))))) : el("div", { class: "muted" }, "未采集本次运行计时"));
  const fs = detail.stage_timeline || [];
  if (fs.length) {
    out.push(el("h3", {}, "阶段时间线"), el("div", { class: "table-wrap" }, el("table", { class: "grid" },
      el("thead", {}, el("tr", {}, ["泳道 / 阶段", "状态", "相对接纳时间", "依赖等待", "资源等待", "运行", "worker / 冷暖"].map(t => el("th", {}, t)))),
      el("tbody", {}, fs.map(f => { const begin = f.started ?? f.dispatched, end = f.finished; return el("tr", {}, el("td", {}, stageLabel(f)), el("td", {}, f.state),
        el("td", {}, begin == null ? "未采集" : secs(begin - f.admitted)), el("td", {}, f.ready == null ? "未采集" : secs(f.ready - f.admitted)),
        el("td", {}, f.ready == null || f.dispatched == null ? "未采集" : secs(f.dispatched - f.ready)), el("td", {}, begin == null || end == null ? "未采集" : secs(end - begin)),
        el("td", {}, (f.worker_id || "未采集") + " / " + (f.temperature || "未采集"))); })))));
    const timed = fs.filter(f => f.started != null && f.finished != null);
    if (timed.length) {
      const origin = Math.min(...timed.map(f => f.admitted)), extent = Math.max(1, ...timed.map(f => f.finished - origin));
      const lanes = el("div", { class: "stage-lanes" });
      for (const f of timed) {
        const track = el("div", { class: "stage-track" });
        const add = (a, b, cls) => { if (b <= a) return; const bar = el("span", { class: cls, title: stageName(f.kind) + " " + secs(b - a) }); bar.style.left = (100 * (a - origin) / extent) + "%"; bar.style.width = (100 * (b - a) / extent) + "%"; track.append(bar); };
        add(f.admitted, f.ready ?? f.started, "stage-dependency"); add(f.ready ?? f.started, f.dispatched ?? f.started, "stage-resource"); add(f.started, f.finished, f.state === "failed" ? "stage-failed" : "stage-running");
        lanes.append(el("div", { class: "stage-lane" }, el("span", {}, stageLabel(f)), track));
      }
      lanes.append(el("div", { class: "muted small" }, "灰：依赖等待 · 黄：资源等待 · 绿：运行 · 红：失败；共享生成/参考不摊分为单 Shape。"));
      out.push(lanes);
    }
  }
  const life = detail.lifecycle || [];
  out.push(el("div", { class: "muted small", "data-tooltip": life.map(e => e.event + " " + e.wall_time).join(" → ") },
    "生命周期（UTC）：" + (life.map(e => e.event + " " + String(e.wall_time || "").slice(11, 23)).join(" → ") || "未采集") + "。端到端延迟：" + (secs(detail.end_to_end_ns) || "未采集")));
  if (!d.detail) out.push(el("div", { class: "muted small" }, "该执行未上传详细诊断（旧客户端或诊断不可用）。"));
  return out;
}

// DAG layout ported from the original detail page (node labels use operator display names).
function caseDag(meta, sample) {
  const graph = meta.graph || {};
  if (!graph.nodes || !graph.nodes.length) return { nodes: [], edges: [] };
  const nodes = [], byName = new Map(), edges = [];
  const add = (name, labelText, tensor = {}, kind = "op", inputs = [], attrs = {}) => { const node = { name, label: labelText, ...tensor, kind, inputs, attrs, level: 0 }; nodes.push(node); byName.set(name, node); return node; };
  const shapes = sample.input_shapes || meta.input_shapes || {}, types = sample.input_dtypes || meta.input_dtypes || {};
  for (const [name, shp] of Object.entries(shapes)) add(name, "Input", { shape: shp, dtype: types[name] }, "input");
  for (const w of meta.weights || []) add(w.name, "Weight", w, "weight");
  for (const node of graph.nodes) {
    const outputs = node.outputs || [node.output];
    const labelText = node.display_name || node.op;
    for (const [i, name] of outputs.entries()) add(name, outputs.length > 1 ? `${labelText} [${i}]` : labelText, {}, "op", node.inputs || [], node.attrs || {});
  }
  for (const node of [...nodes]) for (const name of node.inputs) if (!byName.has(name)) add(name, "未记录输入", {}, "input");
  const pending = new Set(nodes), resolved = new Set();
  while (pending.size) {
    let progress = false;
    for (const node of pending) {
      if (node.inputs.some(n => !resolved.has(n))) continue;
      const inputs = node.inputs.map(n => byName.get(n));
      node.level = inputs.length ? Math.max(...inputs.map(n => n.level)) + 1 : 0;
      if (node.kind === "op") {
        const first = inputs[0];
        node.dtype = inputs.length && inputs.every(n => n.dtype === first.dtype) ? first.dtype : undefined;
      }
      const outputShape = (sample.output_shapes || meta.output_shapes || {})[node.name];
      if (outputShape) { node.shape = outputShape; node.dtype = (sample.output_dtypes || meta.output_dtypes || {})[node.name] || node.dtype; node.inferred = false; }
      node.isOutput = (graph.outputs || []).includes(node.name);
      resolved.add(node.name); pending.delete(node); progress = true;
    }
    if (!progress) return { nodes: [], edges: [], error: "已保存节点依赖存在环，无法绘制 DAG" };
  }
  for (const node of nodes) for (const name of node.inputs) edges.push({ from: byName.get(name), to: node });
  return { nodes, edges };
}
function renderCaseDag(meta, sample) {
  const graph = caseDag(meta, sample);
  if (!graph.nodes.length) return el("div", { class: "muted" }, graph.error || "未保存节点依赖，无法绘制 DAG");
  const levels = [];
  for (const node of graph.nodes) (levels[node.level] = levels[node.level] || []).push(node);
  const W = 270, H = 150;
  const width = Math.max(640, ...levels.map(l => l.length * W + 32)), height = levels.length * H + 32;
  const canvas = svg("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": "用例节点 DAG，箭头表示张量依赖" });
  const marker = svg("marker", { id: "dag-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, svg("path", { d: "M 0 0 L 10 5 L 0 10 z", class: "dag-arrowhead" }));
  canvas.append(svg("defs", {}, marker));
  for (const layer of levels) layer.forEach((node, i) => { node.x = (width - layer.length * W) / 2 + i * W + 10; node.y = node.level * H + 16; });
  for (const { from, to } of graph.edges) {
    const x = from.x + 125, y = from.y + 100, ex = to.x + 125, ey = to.y;
    canvas.append(svg("path", { d: `M ${x} ${y} C ${x} ${y + 25}, ${ex} ${ey - 25}, ${ex} ${ey}`, class: "dag-edge", "marker-end": "url(#dag-arrow)" }, svg("title", {}, from.name + " → " + to.name)));
  }
  for (const node of graph.nodes) {
    const tensor = (node.shape ? "[" + node.shape.join(", ") + "]" : "Shape 未采集") + " · " + (node.dtype || "Dtype 未采集") + (node.inferred ? " *" : "");
    const title = node.label + " · " + node.name + (node.isOutput ? " · Output" : "");
    const tip = title + "\n" + tensor + (Object.keys(node.attrs).length ? "\n" + JSON.stringify(node.attrs) : "");
    const g = svg("g", { transform: `translate(${node.x},${node.y})`, class: "dag-node " + node.kind, tabindex: 0, "aria-label": tip, "data-tooltip": tip }, svg("rect", { width: 250, height: 100, rx: 10 }));
    const line = (y, text, cls) => g.append(svg("text", { x: 12, y, class: cls }, text.length > 32 ? text.slice(0, 31) + "…" : text));
    line(26, title, "dag-op"); line(50, node.name + (node.isOutput ? " → Output" : ""), "dag-name"); line(76, tensor, "dag-tensor");
    canvas.append(g);
  }
  return el("div", { class: "dag-scroll" }, canvas);
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
      tasks.replaceChildren(el("table", { class: "grid" }, el("tbody", {}, list.map(t => el("tr", {}, el("td", {}, el("a", { href: "#", class: "mono small", onclick: e => { e.preventDefault(); openCase(t.case_key); } }, t.case_key)), el("td", {}, archLabel(t.npu_arch)), el("td", {}, t.result || t.status), el("td", { class: "small muted" }, t.detail || ""))))));
    } }, "查看任务"), tasks);
    rounds.append(box);
  }
  const adminBox = isAdmin() && !["resolved", "closed"].includes(issue.status) ? el("div", { class: "toolbar" },
    el("button", { class: "danger", onclick: async () => { const reason = prompt("关闭原因（不修复/误报/重复）："); if (!reason) return; try { await api(`/api/admin/issues/${issueId}/close`, { method: "POST", body: { request_key: rkey(), reason } }); issueDrawer(issueId); } catch (err) { alert(err.message); } } }, "关闭问题"),
    el("button", { onclick: async () => { const into = prompt("合并到哪个问题 ID？"); if (!into) return; try { await api(`/api/admin/issues/${issueId}/merge`, { method: "POST", body: { request_key: rkey(), into: into.trim() } }); issueDrawer(issueId); } catch (err) { alert(err.message); } } }, "合并到…")) : null;
  openDrawer(el("h2", {}, issue.issue_id, " ", el("span", { class: "chip info" }, STATUS_LABEL[issue.status])), el("div", {}, issue.title), subway(issue),
    el("div", { class: "kv", style: { marginTop: "12px" } }, [["维度", { function: "功能", accuracy: "精度", performance: "性能" }[issue.dimension]], ["后端", label(issue.backend)], ["分类", issue.category],
      ["用例数", fmt(issue.case_count)], ["芯片", issue.arches.map(archLabel).join(", ") || "—"], ["领取", issue.owner_user ? `${issue.owner_user} / ${issue.agent_id}` : "—"],
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
    el("td", {}, el("a", { href: "#", class: "mono small", onclick: e => { e.preventDefault(); openCase(c.case_key); } }, c.case_key)),
    el("td", {}, archLabel(c.npu_arch)), el("td", {}, c.pick_reason || ""), el("td", {}, c.stale ? el("span", { class: "chip gray" }, "旧图") : ""),
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
    el("h3", {}, "关联"), el("div", {}, k.links.map(l => l.target_kind === "issue" ? el("a", { href: "#", onclick: e => { e.preventDefault(); issueDrawer(l.target_id); }, style: { marginRight: "10px" } }, l.target_id) : el("a", { href: "#", class: "mono small", style: { marginRight: "10px" }, onclick: e => { e.preventDefault(); openCase(l.target_id); } }, l.target_id))));
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
    arches.length ? select(arches.map(a => [a, archLabel(a)]), arch, v => set({ arch: v })) : null);
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
      opPanel(`${label(x)} 优势算子`, pat.best_ops.filter(o => o.log2 < 0)), opPanel(`${label(x)} 劣势算子`, pat.worst_ops.filter(o => o.log2 > 0)));
    if (isAdmin()) side.append(el("div", { class: "toolbar" }, el("button", { onclick: async () => {
        try { const r = await api("/api/admin/profile/drafts", { method: "POST", body: { request_key: rkey(), x, y, npu_arch: arch || null } }); alert(`生成 ${r.created.length} 条知识草稿（候选 ${r.candidates}），请到知识库审核发布`); }
        catch (err) { alert(err.message); } } }, "生成画像知识草稿")));
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
  const discreteX = new Set(pts.map(p => p[0])).size <= 64;
  const size = pts.length > 5000 ? 3 : 6;
  const jitter = discreteX ? Math.min(14, (W - 2 * pad) / Math.max(8, new Set(pts.map(p => p[0])).size) / 3) : 0;
  for (const p of pts) {
    const r = Math.max(-2, Math.min(2, p[2]));
    ctx.fillStyle = Math.abs(r) < 0.07 ? tie : r < 0 ? ok : bad;
    ctx.globalAlpha = 0.35 + 0.55 * Math.min(1, Math.abs(r) / 1.5);
    const jx = jitter ? (((p[3] * 2654435761) % 1000) / 1000 - 0.5) * 2 * jitter : 0;  // stable per case
    ctx.beginPath(); ctx.arc(px(p[0]) + jx, py(p[1]), size / 2, 0, 2 * Math.PI); ctx.fill();
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
    el("div", { class: "table-wrap" }, el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["", "状态", "节点", "用户", "主机", "芯片", "afx", "后端版本", "近 24h 执行", "上传失败", "待上传", "最后心跳"].map(t => el("th", {}, t)))),
      el("tbody", {}, data.clients.map(c => el("tr", {},
        el("td", {}, isAdmin() ? el("input", { type: "checkbox", onchange: e => e.target.checked ? selected.add(c.client_id) : selected.delete(c.client_id) }) : ""),
        el("td", {}, el("span", { class: "chip " + (c.online ? "ok" : "gray") }, c.online ? (c.busy ? "复测中" : "在线") : "离线")),
        el("td", {}, el("b", {}, c.display_name || "—")), el("td", {}, c.user_id), el("td", { class: "small", "data-tooltip": c.machine_id }, (c.hostname || c.machine_id).slice(0, 16)), el("td", {}, archLabel(c.npu_arch)),
        el("td", {}, c.afx_version || "—", c.afx_version && data.latest_afx_version && c.afx_version !== data.latest_afx_version ? el("span", { class: "chip warn", style: { marginLeft: "4px" } }, "落后") : null,
          c.target_afx_version ? el("span", { class: "chip info", style: { marginLeft: "4px" } }, "→ " + c.target_afx_version) : null),
        el("td", { class: "small" }, Object.entries(c.backend_versions).filter(([, v]) => v).map(([k, v]) => `${label(k)} ${v}`).join(" · ") || "—"),
        el("td", {}, fmt(c.last_24h.executions)), el("td", {}, fmt(c.last_24h.upload_errors)), el("td", {}, fmt(c.pending_uploads)),
        el("td", { class: "small" }, time(c.last_heartbeat_at || c.last_seen_at)))))))));
  root.append(el("section", { class: "panel" }, el("h2", {}, "近 48 小时吞吐（执行/小时）"), hourly(data.hourly)));
  root.append(fleetPanel(data.clients));
  const effBox = el("section", { class: "panel" });
  root.append(effBox);
  renderEfficiency(effBox, (await api("/api/ops/efficiency")).windows);
  if (isAdmin()) root.append(await settingsPanel());
};
function fleetPanel(clients) {
  const live = clients.filter(c => (c.runtime || []).length);
  const box = el("section", { class: "panel" }, el("h2", {}, "节点运行情况", el("span", { class: "muted small" }, "来自客户端心跳中的运行快照，约每分钟更新")));
  if (!live.length) { box.append(el("div", { class: "empty" }, "暂无正在上报运行快照的节点")); return box; }
  for (const c of live) {
    const card = el("div", { class: "fleet-card" }, el("h3", {}, c.display_name || c.machine_id, el("span", { class: "muted small" }, ` · ${c.user_id} · ${archLabel(c.npu_arch)} · 快照于 ${time(c.runtime_at)}`)));
    const rows = c.runtime.map(r => el("tr", {},
      el("td", { class: "mono small", "data-tooltip": r.run_id }, String(r.run_id || "").slice(0, 8)),
      el("td", {}, el("span", { class: "chip " + (r.state === "running" ? "ok" : "gray") }, r.state)),
      el("td", {}, (r.backend || []).map(label).join(" · ")),
      el("td", {}, el("div", { class: "device-dots" }, (r.devices || []).map(d => el("span", { class: "device-dot" + (d.busy ? " busy" : ""), "data-tooltip": d.uuid }, `${d.logical} ${d.model || ""}`)))),
      el("td", {}, Object.entries(r.stages || {}).map(([k, v]) => `${stageName(k)} ×${v}`).join("，") || "—"),
      el("td", {}, `${fmt(r.success)} / ${fmt(r.completed)}`),
      el("td", {}, r.elapsed > 0 ? (r.completed * 60 / r.elapsed).toFixed(2) : "—"),
      el("td", {}, fmt(r.queued)), el("td", {}, fmt(r.case_errors)), el("td", { class: "small" }, Math.round(r.elapsed / 60) + " 分钟")));
    card.append(el("div", { class: "table-wrap" }, el("table", { class: "grid" },
      el("thead", {}, el("tr", {}, ["运行", "状态", "后端", "设备（绿=忙）", "进行中阶段", "成功 / 完成 Shape", "Shape/min", "排队", "Case 错误", "已运行"].map(t => el("th", {}, t)))), el("tbody", {}, rows))));
    box.append(card);
  }
  return box;
}

// Tool efficiency (ported): finished stage windows grouped by backend / environment / Shape mode.
function renderEfficiency(box, windows) {
  box.replaceChildren(el("h2", {}, "工具效率", el("span", { class: "muted small" }, "已完成运行窗口的吞吐与延迟分位数")));
  if (!windows.length) { box.append(el("div", { class: "empty" }, "暂无已完成运行窗口的效率数据（持续运行的任务在窗口结束后上报）。")); return; }
  let chosen = windows[0];
  const pick = select(windows.map((w, i) => [w.window_id + "|" + w.client_id, `${i === 0 ? "最新" : "历史 " + i} · ${w.client_name || w.client_id} · ${w.window_id.slice(0, 12)}`]), "", v => { chosen = windows.find(w => w.window_id + "|" + w.client_id === v); draw(); });
  const filter = el("input", { type: "search", placeholder: "筛选后端 / 环境 / static / dynamic", oninput: () => draw() });
  const body = el("div");
  box.append(el("div", { class: "toolbar" }, "运行窗口", pick, filter), body);
  const seconds = v => v == null ? "未采集" : (v / 1e9).toFixed(2) + "s";
  const draw = () => {
    const rows = chosen.rows || [];
    if (!rows.length) { body.replaceChildren(el("div", { class: "muted" }, "该窗口没有可统计的记录")); return; }
    const w = rows[0], duration = Number(w.window_seconds);
    const devices = Object.entries(w.npu_slot_utilization || {});
    const tokens = filter.value.toLowerCase().split(/\s+/).filter(Boolean);
    const groups = new Map();
    for (const r of rows) {
      const k = JSON.stringify([r.backend, r.environment_id, r.mode]);
      if (!groups.has(k)) groups.set(k, { backend: r.backend, environment_id: r.environment_id || "", mode: r.mode, rows: [], success: 0, failed: 0, case_errors: 0, completed: 0 });
      const g = groups.get(k); g.rows.push(r);
      for (const n of ["success", "failed", "case_errors", "completed"]) g[n] += Number(r[n] || 0);
    }
    const list = [...groups.values()].filter(g => tokens.every(t => [g.backend, g.environment_id, g.mode, ...g.rows.map(r => r.hardware || "")].join(" ").toLowerCase().includes(t)));
    const rate = n => duration > 0 ? (n * 60 / duration).toFixed(2) : "未采集";
    const latency = (g, k) => g.rows.length === 1 ? `${seconds(g.rows[0][k + "_p50_ns"])} / ${seconds(g.rows[0][k + "_p95_ns"])}` : "见明细";
    body.replaceChildren(
      el("div", { class: "muted small" }, `窗口 ${chosen.window_id} · 节点 ${chosen.client_name || chosen.client_id} · 时长 ${Number.isFinite(duration) ? duration.toFixed(1) + " s" : "未采集"}。吞吐按完整窗口时长计算；资源占用属于整个窗口，不随筛选变化。`),
      el("div", { class: "tiles" }, [["CPU 资源池使用率", w.cpu_pool_utilization == null ? "未采集" : (100 * w.cpu_pool_utilization).toFixed(1) + "%"],
        ...devices.map(([uuid, v], i) => [`NPU ${i + 1} 执行槽使用率`, (100 * v).toFixed(1) + "%", uuid]),
        ["窗口取消 / 写库失败", `${w.cancelled_attempts ?? "未采集"} / ${w.record_failed_attempts ?? "未采集"}`]].map(([k, v, tip]) => el("div", { class: "tile", "data-tooltip": tip || null }, el("div", { class: "v" }, v), el("div", { class: "k" }, k)))),
      el("div", { class: "table-wrap" }, el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["后端 / Shape 模式", "环境", "成功 / 失败 Shape / Case 错误", "成功 Shape/min", "已完成 Shape/min", "处理 P50 / P95", "端到端 P50 / P95"].map(t => el("th", {}, t)))),
        el("tbody", {}, list.map(g => el("tr", {}, el("td", {}, `${label(g.backend)} / ${g.mode}`), el("td", { class: "mono small", "data-tooltip": g.environment_id }, g.environment_id.slice(0, 12)),
          el("td", {}, `${g.success} / ${g.failed} / ${g.case_errors}`), el("td", {}, rate(g.success)), el("td", {}, rate(g.completed)), el("td", {}, latency(g, "processing")), el("td", {}, latency(g, "e2e"))))))),
      el("details", {}, el("summary", {}, "诊断明细：冷 / 热、状态、延迟与环境"),
        ...list.map(g => el("div", {}, el("h3", {}, `${label(g.backend)} / ${g.mode} · ${g.environment_id.slice(0, 12)}`),
          el("pre", { class: "log" }, JSON.stringify({ environment_id: g.environment_id, hardware: g.rows[0].hardware, software: g.rows[0].software }, null, 2)),
          el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["冷 / 热 · 状态", "成功 / 失败 / Case 错误", "处理 P50 / P95", "端到端 P50 / P95", "唯一 Shape / 重试", "摊销秒/Shape", "冷 / 热调用数"].map(t => el("th", {}, t)))),
            el("tbody", {}, g.rows.map(r => el("tr", {}, el("td", {}, `${r.temperature} / ${r.status}`), el("td", {}, `${r.success} / ${r.failed} / ${r.case_errors}`),
              el("td", {}, `${seconds(r.processing_p50_ns)} / ${seconds(r.processing_p95_ns)}`), el("td", {}, `${seconds(r.e2e_p50_ns)} / ${seconds(r.e2e_p95_ns)}`),
              el("td", {}, `${r.unique_shapes ?? "—"} / ${r.retry_attempts ?? "—"}`), el("td", {}, r.amortized_seconds_per_shape == null ? "—" : Number(r.amortized_seconds_per_shape).toFixed(2)),
              el("td", {}, `${r.cold ?? "—"} / ${r.warm ?? "—"}`)))))))));
  };
  draw();
}

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
    pins.length ? el("table", { class: "grid" }, el("tbody", {}, pins.map(p => el("tr", {}, el("td", {}, p.scope === "global" ? "全局" : p.scope_value), el("td", {}, el("a", { href: "#", class: "mono small", onclick: e => { e.preventDefault(); openCase(p.case_key); } }, p.case_key)),
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

// ---------- field tooltip (ported: full text for truncated cells, logs and DAG nodes) ----------
const fieldTooltip = el("div", { class: "field-tooltip", role: "tooltip", id: "field-tooltip" });
fieldTooltip.hidden = true;
document.body.append(fieldTooltip);
let tooltipTarget = null;
function hideFieldTooltip() { if (tooltipTarget) tooltipTarget.removeAttribute("aria-describedby"); tooltipTarget = null; fieldTooltip.hidden = true; }
function showFieldTooltip(event) {
  const target = event.target.closest && event.target.closest("[data-tooltip], [title]");
  if (!target || target.closest("#field-tooltip")) return;
  if (target.hasAttribute("title")) { target.dataset.tooltip = target.getAttribute("title"); target.removeAttribute("title"); }
  const content = target.dataset.tooltip;
  if (!content || !content.trim()) return;
  hideFieldTooltip(); tooltipTarget = target;
  fieldTooltip.textContent = content; fieldTooltip.hidden = false; target.setAttribute("aria-describedby", "field-tooltip");
  const tip = fieldTooltip.getBoundingClientRect(), bounds = target.getBoundingClientRect();
  const x = event.clientX ?? bounds.left, y = event.clientY ?? bounds.bottom;
  fieldTooltip.style.left = Math.max(8, Math.min(x, innerWidth - tip.width - 8)) + "px";
  fieldTooltip.style.top = Math.max(8, Math.min(y + 14, innerHeight - tip.height - 8)) + "px";
}
document.addEventListener("pointerover", showFieldTooltip);
document.addEventListener("focusin", showFieldTooltip);
document.addEventListener("pointerout", e => { if (tooltipTarget && !tooltipTarget.contains(e.relatedTarget) && !fieldTooltip.contains(e.relatedTarget)) hideFieldTooltip(); });
document.addEventListener("keydown", e => { if (e.key === "Escape") hideFieldTooltip(); });
window.addEventListener("scroll", hideFieldTooltip, true);

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (route().page === "overview" || route().page === "") render(); }, 300);
});

// ---------- global search: field picker / autocomplete ----------
// Most people don't remember the field:value syntax, so focusing the (empty) box lists every
// field with a one-line example, and typing narrows it to matching fields, then matching values.
const searchPanel = el("div", { class: "search-suggest hidden", role: "listbox" });
document.body.append(searchPanel);
let searchActiveIndex = -1;
function currentTerm(input) {
  const pos = input.selectionStart ?? input.value.length, text = input.value;
  const start = text.lastIndexOf(" ", pos - 1) + 1;
  let end = text.indexOf(" ", pos);
  if (end === -1) end = text.length;
  return { start, end, term: text.slice(start, end) };
}
function searchMatches(term) {
  const negate = term.startsWith("-") ? "-" : "";
  const body = negate ? term.slice(1) : term;
  const colon = body.indexOf(":");
  if (colon === -1) {
    const q = body.toLowerCase();
    return { kind: "field", negate, items: SEARCH_FIELDS.filter(f => !q || f.name.startsWith(q)) };
  }
  const field = SEARCH_FIELDS.find(f => f.name === body.slice(0, colon));
  if (!field || !field.enum) return { kind: "none" };
  const already = body.slice(colon + 1), lastComma = already.lastIndexOf(","), typed = already.slice(lastComma + 1);
  const q = typed.toLowerCase();
  const items = field.enum().filter(v => !q || String(v).toLowerCase().includes(q)).slice(0, 30);
  return { kind: "value", negate, field, keep: already.slice(0, lastComma + 1), items };
}
function applySearchPick(input, start, end, text) {
  input.value = input.value.slice(0, start) + text + input.value.slice(end);
  const pos = start + text.length;
  input.setSelectionRange(pos, pos);
  input.focus();
  updateSearchPanel(input);
}
function renderSearchPanel(input, match, bounds) {
  searchActiveIndex = -1;
  if (match.kind === "none" || !match.items.length) { searchPanel.classList.add("hidden"); return; }
  const rows = match.kind === "field"
    ? match.items.map(f => el("button", { class: "search-sug-item", type: "button",
        onclick: () => applySearchPick(input, bounds.start, bounds.end, match.negate + f.name + ":") },
        el("span", { class: "sf-name" }, f.name + ":"), el("span", { class: "sf-label" }, f.label),
        el("span", { class: "sf-example" }, f.desc || f.example)))
    : match.items.map(v => el("button", { class: "search-sug-item", type: "button",
        onclick: () => applySearchPick(input, bounds.start, bounds.end,
          match.negate + match.field.name + ":" + match.keep + v + " ") },
        el("span", { class: "sf-name" }, String(v)), match.field.text ? el("span", { class: "sf-label" }, match.field.text(v)) : null));
  searchPanel.replaceChildren(...rows);
  const rect = input.getBoundingClientRect();
  searchPanel.style.left = (rect.left + scrollX) + "px";
  searchPanel.style.top = (rect.bottom + scrollY + 4) + "px";
  searchPanel.style.width = Math.max(320, rect.width) + "px";
  searchPanel.classList.remove("hidden");
}
function updateSearchPanel(input) {
  const bounds = currentTerm(input);
  renderSearchPanel(input, searchMatches(bounds.term), bounds);
}
(() => {
  const input = $("#search");
  input.addEventListener("focus", () => { loadSearchFieldValues().then(() => updateSearchPanel(input)); updateSearchPanel(input); });
  input.addEventListener("input", () => updateSearchPanel(input));
  input.addEventListener("click", () => updateSearchPanel(input));
  input.addEventListener("keydown", e => {
    const items = [...searchPanel.children];
    if (searchPanel.classList.contains("hidden") || !items.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      searchActiveIndex = (searchActiveIndex + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      items.forEach((it, i) => it.classList.toggle("active", i === searchActiveIndex));
      items[searchActiveIndex].scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter" && searchActiveIndex >= 0) {
      e.preventDefault();
      items[searchActiveIndex].click();
    } else if (e.key === "Escape") {
      searchPanel.classList.add("hidden");
    }
  });
  // A suggestion click re-renders the panel's children (detaching the clicked button) and
  // refocuses the input before this bubbles here, so check focus rather than DOM containment.
  document.addEventListener("click", e => { if (e.target !== input && document.activeElement !== input) searchPanel.classList.add("hidden"); });
})();

// ---------- boot ----------
window.addEventListener("hashchange", render);
$("#search-form").addEventListener("submit", e => { e.preventDefault(); const { page } = route(); go(page === "models" ? "discovery" : page, { q: $("#search").value.trim() || null }); searchPanel.classList.add("hidden"); });
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
