"use strict";
// 问题发现: execution records grouped by case (expanded/pivot views) + manual retests.
import { el, store, BACKEND_LABEL, VIEW_KEY, COLS_KEY, api, qs, rkey, errorBox, label, archLabel, time, ns } from "./core.js";
import { resultFilterParams } from "./result-filter.js";
import { select, stackBar, statusChip, pagination } from "./widgets.js";
import { PAGES, go, render, caseDrawer, openCase } from "./router.js";
import { isAdmin, me } from "./auth.js";
import { showActionBar, hideActionBar, popMenu, openDrawer } from "./ui.js";

// Keep copied commands tied to the backend represented by the row.
export function replayCommand(caseKey, executions = []) {
  const target = executions.find(e => e.backend !== "cpu" &&
    (e.function_status === "fail" || e.accuracy_status === "fail" || e.status === "error"))
    || executions.find(e => e.backend !== "cpu") || executions[0];
  return target && target.backend ? "afx fuzz replay " + caseKey + " --backend " + target.backend
    : "afx fuzz replay " + caseKey;
}

export const COLUMNS = [
  ["user_id", "用户"], ["client", "节点"], ["backend", "后端"], ["npu_arch", "芯片"], ["status", "结果"], ["log_summary", "日志摘要"],
  ["compile_ns", "编译"], ["kernel_ns", "Kernel"], ["e2e_ns", "E2E"], ["run", "批次"], ["created_at", "时间"],
];
let discoverySelection = { filter: null, keys: new Set() };
PAGES.discovery = async (root, params) => {
  const tabs = el("div", { class: "tabs" },
    el("button", { class: params.get("sub") !== "retests" ? "active" : "", onclick: () => go("discovery", {}) }, "执行记录"),
    el("button", { class: params.get("sub") === "retests" ? "active" : "", onclick: () => go("discovery", { sub: "retests" }) }, "复测记录"));
  root.append(tabs);
  if (params.get("sub") === "retests") return renderRetests(root);
  const filters = resultFilterParams(params);
  const view = store.get(VIEW_KEY, "expanded");
  const visible = new Set(store.get(COLS_KEY, COLUMNS.map(c => c[0]).filter(k => k !== "client" && k !== "run")));
  const panel = el("section", { class: "panel" });
  root.append(panel);
  const filterKey = JSON.stringify([me?.user_id, filters]);
  if (discoverySelection.filter !== filterKey) discoverySelection = { filter: filterKey, keys: new Set() };
  const selected = discoverySelection.keys;
  const table = el("div", { class: "table-wrap" });
  const pagerTop = el("div", { class: "toolbar" });
  const more = el("div", { class: "toolbar", style: { marginTop: "10px" } });
  const colpick = el("div", { class: "colpick hidden" }, COLUMNS.map(([k, t]) => el("label", {},
    el("input", { type: "checkbox", checked: visible.has(k), onchange: e => { e.target.checked ? visible.add(k) : visible.delete(k); store.set(COLS_KEY, [...visible]); redraw(); } }), t)));
  panel.append(el("h2", {}, "执行记录", el("span", { class: "spacer" }),
    select([["expanded", "展开视图"], ["pivot", "后端透视"]], view, v => { store.set(VIEW_KEY, v); render(); }),
    el("button", { onclick: () => { colpick.classList.toggle("hidden"); if (colpick.classList.contains("hidden")) delete panel.dataset.custom; else panel.dataset.custom = "1"; } }, "显示列")), colpick, pagerTop, table, more);
  let selecting = false;
  let selectionGeneration = 0;
  const clearSelection = () => { selectionGeneration++; selected.clear(); redraw(); updateBar(); };
  const selectAll = el("button", { title: "选择当前筛选命中的全部用例（包括其他页）", onclick: async () => {
    const generation = ++selectionGeneration;
    selecting = true; selectAll.disabled = true; selectAll.textContent = "选择中…";
    panel.dataset.custom = "1";
    try {
      const result = await api("/api/cases/selection" + qs(filters));
      if (!panel.isConnected || generation !== selectionGeneration) return;
      selected.clear(); for (const key of result.case_keys) selected.add(key);
      redraw();
    } catch (err) { alert(err.message); }
    finally { selecting = false; selectAll.disabled = false; selectAll.textContent = "选择全部"; if (panel.isConnected) updateBar(); }
  } }, "选择全部");
  panel.insertBefore(el("div", { class: "toolbar" }, selectAll,
    el("button", { onclick: clearSelection }, "清空选择")), pagerTop);
  let groups = [];
  const requestedPage = Number(params.get("page") || 1);
  const pageNumber = Number.isSafeInteger(requestedPage) && requestedPage > 0 ? requestedPage : 1;
  const load = async () => {
    const page = await api("/api/cases" + qs({ ...filters, page: pageNumber, limit: 30 }));
    groups = page.groups;
    redraw();
    for (const target of [pagerTop, more]) target.replaceChildren(
      pagination(page.page, page.total_pages, page.total, targetPage =>
        go("discovery", { ...Object.fromEntries(params), page: targetPage }), "用例"));
  };
  const updateBar = () => {
    if (selected.size || selecting) panel.dataset.custom = "1"; else delete panel.dataset.custom;  // keep the selection through auto refresh
    if (!selected.size) return hideActionBar();
    showActionBar(el("span", {}, `已选 ${selected.size} 个用例`),
      el("button", { onclick: () => retestDrawer([...selected], groups) }, "复测"),
      isAdmin() ? el("button", { onclick: () => pinGlobal([...selected]) }, "加入回归集") : null,
      el("button", { onclick: clearSelection }, "取消"));
  };
  const redraw = () => {
    table.replaceChildren(groups.length ? (view === "pivot" ? pivotTable(groups, selected, updateBar) : expandedTable(groups, visible, selected, updateBar))
      : el("div", { class: "empty" }, "没有匹配的用例"));
  };
  await load();
  updateBar();
};

function caseCell(group, selected, updateBar, rowSpan) {
  const c = group.case;
  const box = el("input", { type: "checkbox", checked: selected.has(c.case_key), "aria-label": "选择用例",
    onchange: e => { e.target.checked ? selected.add(c.case_key) : selected.delete(c.case_key); updateBar(); } });
  const copyBtn = el("button", { class: "mini-btn" }, "复现命令");
  copyBtn.addEventListener("click", () => { if (!navigator.clipboard) return; navigator.clipboard.writeText(replayCommand(c.case_key, group.executions)); copyBtn.textContent = "已复制"; setTimeout(() => { copyBtn.textContent = "复现命令"; }, 1200); });
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
export function batchClasses(executions) {
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
      for (const [k] of cols) tr.append(el("td", { class: cls + (k === cols[0][0] ? " batch" : "") }, execValue(e, k, group.case.case_key)));
      body.append(tr);
    });
    if (group.more_executions) body.append(el("tr", {}, el("td", { colspan: cols.length + 1, class: "muted small" }, "仅显示最近 10 条，更多见用例详情")));
  }
  return el("table", { class: "grid" }, el("thead", {}, el("tr", {}, el("th", {}, "用例"), cols.map(([, t]) => el("th", {}, t)))), body);
}
function executionLink(e, caseKey) {
  return el("a", { class: "execution-link", href: "#case?key=" + encodeURIComponent(caseKey) + "&exec=" + encodeURIComponent(e.exec_uuid),
    title: "查看执行详情", "aria-label": "查看执行详情" }, statusChip(e));
}
export function execValue(e, k, caseKey = e.case_key) {
  switch (k) {
    case "backend": return el("span", {}, label(e.backend), e.is_head ? null : el("span", { class: "muted small", title: "非当前结果（历史/复测/验证）" }, " ·历史"), e.stale_graph ? el("span", { class: "chip gray" }, "旧图") : null);
    case "status": return executionLink(e, caseKey);
    case "log_summary":
      if (e.function_status === "pass") return "";
      return el("span", { class: "small mono", title: e.log_summary || "" }, (e.log_summary || "未采集到错误日志").slice(0, 80));
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
      for (const b of backends) { const e = cells[b]; tr.append(el("td", { class: run ? batchOf(run) : "" }, e ? el("div", {}, executionLink(e, group.case.case_key), el("div", { class: "muted small" }, e.kernel_ns ? "kernel " + ns(e.kernel_ns) : "")) : "—")); }
      body.append(tr);
    });
  }
  return el("table", { class: "grid" }, el("thead", {}, el("tr", {}, el("th", {}, "用例"), el("th", {}, "批次"), backends.map(b => el("th", {}, label(b))))), body);
}

export async function retestDrawer(keys, groups) {
  const known = new Set(groups.flatMap(g => g.executions.map(e => e.backend)).filter(b => b !== "cpu"));
  const backends = Object.keys(BACKEND_LABEL).filter(b => b !== "cpu");
  const checks = backends.map(b => el("label", {}, el("input", { type: "checkbox", value: b, checked: known.size ? known.has(b) : b === "af-mlir" }), " ", label(b)));
  const force = el("input", { type: "checkbox" });
  const reps = el("input", { type: "number", min: 1, max: 5, value: 1, style: { width: "80px" } });
  const out = el("div");
  const userPanel = el("div");
  const userChecks = [];
  const admin = isAdmin();
  let usersReady = !admin;
  let submitted = false;
  const requestKey = rkey();
  const submit = async () => {
    if (submitted || !usersReady) return;
    const chosen = checks.map(c => c.querySelector("input")).filter(i => i.checked).map(i => i.value);
    if (!chosen.length) { out.replaceChildren(errorBox("至少选择一个后端")); return; }
    const users = userChecks.filter(i => i.checked).map(i => i.value);
    if (admin && !users.length) { out.replaceChildren(errorBox("至少选择一个复跑用户")); return; }
    submitted = true; submitButton.disabled = true;
    try {
      const res = await api("/api/retests", { method: "POST", body: { request_key: requestKey, case_keys: keys, ...(admin ? { user_ids: users } : {}), backends: chosen, force: force.checked, repetitions: Number(reps.value) || 1 } });
      out.replaceChildren(...[el("div", { class: "chip ok" }, `已创建 ${res.tasks} 个复测任务`), res.skipped_no_coverage ? el("div", { class: "muted small" }, "已跳过 " + res.skipped_no_coverage + " 个无 Autofuse 覆盖的目标") : null, res.skipped_without_arch ? el("div", { class: "muted small" }, `${res.skipped_without_arch} 项任务没有可用的 NPU 执行记录，无法确定芯片`) : null,
        el("div", { class: "toolbar" }, el("button", { onclick: () => go("discovery", { sub: "retests" }) }, "查看复测记录"))].filter(Boolean));
    } catch (err) { submitted = false; submitButton.disabled = false; out.replaceChildren(errorBox(err)); }
  };
  const submitButton = el("button", { class: "primary", onclick: submit, disabled: !usersReady }, "提交复测");
  openDrawer(el("h2", {}, `复测 ${keys.length} 个用例`),
    el("div", { class: "muted small" }, "任务派发给对应芯片上在线的客户端，复测结果会作为新批次出现在原用例下，并更新当前结果。"),
    el("div", { class: "form-grid", style: { marginTop: "12px" } }, el("div", {}, "后端"), el("div", { class: "toolbar" }, checks), el("div", {}, "重复次数"), reps),
    el("label", {}, force, " 强制复测已标记无 Autofuse 覆盖的目标"),
    userPanel, el("div", { class: "toolbar", style: { marginTop: "12px" } }, submitButton), out);
  if (admin) {
    userPanel.textContent = "正在加载复跑用户…";
    try {
      const data = await api("/api/admin/users");
      const users = data.users.filter(u => u.status === "active" && ["admin", "developer"].includes(u.role));
      const labels = users.map(u => {
        const input = el("input", { type: "checkbox", value: u.user_id, checked: u.user_id === me?.user_id });
        userChecks.push(input); return el("label", {}, input, " ", u.user_id);
      });
      userPanel.replaceChildren(el("h3", {}, "复跑用户"),
        el("div", { class: "muted small" }, "任务平均分配到所选用户；更换客户端后仍可领取。用户需要有匹配后端和芯片的客户端。"),
        el("div", { class: "toolbar" },
          el("button", { onclick: () => userChecks.forEach(i => { i.checked = true; }) }, "全选"),
          el("button", { onclick: () => userChecks.forEach(i => { i.checked = false; }) }, "清空选择")),
        el("div", { class: "toolbar" }, labels));
      usersReady = true; submitButton.disabled = false;
    } catch (err) { userPanel.replaceChildren(errorBox(err)); }
  }
}
export async function pinGlobal(keys) {
  const reason = prompt(`把 ${keys.length} 个用例加入全局必选回归集，原因：`);
  if (reason === null) return;
  try { const r = await api("/api/regression/pins", { method: "POST", body: { request_key: rkey(), scope: "global", case_keys: keys, reason } }); alert(`已加入 ${r.pinned} 个用例`); }
  catch (err) { alert(err.message); }
}
const expandedRetestBatches = new Set();
async function renderRetests(root) {
  const rows = await api("/api/retests");
  const body = el("tbody");
  for (const r of rows) {
    const detailId = "retest-details-" + encodeURIComponent(r.batch);
    const detail = el("tr", { class: "retest-details", id: detailId },
      el("td", { colspan: 8 }, el("div", { class: "retest-details-inner" },
          el("div", { class: "muted small" }, (r.assignments || []).map(a => a.user_id + ": " + a.tasks + " 个任务").join(" · ")),
          r.items_truncated ? el("div", { class: "muted small" }, "仅显示前 100 个任务，统计包含整个批次。") : null,
          el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["Seed", "后端 / 芯片", "状态", "结果详情"].map(t => el("th", {}, t)))),
            el("tbody", {}, (r.items || []).map(t => el("tr", {},
              el("td", {}, el("a", { class: "mono small", href: "#case?key=" + encodeURIComponent(t.case_key), onclick: e => { e.preventDefault(); openCase(t.case_key); } }, t.case_key)),
              el("td", {}, t.backends.split(",").map(label).join(" / "), el("div", { class: "muted small" }, archLabel(t.npu_arch)), el("div", { class: "muted small" }, "分配用户：" + (t.assigned_user || "—"))),
              el("td", {}, ({ pass: "通过", fail: "失败", invalid: "无效", skip: "跳过" }[t.result] || { pending: "待执行", leased: "执行中", done: "已完成", invalid: "无效" }[t.status] || t.status),
                t.detail ? el("div", { class: "muted small" }, t.detail) : null),
              el("td", {}, t.executions.length ? t.executions.map(x => el("a", { href: "#case?key=" + encodeURIComponent(t.case_key) + "&exec=" + encodeURIComponent(x.exec_uuid), style: { marginRight: "10px" } }, label(x.backend) + " 详情")) : "暂无执行结果")
            )))))));
    detail.hidden = !expandedRetestBatches.has(r.batch);
    const toggle = el("button", { class: "retest-batch-toggle mono small", "aria-controls": detailId,
      "aria-expanded": String(!detail.hidden), title: r.batch, onclick: () => {
        detail.hidden = !detail.hidden;
        if (detail.hidden) expandedRetestBatches.delete(r.batch); else expandedRetestBatches.add(r.batch);
        toggle.setAttribute("aria-expanded", String(!detail.hidden));
      } }, el("span", { class: "retest-chevron", "aria-hidden": "true" }, "▶"), (r.batch || "").slice(0, 12));
    body.append(el("tr", {}, el("td", {}, toggle), el("td", {}, time(r.created_at)),
      el("td", {}, r.tasks), el("td", {}, r.running), el("td", {}, r.passed), el("td", {}, r.failed), el("td", {}, r.invalid),
      el("td", { style: { minWidth: "160px" } }, el("div", { class: "hbar", style: { gridTemplateColumns: "1fr" } },
        stackBar([["c-pass", r.passed, "通过"], ["c-func", r.failed, "失败"], ["c-uns", r.invalid, "无效"], ["c-acc", r.running, "进行中"]], r.tasks)))), detail);
  }
  root.append(el("section", { class: "panel" }, el("h2", {}, "我提交的复测"),
    rows.length ? el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["批次", "提交时间", "任务", "进行中", "通过", "失败", "无效", "进度"].map(t => el("th", {}, t)))), body)
      : el("div", { class: "empty" }, "还没有复测记录。在执行记录中选中用例后点击“复测”。")));
}
