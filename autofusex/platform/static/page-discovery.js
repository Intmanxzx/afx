"use strict";
// 问题发现: execution records grouped by case (expanded/pivot views) + manual retests.
import { el, store, VIEW_KEY, COLS_KEY, api, qs, rkey, errorBox, label, archLabel, time, ns } from "./core.js";
import { select, stackBar, statusChip } from "./widgets.js";
import { PAGES, go, render, caseDrawer, openCase } from "./router.js";
import { isAdmin } from "./auth.js";
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
  panel.append(el("h2", {}, "执行记录", el("span", { class: "spacer" }),
    select([["expanded", "展开视图"], ["pivot", "后端透视"]], view, v => { store.set(VIEW_KEY, v); render(); }),
    el("button", { onclick: () => { colpick.classList.toggle("hidden"); if (colpick.classList.contains("hidden")) delete panel.dataset.custom; else panel.dataset.custom = "1"; } }, "显示列")), colpick, table, more);
  let groups = [];
  let cursor = null, loadedOnce = false;
  const load = async () => {
    const page = await api("/api/cases" + qs({ q, after: cursor, limit: 30 }));
    groups = groups.concat(page.groups);
    cursor = page.next_cursor;
    if (loadedOnce) panel.dataset.custom = "1";  // extra pages loaded: a refresh would drop them
    loadedOnce = true;
    redraw();
    more.replaceChildren(cursor ? el("button", { onclick: load }, "加载更多") : el("span", { class: "muted small" }, groups.length ? "已全部加载" : ""));
  };
  const updateBar = () => {
    if (selected.size) panel.dataset.custom = "1"; else delete panel.dataset.custom;  // keep the selection through auto refresh
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
      for (const [k] of cols) tr.append(el("td", { class: cls + (k === cols[0][0] ? " batch" : "") }, execValue(e, k)));
      body.append(tr);
    });
    if (group.more_executions) body.append(el("tr", {}, el("td", { colspan: cols.length + 1, class: "muted small" }, "仅显示最近 10 条，更多见用例详情")));
  }
  return el("table", { class: "grid" }, el("thead", {}, el("tr", {}, el("th", {}, "用例"), cols.map(([, t]) => el("th", {}, t)))), body);
}
export function execValue(e, k) {
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

export function retestDrawer(keys, groups) {
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
      out.replaceChildren(...[el("div", { class: "chip ok" }, `已创建 ${res.tasks} 个复测任务`), res.skipped_without_arch ? el("div", { class: "muted small" }, `${res.skipped_without_arch} 个用例没有该后端的执行记录，无法确定芯片`) : null,
        el("div", { class: "toolbar" }, el("button", { onclick: () => go("discovery", { sub: "retests" }) }, "查看复测记录"))].filter(Boolean));
    } catch (err) { out.replaceChildren(errorBox(err)); }
  };
  openDrawer(el("h2", {}, `复测 ${keys.length} 个用例`),
    el("div", { class: "muted small" }, "任务派发给对应芯片上在线的客户端，复测结果会作为新批次出现在原用例下，并更新当前结果。"),
    el("div", { class: "form-grid", style: { marginTop: "12px" } }, el("div", {}, "后端"), el("div", { class: "toolbar" }, checks), el("div", {}, "重复次数"), reps),
    el("div", { class: "toolbar", style: { marginTop: "12px" } }, el("button", { class: "primary", onclick: submit }, "提交复测")), out);
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
          el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["Seed", "后端 / 芯片", "状态", "结果详情"].map(t => el("th", {}, t)))),
            el("tbody", {}, (r.items || []).map(t => el("tr", {},
              el("td", {}, el("a", { class: "mono small", href: "#case?key=" + encodeURIComponent(t.case_key), onclick: e => { e.preventDefault(); openCase(t.case_key); } }, t.case_key)),
              el("td", {}, t.backends.split(",").map(label).join(" / "), el("div", { class: "muted small" }, archLabel(t.npu_arch))),
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
