import { el, BACKEND_LABEL, archOptions, STATUS_LABEL } from "./core.js";
export const RESULT_GROUPS = [
  ["构图失败", [
    ["nnsmith", "c-build", "NNSmith", "构图失败 · NNSmith"],
    ["gen_fail", "c-build", "Inductor", "Inductor 构图失败"],
    ["ascgraph_fail", "c-build", "AF前端", "构图失败 · AF前端"]
  ]],
  ["编译失败", [
    ["schedule", "c-compile", "Schedule", "Schedule 失败"],
    ["legacy_codegen", "c-compile", "Legacy Codegen", "Legacy Codegen 失败"],
    ["mlir", "c-compile", "MLIR Codegen", "MLIR Codegen 失败"],
    ["triton_ascend", "c-compile", "TritonNPU", "TritonNPU 编译失败"],
    ["af_common", "c-compile", "AF其他", "AF其他失败"],
    ["compile_unknown", "c-compile", "未知", "编译失败 · 未知"]
  ]],
  ["运行失败", [
    ["aclnn", "c-runtime", "aclNN单算子", "运行失败 · aclNN单算子"],
    ["aicpu", "c-runtime", "AICPU", "运行失败 · AICPU"],
    ["fused_kernel", "c-runtime", "融合算子", "运行失败 · 融合算子"],
    ["runtime_unknown", "c-runtime", "未知", "运行失败 · 未知"]
  ]],
  ["精度结果", [
    ["pass", "c-pass", "精度通过", "精度通过"],
    ["accuracy_fail", "c-acc", "精度失败", "精度失败"],
    ["unchecked", "c-uns", "未知", "精度结果 · 未知"]
  ]],
  ["其他", [
    ["other", "c-uns", "不支持", "不支持"],
    ["environment", "c-env", "环境问题", "环境问题"],
    ["timeout", "c-env", "超时", "超时"],
    ["unknown", "c-uns", "未知", "未知"],
    ["unknown_retest", "c-uns", "未知·待复测", "存量证据不足 · 待复测"]
  ]]
];
export const RESULT_ITEMS = RESULT_GROUPS.flatMap(([, items]) => items);
export const DEFAULT_RESULTS = RESULT_ITEMS.map(([key]) => key).filter(key => ![
  "nnsmith", "gen_fail", "compile_unknown", "runtime_unknown", "unchecked", "aclnn", "aicpu",
  "other", "environment", "timeout", "unknown", "unknown_retest"
].includes(key));
export const RESULT_COLORS = {"c-build":"var(--loss-build)","c-compile":"var(--loss-stage)","c-runtime":"#ef4444","c-env":"#f97316","c-acc":"var(--warn)","c-pass":"var(--ok)","c-uns":"var(--tie)"};

export function selectedResults(params) {
  return new Set(params.has("results") ? params.get("results").split(",").filter(k => RESULT_ITEMS.some(([v]) => k === v)) : DEFAULT_RESULTS);
}
export function resultFilterParams(params) {
  return { q: params.get("q"), results: params.get("results") || DEFAULT_RESULTS.join(","), selected_backends: params.get("backends"), chips: params.get("chips") || "a2a3" };
}
let filterTimer = null;
export function cancelFilterChange() { clearTimeout(filterTimer); }
let collapsed = matchMedia("(max-width:760px)").matches;
export const ISSUE_CATEGORIES = ["Reduce", "Transpose", "Concat", "Split", "Gather", "Scatter", "View", "Reshape", "Broadcast", "Elementwise"];
export const TREATMENTS = [["ignore", "忽略"], ["low", "低级"], ["normal", "中等"], ["high", "高级"]];
export const REPAIR_DEFAULT_RESULTS = ["mlir", "fused_kernel"];
export const REPAIR_DEFAULT_STATUSES = Object.keys(STATUS_LABEL).filter(key => !["resolved", "closed"].includes(key));
export function resultFilter(params, change, arches = [], options = {}) {
  const repair = options.repair;
  const groups = repair ? RESULT_GROUPS.filter(([name]) => name !== "精度结果") : RESULT_GROUPS;
  const items = groups.flatMap(([, entries]) => entries);
  const defaults = repair ? REPAIR_DEFAULT_RESULTS : DEFAULT_RESULTS;
  params = new URLSearchParams(params);
  const notify = change;
  change = patch => {
    for (const [key, value] of Object.entries(patch)) value == null ? params.delete(key) : params.set(key, value);
    const scroll = aside.scrollTop;
    const fresh = resultFilter(params, notify, arches, options);
    aside.replaceWith(fresh); fresh.scrollTop = scroll;
    clearTimeout(filterTimer);
    filterTimer = setTimeout(() => notify({ ...Object.fromEntries(params),
      ...Object.fromEntries(["q", "backends", "chips", "results",
        ...(repair ? ["categories", "treatment", "status", "scope"] : [])].map(key => [key, params.get(key)]))
    }), 180);
  };
  const active = new Set(params.has("results") ? params.get("results").split(",").filter(k => items.some(([v]) => v === k)) : defaults);
  const aside = el("aside", { class: "result-filter", "aria-label": "数据筛选" });
  const body = el("div", { class: "result-filter-body", id: "result-filter-body" });
  const toggle = el("button", { type: "button", class: "result-filter-toggle", "aria-controls": "result-filter-body", onclick: () => { collapsed = !collapsed; update(); } });
  function update() {
    aside.classList.toggle("collapsed", collapsed); body.hidden = collapsed;
    toggle.setAttribute("aria-expanded", String(!collapsed)); toggle.textContent = collapsed ? "数据筛选 ›" : "‹ 收起筛选";
  }
  function field(title, key, options, chosen) {
    const selected = new Set(chosen);
    return el("section", { class: "visual-filter-field" }, el("h3", {}, title),
      el("div", { class: "legend-group-items" }, options.map(([value, text]) => el("button", {
        type: "button", class: "legend-toggle", "aria-pressed": String(selected.has(value)), onclick: () => {
          selected.has(value) ? selected.delete(value) : selected.add(value);
          change({ [key]: key === "backends" && selected.size === options.length ? null : [...selected].join(",") || "none" });
        }
      }, text))));
  }
  const backendOptions = Object.entries(BACKEND_LABEL);
  let chipOptions = [["a2a3", "A2A3"], ["a5", "A5"], ...arches.filter(a => !/910|950/i.test(a)).map(a => [a, a])];
  const allFilters = selected => ({
    backends: selected ? backendOptions.map(([key]) => key).join(",") : "none",
    chips: selected ? chipOptions.map(([key]) => key).join(",") : "none",
    results: selected ? items.map(([key]) => key).join(",") : "none",
    ...(repair ? {
      ...(selected ? { scope: "all" } : {}),
      categories: selected ? ISSUE_CATEGORIES.join(",") : "none",
      treatment: selected ? TREATMENTS.map(([key]) => key).join(",") : "none",
      status: selected ? Object.keys(STATUS_LABEL).join(",") : "none"
    } : {})
  });
  body.append(el("div", { class: "toolbar result-filter-actions", role: "group", "aria-label": "全部筛选操作" },
    el("button", { type: "button", onclick: () => change(allFilters(true)) }, "全部选择"),
    el("button", { type: "button", onclick: () => change(allFilters(false)) }, "全部清空"),
    el("button", { type: "button", onclick: () => change({ backends: null, chips: null, results: null,
      ...(repair ? { categories: null, treatment: null, status: null, scope: null } : {}) }) }, "恢复默认")));
  body.append(
    field("后端", "backends", backendOptions, params.has("backends") ? params.get("backends").split(",") : backendOptions.map(([v]) => v)),
    field("芯片", "chips", chipOptions, (params.get("chips") || (repair ? chipOptions.map(([v]) => v).join(",") : "a2a3")).split(",")));
  const legend = el("div", { class: "legend capability-legend" });
  const commit = () => change({ results: [...active].join(",") || "none" });
  for (const [name, items] of groups) {
    const count = items.filter(([k]) => active.has(k)).length;
    legend.append(el("div", { class: "legend-group", role: "group", "aria-label": name },
      el("button", { type: "button", class: "legend-group-toggle", "aria-pressed": count === items.length ? "true" : count ? "mixed" : "false", onclick: () => {
        const hide = items.every(([k]) => active.has(k)); for (const [k] of items) hide ? active.delete(k) : active.add(k); commit();
      } }, name), el("div", { class: "legend-group-items" }, items.map(([key, cls, text, title]) => el("button", {
        type: "button", class: "legend-toggle", "data-result": key, "aria-label": title, title, "aria-pressed": String(active.has(key)),
        onclick: () => { active.has(key) ? active.delete(key) : active.add(key); commit(); }
      }, el("i", { class: cls }), text)))));
  }
  if (!repair || (params.get("dimension") || "function") === "function") body.append(
    el("h3", {}, "结果与阶段"),
    el("div", { class: "muted small" }, "已选 " + active.size + "/" + items.length), legend);
  if (repair) {
    body.append(field("算子类别", "categories", ISSUE_CATEGORIES.map(v => [v, v]),
      params.has("categories") ? params.get("categories").split(",") : ISSUE_CATEGORIES),
      field("处理标签", "treatment", TREATMENTS, (params.get("treatment") || "low,normal,high").split(",")),
      field("修复状态", "status", Object.entries(STATUS_LABEL).filter(([key]) => key !== "resolved").map(([key, text]) => [key === "closed" ? "resolved,closed" : key, text]), params.has("status") ? params.get("status").replace(/resolved,closed|closed,resolved|resolved|closed/g, "terminal").split(",").map(key => key === "terminal" ? "resolved,closed" : key) : REPAIR_DEFAULT_STATUSES));
  }
  aside.append(toggle, body);
  aside.addEventListener("keydown", e => { if (e.key === "Escape") { collapsed = true; update(); toggle.focus(); } });
  archOptions().then(values => {
    if (!aside.isConnected) return;
    arches = values;
    chipOptions = [["a2a3", "A2A3"], ["a5", "A5"], ...values.filter(a => !/910|950/i.test(a)).map(a => [a, a])];
    body.querySelectorAll(".visual-filter-field")[1].replaceWith(field("芯片", "chips", chipOptions, (params.get("chips") || (repair ? chipOptions.map(([v]) => v).join(",") : "a2a3")).split(",")));
  }).catch(() => {});
  update(); return aside;
}
