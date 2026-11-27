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
export function cancelFilterChange() { clearTimeout(filterTimer); filterTimer = null; }
let collapsed = matchMedia("(max-width:760px)").matches;
export const ISSUE_CATEGORIES = ["Reduce", "Transpose", "Concat", "Split", "Gather", "Scatter", "View", "Reshape", "Broadcast", "Elementwise"];
export const TREATMENTS = [["ignore", "忽略"], ["low", "低级"], ["normal", "中等"], ["high", "高级"]];
export const REPAIR_DEFAULT_RESULTS = ["mlir", "fused_kernel"];
export const REPAIR_DEFAULT_STATUSES = Object.keys(STATUS_LABEL).filter(key => !["resolved", "closed"].includes(key));
// A panel owns its draft until the user pauses; only then does routing load data.
export function isFilterChangePending() { return filterTimer !== null; }
export const FILTER_DELAY_MS = 450;
export function resultFilter(params, notify, arches = [], options = {}) {
  const repair = !!options.repair;
  const groups = repair ? RESULT_GROUPS.filter(([name]) => name !== "精度结果") : RESULT_GROUPS;
  const items = groups.flatMap(([, entries]) => entries);
  const defaults = repair ? REPAIR_DEFAULT_RESULTS : DEFAULT_RESULTS;
  const keys = ["q", "backends", "chips", "results", ...(repair ? ["categories", "treatment", "status", "scope"] : [])];
  params = new URLSearchParams(params);
  const fields = [];
  const aside = el("aside", { class: "result-filter", "aria-label": "数据筛选" });
  const body = el("div", { class: "result-filter-body", id: "result-filter-body" });
  const toggle = el("button", { type: "button", class: "result-filter-toggle", "aria-controls": "result-filter-body", onclick: () => { collapsed = !collapsed; update(); } });
  const countLabel = el("div", { class: "muted small" });
  const activeResults = () => new Set(params.has("results") ? params.get("results").split(",").filter(k => items.some(([v]) => v === k)) : defaults);
  // Keep the panel, its scroll position and focused button through filter loads
  // and pagination. Other route changes create a fresh panel.
  const signature = value => JSON.stringify([...value].filter(([key]) => key !== "page").sort(([a], [b]) => a.localeCompare(b)));
  aside.matchesFilter = (next, nextRepair) => repair === nextRepair && signature(params) === signature(next);
  function update() {
    aside.classList.toggle("collapsed", collapsed); body.hidden = collapsed;
    toggle.setAttribute("aria-expanded", String(!collapsed)); toggle.textContent = collapsed ? "数据筛选 ›" : "‹ 收起筛选";
  }
  function sync() {
    fields.forEach(field => field.sync());
    countLabel.textContent = "已选 " + activeResults().size + "/" + items.length;
  }
  function change(patch) {
    for (const [key, value] of Object.entries(patch)) value == null ? params.delete(key) : params.set(key, value);
    sync();
    cancelFilterChange();
    filterTimer = setTimeout(() => {
      filterTimer = null;
      if (aside.isConnected) notify({ ...Object.fromEntries(params), ...Object.fromEntries(keys.map(key => [key, params.get(key)])) });
    }, FILTER_DELAY_MS);
  }
  function actions(title, selectAll, clear, reset) {
    return el("div", { class: "filter-field-actions", role: "group", "aria-label": title + "筛选操作" },
      [["全部选择", selectAll], ["全部清空", clear], ["恢复默认", reset]].map(([text, onclick]) =>
        el("button", { type: "button", onclick }, text)));
  }
  function field(title, key, entries, fallback, resultGroup = false) {
    const values = entries.map(([value]) => value);
    const selected = () => {
      const raw = new Set(params.has(key) ? params.get(key).split(",") : fallback);
      return new Set(values.filter(value => value.split(",").some(part => raw.has(part))));
    };
    const apply = chosen => {
      const next = resultGroup ? activeResults() : new Set();
      values.forEach(value => next.delete(value));
      chosen.forEach(value => next.add(value));
      change({ [key]: [...next].join(",") || "none" });
    };
    const heading = resultGroup ? el("button", { type: "button", class: "legend-group-toggle", onclick: () =>
      apply(selected().size === values.length ? [] : values) }, title) : el("h3", {}, title);
    const buttons = entries.map(([value, text, cls, description]) => el("button", {
      type: "button", class: "legend-toggle", ...(resultGroup ? { "data-result": value, "aria-label": description, title: description } : {}),
      onclick: () => { const next = selected(); next.has(value) ? next.delete(value) : next.add(value); apply(next); }
    }, cls ? el("i", { class: cls }) : null, text));
    const control = { sync() {
      const current = selected();
      buttons.forEach((button, i) => button.setAttribute("aria-pressed", String(current.has(values[i]))));
      if (resultGroup) heading.setAttribute("aria-pressed", current.size === values.length ? "true" : current.size ? "mixed" : "false");
    }};
    fields.push(control);
    const section = el(resultGroup ? "div" : "section", { class: resultGroup ? "legend-group" : "visual-filter-field", role: "group", "aria-label": title },
      el("div", { class: "filter-field-heading" }, heading, actions(title, () => apply(values), () => apply([]),
        () => resultGroup ? apply(values.filter(value => fallback.includes(value))) : change({ [key]: null }))),
      el("div", { class: "legend-group-items" }, buttons));
    section.filterControl = control;
    control.sync();
    return section;
  }
  const backendOptions = Object.entries(BACKEND_LABEL);
  const chips = values => [["a2a3", "A2A3"], ["a5", "A5"], ...[...new Set(values)].filter(a => !/910|950/i.test(a) && !["a2a3", "a5"].includes(a)).map(a => [a, a])];
  let chipOptions = chips(arches);
  const chipField = () => field("芯片", "chips", chipOptions, repair ? chipOptions.map(([v]) => v) : ["a2a3"]);
  let chipSection = chipField();
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
  body.append(field("后端", "backends", backendOptions, backendOptions.map(([v]) => v)), chipSection);
  if (!repair || (params.get("dimension") || "function") === "function") {
    body.append(el("div", { class: "filter-field-heading" }, el("h3", {}, "结果与阶段"),
      actions("结果与阶段", () => change({ results: items.map(([key]) => key).join(",") }),
        () => change({ results: "none" }), () => change({ results: null }))), countLabel,
      el("div", { class: "legend capability-legend" }, groups.map(([name, entries]) =>
        field(name, "results", entries.map(([key, cls, text, title]) => [key, text, cls, title]), defaults, true))));
  }
  if (repair) body.append(
    field("算子类别", "categories", ISSUE_CATEGORIES.map(v => [v, v]), ISSUE_CATEGORIES),
    field("处理标签", "treatment", TREATMENTS, ["low", "normal", "high"]),
    field("修复状态", "status", Object.entries(STATUS_LABEL).filter(([key]) => key !== "resolved").map(([key, text]) => [key === "closed" ? "resolved,closed" : key, text]), REPAIR_DEFAULT_STATUSES));
  aside.append(toggle, body);
  aside.addEventListener("keydown", e => { if (e.key === "Escape") { collapsed = true; update(); toggle.focus(); } });
  archOptions().then(values => {
    if (!aside.isConnected) return;
    const next = chips(values);
    if (JSON.stringify(next) === JSON.stringify(chipOptions)) return;
    chipOptions = next;
    fields.splice(fields.indexOf(chipSection.filterControl), 1);
    const fresh = chipField();
    chipSection.replaceWith(fresh); chipSection = fresh;
  }).catch(() => {});
  sync(); update(); return aside;
}
