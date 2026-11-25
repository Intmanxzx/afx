"use strict";
// Small shared chart/table widgets used by several pages.
import { el, fmt, pct, STAGE_LABEL } from "./core.js";

export function stackBar(parts, total) {
  const track = el("div", { class: "track" });
  for (const [cls, n, title] of parts) if (n > 0) track.append(el("i", { class: cls, title: `${title} ${fmt(n)} (${pct(n, total)})`, style: { width: (100 * n / total) + "%" } }));
  return track;
}
export function legend(items) { return el("div", { class: "legend" }, items.map(([cls, t]) => el("span", {}, el("i", { class: cls }), t))); }
export function select(options, value, onchange, attrs = {}) {
  const s = el("select", attrs, options.map(([v, t]) => el("option", { value: v, selected: v === value }, t)));
  s.addEventListener("change", () => onchange(s.value));
  return s;
}

export function pagination(current, pages, total, onchange, subject = "问题") {
  const nav = el("nav", { class: "issue-pagination", "aria-label": subject + "列表分页" },
    el("span", { class: "muted small", role: "status" }, `共 ${fmt(total)} 条 · 共 ${fmt(pages)} 页 · 当前第 ${pages ? current : 0} 页`));
  const button = (n, text, disabled = false) => el("button", { type: "button", disabled,
    "aria-label": text || `第 ${n} 页`, "aria-current": !text && n === current ? "page" : null,
    onclick: () => onchange(n) }, text || String(n));
  nav.append(button(current - 1, "上一页", current <= 1 || !pages));
  const visible = new Set();
  for (let n = 1; n <= Math.min(3, pages); n++) visible.add(n);
  for (let n = Math.max(1, current - 2); n <= Math.min(pages, current + 2); n++) visible.add(n);
  for (let n = Math.max(1, pages - 3); n <= pages; n++) visible.add(n);
  let previous = 0;
  for (const n of [...visible].sort((a, b) => a - b)) {
    if (n - previous === 2) nav.append(button(n - 1));
    else if (n - previous > 2) nav.append(el("span", { class: "muted", "aria-hidden": "true" }, "…"));
    nav.append(button(n, null, n === current));
    previous = n;
  }
  nav.append(button(current + 1, "下一页", current >= pages));
  return nav;
}
export function statusChip(e) {
  if (e.backend === "cpu" && e.function_status === "pass") return el("span", { class: "chip gray", "data-tooltip": "CPU 结果作为精度标杆" }, "参考");
  if (e.function_status === "pass") {
    if (e.accuracy_status === "fail") return el("span", {}, el("span", { class: "chip warn" }, "精度失败"), e.accuracy_ref === "eager-npu" ? el("span", { class: "float-tag", title: "CPU 不支持，以 EagerNPU 为标杆" }, "EagerNPU") : null);
    return el("span", { class: "chip ok" }, e.accuracy_status === "pass" ? "通过" : "功能通过");
  }
  if (e.function_status === "unsupported") return el("span", { class: "chip gray" }, "不支持");
  const map = { gen_fail: "构图失败", ascgraph_fail: "构图失败", build_fail: "构图失败", compile_fail: "编译失败", runtime_fail: "运行失败", timeout: "超时" };
  const result = map[e.function_status] || e.function_status;
  const stage = STAGE_LABEL[e.fail_stage] || e.fail_stage;
  return el("span", { class: "chip bad", title: e.fail_stage || "" }, result + (stage && stage !== result ? " · " + stage : ""));
}
