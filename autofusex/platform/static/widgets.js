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
export function statusChip(e) {
  if (e.backend === "cpu" && e.function_status === "pass") return el("span", { class: "chip gray", "data-tooltip": "CPU 结果作为精度标杆" }, "参考");
  if (e.function_status === "pass") {
    if (e.accuracy_status === "fail") return el("span", {}, el("span", { class: "chip warn" }, "精度失败"), e.accuracy_ref === "eager-npu" ? el("span", { class: "float-tag", title: "CPU 不支持，以 EagerNPU 为标杆" }, "EagerNPU") : null);
    return el("span", { class: "chip ok" }, e.accuracy_status === "pass" ? "通过" : "功能通过");
  }
  if (e.function_status === "unsupported") return el("span", { class: "chip gray" }, "不支持");
  const map = { build_fail: "构图失败", compile_fail: "编译失败", runtime_fail: "运行失败", timeout: "超时" };
  return el("span", { class: "chip bad", title: e.fail_stage || "" }, (map[e.function_status] || e.function_status) + (e.fail_stage ? " · " + (STAGE_LABEL[e.fail_stage] || e.fail_stage) : ""));
}
