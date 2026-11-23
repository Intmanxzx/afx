"use strict";
// 知识库: search/filter list + entry detail drawer + JSONL export.
import { el, api, qs, rkey, time, label, fmt, BACKEND_LABEL } from "./core.js";
import { select } from "./widgets.js";
import { PAGES, go, openCase } from "./router.js";
import { isAdmin } from "./auth.js";
import { openDrawer } from "./ui.js";
// Circular with page-repair.js (it imports knowledgeDrawer from here): safe because both
// sides only call the other's export lazily, from inside a click handler, long after both
// modules have finished loading — see the issueDrawer/knowledgeDrawer hoisted declarations.
import { issueDrawer } from "./page-repair.js";

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
    el("div", { class: "toolbar", style: { marginTop: "12px" } }, el("button", { onclick: exportKb }, "导出 JSONL")));
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
export async function knowledgeDrawer(id) {
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
