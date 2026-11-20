"use strict";
// 能力画像: Kernel performance scatter plot across backends, by feature axes.
import { el, api, qs, rkey, archLabel, label, fmt, archOptions, DIM_LABEL } from "./core.js";
import { select } from "./widgets.js";
import { PAGES, go, caseDrawer } from "./router.js";
import { isAdmin } from "./auth.js";

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
  } catch (err) { info.replaceChildren(el("div", { class: "error-box" }, String(err.message || err))); }
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
