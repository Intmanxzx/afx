"use strict";
// 用例详情: DAG, Shape, backend comparison, execution environment, timing, stage
// timeline, failure details, ASCIR, history.
import { el, svg, api, errorBox, fmt, label, archLabel, time, ns } from "./core.js";
import { statusChip } from "./widgets.js";
import { PAGES, go } from "./router.js";
import { isAdmin } from "./auth.js";
import { retestDrawer, pinGlobal, batchClasses, replayCommand } from "./page-discovery.js";

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
  root.classList.add("case-page");
  const key = params.get("key");
  if (!key) { root.append(el("div", { class: "empty" }, "缺少用例 ID")); return; }
  const [spec, history] = await Promise.all([api(`/api/cases/${encodeURIComponent(key)}/spec`), api(`/api/cases/${encodeURIComponent(key)}/history?limit=50`)]);
  const c = history.case;
  let executions = history.executions, cursor = history.next_cursor;
  const shapes = (spec.shape_variants || []).length;
  let shape = Number(params.get("shape") || (shapes ? 0 : -1));
  const command = replayCommand(c.case_key, executions);
  const copy = el("button", { onclick: () => navigator.clipboard && navigator.clipboard.writeText(command).then(() => { copy.textContent = "已复制"; }) }, "复制");
  root.append(el("button", { class: "back-btn", "aria-label": "返回上一页",
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
    cmpBox.replaceChildren(el("h2", {}, "后端对比"),
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
  root.append(el("section", { class: "panel" }, el("h2", {}, `执行历史（${fmt(history.total)}）`),
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
  const diagnostics = d.diagnostics || {};
  const out = [el("h2", {}, `执行详情 · ${label(d.backend)}`, " ", statusChip(d), el("span", { class: "muted small" }, ` · ${d.user_name || d.user_id} · ${d.client_name || ""} · ${time(d.created_at)}`))];
  // per-shape results
  if (d.shapes && d.shapes.length) out.push(el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["Shape", "功能", "精度", "最大绝对误差", "最大相对误差", "Kernel", "E2E"].map(t => el("th", {}, t)))),
    el("tbody", {}, d.shapes.map(x => el("tr", { class: x.shape_index === shape ? "bt0" : "" }, el("td", {}, "S" + x.shape_index), el("td", {}, x.function_status), el("td", {}, x.accuracy_status || "—"),
      el("td", {}, x.max_abs_err ?? "—"), el("td", {}, x.max_rel_err ?? "—"), el("td", {}, ns(x.kernel_ns)), el("td", {}, ns(x.e2e_ns)))))));
  const logs = diagnostics.error_logs || [];
  out.push(el("h3", {}, "错误日志"), ...(logs.length ? logs.map(s =>
    el("details", { open: true }, el("summary", {}, s.name), el("pre", { class: "log" }, s.text))) : [el("div", { class: "muted" }, "未保存错误日志")]));
  const graphs = (diagnostics.ascir_graphs || []).filter(g => g.shape_index === shape);
  out.push(el("h3", {}, "ASCIR 表达的图结构"), el("div", { class: "muted small" }, "编译时实际生成的 ASCIR 构图代码（Schedule 之前）。"),
    el("pre", { class: "log" }, graphs.map(g => g.source).join("\n\n") || "未采集当前 Shape 的 ASCIR 构图代码（旧客户端、编译缓存命中或未进入 AF 构图）。错误日志不足以还原完整图结构。"));
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
    const title = node.label;
    const tip = title + "\n" + tensor + (Object.keys(node.attrs).length ? "\n" + JSON.stringify(node.attrs) : "");
    const g = svg("g", { transform: `translate(${node.x},${node.y})`, class: "dag-node " + node.kind, tabindex: 0, "aria-label": tip, "data-tooltip": tip }, svg("rect", { width: 250, height: 100, rx: 10 }));
    const line = (y, text, cls) => g.append(svg("text", { x: 12, y, class: cls }, text.length > 32 ? text.slice(0, 31) + "…" : text));
    line(26, title, "dag-op");
    line(50, node.shape ? "[" + node.shape.join(", ") + "]" : "Shape 未采集", "dag-tensor");
    line(76, node.dtype || "Dtype 未采集", "dag-tensor");
    canvas.append(g);
  }
  return el("div", { class: "dag-scroll" }, canvas);
}
