"use strict";
// 平台运维: client fleet, per-user throughput, hourly chart, platform settings and
// per-user task configuration.
import { el, svg, api, qs, rkey, store, FLEET_SCOPE_KEY, time, fmt, label, archLabel, BACKEND_LABEL } from "./core.js";
import { PAGES, render, openCase, go } from "./router.js";
import { me, isAdmin } from "./auth.js";

PAGES.ops = async (root, params) => {
  const fleetScope = isAdmin() || !me ? "all" : store.get(FLEET_SCOPE_KEY, "all");
  const data = await api("/api/ops/summary" + qs({ scope: fleetScope === "all" ? "all" : null }));
  const mb = Object.values(data.db_bytes).reduce((a, b) => a + b, 0) / 1048576;
  root.append(el("div", { class: "tiles" }, [["在线客户端", fmt(data.online_clients)], ["近 24 小时执行", fmt(data.executions_24h), "按执行发生时间统计，不含 CPU 基线；含复测"], ["待上传", fmt(data.pending_uploads)],
    ["复测队列", `${fmt(data.retest_queue.pending || 0)} 等待 / ${fmt(data.retest_queue.leased || 0)} 执行中`], ["数据库", mb.toFixed(1) + " MB"]].map(([k, v, tip]) =>
    el("div", { class: "tile", "data-tooltip": tip }, el("div", { class: "v" }, v), el("div", { class: "k" }, k)))));
  const clientRows = data.clients.map(c => el("tr", {},
    el("td", {}, el("span", { class: "chip " + (c.online ? "ok" : "gray") }, c.online ? (c.busy ? "复测中" : "在线") : "离线")),
    el("td", {}, el("b", {}, c.display_name || "—")),
    el("td", { class: "small", style: { maxWidth: "280px", overflowWrap: "anywhere" }, "data-tooltip": c.client_home || "客户端尚未上报目录" }, c.client_home || "未上报"),
    el("td", {}, archLabel(c.npu_arch)),
    el("td", {}, c.card_count == null ? "未上报" : fmt(c.card_count)),
    el("td", {}, c.afx_version || "—", c.afx_version && data.latest_afx_version && c.afx_version !== data.latest_afx_version ? el("span", { class: "chip warn", style: { marginLeft: "4px" } }, "落后") : null,
      c.target_afx_version && c.update_status ? el("span", { class: "chip info" }, ({queued:"等待升级",running:"升级中",activating:"切换中"})[c.update_status.state] || "升级中") : null,
      c.target_afx_version ? el("span", { class: "chip info", style: { marginLeft: "4px" } }, "→ " + c.target_afx_version) : null),
    el("td", { class: "small" }, Object.entries(c.backend_versions).filter(([, v]) => v).map(([k, v]) => `${label(k)} ${v}`).join(" · ") || "—"),
    el("td", {}, clientControls(c, data.latest_afx_version)),
    el("td", { class: "small", "data-tooltip": "客户端主动上报的心跳时间；在线状态按此字段判断" }, c.last_heartbeat_at ? time(c.last_heartbeat_at) : "未收到心跳")
  ));
  const clientsTable = el("table", { class: "grid" },
    el("thead", {}, el("tr", {}, ["状态", "节点", "客户端目录", "芯片", "可见 NPU 数", "afx", "后端版本", "操作", "最后心跳"].map(t => el("th", {}, t)))),
    el("tbody", {}, clientRows)
  );
  const clientsPanel = el("section", { class: "panel" },
    el("h2", {}, "客户端", el("span", { class: "spacer" }), data.latest_afx_version ? el("span", { class: "small" }, "最新版本 " + data.latest_afx_version) : null),
    el("div", { class: "table-wrap" }, clientsTable)
  );
  root.append(clientsPanel);
  const efficiencyData = await api("/api/ops/efficiency");
  const efficiencyUsers = new Map((efficiencyData.users || []).map(u => [u.user_id, u]));
  const overviewRows = (data.users || []).map(u => {
    const e = efficiencyUsers.get(u.user_id) || {};
    const clients = data.clients.filter(c => c.user_id === u.user_id);
    const pendingUploads = clients.reduce((sum, c) => sum + (c.pending_uploads || 0), 0);
    return el("tr", {},
      el("td", {}, u.user_id), el("td", {}, fmt(u.online_clients) + "/" + fmt(u.clients)),
      el("td", {}, u.npu_total == null ? "未上报" : fmt(u.npu_used) + "/" + fmt(u.npu_total)), el("td", {}, fmt(u.executions_10m)),
      el("td", {}, Number(u.cases_per_minute || 0).toFixed(2)),
      el("td", {}, fmt(u.executions_24h) + "/" + fmt(u.uploads_24h)), el("td", {}, fmt(pendingUploads)), el("td", {}, fmt(e.success)),
      el("td", {}, fmt(e.completed)), el("td", {}, fmt(e.failed)), el("td", {}, fmt(e.case_errors)),
      el("td", {}, e.success_per_minute == null ? "—" : Number(e.success_per_minute).toFixed(2)));
  });
  const overviewPanel = el("section", { class: "panel" }, el("h2", {}, "用户运行概览"),
    el("p", { class: "muted small" }, "用例指标按总览页面口径：同一 case 在不同后端或重复执行只计 1 个唯一用例；近 10 分钟用例按最近执行过的唯一 case 计数。客户端显示在线数/总数，NPU 显示当前运行任务使用数/机器总数。24 小时执行/上传中的执行和上传均为最近 24 小时累计。完成 Shape = 成功 + 失败；成功要求功能通过且精度未失败，Shape 按后端和运行计数，动态用例可包含多个 Shape。Case 错误指唯一用例在统计窗口内没有任何 Shape 明细且至少一次执行失败。"),
    el("div", { class: "table-wrap" }, el("table", { class: "grid" },
      el("thead", {}, el("tr", {}, ["用户", "客户端（在线/总数）", "NPU 使用/总数", "近 10 分钟用例", "用例/分钟", "24 小时执行/上传", "待上传", "成功 Shape", "完成 Shape", "失败 Shape", "错误用例", "成功 Shape/min"].map(t => el("th", {}, t)))),
      el("tbody", {}, overviewRows.length ? overviewRows : [el("tr", {}, el("td", { colspan: 12, class: "empty" }, "暂无用户运行数据"))]))));
  root.append(overviewPanel);
  root.append(el("section", { class: "panel" }, el("h2", {}, "近 48 小时吞吐（用例/小时）"), hourly(data.hourly)));
  if (!me) return;
  const tabs = [["tasks","任务配置"], ...(isAdmin() ? [["platform","平台配置"],["rules","分类规则"],["jobs","刷新记录"]] : [])];
  const area = el("section"); area.dataset.custom = "1";
  const body = el("div", {role:"tabpanel"});
  const cache = new Map(); let ticket = 0;
  const nav = el("nav", {class:"toolbar", role:"tablist"});
  const select = async (id, job=null, updateUrl=true) => {
    if (!tabs.some(t => t[0] === id)) id = "tasks";
    const current = ++ticket;
    if (updateUrl) {
      const query = new URLSearchParams({tab:id}); if (job) query.set("job",job);
      history.replaceState(null,"", "#ops?" + query);
    }
    for (const button of nav.children) {
      button.classList.toggle("primary",button.dataset.tab===id);
      button.setAttribute("aria-selected",String(button.dataset.tab===id));
    }
    const key = id === "jobs" ? id + ":" + (job || "") : id;
    if (cache.has(key)) { const panel = cache.get(key); body.replaceChildren(panel); if (panel.refreshJob) panel.refreshJob(); return; }
    body.replaceChildren(el("p", {role:"status"}, "加载配置中……"));
    try {
      const panel = id === "tasks" ? await taskConfigurationPanel() : id === "jobs" ? await classificationJobsPanel(job,select) : await settingsPanel(id,select);
      cache.set(key,panel);
      if (current === ticket) body.replaceChildren(panel);
    } catch (err) {
      if (current === ticket) body.replaceChildren(el("div",{class:"error-box"},err.message),el("button",{onclick:() => select(id,job)},"重试"));
    }
  };
  for (const [id,title] of tabs) nav.append(el("button", {role:"tab", "data-tab":id, onclick:() => select(id)},title));
  area.append(nav,body); root.append(area);
  await select(params.get("tab") || "tasks",params.get("job"),false);
};
function clientControls(client, latestVersion) {
  const fresh = client.online && client.daemon_state_at && Date.now() - client.daemon_state_at < 300000;
  const state = { running: "运行中", stopped: "已停止" }[client.daemon_state];
  let status = el("span", { class: "chip " + (fresh && client.daemon_state === "running" ? "ok" : "gray"),
    "data-tooltip": client.daemon_state_at ? "状态上报时间：" + time(client.daemon_state_at) : "等待客户端上报任务状态" },
    state ? (fresh ? state : "上次状态：" + state) : "状态未上报");
  if (client.workload_states?.fuzz && client.workload_states?.retest) {
    const labels = { running: "已开启", stopped: "已停止", stopping: "停止中", error: "异常" };
    status = el("span", {style: {display: "inline-flex", gap: "4px"}},
      ...[["fuzz", "Fuzz"], ["retest", "复测"]].map(([name, label]) => {
        const value = client.workload_states[name].state;
        return el("span", {class: "chip " + (fresh && value === "running" ? "ok" : "gray")},
          `${fresh ? "" : "上次状态："}${label}：${labels[value] || "未知"}`);
      }));
  }
  if (!me || !(isAdmin() || client.mine)) return status;
  const actions = el("div", { style: { display: "flex", gap: "4px", flexWrap: "nowrap", whiteSpace: "nowrap", width: "max-content" } });
  const supported = !!client.workload_states?.fuzz && !!client.workload_states?.retest;
  for (const [workload, label] of [["fuzz", "fuzz"], ["retest", "复测"]]) {
    const pending = client.workload_controls?.[workload];
    const actual = client.workload_states?.[workload]?.state;
    const running = (pending?.state || actual || client.daemon_state) === "running";
    const state = running ? "stopped" : "running";
    actions.append(el("button", { class: "mini-btn", disabled: !supported || !!client.target_afx_version || actual === "stopping",
      "data-tooltip": !supported ? "客户端需升级后才能独立控制 Fuzz 和复测" : pending ? "指令已下发，等待客户端确认" : "", onclick: async event => {
      const button = event.currentTarget;
      button.disabled = true;
      try {
        await api("/api/clients/daemon", { method: "POST", body: { request_key: rkey(), client_id: client.client_id, workload, state } });
        render();
      } catch (err) { button.disabled = false; alert(err.message); }
    } }, (running ? "停" : "启动") + label));
  }
  const canUpgrade = isAdmin() && latestVersion && client.afx_version !== latestVersion && !client.target_afx_version;
  actions.append(el("button", { class: "mini-btn", disabled: !canUpgrade,
    "data-tooltip": client.target_afx_version ? "正在升级" : !isAdmin() ? "仅管理员可升级" : !latestVersion ? "等待获取最新版本" : client.afx_version === latestVersion ? "已是最新版本" : "", onclick: async event => {
    const button = event.currentTarget;
    if (!confirm(`将客户端 ${client.display_name || client.client_id} 停止后升级到 ${latestVersion}，继续吗？`)) return;
    button.disabled = true;
    try { await api("/api/admin/clients/update", { method: "POST", body: { request_key: rkey(), client_ids: [client.client_id] } }); render(); }
    catch (err) { button.disabled = false; alert(err.message); }
  } }, "升级"));
  if (isAdmin() && !client.online) actions.append(el("button", { class: "danger mini-btn", onclick: async event => {
    if (!confirm(`删除离线客户端 ${client.display_name || client.client_id}？`)) return;
    const button = event.currentTarget;
    button.disabled = true;
    try { await api(`/api/admin/clients/${client.client_id}`, { method: "DELETE" }); render(); }
    catch (err) { button.disabled = false; alert(err.message); }
  } }, "删除"));
  const pending = { running: "启动中（等待回执）", stopped: "停止中（等待回执）", restart: "重启中（等待回执）" }[client.target_daemon_state];
  const statuses = el("div", { style: { display: "flex", gap: "4px", alignItems: "center", flexWrap: "nowrap", whiteSpace: "nowrap", width: "max-content" } },
    status, ...Object.entries(client.workload_controls || {}).map(([name, command]) =>
      el("span", {class: "chip info"}, `${name === "fuzz" ? "Fuzz" : "复测"}${command.state === "running" ? "启动" : "停止"}中（等待回执）`)), pending ? el("span", { class: "chip info" }, pending) : null,
    client.target_afx_version && client.update_status?.phase === "stop_timeout" ? el("span", { class: "chip danger" }, "停止失败") : null,
    client.target_afx_version && client.update_status?.phase === "upgrade_timeout" ? el("span", { class: "chip danger" }, "升级失败") : null);
  return el("div", {}, statuses, actions);
}
// One bar per hour for the last 48 hours, counted by unique case IDs when they ran.
function hourly(series) {
  if (!series.length) return el("div", { class: "empty" }, "暂无执行数据");
  const W = 900, H = 170, pad = 26, bottom = 22, max = Math.max(1, ...series.map(s => s.cases || 0));
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "hourly throughput" });
  const bw = (W - 2 * pad) / 48, plotH = H - pad - bottom;
  const last = Math.floor(Date.now() / 3600000);
  const hourText = h => { const d = new Date(h * 3600000); return `${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:00`; };
  root.append(svg("line", { x1: pad, x2: W - pad, y1: H - bottom, y2: H - bottom, stroke: "var(--line, #8884)", "stroke-width": 1 }));
  const byHour = new Map(series.map(s => [s.hour, s]));
  for (let i = 0; i < 48; i++) {
    const hour = last - (47 - i), s = byHour.get(hour) || { cases: 0 };
    const h = s.cases ? Math.max(2, plotH * s.cases / max) : 0;
    if (h) root.append(svg("rect", { x: pad + i * bw, y: H - bottom - h, width: Math.max(1, bw - 2), height: h, fill: "var(--accent)" }));
    // Full-height transparent hit area per hour, so empty and tiny bars can be hovered too (the custom tooltip reads data-tooltip).
    root.append(svg("rect", { x: pad + i * bw, y: pad, width: bw, height: H - bottom - pad, fill: "transparent", "pointer-events": "all",
      "data-tooltip": `${hourText(hour)}\n${s.cases || 0} 个用例` }));
  }
  for (const k of [0, 12, 24, 36, 47]) root.append(svg("text", { x: pad + k * bw + bw / 2, y: H - 6, "text-anchor": k === 0 ? "start" : k === 47 ? "end" : "middle", class: "muted small" }, hourText(last - (47 - k))));
  root.append(svg("text", { x: pad, y: 14 }, `峰值 ${max} 个用例/小时`));
  return root;
}
async function settingsPanel(mode = "platform", selectTab = (tab,job) => go("ops", {tab,job})) {
  const [settings, pins, ruleData] = await Promise.all([api("/api/settings"), api("/api/regression/pins"), api("/api/admin/classification/rules")]);
  const fields = [["perf.tie_threshold", "Kernel 持平阈值", "number", "0.01", "判定性能差异接近持平的比例阈值。"], ["regression.budget", "回归默认预算", "number", "1", "每个问题默认抽取的回归用例总数。"], ["regression.max_budget", "单问题回归上限", "number", "1", "单个问题允许的最大回归用例数。"],
    ["issue.perf_anchor", "性能对比基准后端", "select", null, "自定义性能对比的基准后端，不是测试任务的执行后端。", ["cpu", "af-legacy", "af-mlir", "triton-npu", "eager-npu"]], ["issue.perf_slowdown", "性能问题阈值（倍）", "number", "0.1", "被测后端慢于基准达到该倍数时生成性能问题。"], ["issue.lease_days", "问题租约（天）", "number", "1", "Agent 领取问题后的最长占用时间。"], ["verify.max_cases", "单轮验证用例上限", "number", "1", "一次验证最多执行多少个用例。"]];
  const inputs = {};
  const grid = el("div", { class: "config-grid" });
  for (const [key, text, type, step, desc, options] of fields) {
    inputs[key] = type === "select" ? el("select", {}, options.map(v => el("option", {value:v}, BACKEND_LABEL[v] || v))) : el("input", { type, step, value: settings[key].value });
    inputs[key].value = settings[key].value;
    grid.append(el("label", {class:"config-field"}, el("span", {class:"config-label"}, text), el("span", {class:"config-help"}, desc), inputs[key]));
  }
  const shares = settings["regression.shares"].value;
  const shareInputs = {};
  grid.append(el("label", {class:"config-field config-share-field"}, el("span", {class:"config-label"}, "回归四层占比"), el("span", {class:"config-help"}, "控制邻域、哨兵、历史和随机用例在回归任务中的比例。"), el("span", {class:"toolbar"}, Object.entries({ neighbor: "邻域", sentinel: "哨兵", history: "历史", random: "随机" }).map(([k, t]) => {
    shareInputs[k] = el("input", { type: "number", min: 0, max: 100, value: shares[k], style: { width: "70px" } }); return el("label", {}, t, " ", shareInputs[k]); }))));
  const out = el("div");
  const rulesText = el("textarea", { rows: 8, style: { width: "100%", fontFamily: "monospace" }, value: JSON.stringify(ruleData.rules || [], null, 2) });
  rulesText.value = JSON.stringify(ruleData.rules || [], null, 2);
  const ruleOut = el("div", {role: "status", "aria-live": "polite", class: "panel"}, "预览结果将在这里显示。预览不会修改数据。");
  let ruleBusy = false;
  let ruleRevision = ruleData.revision;
  let preview = null;
  const backups = el("select", {}, el("option", {value:""}, "选择历史备份"), (ruleData.history || []).map(h => el("option", {value:String(h.id)}, "版本 " + h.revision + " · " + time(h.at))));
  const restoreBackup = () => {
    const h = (ruleData.history || []).find(h => String(h.id) === backups.value);
    if (h) { rulesText.value = JSON.stringify(h.rules, null, 2); ruleOut.textContent = "已载入备份，保存后恢复生效"; }
  };
  const saveRules = async apply => {
    if (ruleBusy) return;
    ruleBusy = true;
    ruleOut.textContent = !apply ? "正在预览，请稍候……只读取日志摘要，不会修改数据。" : "正在保存规则，请稍候……";
    try {
      const rules = JSON.parse(rulesText.value);
      if (apply === true && (!preview || preview.text !== rulesText.value)) throw new Error('请先预览当前规则，再提交已确认的记录');
      if (!apply) {
        const r = await api("/api/admin/classification/refresh", { method: "POST", body: { rules, apply: false } });
        preview = {id:r.preview_id, text:rulesText.value};
        const counts = new Map();
        for (const c of r.changes || []) counts.set(c.rule_id, (counts.get(c.rule_id) || 0) + 1);
        ruleOut.replaceChildren(el("h4", {}, `预览完成：待更新 ${r.count} 条`),
          el("p", {}, r.count ? "预览未修改数据。以下展示前 50 条记录。" : "没有需要更新的记录：规则未命中，或命中记录已是目标分类。"),
          el("div", {class:"toolbar"}, [...counts].map(([id,n]) => el("span", {class:"chip info"}, `${id}：${n}`))),
          el("div", {class:"table-wrap"}, el("table", {class:"grid"},
            el("thead", {}, el("tr", {}, ["用例","规则","原分类","新分类","日志摘要"].map(x => el("th", {}, x)))),
            el("tbody", {}, (r.changes || []).slice(0,50).map(c => el("tr", {},
              el("td", {}, el("a", {href:"#case?key=" + encodeURIComponent(c.case_key) + "&exec=" + encodeURIComponent(c.exec_uuid)}, c.case_key || c.exec_uuid)),
              el("td", {}, c.rule_id), el("td", {}, c.before.slice(0,3).join(" / ")), el("td", {}, c.after.slice(0,3).join(" / ")), el("td", {}, c.log_summary)))))));
        return;
      }
      const saved = await api("/api/admin/classification/rules", { method: "PUT", body: { rules, base_revision: ruleRevision } });
      ruleRevision = saved["classification.rules"].revision;
      rulesText.value = JSON.stringify(saved["classification.rules"].value, null, 2);
      const latest = await api("/api/admin/classification/rules");
      ruleData.history = latest.history;
      backups.replaceChildren(el("option", {value:""}, "选择历史备份"), latest.history.map(h => el("option", {value:String(h.id)}, "版本 " + h.revision + " · " + time(h.at))));
      if (apply === "save") { ruleOut.textContent = "已保存并自动备份上一版本，新上报数据立即生效"; return; }
      const job = await api("/api/admin/classification/jobs", {method:"POST", body:{request_key:rkey(), rules:saved["classification.rules"].value, preview_id:preview.id}});
      await selectTab("jobs",job.job_id);
    } catch (err) { ruleOut.replaceChildren(el("div", {class:"error-box",role:"alert"}, String(err.message || err))); }
    finally { ruleBusy = false; }
  };
  const save = async () => {
    const changes = {}, base = {};
    for (const [key, , type] of fields) { const v = type === "number" ? Number(inputs[key].value) : inputs[key].value.trim(); if (JSON.stringify(v) !== JSON.stringify(settings[key].value)) { changes[key] = v; base[key] = settings[key].revision; } }
    const newShares = Object.fromEntries(Object.entries(shareInputs).map(([k, i]) => [k, Number(i.value)]));
    if (JSON.stringify(newShares) !== JSON.stringify(shares)) { changes["regression.shares"] = newShares; base["regression.shares"] = settings["regression.shares"].revision; }
    if (!Object.keys(changes).length) { out.replaceChildren(el("span", { class: "muted" }, "没有改动")); return; }
    try { await api("/api/admin/settings", { method: "PUT", body: { changes, base_revisions: base } }); out.replaceChildren(el("span", { class: "chip ok" }, "已保存")); }
    catch (err) { out.replaceChildren(el("div", { class: "error-box" }, String(err.message || err))); }
  };
  const loadBuiltin = () => {
    try {
      const current = JSON.parse(rulesText.value), ids = new Set(current.map(r => r.id));
      rulesText.value = JSON.stringify([...current, ...(ruleData.builtin_rules || []).filter(r => !ids.has(r.id))], null, 2);
      ruleOut.textContent = "已补全内置规则，现有编辑已保留";
    } catch (err) { ruleOut.textContent = err.message; }
  };
  const platform = el("section", { class: "panel config-panel" }, el("h2", {}, "平台配置"), el("div", { class: "config-actions config-actions-top" }, el("button", { class: "primary", onclick: save }, "保存配置"), out), grid);
  if (mode === "rules") return el("section", {class:"panel config-panel"}, el("h3", {}, "日志分类规则"), el("p", { class: "config-intro" }, "规则会同时用于新上报数据和存量刷新；匹配字段是页面显示的日志摘要。"), el("div", { class: "config-actions" }, el("button", { onclick: loadBuiltin }, "补全内置规则"), backups, el("button", {onclick:restoreBackup}, "载入备份")), rulesText, el("div", { class: "config-actions" }, el("button", { onclick: () => saveRules(false) }, "预览命中"), el("button", {onclick:() => saveRules("save")}, "保存规则"), el("button", { class: "primary", onclick: () => saveRules(true) }, "保存并刷新"), ruleOut));
  return el("div", {}, platform,
    el("h3", {}, `全局与特征必选回归用例（${pins.length}）`),
    pins.length ? el("table", { class: "grid" }, el("tbody", {}, pins.map(p => el("tr", {}, el("td", {}, p.scope === "global" ? "全局" : p.scope_value), el("td", {}, el("a", { href: "#", class: "mono small", onclick: e => { e.preventDefault(); openCase(p.case_key); } }, p.case_key)),
      el("td", { class: "small" }, p.reason || ""), el("td", { class: "small" }, p.added_by), el("td", {}, el("button", { class: "danger", onclick: async () => {
        try { await api("/api/regression/pins/" + p.pin_id, { method: "DELETE", body: { request_key: rkey() } }); render(); } catch (err) { alert(err.message); } } }, "移除")))))) : el("div", { class: "muted" }, "在问题发现页选中用例后可加入回归集"));
}

async function taskConfigurationPanel() {
  const panel = el("section", {class:"panel config-panel"}, el("h2", {}, "测试任务配置"),
    el("p", {class:"config-intro"}, "配置客户端下一次启动后执行什么任务。常用选项直接展示，生成细节收在“高级参数”中。"));
  const target = el("select", {}, el("option", {value:"me"}, "我的配置"), isAdmin() ? el("option", {value:"default"}, "管理员默认配置") : null);
  const body = el("div", {}), notice = el("div", {role:"status"});
  let config, collect;
  async function load() {
    try {
      const [result, catalog] = await Promise.all([api("/api/config/"+target.value),api("/api/config/operators")]);
      config = result.config;
      const gen = config.workload.generation ||= {};
      const cleanup = config.cleanup ||= {debug_days:2, runs_days:2};
      const controls = [];
      const common = el("div", {class:"config-grid"});
      const details = el("details", {class:"config-advanced"}, el("summary", {}, "高级参数：生成范围、超时和性能"));
      const advanced = el("div", {class:"config-grid"}); details.append(advanced);
      const help = {
        "静态 / 动态": "生成固定 Shape、动态 Shape，或两者混合。", "Tensor 规模": "一组预设范围，会联动生成范围参数。",
        "数据类型（仅保留全部）": "当前统一使用全部数据类型，客户端会根据算子能力生成。", "最大节点数（上限 64）": "单个计算图最多包含多少个算子节点，最大不能超过 64。",
        "后端（多选）": "选择要执行的后端，可同时选择多个。", "设备": "客户端使用的设备，例如 npu:0,npu:1。",
        "任务规模": "continuous 持续执行；finite 执行指定数量后停止。", "有限任务用例数": "finite 模式下要执行的用例数。",
        "算子（多选；不选表示全部）": "只生成选中的算子；不选择表示使用全部算子。", "性能测试": "是否额外采集性能数据，会增加执行时间。",
        "上传结果": "是否把执行结果上传到平台。", "执行环境": "任务在哪类环境中执行。",
        "保存本地结果": "是否在客户端本地保留结果数据库。",
      };
      function field(parent, title, object, key, values, fallback, multiple=false) {
        const node = values ? el("select", {multiple}, ...values.map(v=>el("option",{value:v},v))) : el("input", {type:typeof fallback === "number" ? "number":"text"});
        if (title.startsWith("最大节点数")) node.max = "64";
        const value=object[key] ?? fallback;
        if (multiple) for (const option of node.options) option.selected=value.includes(option.value);
        else node.value=value;
        controls.push(()=>object[key]=multiple ? [...node.selectedOptions].map(o=>o.value) : typeof fallback === "number" ? Number(node.value) : node.value);
        parent.append(el("label",{class:"config-field"},el("span",{class:"config-label"},title),el("span",{class:"config-help"},help[title] || "高级生成参数，可按需要调整。"),node)); return node;
      }
      function checkboxGroup(parent, title, object, key, values, fallback) {
        const current = object[key] || fallback;
        const checks = values.map(([value, text]) => { const input=el("input",{type:"checkbox",value,checked:current.includes(value)}); return el("label",{class:"config-check"},input,el("span",{},text)); });
        parent.append(el("div",{class:"config-field config-check-group"},el("span",{class:"config-label"},title),el("span",{class:"config-help"},help[title]),el("div",{class:"config-checks"},checks)));
        controls.push(()=>object[key]=checks.map(x=>x.querySelector('input')).filter(x=>x.checked).map(x=>x.value));
      }
      field(common,"静态 / 动态",gen,"shape_mode",["static","dynamic","mixed"],"mixed");
      const scale = field(common,"Tensor 规模",gen,"tensor_scale",["default","small","medium","large","16k","32k","64k","128k"],"128k");
      field(common,"数据类型（仅保留全部）",gen,"dtype_mode",["all"],"all");
      field(common,"最大节点数（上限 64）",gen,"max_nodes",null,10);
      checkboxGroup(common,"后端（多选）",config.execution,"backends",Object.keys(BACKEND_LABEL).map(v=>[v,BACKEND_LABEL[v]]),["cpu","af-legacy","af-mlir"]);
      const deviceValue=config.execution.devices.includes("all")?"all":(config.execution.devices[0]||"all");
      const devices=el("select",{},[["all","全部本地设备（自适应）"],["npu:0","npu:0"],["npu:1","npu:1"]].map(([v,t])=>el("option",{value:v},t))); devices.value=deviceValue;
      common.append(el("label",{class:"config-field"},el("span",{class:"config-label"},"设备"),el("span",{class:"config-help"},help["设备"]),devices));
      controls.push(()=>config.execution.devices=[devices.value]);
      const term=config.termination;
      const taskMode=field(common,"任务规模",term,"mode",["continuous","finite"],"continuous");
      const maxCases=field(common,"有限任务用例数",term,"max_cases",null,100);
      const updateMaxCases=()=>{maxCases.disabled=taskMode.value==="continuous";}; taskMode.addEventListener("change",updateMaxCases); updateMaxCases();
      // 算子目录数量较大，暂不在任务配置页展开；已有 ops 配置继续原样保留。
      for (const [title,obj,key,values,fallback] of [["性能测试",config.evaluation.performance,"enabled",[["false","不启用"],["true","启用"]],"false"],["上传结果",config.results,"upload",[["true","上传"],["false","不上传"]],"true"]]) {
        const select=el("select",{},values.map(([v,t])=>el("option",{value:v},t))); select.value=String(obj[key]??fallback); common.append(el("label",{class:"config-field"},el("span",{class:"config-label"},title),el("span",{class:"config-help"},help[title]),select)); controls.push(()=>obj[key]=select.value==="true");
      }
      for (const [title,key,value] of [["每个维度 Shape（最小）","dim_min",1],["每个维度 Shape（最大）","dim_max",131072],["单 Tensor 元素上限","max_elements_per_tensor",524288],["动态 Shape 数量","dynamic_variants",3]]) field(advanced,title,gen,key,null,value);
      field(advanced,"维度分布",gen,"dim_dist_mode",["uniform","half_normal","log_normal"],"half_normal");
      field(advanced,"数值分布",gen,"value_dist_mode",["log_scale","llm","recommendation"],"log_scale");
      help["构图方法"] = "symbolic-cinit：符号构图并初始化；symbolic：纯符号构图；concolic：符号与具体值结合。";
      field(advanced,"构图方法",gen,"method",["symbolic-cinit","symbolic","concolic"],"symbolic-cinit");
      help["生成超时（秒）"] = "单个用例的 NNSmith 构图与生成阶段最长时间，不是整批任务时间。";
      const generationTimeout=field(advanced,"生成超时（秒）",gen,"timeout",null,60); generationTimeout.value=parseInt(gen.timeout||60,10)||60; controls.push(()=>gen.timeout=Math.max(1,Number(generationTimeout.value))+"s");
      help["执行环境"] = "客户端已安装的执行环境；local 表示客户端本地默认环境。"; field(advanced,"执行环境",config,"environment",["local"],"local");
      help["执行超时（秒）"] = "单个用例从执行开始到结束的最长时间。"; const executionTimeout=field(advanced,"执行超时（秒）",config.execution,"timeout",null,120); executionTimeout.value=parseInt(config.execution.timeout||120,10)||120; controls.push(()=>config.execution.timeout=Math.max(1,Number(executionTimeout.value))+"s");
      help["Worker 回收阶段数"] = "每个 Worker 最多连续处理多少个任务，达到后回收重启，用于控制内存增长。"; field(advanced,"Worker 回收阶段数",config.execution.resources,"worker_max_tasks",null,100);
      const save=el("select",{},[["true","保存"],["false","不保存"]].map(([v,t])=>el("option",{value:v},t))); save.value=String(config.results.save);
      advanced.append(el("label",{class:"config-field"},el("span",{class:"config-label"},"保存本地结果"),el("span",{class:"config-help"},help["保存本地结果"]),save)); controls.push(()=>config.results.save=save.value==="true");
      help["预热次数"] = "每个 Shape 先执行的预热次数，结果不计入性能统计。"; field(advanced,"预热次数",config.evaluation.performance,"warmup_steps",null,3);
      help["测量次数"] = "预热后正式测量的次数；默认是预热 3 次、测量 50 次。"; field(advanced,"测量次数",config.evaluation.performance,"measured_steps",null,50);
      const retention = el("div", {class:"config-grid"});
      for (const [title, key] of [["Debug 保留天数", "debug_days"], ["Runs 保留天数", "runs_days"]]) {
        help[title] = "默认保留最近 2 天；0 关闭清理。同步后生效，每小时检查。使用中的目录和未完成上传的任务会保留。";
        const input = field(retention,title,cleanup,key,null,2); input.min="0"; input.max="3650"; input.step="1";
      }
      const presets={small:[1,64,65536,10,"uniform"],medium:[8,4096,1048576,30,"uniform"],large:[8,8192,16777216,60,"uniform"],"16k":[8,16384,65536,90,"log_normal"],"32k":[8,32768,131072,120,"log_normal"],"64k":[8,65536,262144,180,"log_normal"],"128k":[8,131072,524288,300,"log_normal"],default:[1,131072,524288,300,"half_normal"]};
      scale.addEventListener("change",()=>{
        const values=presets[scale.value];
        const labels=[...advanced.querySelectorAll('label')];
      for(const [label,value] of [["每个维度 Shape（最小）",values[0]],["每个维度 Shape（最大）",values[1]],["单 Tensor 元素上限",values[2]],["生成超时（秒）",values[3]],["维度分布",values[4]]]) {
          const row=labels.find(x=>x.querySelector('.config-label')?.textContent===label);if(row)row.querySelector('input,select').value=value;
        }
      });
      collect=()=>{controls.forEach(fn=>fn());if(term.mode==='continuous')delete term.max_cases;return config;};
      body.replaceChildren(common,el("h3", {}, "客户端自动清理"),retention,details); notice.textContent=result.source==='personal' ? "使用个人配置":"继承管理员默认配置";
    } catch(err) {notice.textContent=err.message;}
  }
  target.addEventListener("change",load);
  const actions = el("div", {class:"config-actions"}, el("label", {class:"config-target"}, el("span", {}, "配置来源"), target),
    el("button",{class:"primary",onclick:async()=>{try {await api("/api/config/"+target.value,{method:"POST",body:{config:collect()}});notice.textContent="已保存：清理配置同步后生效，任务参数下次启动生效";}catch(err){notice.textContent=err.message;}}},"保存配置"),
    el("button",{onclick:async()=>{try {await api("/api/config/"+target.value,{method:"POST",body:{config:null}});await load();}catch(err){notice.textContent=err.message;}}},"恢复默认"));
  panel.append(actions,body,notice);
  await load();return panel;
}

async function classificationJobsPanel(jobId, selectTab) {
  const panel = el("section", {class:"panel"}, el("h2", {}, "分类刷新任务"));
  const content = el("div", {role:"status", "aria-live":"polite"}); panel.append(content);
  const phases = {queued:"等待执行",scanning:"扫描日志摘要",applying:"更新分类",issues:"更新关联问题",statistics:"重建统计",completed:"已完成",failed:"失败"};
  let timer, loading = false;
  const load = async () => {
    if (loading) return;
    loading = true; clearTimeout(timer);
    try {
      const jobs = await api("/api/admin/classification/jobs");
      const job = jobId ? await api("/api/admin/classification/jobs/" + encodeURIComponent(jobId)) : null;
      content.replaceChildren(el("div", {class:"toolbar"}, el("button", {onclick:load}, "刷新状态"), el("a", {href:"#ops?tab=rules", onclick:e => { if (selectTab) { e.preventDefault(); selectTab("rules"); } }}, "返回分类规则")),
        el("table", {class:"grid"}, el("thead", {}, el("tr", {}, ["任务","阶段","已扫描","待更新","提交时间"].map(x => el("th", {}, x)))),
          el("tbody", {}, jobs.map(j => el("tr", {}, el("td", {}, el("a", {href:"#ops?tab=jobs&job="+j.job_id, onclick:e => { if (selectTab) { e.preventDefault(); selectTab("jobs",j.job_id); } }},j.job_id)), el("td", {}, phases[j.phase] || j.phase), el("td", {},fmt(j.scanned)), el("td", {},fmt(j.matched)), el("td", {},time(j.created_at)))))));
      if (job) {
        content.append(el("h3", {}, "任务 " + job.job_id), el("p", {}, phases[job.phase] || job.phase), el("p", {}, "离开或刷新页面不会取消任务。服务重启后会继续处理未完成任务。"));
        if (job.error) content.append(el("div", {class:"error-box"},job.error));
        if (job.result) {
          content.append(el("p", {}, "已完成：更新 " + job.result.count + " 条；审计批次 " + job.result.batch_id),
            el("div", {class:"toolbar"},Object.entries(job.result.rules).map(([id,n]) => el("span", {class:"chip info"},id+"："+n))),
            el("h4", {}, "结果样本（最多 50 条）"),
            el("table", {class:"grid"}, el("tbody", {},job.result.samples.map(c => el("tr", {},el("td", {},el("a", {href:"#case?key="+encodeURIComponent(c.case_key)+"&exec="+c.exec_uuid},c.case_key)),el("td", {},c.rule_id),el("td", {},c.before.slice(0,3).join(" / ")),el("td", {},c.after.slice(0,3).join(" / ")))))));
        }
      }
      if (jobs.some(j => ["queued","running"].includes(j.status))) timer = setTimeout(() => { if (panel.isConnected) load(); },3000);
    } catch (err) { content.replaceChildren(el("div", {class:"error-box"},err.message),el("button",{onclick:load},"重试")); }
    finally { loading = false; }
  };
  panel.refreshJob = load;
  await load(); return panel;
}
