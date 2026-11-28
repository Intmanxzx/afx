"use strict";
// 修复闭环: issue list with a subway-style status track, and the issue detail drawer.
import { el, api, qs, rkey, time, label, fmt, archLabel, STATUS_LABEL } from "./core.js";
import { select, pagination, statusChip } from "./widgets.js";
import { PAGES, go, openCase } from "./router.js";
import { TREATMENTS, REPAIR_DEFAULT_RESULTS, REPAIR_DEFAULT_STATUSES } from "./result-filter.js";
import { isAdmin } from "./auth.js";
import { openDrawer } from "./ui.js";
import { knowledgeDrawer } from "./page-knowledge.js";

const STATIONS = [["claimed", "领取"], ["verifying", "验证"], ["regressing", "回归"], ["summarizing", "总结"], ["resolved", "关闭"]];
function issueClassification(issue) {
  if (issue.dimension === "accuracy") return el("span", { class: "chip warn" }, "精度失败");
  if (issue.dimension === "performance") return el("span", { class: "chip bad" }, "性能问题");
  const classifications = issue.classifications || [];
  return classifications.length ? el("span", {}, classifications.map(statusChip))
    : el("span", { class: "chip gray" }, issue.current_case_count === 0 ? "无当前关联" : "未知");
}
function accuracyReason(issue) {
  const raw = issue.category || issue.primary_category || "";
  const parts = raw.split("·").map(value => value.trim());
  if (parts.length < 5) return raw;
  const [primary, anomaly, severity, ratio, pattern, ...features] = parts;
  const anomalies = {finite:"有限值", nan:"NaN", inf:"Inf", "nan+inf":"NaN+Inf", unknown:"未知数值"};
  const patterns = {none:"无", all:"全量", contiguous:"连续", periodic:"周期", scattered:"离散"};
  const compact = value => ({nonfinite:"非有限值", unknown:"未知"}[value] || value.replaceAll("<=", "≤").replaceAll("-", "–"));
  return [primary, anomalies[anomaly] || anomaly, `超容差:${compact(severity)}`,
    `错误比例:${compact(ratio)}`, `错误形态:${patterns[pattern] || pattern}`,
    features.filter(Boolean).length ? `特征:${features.filter(Boolean).join("、")}` : ""].filter(Boolean).join("/");
}
function treatmentChip(issue) {
  return el("span", { class: "chip priority-" + issue.treatment }, new Map(TREATMENTS).get(issue.treatment));
}
function similarLink(issue) {
  return issue.similar_to ? el("a", { href: "#", class: "chip warn", onclick: e => {
    e.preventDefault(); issueDrawer(issue.similar_to);
  } }, "与问题 " + issue.similar_to + " 相似") : null;
}
function issueCell(issue) {
  const reason = issue.dimension === "function" ? issue.log_summary || "" : issue.dimension === "accuracy" ? accuracyReason(issue) : issue.category || "";
  return el("td", { class: "issue-cell" },
    el("a", { href: "#", class: "issue-summary-link", "aria-label": `查看问题 ${issue.issue_id} 详情`,
      onclick: e => { e.preventDefault(); issueDrawer(issue.issue_id); } },
      el("span", { class: "issue-summary" },
        el("span", { class: "issue-number" + (issue.similar_to ? " similar" : "") }, issue.issue_id),
        treatmentChip(issue),
        issueClassification(issue),
        el("span", { class: "issue-reason", title: reason }, reason.slice(0, 110)))));
}
function subway(issue) {
  const index = { open: -1, claimed: 0, verifying: 1, regressing: 2, summarizing: 3, resolved: 4, closed: 4 }[issue.status];
  const lastRound = (issue.rounds || [])[issue.rounds ? issue.rounds.length - 1 : 0];
  const bounced = issue.status === "claimed" && issue.round_no > 0;
  const box = el("div", { class: "subway", title: STATUS_LABEL[issue.status] + (bounced ? `（第 ${issue.round_no} 轮未通过，退回修复）` : "") });
  STATIONS.forEach(([key, text], i) => {
    if (i) box.append(el("div", { class: "line" + (i <= index ? " done" : "") }));
    let cls = i < index ? "done" : i === index ? (issue.status === "resolved" || issue.status === "closed" ? "done" : "now") : "";
    if (bounced && i === 1) cls = "back";
    box.append(el("div", { class: "st " + cls }, el("div", { class: "dot" }), text));
  });
  void lastRound;
  return box;
}
PAGES.repair = async (root, params) => {
  const dimension = params.get("dimension") || "function";
  const status = params.get("status") || REPAIR_DEFAULT_STATUSES.join(",");
  const navigate = patch => go("repair", {...Object.fromEntries(params), ...patch});
  const q = params.get("q") || "";
  const requestedPage = Number(params.get("page") || 1);
  const pageNumber = Number.isSafeInteger(requestedPage) && requestedPage > 0 ? requestedPage : 1;
  const tabs = el("div", { class: "tabs" }, [["function", "功能问题"], ["accuracy", "精度问题"], ["performance", "性能问题"]].map(([k, t]) =>
    el("button", { class: k === dimension ? "active" : "", onclick: () => navigate({ dimension: k, results: null, page: null }) }, t)));
  const issueSearch = el("input", { class: "issue-search", placeholder: "搜索问题 ID、标题或日志", value: q, maxlength: 128 });
  issueSearch.addEventListener("keydown", e => { if (e.key === "Enter") navigate({ q: issueSearch.value.trim() || null, page: null }); });
  const panel = el("section", { class: "panel" }, el("h2", {}, "问题列表", el("span", { class: "spacer" }), issueSearch), tabs);
  root.append(panel);
  const body = el("tbody");
  const table = el("table", { class: "grid" }, el("thead", {}, el("tr", {}, ["问题", "涉及后端", "匹配 / 总用例", "状态", "领取人", "更新时间"].map(t => el("th", {}, t)))), body);
  const pagerTop = el("div", { class: "toolbar" });
  const more = el("div", { class: "toolbar", style: { marginTop: "10px" } });
  panel.append(pagerTop, el("div", { class: "table-wrap" }, table), more);
  const load = async () => {
    const page = await api("/api/issues" + qs({ dimension, status, q, page: pageNumber, limit: 50, selected_backends:params.get("backends"), chips:params.get("chips"), results:dimension === "function" ? params.get("results") || REPAIR_DEFAULT_RESULTS.join(",") : null, categories:params.get("categories"), treatment:params.get("treatment") || "low,normal,high", scope:params.get("scope") || "current" }));
    for (const issue of page.issues) body.append(el("tr", {},
      issueCell(issue),
      el("td", {}, (issue.backends || [issue.backend]).map(label).join(" · ")), el("td", {}, fmt(issue.matched_case_count) + " / " + fmt(issue.case_count)), el("td", {}, subway(issue)),
      el("td", { class: "small" }, issue.owner_user ? `${issue.owner_user} / ${issue.agent_id}` : "—"), el("td", { class: "small" }, time(issue.updated_at))));
    if (!body.children.length) body.append(el("tr", {}, el("td", { colspan: 6, class: "empty" }, "暂无问题")));
    for (const target of [pagerTop, more]) target.replaceChildren(pagination(page.page, page.total_pages, page.total, pageNumber =>
      navigate({ page: pageNumber })));
  };
  await load();
};
export async function issueDrawer(issueId) {
  openDrawer(el("div", { class: "empty" }, "加载中…"));
  const [issue, cases, history, regression] = await Promise.all([api("/api/issues/" + issueId),
    api("/api/issues/" + issueId + "/cases?scope=current"),
    api("/api/issues/" + issueId + "/cases?scope=history"),
    api("/api/issues/" + issueId + "/cases?role=regression")]);
  const rounds = el("div");
  for (const r of issue.rounds) rounds.append(roundPanel(issueId, r));
  const adminBox = isAdmin() && !["resolved", "closed"].includes(issue.status) ? el("div", { class: "toolbar" },
    el("button", { class: "danger", onclick: async () => { const reason = prompt("关闭原因（不修复/误报/重复）："); if (!reason) return; try { await api(`/api/admin/issues/${issueId}/close`, { method: "POST", body: { request_key: rkey(), reason } }); issueDrawer(issueId); } catch (err) { alert(err.message); } } }, "关闭问题"),
    el("button", { onclick: async () => { const into = prompt("合并到哪个问题 ID？"); if (!into) return; try { await api(`/api/admin/issues/${issueId}/merge`, { method: "POST", body: { request_key: rkey(), into: into.trim() } }); issueDrawer(issueId); } catch (err) { alert(err.message); } } }, "合并到…")) : null;
  openDrawer(el("h2", {}, issue.issue_id, " ", el("span", { class: "chip info" }, STATUS_LABEL[issue.status]), " ", similarLink(issue)), subway(issue),
    el("div", { class: "kv", style: { marginTop: "12px" } }, [["维度", { function: "功能", accuracy: "精度", performance: "性能" }[issue.dimension]], ["后端", (issue.backends || [issue.backend]).map(label).join(" · ")], ["分类", issueClassification(issue)], ["主算子类别", issue.primary_category], ["处理标签", treatmentChip(issue)],
      ["当前 / 累计用例", fmt(issue.current_case_count) + " / " + fmt(issue.case_count)], ["芯片", issue.arches.map(archLabel).join(", ") || "—"], ["领取", issue.owner_user ? `${issue.owner_user} / ${issue.agent_id}` : "—"],
      ["修复版本", issue.fix_version || "—"], ["代码链接", issue.fix_ref ? el("a", { href: issue.fix_ref, target: "_blank", rel: "noopener noreferrer" }, issue.fix_ref) : "—"],
      ["知识", issue.knowledge_id ? el("a", { href: "#", onclick: e => { e.preventDefault(); knowledgeDrawer(issue.knowledge_id); } }, issue.knowledge_id) : "—"],
      ["首次出现", time(issue.first_seen_at)]].flat().map(v => el("div", {}, v))),
    issue.dimension === "accuracy" ? el("p", {class:"small"}, accuracyReason(issue)) : null,
    issue.log_summary ? el("pre", { class: "log" }, issue.log_summary) : null,
    isAdmin() ? el("div", {class:"toolbar"}, "处理标签", select(TREATMENTS, issue.treatment, async treatment => {
      try { await api('/api/admin/issues/' + issueId + '/treatment', {method:"POST",body:{request_key:rkey(),treatment}}); await issueDrawer(issueId); }
      catch(err) { alert(err.message); }
    })) : null, adminBox, isAdmin() ? adminStatusEditor(issue) : null,
    el("h3", {}, "验证与回归"), issue.rounds.length ? rounds : el("div", { class: "muted" }, "尚未提交验证"),
    caseSection(issueId, cases, "current", "当前关联用例"),
    el("p", {class:"muted small"}, "最新执行结果用于判断用例目前是否通过；历史记录仅说明当时的结果。候选修复包的结果见验证与回归轮次。"),
    caseSection(issueId, history, "history", "历史关联用例"),
    regression.cases.length ? caseSection(issueId, regression, "regression", "回归证据") : null,
    el("h3", {}, "事件"), el("ul", { class: "timeline" }, issue.events.map(e => el("li", {}, `${time(e.at)} · ${e.actor} · ${e.kind}`, e.to_status ? ` → ${STATUS_LABEL[e.to_status] || e.to_status}` : "", e.detail ? el("div", { class: "muted" }, e.detail) : null))));
}
function roundTaskStatus(task) {
  const states = {pass:["ok", "通过"], fail:["bad", "失败"], invalid:["warn", "无效"],
    skip:["gray", "跳过"], pending:["gray", "待执行"], leased:["info", "执行中"],
    done:["gray", "已完成"], cancelled:["gray", "已取消"]};
  const [tone, text] = states[task.result || task.status] || ["gray", task.result || task.status || "未知"];
  return el("span", {class:"chip " + tone}, text);
}
function roundTaskDetail(task) {
  const detail = task.detail || "";
  const destination = /^→ *([A-Z][0-9]+)$/.exec(detail);
  if (destination) return el("a", {href:"#", onclick:e => {
    e.preventDefault(); issueDrawer(destination[1]);
  }}, "转至问题 " + destination[1]);
  const [status, accuracy] = detail.split("/");
  const labels = {gen_fail:"构图失败", ascgraph_fail:"构图失败", build_fail:"构图失败",
    compile_fail:"编译失败", runtime_fail:"运行失败", timeout:"超时", unsupported:"不支持"};
  const text = labels[status] ? labels[status] + (accuracy === "fail" ? " · 精度失败" : "")
    : status === "pass" && accuracy === "fail" ? "精度失败" : detail;
  return el("span", {class:"muted", title:detail || null}, text || "—");
}
function roundPanel(issueId, round) {
  const tasks = el("div", {class:"round-tasks", hidden:true});
  let loaded = false;
  const toggle = el("button", {class:"round-toggle", "aria-expanded":"false", onclick:async () => {
    if (loaded && !tasks.hidden) {
      tasks.hidden = true; toggle.textContent = "查看任务"; toggle.setAttribute("aria-expanded", "false"); return;
    }
    toggle.disabled = true; toggle.textContent = "加载中…";
    try {
      const list = await api("/api/issues/" + issueId + "/rounds/" + round.round_no + "/tasks");
      tasks.replaceChildren(list.length ? el("div", {class:"table-wrap"},
        el("table", {class:"grid round-task-table"},
          el("thead", {}, el("tr", {}, ["用例", "芯片", "状态", "结果说明"].map(text => el("th", {scope:"col"}, text)))),
          el("tbody", {}, list.map(task => el("tr", {},
            el("td", {}, el("a", {href:"#", class:"mono small", onclick:e => {e.preventDefault(); openCase(task.case_key);}}, task.case_key)),
            el("td", {class:"small"}, archLabel(task.npu_arch)),
            el("td", {}, roundTaskStatus(task)), el("td", {class:"small round-task-detail"}, roundTaskDetail(task)))))))
        : el("div", {class:"empty"}, "暂无任务"));
      loaded = true; tasks.hidden = false;
      toggle.textContent = "收起任务"; toggle.setAttribute("aria-expanded", "true");
    } catch (err) {
      tasks.hidden = false; tasks.replaceChildren(el("div", {class:"error-box", role:"alert"}, err.message));
      toggle.textContent = "重试加载"; toggle.setAttribute("aria-expanded", "true");
    } finally { toggle.disabled = false; }
  }}, "查看任务");
  return el("section", {class:"round-card"},
    el("div", {class:"round-header"},
      el("div", {class:"round-heading"}, el("strong", {}, "第 " + round.round_no + " 轮"),
        el("span", {class:"muted small"}, round.kind === "verify" ? "验证" : "回归"),
        el("span", {class:"chip " + ({passed:"ok", failed:"bad", invalid:"warn"}[round.outcome] || "info")},
          {passed:"通过", failed:"失败", invalid:"无效"}[round.outcome] || "进行中")), toggle),
    el("div", {class:"round-meta small muted"}, el("span", {}, "版本 " + (round.fix_version || "—")),
      el("span", {}, "通过 " + fmt(round.pass_count) + " · 失败 " + fmt(round.fail_count) + " · 无效 " + fmt(round.invalid_count))), tasks);
}
function evidenceVerdict(c) {
  const verdicts = {pass:["ok", "通过"], fail:["bad", "失败"], skip:["gray", "跳过"], invalid:["warn", "无效"]};
  const [cls, text] = verdicts[c.result] || ["gray", "未记录"];
  return el("span", {class:"chip " + cls}, text);
}
function caseList(cases, mode = "evidence") {
  const current = mode === "current";
  const headers = current ? ["用例", "芯片", "后端", "最新执行结果"]
    : ["用例", "芯片", "后端", "最新执行结果", "历史记录（当时）"];
  const currentStatus = c => c.head_function_status ? statusChip({backend:c.backend, function_status:c.head_function_status,
    fail_stage:c.head_fail_stage, accuracy_status:c.head_accuracy_status, accuracy_ref:c.head_accuracy_ref,
    invalid_reason:c.head_invalid_reason}) : current && c.function_status ? statusChip(c) : "暂无结果";
  const source = c => ({verify:"候选包验证", regress:"候选包回归", fuzz:"普通执行", retest:"普通复测", import:"导入执行"}[c.run_kind] || "历史证据");
  return el("table", { class: "grid issue-cases" },
    el("thead", {}, el("tr", {}, headers.map(t => el("th", {}, t)))),
    el("tbody", {}, cases.map(c => el("tr", {},
      el("td", {}, el("a", { href: "#", class: "mono small", onclick: e => { e.preventDefault(); openCase(c.case_key); } }, c.case_key)),
      el("td", {}, archLabel(c.npu_arch)), el("td", {}, label(c.backend)),
      el("td", {}, el("div", {class:"issue-case-status"}, currentStatus(c)), !current && c.current_issue_id ?
        el("div", {class:"muted small issue-case-meta"}, el("a", {href:"#", onclick:e => {
          e.preventDefault(); issueDrawer(c.current_issue_id); }}, "当前问题 " + c.current_issue_id)) : null),
      ...(current ? [] : [el("td", {}, el("details", {},
        el("summary", {}, "查看历史"),
        el("div", {class:"muted small issue-case-meta"}, source(c)),
        el("div", {class:"issue-case-status"}, c.function_status ? statusChip(c) : "暂无结果"),
        el("div", {class:"issue-case-meta small"}, "当时对本问题的结论：", evidenceVerdict(c)),
        c.stale ? el("div", {class:"muted small"}, "旧图") : null,
        c.pick_reason ? el("div", {class:"muted small"}, c.pick_reason) : null))])))));
}
function caseSection(issueId, page, scope, title) {
  const content = el("div", {class:"table-wrap"}, page.cases.length ? caseList(page.cases, scope === "current" ? "current" : "evidence")
    : el("div", {class:"muted"}, "无"));
  const next = el("button", {class:"icon-btn", onclick:async () => {
    const query = scope === "regression" ? "role=regression" : "scope=" + scope;
    const more = await api("/api/issues/" + issueId + "/cases?" + query + "&after=" + page.next_cursor);
    content.append(caseList(more.cases, scope === "current" ? "current" : "evidence"));
    page = more; next.hidden = !page.next_cursor;
  }}, "加载更多");
  next.hidden = !page.next_cursor;
  return el("section", {}, el("h3", {}, title), content, next);
}

function adminStatusEditor(issue) {
  let target = issue.status === 'resolved' ? 'closed' : issue.status;
  const reason = el('input', {placeholder:'状态修改原因', maxlength:500, 'aria-label':'状态修改原因'});
  const hints = {open:'取消未完成任务并释放领取；保留封存集合及历史记录。', claimed:'取消未完成任务并撤销旧租约；保留原领取人，无人领取时归当前管理员。', verifying:'使用已有候选包创建验证任务；相同包可复用已通过的回归。', regressing:'最近验证通过后创建回归任务；封存问题沿用原回归集合。', summarizing:'需要最近轮次通过，已有回归集合需要回归通过。', closed:'取消未完成任务、撤销租约并封存集合，保留领取人。'};
  const hint = el('div', {class:'muted small'}, hints[target]);
  const picker = select(Object.entries(STATUS_LABEL).filter(([key]) => key !== 'resolved'), target, value => {
    target = value; hint.textContent = hints[value];
  }, {'aria-label':'问题状态'});
  const save = el('button', {onclick: async () => {
    if (!reason.value.trim()) { alert('请填写状态修改原因'); return; }
    save.disabled = true;
    try {
      await api('/api/admin/issues/' + issue.issue_id + '/status', {method:'POST', body:{request_key:rkey(), status:target, expected_status:issue.status, reason:reason.value.trim()}});
      await issueDrawer(issue.issue_id);
    } catch (err) { alert(err.message); save.disabled = false; }
  }}, '修改状态');
  return el('div', {class:'panel'}, el('div', {class:'toolbar'}, '问题状态', picker, reason, save), hint);
}
