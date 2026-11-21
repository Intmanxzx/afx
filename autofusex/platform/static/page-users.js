"use strict";
import { el, api, time, errorBox, store } from "./core.js";
import { PAGES, go } from "./router.js";
import { isAdmin } from "./auth.js";

function copy(text) {
  return el("button", { onclick: async e => {
    try { await navigator.clipboard.writeText(text); e.target.textContent = "已复制"; }
    catch { e.target.textContent = "请选中文字手动复制"; }
  } }, "复制");
}
function secret(root, token) {
  const command = "afx init --token " + token + String.fromCharCode(10) + "afx start";
  root.replaceChildren(el("h2", {}, "token 已生成"),
    el("p", {}, "仅展示这一次，请立即保存。不要将 token 发到公开聊天或截图中。"),
    el("pre", {}, token), copy(token), el("pre", {}, command), copy(command),
    el("p", {}, "已有客户端重置 token 后需更新凭据，无需重新初始化。"));
}
function button(label, action, notice) {
  return el("button", { onclick: async e => {
    e.target.disabled = true; notice.replaceChildren();
    try { await action(); } catch (err) { notice.append(errorBox(err)); }
    finally { e.target.disabled = false; }
  } }, label);
}

PAGES.apply = async root => {
  const user = el("input", { placeholder: "用户名", maxlength: 64, "aria-label": "用户名" });
  const notice = el("div");
  const panel = el("div", { class: "panel login" }, el("h2", {}, "申请加入 AFX"),
    el("p", {}, "填写用户名，管理员审核后即可领取 token。"), user, notice);
  panel.append(button("提交申请", async () => {
    const data = await api("/api/registration/apply", { method: "POST", body: { user_id: user.value.trim() } });
    store.set("afx.application", data.receipt);
    go("application", { receipt: data.receipt, q: null });
  }, notice));
  const saved = store.get("afx.application", null);
  if (saved) panel.append(el("button", { onclick: () => go("application", { receipt: saved, q: null }) }, "查看已有申请"));
  root.append(panel);
};

PAGES.application = async (root, params) => {
  const receipt = params.get("receipt");
  const panel = el("div", { class: "panel login" }); root.append(panel);
  if (!receipt) { panel.append(el("p", {}, "请使用提交申请后保存的进度链接。")); return; }
  const link = location.origin + location.pathname + "#application?receipt=" + encodeURIComponent(receipt);
  const data = await api("/api/registration/status", { method: "POST", body: { receipt } });
  const labels = { pending: "等待审核", approved: "已批准，可以领取 token", rejected: "申请未通过", expired: "申请已过期，请重新申请；已有账号请联系管理员", claimed: "token 已领取；如已遗失，请联系管理员重置" };
  const notice = el("div");
  panel.append(el("h2", {}, data.user_id + " · 申请进度"), el("p", {}, labels[data.status]),
    el("p", {}, "进度链接有效至 " + time(data.expires_at) + "。此链接可用于领取 token，请私下保存，不要分享。"),
    el("textarea", { readonly: true, "aria-label": "申请进度链接" }, link), copy(link), notice);
  if (data.status === "pending") panel.append(button("刷新状态", async () => { root.replaceChildren(); await PAGES.application(root, params); }, notice));
  if (data.status === "approved") panel.append(button("领取 token（仅一次）", async () => {
    const result = await api("/api/registration/claim", { method: "POST", body: { receipt } });
    store.set("afx.application", null);
    secret(panel, result.token);
    history.replaceState(null, "", location.pathname + "#application");
  }, notice));
};

PAGES.users = async (root, params) => {
  if (!isAdmin()) { root.append(el("p", {}, "仅管理员可管理用户。")); return; }
  const data = await api("/api/admin/users");
  const tab = params.get("tab") || "requests";
  root.append(el("h1", {}, "用户管理"), el("div", { class: "toolbar" },
    el("button", { onclick: () => go("users", { tab: "requests", q: null }) }, "待审核（" + data.requests.length + "）"),
    el("button", { onclick: () => go("users", { tab: "users", q: null }) }, "已有用户")));
  const notice = el("div"); root.append(notice);
  const reload = async () => { root.replaceChildren(); await PAGES.users(root, params); };
  const roles = value => {
    const select = el("select", { "aria-label": "角色" }, el("option", { value: "developer" }, "developer"), el("option", { value: "admin" }, "admin"));
    select.value = value; return select;
  };
  if (tab === "requests") {
    if (!data.requests.length) root.append(el("p", {}, "暂无待审核申请"));
    for (const req of data.requests) {
      const role = roles("developer");
      const review = async action => {
        await api("/api/admin/users/review", { method: "POST", body: { id: req.id, action, role: role.value } }); await reload();
      };
      root.append(el("div", { class: "panel" }, el("h2", {}, req.user_id),
        el("p", {}, time(req.created_at)), role,
        button("批准", () => review("approve"), notice), button("拒绝", () => review("reject"), notice)));
    }
  } else {
    for (const user of data.users) {
      const role = roles(user.role);
      const change = async (action, extra = {}) => {
        const result = await api("/api/admin/users/manage", { method: "POST", body: { user_id: user.user_id, action, ...extra } });
        if (result.token) secret(root, result.token); else await reload();
      };
      root.append(el("div", { class: "panel" }, el("h2", {}, user.user_id),
        el("p", {}, "角色：" + user.role + " · " + (user.status === "active" ? "启用" : "禁用")),
        el("p", {}, "创建：" + time(user.created_at) + " · 最近使用：" + time(user.last_seen_at)), role,
        button("保存角色", () => change("role", { role: role.value }), notice),
        button(user.status === "active" ? "禁用" : "启用", () => change(user.status === "active" ? "disable" : "enable"), notice),
        button("遗失后重置 token", async () => {
          if (confirm("仅在用户遗失 token 时重置。旧 token 将立即失效，已有客户端需更新凭据。继续？")) await change("reset-token");
        }, notice)));
    }
  }
};
