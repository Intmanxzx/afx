"use strict";
// Login/logout state. core.js's api() dynamically imports showLogin/clearSession
// from here when a request 401s — see the comment in core.js for why that is
// a dynamic (not static) import, so this module can statically import api()
// from core.js without creating a real circular module graph.
import { $, el, api, TOKEN_KEY } from "./core.js";
import { closeDrawer, hideActionBar } from "./ui.js";
import { render } from "./router.js";

export let me = null;

export function clearSession() { me = null; updateWho(); }

export function showLogin(message) {
  closeDrawer();
  hideActionBar();
  const input = el("textarea", { placeholder: "afx_…", "aria-label": "token" });
  const msg = el("div", { class: "muted small" }, message || "使用管理员离线发放的 token 登录。token 只保存在当前标签页。");
  const submit = async () => {
    const value = input.value.trim();
    if (!value.startsWith("afx_")) { msg.textContent = "token 应以 afx_ 开头"; return; }
    try { sessionStorage.setItem(TOKEN_KEY, value); } catch { msg.textContent = "浏览器禁止会话存储"; return; }
    try { await loadMe(); render(); } catch (err) { try { sessionStorage.removeItem(TOKEN_KEY); } catch { /* ignore */ } msg.textContent = "登录失败：" + err.message; }
  };
  $("#main").replaceChildren(el("div", { class: "panel login" }, el("h2", {}, "登录 AutofuseX 平台"), input, msg,
    el("div", { class: "toolbar", style: { marginTop: "10px" } }, el("button", { class: "primary", onclick: submit }, "登录"),
      el("button", { onclick: () => render() }, "返回浏览"))));
}
export async function loadMe() {
  me = await api("/api/me");
  updateWho();
}
export function updateWho() {
  $("#who").textContent = me ? `${me.user_id} · ${me.role === "admin" ? "管理员" : me.role}` : "";
  $("#logout").textContent = me ? "退出" : "登录";
}
export const isAdmin = () => me && me.role === "admin";
