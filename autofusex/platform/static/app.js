"use strict";
// AutofuseX platform UI entry point. Plain DOM; every server value goes through
// textContent. This file only wires up the page shell (search form, logout,
// hashchange, boot); everything else lives in its own module — see core.js,
// router.js, auth.js, ui.js, widgets.js, search-ui.js and the page-*.js files.
import { $, token, TOKEN_KEY } from "./core.js";
import { route, go, render, initAutoRefresh } from "./router.js";
import { me, loadMe, updateWho, showLogin, clearSession } from "./auth.js";
import { hideSearchPanel } from "./search-ui.js";
import "./ui.js";

// Side-effect imports: each page module registers itself into router.js's PAGES.
import "./page-overview.js";
import "./page-discovery.js";
import "./page-case.js";
import "./page-repair.js";
import "./page-knowledge.js";
import "./page-profile.js";
import "./page-ops.js";
import "./page-models.js";
import "./page-users.js";

window.addEventListener("hashchange", render);
$("#search-form").addEventListener("submit", e => {
  e.preventDefault();
  const { page } = route();
  go(page === "models" ? "discovery" : page, { q: $("#search").value.trim() || null });
  hideSearchPanel();
});
$("#logout").addEventListener("click", () => {
  if (!me) return showLogin();
  try { sessionStorage.removeItem(TOKEN_KEY); } catch { /* ignore */ }
  clearSession();
  render();
});
if (matchMedia("(hover: none)").matches) document.body.classList.add("touch");
initAutoRefresh();
(async () => {
  updateWho();
  if (token()) { try { await loadMe(); } catch { clearSession(); } }
  render();
})();
