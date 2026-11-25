"use strict";
// Hash-based routing, the page registry, and the silent auto-refresh loop.
// Page modules register themselves into PAGES as a side effect of being
// imported (see app.js), so this module never imports a page module itself.
import { $, el, errorBox, ApiError, setQuietRefresh } from "./core.js";
import { resultFilter, cancelFilterChange } from "./result-filter.js";
import { closeDrawer, hideActionBar, isMenuOpen } from "./ui.js";

// ---------- routing ----------
export function route() {
  const raw = location.hash.replace(/^#/, "") || "overview";
  const [page, query] = raw.split("?");
  return { page, params: new URLSearchParams(query || "") };
}
export function go(page, params = {}) {
  const current = route().params;
  const merged = new URLSearchParams();
  if (current.get("q") && params.q === undefined) merged.set("q", current.get("q"));
  for (const key of ["results", "backends", "chips"]) if (current.has(key) && params[key] === undefined) merged.set(key, current.get(key));
  for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined && v !== "") merged.set(k, v);
  const s = merged.toString();
  location.hash = page + (s ? "?" + s : "");
}
export const PAGES = {};
export function openCase(key, exec) {
  location.hash = "case?key=" + encodeURIComponent(key) + (exec ? "&exec=" + encodeURIComponent(exec) : "");
}
export function caseDrawer(group) { openCase(group.case.case_key); }

let renderToken = 0;
let overviewRedraw = null;
let loadingTimer = null, loadingToast = null;
export async function render() {
  cancelFilterChange();
  $("#top").classList.remove("hidden");
  const { page, params } = route();
  const ticket = ++renderToken;
  for (const a of document.querySelectorAll("#nav a")) a.classList.toggle("active", a.dataset.page === page);
  $("#search").value = params.get("q") || "";
  closeDrawer();
  hideActionBar();
  overviewRedraw = null;
  document.querySelector(".result-filter")?.remove();
  const filterable = ["overview", "discovery"].includes(page) && params.get("sub") !== "retests";
  document.body.classList.toggle("has-result-filter", filterable);
  if (filterable) document.body.append(resultFilter(params, changes => go(page, { ...Object.fromEntries(params), ...changes, page: null })));
  const main = $("#main");
  const q = params.get("q");
  const content = el("div");
  if (q) content.append(el("div", { class: "filter-note" }, "已过滤：", el("span", { class: "mono" }, q),
    el("button", { onclick: () => go(page, { ...Object.fromEntries(params), q: null }) }, "清除")));
  const visualKeys = new Set(["results", "backends", "chips"]);
  const samePage = view && view.page === page && view.content.isConnected;
  const visualChange = samePage && [...new Set([...params.keys(), ...view.params.keys()])]
    .every(key => visualKeys.has(key) || params.get(key) === view.params.get(key));
  // Keep existing charts/rows visible while a new filter cohort is loading.
  if (!visualChange) main.replaceChildren(content);
  else {
    content.style.cssText = "position:absolute;left:-100000px;visibility:hidden;width:" + view.content.getBoundingClientRect().width + "px";
    main.append(content);
  }
  clearTimeout(loadingTimer);
  loadingToast?.remove();
  loadingToast = null;
  // Delay the floating status so fast navigation does not flash a message.
  loadingTimer = setTimeout(() => {
    if (ticket !== renderToken) return;
    loadingToast = el("div", { class: "page-loading-toast", role: "status", "aria-live": "polite" }, "加载中…");
    document.body.append(loadingToast);
  }, 200);
  try {
    await (PAGES[page] || PAGES.overview)(content, params);
    if (visualChange && content.settled) await content.settled;
  } catch (err) {
    if (!(err instanceof ApiError && err.status === 401)) content.append(errorBox(err));
  } finally {
    if (ticket === renderToken) {
      clearTimeout(loadingTimer);
      loadingToast?.remove();
      loadingToast = null;
    }
  }
  if (ticket === renderToken) {
    if (visualChange) { content.style.cssText = ""; main.replaceChildren(content); }
    overviewRedraw = content.redraw || null;
    view = { ticket, page: PAGES[page] ? page : "overview", params, content };
    lastRefresh = Date.now();
  } else content.remove();
}

// Mobile browsers fire resize on every scroll as the URL bar collapses/expands (height only).
// Only a real width change matters, and it redraws in place instead of re-rendering the page.
let resizeTimer = null, lastWidth = innerWidth;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (innerWidth === lastWidth) return;
    lastWidth = innerWidth;
    if (overviewRedraw) overviewRedraw();
  }, 300);
});

// ---------- silent auto refresh ----------
// Re-renders the current page off-screen, then swaps in only the sections whose markup changed,
// so there is no loading flash and the scroll position is untouched. It backs off whenever the
// user is busy (tab hidden, drawer open, typing, picking) and keeps the old content on any error.
const AUTO_REFRESH_MS = 60000;
const AUTO_REFRESH_PAGES = new Set(["overview", "discovery", "repair", "ops"]);
let view = null, lastRefresh = 0, refreshing = false;
// manual = the user pressed 刷新: any page, and sections they customised are refreshed too.
export async function softRefresh(manual = false) {
  if (refreshing || !view || view.ticket !== renderToken || !view.content.isConnected) return false;
  if (!manual) {
    if (document.hidden || !AUTO_REFRESH_PAGES.has(view.page)) return false;
    if (!$("#drawer").classList.contains("hidden") || isMenuOpen() || !$("#actionbar").classList.contains("hidden")) return false;
    const a = document.activeElement;
    if (a && a.closest("#main") && /^(INPUT|SELECT|TEXTAREA)$/.test(a.tagName)) return false;
  }
  const { ticket, content, params } = view;
  refreshing = true; setQuietRefresh(true);
  const staging = el("div");
  // Laid out at the live width (so canvases/SVGs measure correctly) but invisible and off-screen.
  staging.style.cssText = `position:absolute;left:0;top:-100000px;visibility:hidden;pointer-events:none;width:${content.getBoundingClientRect().width}px`;
  try {
    $("#main").append(staging);
    staging.forceRefresh = manual;
    await PAGES[view.page](staging, params);
    if (staging.settled) await staging.settled;
    // Pages that fill sections lazily: give them a moment to leave their loading placeholder.
    for (let i = 0; i < 40 && staging.textContent.includes("加载中…"); i++) await new Promise(r => setTimeout(r, 250));
    const fresh = [...staging.children], old = [...content.children];
    if (ticket !== renderToken || fresh.length !== old.length || staging.querySelector(".error-box")) return false;
    let heroSwapped = false;
    fresh.forEach((n, i) => {
      const o = old[i];
      if ((o.dataset.custom && !manual) || n.innerHTML === o.innerHTML) return;
      const sx = o.querySelector(".cap-scroll")?.scrollLeft || 0;
      o.replaceWith(n);
      const box = n.querySelector(".cap-scroll");
      if (box) box.scrollLeft = sx;
      if (n.classList.contains("hero")) heroSwapped = true;
    });
    // The redraw closure is bound to its own hero node, so adopt it only if that hero moved over.
    if (heroSwapped) { overviewRedraw = staging.redraw || null; content.redraw = overviewRedraw; }
    lastRefresh = Date.now();
    return true;
  } catch { return false; /* keep what is on screen; try again next tick */ }
  finally { staging.remove(); refreshing = false; setQuietRefresh(false); }
}

export function initAutoRefresh() {
  const refreshBtn = $("#refresh");
  refreshBtn.addEventListener("click", async () => {
    refreshBtn.disabled = true; refreshBtn.textContent = "刷新中…";
    const ok = await softRefresh(true);
    refreshBtn.disabled = false; refreshBtn.textContent = ok ? "已刷新" : "刷新";
    if (ok) setTimeout(() => { refreshBtn.textContent = "刷新"; }, 1200);
    else if (view && view.ticket === renderToken) render();  // could not refresh in place: fall back to a full render
  });
  setInterval(() => { if (Date.now() - lastRefresh >= AUTO_REFRESH_MS) softRefresh(); }, 5000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden && Date.now() - lastRefresh >= AUTO_REFRESH_MS) softRefresh(); });
}
