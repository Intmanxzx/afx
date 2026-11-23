"use strict";
// Drawer, action bar, popup menu and the shared field tooltip. No dependency
// on routing or auth — pages and router.js both import from here.
import { $, el } from "./core.js";

// ---------- drawer & action bar ----------
let drawerOutsideHandler = null;
export function openDrawer(...children) {
  const d = $("#drawer");
  d.replaceChildren(el("button", { class: "close", onclick: closeDrawer }, "关闭"),
    ...children.flat().filter(child => child !== null && child !== undefined && child !== false));
  d.classList.remove("hidden");
  d.scrollTop = 0;
  if (drawerOutsideHandler) document.removeEventListener("click", drawerOutsideHandler);
  // Registered next tick so the click that opened the drawer doesn't immediately close it.
  setTimeout(() => {
    drawerOutsideHandler = e => { if (!d.contains(e.target)) closeDrawer(); };
    document.addEventListener("click", drawerOutsideHandler);
  }, 0);
}
export function closeDrawer() {
  $("#drawer").classList.add("hidden");
  if (drawerOutsideHandler) { document.removeEventListener("click", drawerOutsideHandler); drawerOutsideHandler = null; }
}
export function showActionBar(...children) {
  const b = $("#actionbar");
  b.replaceChildren(...children.flat().filter(child => child !== null && child !== undefined && child !== false));
  b.classList.remove("hidden");
}
export function hideActionBar() { $("#actionbar").classList.add("hidden"); }
let openMenu = null;
export function popMenu(anchor, items) {
  if (openMenu) openMenu.remove();
  const rect = anchor.getBoundingClientRect();
  const menu = el("div", { class: "menu", style: { top: (rect.bottom + scrollY) + "px", left: Math.max(8, rect.right + scrollX - 160) + "px" } },
    items.map(([text, fn]) => el("button", { onclick: () => { menu.remove(); openMenu = null; fn(); } }, text)));
  document.body.append(menu);
  openMenu = menu;
  setTimeout(() => document.addEventListener("click", function off(e) { if (!menu.contains(e.target)) { menu.remove(); openMenu = null; document.removeEventListener("click", off); } }), 0);
}
export function isMenuOpen() { return openMenu != null; }

// ---------- field tooltip (full text for truncated cells, logs and DAG nodes) ----------
const fieldTooltip = el("div", { class: "field-tooltip", role: "tooltip", id: "field-tooltip" });
fieldTooltip.hidden = true;
document.body.append(fieldTooltip);
let tooltipTarget = null;
export function hideFieldTooltip() { if (tooltipTarget) tooltipTarget.removeAttribute("aria-describedby"); tooltipTarget = null; fieldTooltip.hidden = true; }
function showFieldTooltip(event) {
  const target = event.target.closest && event.target.closest("[data-tooltip], [title]");
  if (!target || target.closest("#field-tooltip")) return;
  if (target.hasAttribute("title")) { target.dataset.tooltip = target.getAttribute("title"); target.removeAttribute("title"); }
  const content = target.dataset.tooltip;
  if (!content || !content.trim()) return;
  hideFieldTooltip(); tooltipTarget = target;
  fieldTooltip.textContent = content; fieldTooltip.hidden = false; target.setAttribute("aria-describedby", "field-tooltip");
  const tip = fieldTooltip.getBoundingClientRect(), bounds = target.getBoundingClientRect();
  const x = event.clientX ?? bounds.left, y = event.clientY ?? bounds.bottom;
  fieldTooltip.style.left = Math.max(8, Math.min(x, innerWidth - tip.width - 8)) + "px";
  fieldTooltip.style.top = Math.max(8, Math.min(y + 14, innerHeight - tip.height - 8)) + "px";
}
document.addEventListener("pointerover", showFieldTooltip);
document.addEventListener("focusin", showFieldTooltip);
document.addEventListener("pointerout", e => { if (tooltipTarget && !tooltipTarget.contains(e.relatedTarget) && !fieldTooltip.contains(e.relatedTarget)) hideFieldTooltip(); });
document.addEventListener("keydown", e => { if (e.key === "Escape") hideFieldTooltip(); });
window.addEventListener("scroll", hideFieldTooltip, true);
