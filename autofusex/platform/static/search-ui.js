"use strict";
// Global search box autocomplete. Self-contained: wires itself to #search on
// import and needs nothing from the rest of the app beyond the field catalog.
import { $, el, SEARCH_FIELDS, loadSearchFieldValues } from "./core.js";

// Most people don't remember the field:value syntax, so focusing the (empty) box lists every
// field with a one-line example, and typing narrows it to matching fields, then matching values.
const searchPanel = el("div", { class: "search-suggest hidden", role: "listbox" });
document.body.append(searchPanel);
let searchActiveIndex = -1;
function currentTerm(input) {
  const pos = input.selectionStart ?? input.value.length, text = input.value;
  const start = text.lastIndexOf(" ", pos - 1) + 1;
  let end = text.indexOf(" ", pos);
  if (end === -1) end = text.length;
  return { start, end, term: text.slice(start, end) };
}
function searchMatches(term) {
  const negate = term.startsWith("-") ? "-" : "";
  const body = negate ? term.slice(1) : term;
  const colon = body.indexOf(":");
  if (colon === -1) {
    const q = body.toLowerCase();
    return { kind: "field", negate, items: SEARCH_FIELDS.filter(f => !q || f.name.startsWith(q)) };
  }
  const field = SEARCH_FIELDS.find(f => f.name === body.slice(0, colon));
  if (!field || !field.enum) return { kind: "none" };
  const already = body.slice(colon + 1), lastComma = already.lastIndexOf(","), typed = already.slice(lastComma + 1);
  const q = typed.toLowerCase();
  const items = field.enum().filter(v => !q || String(v).toLowerCase().includes(q)).slice(0, 30);
  return { kind: "value", negate, field, keep: already.slice(0, lastComma + 1), items };
}
function applySearchPick(input, start, end, text) {
  input.value = input.value.slice(0, start) + text + input.value.slice(end);
  const pos = start + text.length;
  input.setSelectionRange(pos, pos);
  input.focus();
  updateSearchPanel(input);
}
function renderSearchPanel(input, match, bounds) {
  searchActiveIndex = -1;
  if (match.kind === "none" || !match.items.length) { searchPanel.classList.add("hidden"); return; }
  const rows = match.kind === "field"
    ? match.items.map(f => el("button", { class: "search-sug-item", type: "button",
        onclick: () => applySearchPick(input, bounds.start, bounds.end, match.negate + f.name + ":") },
        el("span", { class: "sf-name" }, f.name + ":"), el("span", { class: "sf-label" }, f.label),
        el("span", { class: "sf-example" }, f.desc || f.example)))
    : match.items.map(v => el("button", { class: "search-sug-item", type: "button",
        onclick: () => applySearchPick(input, bounds.start, bounds.end,
          match.negate + match.field.name + ":" + match.keep + v + " ") },
        el("span", { class: "sf-name" }, String(v)), match.field.text ? el("span", { class: "sf-label" }, match.field.text(v)) : null));
  searchPanel.replaceChildren(...rows);
  const rect = input.getBoundingClientRect();
  searchPanel.style.left = (rect.left + scrollX) + "px";
  searchPanel.style.top = (rect.bottom + scrollY + 4) + "px";
  searchPanel.style.width = Math.max(320, rect.width) + "px";
  searchPanel.classList.remove("hidden");
}
function updateSearchPanel(input) {
  const bounds = currentTerm(input);
  renderSearchPanel(input, searchMatches(bounds.term), bounds);
}
// For the search form's submit handler (app.js): a submitted search is handled by
// navigating, not by picking a suggestion, so the panel needs to be dismissed explicitly.
export function hideSearchPanel() { searchPanel.classList.add("hidden"); }
(() => {
  const input = $("#search");
  input.addEventListener("focus", () => { loadSearchFieldValues().then(() => updateSearchPanel(input)); updateSearchPanel(input); });
  input.addEventListener("input", () => updateSearchPanel(input));
  input.addEventListener("click", () => updateSearchPanel(input));
  input.addEventListener("keydown", e => {
    const items = [...searchPanel.children];
    if (searchPanel.classList.contains("hidden") || !items.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      searchActiveIndex = (searchActiveIndex + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      items.forEach((it, i) => it.classList.toggle("active", i === searchActiveIndex));
      items[searchActiveIndex].scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter" && searchActiveIndex >= 0) {
      e.preventDefault();
      items[searchActiveIndex].click();
    } else if (e.key === "Escape") {
      searchPanel.classList.add("hidden");
    }
  });
  // A suggestion click re-renders the panel's children (detaching the clicked button) and
  // refocuses the input before this bubbles here, so check focus rather than DOM containment.
  document.addEventListener("click", e => { if (e.target !== input && document.activeElement !== input) searchPanel.classList.add("hidden"); });
})();
