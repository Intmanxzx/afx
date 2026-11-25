"use strict";
// Foundational constants, DOM helpers, formatting and the API client. Every
// other module imports from here; this module itself only reaches into
// auth.js, and only lazily (inside api()'s body) to handle a 401 response —
// see the comment there for why that is not a real circular dependency.

export const $ = (sel, root = document) => root.querySelector(sel);
export const TOKEN_KEY = "afx.token";
export const COLS_KEY = "afx.discovery.columns";
export const FLEET_SCOPE_KEY = "afx.fleet.scope";
export const VIEW_KEY = "afx.discovery.view";
export const BACKEND_LABEL = { "cpu": "CPU", "eager-npu": "EagerNPU", "af-legacy": "Legacy", "af-mlir": "MLIR", "triton-npu": "TritonNPU" };
export const STAGE_LABEL = { inductor: "Inductor", nnsmith: "NNSmith", af_fronted: "AF前端", legacy_codegen: "Legacy Codegen", triton_ascend: "TritonNPU", aclnn: "aclNN单算子", fused_kernel: "融合算子", af_common: "AF其他", environment: "环境", schedule: "Schedule", graph_check: "图校验", mlir: "MLIR", lowering: "Lowering", codegen: "Codegen",
  tiling: "Tiling", autofuse: "AF其他", first_call_combined: "首次调用", timeout: "超时", unknown: "未知", unknown_retest: "未知·待复测" };
export const DIM_LABEL = { all: "全部", op_category: "算子类别", op_display: "算子覆盖", dtype: "数据类型", dim_bucket: "最大维度", size_bucket: "张量规模", dynamic: "静态/动态", source: "用例来源" };
export const DIM_ORDER = ["op_display", "dtype", "dim_bucket", "size_bucket", "op_category", "dynamic", "source"];
export const STATUS_LABEL = { open: "未领取", claimed: "修复中", verifying: "验证中", regressing: "回归中", summarizing: "归档中", resolved: "已解决", closed: "已关闭" };

// ---------- global search field catalog (mirrors autofusex/platform/search.py) ----------
export const SEARCH_RESULT_LABEL = { pass: "通过", fail: "失败（功能或精度）", function_fail: "功能失败", accuracy_fail: "精度失败", unsupported: "不支持" };
export const SEARCH_SOURCE_LABEL = { nnsmith: "NNSmith 随机生成", baize_tf: "Baize TF 导入", baize_torch: "Baize Torch 导入", model: "整网模型导入" };
export const SEARCH_DYNAMIC_LABEL = { dynamic: "动态 Shape", static: "静态 Shape" };
export let searchFieldValues = {};  // op/dtype/category/size/dim/arch/user, fetched lazily from /api/search/fields
let searchFieldValuesPromise = null;
export function loadSearchFieldValues() {
  if (!searchFieldValuesPromise) searchFieldValuesPromise = api("/api/search/fields").then(v => { searchFieldValues = v; }).catch(() => {});
  return searchFieldValuesPromise;
}
// ---------- DOM helpers ----------
export function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "style") Object.assign(node.style, v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}
export function svg(tag, attrs, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, String(v));
  for (const c of children.flat()) if (c != null) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}
export const fmt = n => n == null ? "—" : Number(n).toLocaleString("zh-CN");
export const pct = (a, b) => b ? (100 * a / b).toFixed(1) + "%" : "—";
export const label = b => BACKEND_LABEL[b] || b;
export const archLabel = a => { if (!a) return a || "—"; const s = String(a).toLowerCase(); return s.includes("910") ? "A2A3" : s.includes("950") ? "A5" : a; };
export const SEARCH_FIELDS = [
  { name: "backend", label: "后端", example: "backend:af-mlir", enum: () => Object.keys(BACKEND_LABEL), text: label },
  { name: "status", label: "结果", example: "status:fail", enum: () => Object.keys(SEARCH_RESULT_LABEL), text: v => SEARCH_RESULT_LABEL[v] || v },
  { name: "stage", label: "失败阶段", example: "stage:schedule", enum: () => Object.keys(STAGE_LABEL), text: v => STAGE_LABEL[v] || v },
  { name: "arch", label: "芯片", example: "arch:910b", enum: () => searchFieldValues.arch || [], text: archLabel },
  { name: "user", label: "用户", example: "user:alice", enum: () => (searchFieldValues.user || []).map(v => v.id), text: v => ((searchFieldValues.user || []).find(u => u.id === v) || {}).name || v },
  { name: "op", label: "算子", example: "op:aten.add.Tensor", enum: () => searchFieldValues.op || [] },
  { name: "dtype", label: "数据类型", example: "dtype:bfloat16", enum: () => searchFieldValues.dtype || [] },
  { name: "category", label: "算子类别", example: "category:reduce", enum: () => searchFieldValues.category || [] },
  { name: "source", label: "用例来源", example: "source:nnsmith", enum: () => Object.keys(SEARCH_SOURCE_LABEL), text: v => SEARCH_SOURCE_LABEL[v] || v },
  { name: "dynamic", label: "动态/静态", example: "dynamic:static", enum: () => Object.keys(SEARCH_DYNAMIC_LABEL), text: v => SEARCH_DYNAMIC_LABEL[v] || v },
  { name: "size", label: "张量规模", example: "size:<=64K", enum: () => searchFieldValues.size || [] },
  { name: "dim", label: "最大维度", example: "dim:<=512", enum: () => searchFieldValues.dim || [] },
  { name: "error", label: "错误文本", example: "error:tiling", desc: "日志摘要中包含的文本，3..64 个字符" },
  { name: "case", label: "用例 ID", example: "case:E000000000000000000000000A", desc: "精确匹配一个用例 ID" },
];
export const time = ms => ms ? new Date(ms).toLocaleString("zh-CN", { hour12: false }) : "—";
export const ns = v => v == null ? "—" : v >= 1e9 ? (v / 1e9).toFixed(2) + "s" : v >= 1e6 ? (v / 1e6).toFixed(1) + "ms" : (v / 1e3).toFixed(1) + "µs";
export const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};
export function token() { try { return sessionStorage.getItem(TOKEN_KEY); } catch { return null; } }
export function rkey() { return (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random()).slice(0, 64); }

// ---------- API ----------
// true while a background refresh runs: a 401 must not replace the page with the login form
export let quietRefresh = false;
export function setQuietRefresh(v) { quietRefresh = v; }
export class ApiError extends Error { constructor(status, code, message) { super(message); this.status = status; this.code = code; } }
export async function api(path, { method = "GET", body, raw = false } = {}) {
  const headers = {"X-AFX-Capabilities": "cleanup-v1,dump-v1"};
  if (token()) headers["Authorization"] = "Bearer " + token();
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (raw && res.ok) return res.blob();
  let payload = null;
  try { payload = await res.json(); } catch { /* non JSON */ }
  if (res.status === 401) {
    const expired = !!token();
    if (expired) {
      try { sessionStorage.removeItem(TOKEN_KEY); } catch { /* ignore */ }
      // auth.js imports api() from this module; this is the one place that calls back into
      // it, and only once api() actually runs (well after both modules have finished loading),
      // so the circular import is safe — see the module comment at the top of this file.
      const { clearSession } = await import("./auth.js");
      clearSession();
    }
    if (!quietRefresh) { const { showLogin } = await import("./auth.js"); showLogin(expired ? "登录已失效，请重新输入 token；也可以返回以访客身份浏览" : "该操作需要登录：请输入 token，或申请加入平台"); }
    throw new ApiError(401, "unauthorized", "需要登录");
  }
  if (!res.ok || !payload || !payload.ok) {
    const err = (payload && payload.error) || {};
    throw new ApiError(res.status, err.code || "http_error", err.message || res.statusText);
  }
  return payload.data;
}
export const qs = obj => { const p = new URLSearchParams(); for (const [k, v] of Object.entries(obj)) if (v !== null && v !== undefined && v !== "") p.set(k, v); const s = p.toString(); return s ? "?" + s : ""; };
export function errorBox(err) { return el("div", { class: "error-box" }, err instanceof ApiError ? `${err.message}（${err.status} ${err.code}）` : String(err)); }

export async function archOptions() {
  return (await api("/api/overview/pipeline")).arches;
}
