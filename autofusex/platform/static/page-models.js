"use strict";
// 整网分析（预留）
import { el } from "./core.js";
import { PAGES } from "./router.js";

PAGES.models = async root => {
  root.append(el("section", { class: "panel" }, el("h2", {}, "整网分析"), el("div", { class: "empty" },
    "规划中：执行完整推荐类模型与 LLM，接入 autofusex 的图转储、精度对比与 HTML 报告。功能尚未开放。")));
};
