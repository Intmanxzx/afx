from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from autofusedebugger.io import read_json


def _json_for_script(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}e}"
    return "-"


def render_report_html(report: dict[str, Any], title: str = "Autofuse Accuracy Report") -> str:
    summary = report.get("summary", {})
    failed = int(summary.get("failed", 0) or 0)
    total = int(summary.get("total", 0) or 0)
    compared = int(summary.get("compared", total) or 0)
    unmatched = int(summary.get("unmatched", max(0, total - compared)) or 0)
    unmapped = int(summary.get("unmapped", unmatched) or 0)
    not_captured = int(summary.get("not_captured", 0) or 0)
    passed = int(summary.get("passed", 0) or 0)
    strict_passed = int(summary.get("strict_passed", passed) or 0)
    strict_failed = int(summary.get("strict_failed", failed) or 0)
    cann_passed = int(summary.get("cann_passed", strict_passed) or 0)
    cann_failed = int(summary.get("cann_failed", strict_failed) or 0)
    disagreements = int(summary.get("standard_disagreements", 0) or 0)
    pass_rate = (passed / compared * 100.0) if compared else 0.0
    cann_pass_rate = (cann_passed / compared * 100.0) if compared else 0.0
    coverage = float(summary.get("coverage", compared / total if total else 0.0) or 0.0)
    status_text = str(summary.get("dual_status", summary.get("status", "PASS" if failed == 0 else "FAIL")))
    status_class = "pass" if status_text == "PASS" else ("fail" if status_text == "FAIL" else "warn")
    payload = _json_for_script(report)
    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --text: #16202a;
      --muted: #617083;
      --line: #d7dee8;
      --good: #147a4d;
      --bad: #bd2b2b;
      --warn: #a66700;
      --accent: #2457a6;
      --accent-soft: #eaf1ff;
      --shadow: 0 1px 2px rgba(20, 30, 45, 0.08);
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    button, input, select { font: inherit; }
    button {
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      padding: 0 10px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button:focus-visible, input:focus-visible, select:focus-visible, summary:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }
    .app-header {
      flex: 0 0 auto;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 12px 18px 10px;
      position: relative;
      z-index: 10;
    }
    .title-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 { margin: 0; font-size: 20px; line-height: 1.3; font-weight: 700; }
    .quick-status { display: flex; align-items: center; gap: 12px; color: var(--muted); white-space: nowrap; }
    .status-pill {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 0 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }
    .status-pill.pass { color: var(--good); background: #e7f5ee; }
    .status-pill.fail { color: var(--bad); background: #fdecec; }
    .status-pill.warn { color: var(--warn); background: #fff1d6; }
    .quick-stat strong { color: var(--text); }
    .metrics-panel { margin-top: 10px; }
    .metrics-panel summary, .table-panel summary {
      color: var(--muted);
      cursor: pointer;
      user-select: none;
      padding: 3px 0;
    }
    .metrics-panel summary::marker, .table-panel summary::marker { color: var(--accent); }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(140px, 1fr));
      gap: 8px;
      padding-top: 8px;
    }
    .metric {
      min-height: 58px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
    }
    .metric .label { color: var(--muted); font-size: 11px; margin-bottom: 2px; }
    .metric .value { font-size: 17px; line-height: 1.2; font-weight: 700; white-space: nowrap; }
    .metric .value.pass { color: var(--good); }
    .metric .value.fail { color: var(--bad); }
    .metric .value.warn { color: var(--warn); }
    .app-main {
      flex: 1 1 auto;
      height: auto;
      min-height: 0;
      display: flex;
      flex-direction: column;
      padding: 10px 14px 12px;
      gap: 8px;
    }
    .workspace {
      flex: 1 1 auto;
      min-height: 0;
      display: flex;
      position: relative;
      gap: 8px;
    }
    .graph-shell {
      min-width: 0;
      min-height: 0;
      flex: 1 1 auto;
      display: flex;
      flex-direction: column;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .graph-tabs {
      flex: 0 0 auto;
      display: flex;
      gap: 4px;
      overflow-x: auto;
      padding: 7px 10px 0;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }
    .graph-tab {
      flex: 0 0 auto;
      border: 0;
      border-bottom: 2px solid transparent;
      border-radius: 5px 5px 0 0;
      background: transparent;
      color: var(--muted);
      padding: 7px 10px 8px;
      white-space: nowrap;
    }
    .graph-tab:hover { color: var(--text); background: var(--accent-soft); }
    .graph-tab.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 700; }
    .graph-tab-group {
      flex: 0 0 auto;
      display: inline-flex;
      align-items: stretch;
      padding-left: 4px;
      border-left: 1px solid var(--line);
      border-radius: 5px 5px 0 0;
    }
    .graph-tab-group.linked { background: #eaf1ff; box-shadow: inset 0 0 0 1px #9bb7e5; }
    .graph-tab-group.active { background: var(--accent-soft); }
    .graph-kernel-tab, .graph-variant-tab {
      flex: 0 0 auto;
      border: 0;
      border-bottom: 2px solid transparent;
      border-radius: 4px 4px 0 0;
      background: transparent;
      color: var(--muted);
      white-space: nowrap;
    }
    .graph-kernel-tab { max-width: 260px; padding: 7px 7px 8px; overflow: hidden; text-overflow: ellipsis; }
    .graph-variant-tab { padding: 7px 6px 8px; font-size: 11px; }
    .graph-kernel-tab:hover, .graph-variant-tab:hover { color: var(--text); background: #fff; }
    .graph-variant-tab.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 700; }
    .graph-head {
      flex: 0 0 auto;
      min-height: 46px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 12px;
      border-bottom: 1px solid var(--line);
    }
    .graph-title-row { display: flex; align-items: center; gap: 8px; min-width: 0; }
    .graph-title-copy { min-width: 0; }
    .graph-title { font-weight: 700; overflow-wrap: anywhere; }
    .graph-back { width: 28px; height: 28px; padding: 0; font-size: 18px; line-height: 1; }
    .fusion-details-button { height: 28px; padding: 0 9px; white-space: nowrap; color: var(--accent); }
    .graph-head-right { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 8px; min-width: 0; }
    .graph-meta { color: var(--muted); font-size: 12px; white-space: nowrap; }
    .zoom-controls { display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }
    .zoom-controls button { width: 28px; height: 26px; padding: 0; line-height: 1; }
    .zoom-controls .zoom-reset { width: 44px; }
    .zoom-level { min-width: 44px; text-align: center; color: var(--muted); font-size: 12px; }
    .graph-body {
      position: relative;
      flex: 1 1 auto;
      min-height: 420px;
      overflow: hidden;
      background: #fbfcfe;
      cursor: grab;
      touch-action: none;
      user-select: none;
      overscroll-behavior: contain;
    }
    .graph-body.is-panning { cursor: grabbing; }
    .graph-body:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
    .graph-svg { position: absolute; inset: 0; display: block; width: 100%; height: 100%; }
    .graph-svg text { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .graph-edge { transition: opacity 120ms ease, stroke-width 120ms ease; }
    .graph-edge.dimmed { opacity: 0.12; }
    .graph-edge.related { stroke: var(--accent); stroke-width: 2.5; }
    .graph-edge.lifecycle-related { stroke: #2457a6; stroke-width: 2.8; }
    .graph-edge.lifecycle-dimmed { opacity: 0.12; }
    .graph-node { cursor: pointer; transition: opacity 120ms ease; }
    .graph-node.dimmed { opacity: 0.2; }
    .graph-node.lifecycle-dimmed { opacity: 0.2; }
    .graph-node rect { transition: stroke 120ms ease, stroke-width 120ms ease, filter 120ms ease; }
    .graph-node.related rect { stroke: var(--accent); stroke-width: 3; }
    .graph-node.lifecycle-related rect { stroke: var(--accent); stroke-width: 4; filter: drop-shadow(0 0 3px rgba(36, 87, 166, 0.4)); }
    .graph-node.selected rect { stroke: var(--accent); stroke-width: 4; filter: drop-shadow(0 0 4px rgba(36, 87, 166, 0.35)); }
    .graph-node .fusion-mark { fill: var(--accent); cursor: pointer; }
    .graph-node .fusion-mark:focus { stroke: #fff; stroke-width: 2; }
    .graph-node .broadcast-cache-mark { fill: #087f8c; stroke: #fff; stroke-width: 2; pointer-events: none; }
    .graph-node .broadcast-cache-mark-text { fill: #fff; font-size: 8px; font-weight: 800; text-anchor: middle; pointer-events: none; }
    .graph-node .broadcast-cache-label { fill: #087f8c; font-size: 10.5px; font-weight: 700; pointer-events: none; }
    .graph-node .unfused-mark { fill: #b97800; stroke: #fff; stroke-width: 2; pointer-events: none; }
    .graph-node .unfused-mark-text { fill: #fff; font-size: 8px; font-weight: 800; text-anchor: middle; pointer-events: none; }
    .graph-node .unfused-label { fill: #a66700; font-size: 10.5px; font-weight: 700; pointer-events: none; }
    .graph-empty { padding: 24px; color: var(--muted); text-align: center; }
    .autofuse-analytics {
      flex: 0 0 auto;
      min-height: 0;
      position: relative;
      border-top: 1px solid var(--line);
      background: #fbfcfe;
    }
    .autofuse-analytics[hidden] { display: none; }
    .autofuse-analytics[open] { flex-basis: auto; display: block; }
    .autofuse-analytics > summary { padding: 8px 12px; color: var(--muted); font-size: 12px; font-weight: 700; cursor: pointer; user-select: none; }
    .autofuse-analytics > summary::marker { color: var(--accent); }
    .lifecycle-resize-handle { position: absolute; z-index: 8; top: -6px; left: 0; right: 0; height: 12px; cursor: ns-resize; touch-action: none; }
    .lifecycle-resize-handle::after { content: ''; position: absolute; top: 4px; left: 50%; width: 54px; height: 4px; border-radius: 2px; background: #9aa5b1; transform: translateX(-50%); transition: width 120ms ease, background 120ms ease; }
    .lifecycle-resize-handle:hover::after, .lifecycle-resize-handle:focus-visible::after { width: 72px; background: var(--accent); }
    .autofuse-analytics:not([open]) .lifecycle-resize-handle { display: none; }
    body.lifecycle-resizing, body.lifecycle-resizing * { cursor: ns-resize !important; user-select: none !important; }
    .graph-shell.lifecycle-expanded .graph-body { min-height: 0; }
    .lifecycle-summary-stats { margin-left: 8px; color: #7d8794; font-weight: 400; }
    .autofuse-analytics-content { height: var(--lifecycle-height, clamp(260px, 32vh, 340px)); min-height: 0; display: flex; flex-direction: column; gap: 7px; padding: 0 10px 10px; overflow: hidden; }
    .lifecycle-toolbar { flex: 0 0 auto; display: flex; align-items: center; flex-wrap: wrap; gap: 7px; }
    .lifecycle-modes { display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
    .lifecycle-modes button { height: 28px; border: 0; border-right: 1px solid var(--line); border-radius: 0; padding: 0 10px; font-size: 12px; }
    .lifecycle-modes button:last-child { border-right: 0; }
    .lifecycle-modes button.active { color: var(--accent); background: var(--accent-soft); font-weight: 700; }
    .lifecycle-filter { height: 28px; font-size: 12px; }
    .lifecycle-stats { min-width: 0; flex: 1 1 220px; color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .lifecycle-zoom { display: inline-flex; align-items: center; gap: 3px; }
    .lifecycle-zoom button { width: 28px; height: 28px; padding: 0; }
    .lifecycle-zoom .lifecycle-fit { width: 40px; }
    .lifecycle-zoom-level { min-width: 38px; color: var(--muted); font-size: 11px; text-align: center; }
    .lifecycle-scroll { height: 0; min-height: 0; flex: 1 1 auto; overflow: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); overscroll-behavior: contain; scrollbar-gutter: stable; }
    .lifecycle-grid { position: relative; display: grid; min-width: max-content; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .lifecycle-corner, .lifecycle-node-head { position: sticky; top: 0; z-index: 4; display: flex; flex-direction: column; justify-content: center; border-bottom: 1px solid var(--line); background: #f6f8fb; }
    .lifecycle-corner { left: 0; z-index: 6; padding: 0 10px; color: var(--muted); font-size: 11px; border-right: 1px solid var(--line); }
    .lifecycle-node-head { align-items: center; padding: 3px 4px; border-right: 1px solid #e7ebf1; color: var(--muted); font-size: 10px; cursor: pointer; }
    .lifecycle-node-head strong { max-width: 100%; color: #3d4856; font-size: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .lifecycle-cache-mini { position: absolute; top: 2px; right: 3px; color: #087f8c; font: 800 7px/1 system-ui, sans-serif; }
    .lifecycle-node-head.selected { color: var(--accent); background: var(--accent-soft); }
    .lifecycle-node-head.broadcast-cache { box-shadow: inset 0 3px #087f8c; color: #087f8c; background: #eefafb; }
    .lifecycle-node-head.broadcast-cache.selected { background: #dff6f7; }
    .lifecycle-row-label { position: sticky; left: 0; z-index: 3; display: grid; grid-template-columns: 10px minmax(0, 1fr) auto; align-items: center; gap: 7px; min-width: 0; padding: 5px 8px; border-right: 1px solid var(--line); border-bottom: 1px solid #edf0f5; background: #fbfcfe; cursor: pointer; }
    .lifecycle-row-label:hover, .lifecycle-row-label.selected { background: var(--accent-soft); }
    .lifecycle-resource-swatch { width: 9px; height: 28px; border-radius: 3px; background: var(--resource-fill); border: 1px solid var(--resource-stroke); }
    .lifecycle-resource-copy { min-width: 0; }
    .lifecycle-resource-copy strong, .lifecycle-resource-copy small { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .lifecycle-resource-copy strong { font-size: 11px; }
    .lifecycle-resource-copy small { color: var(--muted); font-size: 9.5px; }
    .lifecycle-status { padding: 1px 5px; border-radius: 999px; font: 700 9px/1.5 system-ui, sans-serif; white-space: nowrap; }
    .lifecycle-status.shared { color: #147a4d; background: #e7f5ee; }
    .lifecycle-status.reused { color: #a66700; background: #fff1d6; }
    .lifecycle-status.single { color: #2457a6; background: #eaf1ff; }
    .lifecycle-status.buffer { color: #5e477a; background: #f2eafd; }
    .lifecycle-lane { position: relative; border-bottom: 1px solid #edf0f5; background-image: linear-gradient(to right, transparent calc(var(--cell) - 1px), #edf0f5 calc(var(--cell) - 1px)); background-size: var(--cell) 100%; }
    .lifecycle-lane:hover { background-color: #f8faff; }
    .lifecycle-segment { position: absolute; top: calc((var(--row-height) - 20px) / 2); height: 20px; min-width: 18px; border: 2px solid var(--resource-stroke); border-radius: 5px; background-color: var(--resource-fill); cursor: pointer; transition: opacity 120ms ease, filter 120ms ease; }
    .lifecycle-segment.shared { border-color: #2f8f62; box-shadow: inset 0 0 0 1px #fff, 0 0 0 1px #2f8f62; }
    .lifecycle-segment.reused { border-color: #b97800; border-style: dashed; background-image: repeating-linear-gradient(135deg, transparent 0 5px, rgba(185, 120, 0, 0.2) 5px 8px); }
    .lifecycle-segment.single { border-color: #2457a6; }
    .lifecycle-segment.buffer { border-color: var(--resource-stroke); }
    .lifecycle-segment.selected { filter: drop-shadow(0 0 3px rgba(36, 87, 166, 0.5)); outline: 2px solid var(--accent); outline-offset: 2px; }
    .lifecycle-segment.dimmed, .lifecycle-row-label.dimmed { opacity: 0.24; }
    .lifecycle-marker { position: absolute; top: calc((var(--row-height) - 10px) / 2); z-index: 2; width: 10px; height: 10px; margin-left: -5px; border: 2px solid var(--resource-stroke); border-radius: 50%; background: var(--resource-stroke); pointer-events: none; }
    .lifecycle-marker.consumer { background: #fff; }
    .lifecycle-legend { flex: 0 0 auto; display: flex; align-items: center; flex-wrap: wrap; gap: 12px; color: var(--muted); font-size: 10.5px; }
    .lifecycle-legend span::before { content: ''; display: inline-block; width: 12px; height: 8px; margin-right: 4px; border: 2px solid #2457a6; border-radius: 2px; vertical-align: -1px; }
    .lifecycle-legend .shared::before { border-color: #2f8f62; background: #e7f5ee; }
    .lifecycle-legend .reused::before { border-color: #b97800; border-style: dashed; background: #fff1d6; }
    .lifecycle-legend .producer::before { width: 8px; height: 8px; border-radius: 50%; background: #2457a6; }
    .lifecycle-legend .consumer::before { width: 8px; height: 8px; border-radius: 50%; background: #fff; }
    .load-legend { display: inline-flex; align-items: center; gap: 8px; color: var(--muted); font-size: 11px; font-weight: 400; }
    .load-legend[hidden] { display: none; }
    .load-legend span::before { content: ''; display: inline-block; width: 9px; height: 9px; margin-right: 3px; border-radius: 2px; vertical-align: -1px; }
    .load-legend .pass::before { background: #dff3e9; border: 1px solid #2f8f62; }
    .load-legend .fail::before { background: #fde2e2; border: 1px solid #d34a4a; }
    .load-legend .warning::before { background: #fff1d6; border: 1px solid #b97800; }
    .load-legend .unfused::before { width: 10px; height: 10px; border-radius: 50%; background: #b97800; border: 1px solid #b97800; }
    .load-legend .unmapped::before { background: #eef1f5; border: 1px solid #9aa5b1; }
    .load-legend .internal::before { background: #eaf1ff; border: 1px solid #2457a6; }
    .load-legend .inserted::before { background: #fff1d6; border: 1px solid #a66700; }
    .load-legend .resource::before { background: #dce9ff; border: 1px solid #3e6fb6; }
    .load-legend .cache::before { width: 10px; height: 10px; border-radius: 50%; background: #087f8c; border: 1px solid #087f8c; }
    .load-legend .shared::before { background: #fff; border: 2px solid #2f8f62; }
    .load-legend .reused::before { background: #fff; border: 2px dashed #b97800; }
    .load-legend .single::before { background: #fff; border: 2px solid #2457a6; }
    .details-panel {
      flex: 0 0 360px;
      min-height: 0;
      display: flex;
      flex-direction: column;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .details-panel[hidden] { display: none; }
    .details-head {
      flex: 0 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }
    .details-title { font-weight: 700; min-width: 0; overflow-wrap: anywhere; }
    .details-body { min-height: 0; overflow: auto; padding: 12px; }
    .details-kicker { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
    .details-op-type { margin-top: 2px; color: var(--muted); }
    .linked-autofuse-button { width: 100%; margin-top: 10px; color: var(--accent); font-weight: 700; }
    .details-grid { display: grid; grid-template-columns: auto 1fr; gap: 5px 10px; margin-top: 12px; font-size: 13px; }
    .details-grid dt { color: var(--muted); }
    .details-grid dd { margin: 0; overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .detail-section { margin-top: 16px; padding-top: 12px; border-top: 1px solid #edf0f5; }
    .detail-section-title { margin-bottom: 8px; font-weight: 700; }
    .io-list { display: grid; gap: 6px; }
    .io-item {
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
      overflow-wrap: anywhere;
    }
    .io-item small { display: block; color: var(--muted); margin-top: 2px; }
    .highlight-controls { display: inline-flex; align-items: center; gap: 3px; white-space: nowrap; }
    .highlight-controls .control-label { color: var(--muted); font-size: 12px; margin-right: 2px; }
    .highlight-controls button { height: 26px; padding: 0 8px; font-size: 12px; }
    .highlight-controls button.active { color: var(--accent); background: var(--accent-soft); border-color: var(--accent); }
    .highlight-controls button:disabled { cursor: default; opacity: 0.45; }
    .table-panel {
      flex: 0 0 auto;
      max-height: 46vh;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: auto;
    }
    .table-panel summary { padding: 8px 12px; }
    .table-content { border-top: 1px solid var(--line); }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto auto auto;
      gap: 8px;
      padding: 10px 12px;
      align-items: center;
    }
    input, select { height: 32px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--text); padding: 0 9px; }
    .chart-wrap { padding: 8px 12px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
    canvas { width: 100%; height: 120px; display: block; }
    .table-wrap { overflow: auto; max-height: 34vh; }
    table { width: 100%; border-collapse: collapse; min-width: 1180px; }
    th, td { padding: 7px 9px; border-bottom: 1px solid #edf0f5; text-align: left; vertical-align: top; }
    th { background: #fbfcfe; position: sticky; top: 0; z-index: 2; font-size: 12px; color: var(--muted); user-select: none; cursor: pointer; white-space: nowrap; }
    td { white-space: nowrap; }
    td.op { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    tbody tr { cursor: pointer; }
    tbody tr:hover { background: var(--accent-soft); }
    .badge { display: inline-flex; align-items: center; min-height: 22px; padding: 0 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }
    .badge.pass { color: var(--good); background: #e7f5ee; }
    .badge.fail { color: var(--bad); background: #fdecec; }
    .badge.warn { color: var(--warn); background: #fff1d6; }
    .status-ok { color: var(--good); }
    .status-warn { color: var(--warn); }
    .status-bad { color: var(--bad); }
    .empty { padding: 24px; color: var(--muted); text-align: center; }
    @media (max-width: 1000px) {
      body { overflow: auto; }
      .app-main { min-height: 760px; height: auto; }
      .title-row { align-items: flex-start; flex-direction: column; gap: 8px; }
      .quick-status { flex-wrap: wrap; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .workspace { min-height: 650px; }
      .graph-head { align-items: stretch; flex-direction: column; }
      .graph-title-row { flex-wrap: wrap; }
      .graph-title-copy { flex: 1 1 100%; }
      .graph-head-right { justify-content: flex-start; }
      .graph-meta { white-space: normal; }
      .autofuse-analytics[open] { flex-basis: 440px; }
      .autofuse-analytics-content { height: var(--lifecycle-height, 386px); }
      .details-panel { position: absolute; inset: 8px 8px 8px auto; width: min(360px, calc(100% - 16px)); z-index: 5; }
      .table-panel { max-height: none; }
      .toolbar { grid-template-columns: 1fr; }
      .table-wrap { max-height: none; }
    }
  </style>
</head>
<body>
  <header class="app-header">
    <div class="title-row">
      <h1>__TITLE__</h1>
      <div class="quick-status">
        <span id="resultStatus" class="status-pill __STATUS_CLASS__">__STATUS_TEXT__</span>
        <span class="quick-stat">Strict <strong>__STRICT_PASSED__/__COMPARED__</strong></span>
        <span class="quick-stat">CANN <strong>__CANN_PASSED__/__COMPARED__</strong></span>
        <span class="quick-stat">coverage <strong>__COVERAGE__%</strong></span>
        <button id="toggleMetrics" type="button" aria-expanded="false">指标</button>
      </div>
    </div>
    <details id="metricsPanel" class="metrics-panel">
      <summary>报告指标</summary>
      <div class="metrics">
        <div class="metric"><div class="label">Compared</div><div class="value">__COMPARED__/__TOTAL__</div></div>
        <div class="metric"><div class="label">Coverage</div><div class="value __COVERAGE_CLASS__">__COVERAGE__%</div></div>
        <div class="metric"><div class="label">Strict Elementwise Pass</div><div class="value">__STRICT_PASSED__/__COMPARED__ (__PASS_RATE__%)</div></div>
        <div class="metric"><div class="label">CANN Metrics Pass</div><div class="value">__CANN_PASSED__/__COMPARED__ (__CANN_PASS_RATE__%)</div></div>
        <div class="metric"><div class="label">Strict / CANN Failed</div><div class="value __FAILED_CLASS__">__STRICT_FAILED__ / __CANN_FAILED__</div></div>
        <div class="metric"><div class="label">Standard Disagreements</div><div class="value __DISAGREEMENT_CLASS__">__DISAGREEMENTS__</div></div>
        <div class="metric"><div class="label">Unmapped / No dump</div><div class="value __UNMATCHED_CLASS__">__UNMAPPED__ / __NOT_CAPTURED__</div></div>
        <div class="metric"><div class="label">Max Abs / Max Rel</div><div class="value">__MAX_ABS__ / __MAX_REL__</div></div>
      </div>
    </details>
  </header>
  <main class="app-main">
    <section class="workspace">
      <div class="graph-shell">
        <div class="graph-tabs" id="graphTabs" role="tablist" aria-label="图视图">
          <button class="graph-tab active" type="button" data-graph-tab="main" role="tab" aria-selected="true">CPU&amp;NPU 精度对比</button>
        </div>
        <div class="graph-head">
          <div class="graph-title-row">
            <button class="graph-back" id="graphBack" type="button" title="返回 NPU 主图" hidden>←</button>
            <div class="graph-title-copy">
              <div class="graph-title" id="graphTitle">NPU Autofuse Dump Graph</div>
              <div class="graph-meta" id="npuGraphMeta"></div>
            </div>
            <button class="fusion-details-button" id="fusionDetails" type="button" title="查看父融合算子的精度和输入输出详情" hidden>融合详情</button>
            <div class="load-legend" id="loadLegend" aria-label="精度结果分类">
              <span class="pass">双标准通过</span><span class="warning">仅 Strict 失败</span><span class="fail">CANN 失败</span><span class="unmapped">未映射 / 未采集</span><span class="internal">输入 / 权重</span><span class="unfused">未融合原因</span>
            </div>
          </div>
          <div class="graph-head-right">
            <div class="highlight-controls" aria-label="联动高亮范围">
              <span class="control-label">联动高亮</span>
              <button type="button" data-highlight-mode="depth1" disabled>一层输入输出</button>
              <button type="button" data-highlight-mode="depth2" disabled>两层输入输出</button>
              <button type="button" data-highlight-mode="all" disabled>全部输入输出</button>
              <button type="button" data-highlight-mode="clear" disabled>清除</button>
            </div>
            <div class="zoom-controls" data-graph="npuGraph">
              <button type="button" data-zoom-action="out" title="缩小">-</button>
              <span class="zoom-level" id="npuGraphZoom">100%</span>
              <button type="button" data-zoom-action="in" title="放大">+</button>
              <button class="zoom-reset" type="button" data-zoom-action="fit" title="自动适配并居中">Fit</button>
            </div>
          </div>
        </div>
        <div class="graph-body" id="npuGraph" tabindex="0" aria-label="NPU Autofuse dump graph，可滚轮缩放、拖拽和点击节点"></div>
        <details class="autofuse-analytics" id="autofuseAnalytics" hidden aria-label="AutoFuse 资源生命周期">
          <summary>Queue / Buffer 生命周期 <span class="lifecycle-summary-stats" id="lifecycleSummaryStats"></span></summary>
          <div class="lifecycle-resize-handle" id="lifecycleResizeHandle" role="separator" aria-label="拖动调整生命周期区域高度" aria-orientation="horizontal" aria-valuemin="220" tabindex="0" title="上下拖动调整高度，双击尽量显示全部资源"></div>
          <div class="autofuse-analytics-content">
            <div class="lifecycle-toolbar">
              <div class="lifecycle-modes" role="group" aria-label="资源类型">
                <button type="button" class="active" data-lifecycle-kind="queue">Queue</button>
                <button type="button" data-lifecycle-kind="buffer">Buffer</button>
                <button type="button" data-lifecycle-kind="all">全部</button>
              </div>
              <select class="lifecycle-filter" id="lifecycleFilter" aria-label="生命周期筛选">
                <option value="all">全部状态</option>
                <option value="shared">仅共用</option>
                <option value="reused">仅复用</option>
                <option value="single">仅单独使用</option>
                <option value="current">仅当前节点相关</option>
              </select>
              <div class="lifecycle-stats" id="lifecycleStats"></div>
              <div class="lifecycle-zoom" aria-label="生命周期横轴缩放">
                <button type="button" data-lifecycle-zoom="out" title="缩小">-</button>
                <span class="lifecycle-zoom-level" id="lifecycleZoomLevel">100%</span>
                <button type="button" data-lifecycle-zoom="in" title="放大">+</button>
                <button class="lifecycle-fit" type="button" data-lifecycle-zoom="fit" title="适配宽度">Fit</button>
              </div>
            </div>
            <div class="lifecycle-scroll" id="lifecycleTimeline" tabindex="0" aria-label="AutoFuse Queue 和 Buffer 静态占用区间"></div>
            <div class="lifecycle-legend">
              <span class="shared">共用</span><span class="reused">复用</span>
              <span class="producer">生产节点</span><span class="consumer">直接消费节点</span>
              <span>横条为按拓扑节点推导的静态占用区间</span>
            </div>
          </div>
        </details>
      </div>
      <aside id="detailsPanel" class="details-panel" hidden aria-label="节点详情">
        <div class="details-head">
          <div class="details-title" id="detailsTitle">节点详情</div>
          <button id="dismissDetails" type="button" title="关闭节点详情">×</button>
        </div>
        <div class="details-body">
          <div class="details-kicker" id="detailsKicker">selected NPU Autofuse op</div>
          <div class="details-op-type" id="detailsType">点击图中节点查看</div>
          <button class="linked-autofuse-button" id="linkedAutofuseButton" type="button" hidden></button>
          <dl class="details-grid" id="detailsGrid"></dl>
          <div class="detail-section" id="ioSection">
            <div class="detail-section-title">输入 / 输出</div>
            <div class="io-list" id="detailsIo"></div>
          </div>
        </div>
      </aside>
    </section>
    <details id="tablePanel" class="table-panel">
      <summary>逐算子精度明细 <span id="tableSummary"></span></summary>
      <div class="table-content">
        <div class="toolbar">
          <input id="search" type="search" placeholder="搜索 op / type / status">
          <select id="passFilter">
            <option value="all">全部</option>
            <option value="both_passed">双标准通过</option>
            <option value="failed">Strict 失败</option>
            <option value="cann_failed">CANN 失败</option>
            <option value="disagreement">标准不一致</option>
            <option value="unmatched">全部未对比</option>
            <option value="unmapped">仅未映射</option>
            <option value="not_captured">仅未采集</option>
          </select>
          <select id="typeFilter"><option value="all">全部类型</option></select>
          <button id="exportCsv" type="button">导出 CSV</button>
        </div>
        <div class="chart-wrap"><canvas id="errorChart" height="120"></canvas></div>
        <div class="table-wrap">
          <table id="resultTable">
            <thead><tr>
              <th data-key="passed">结果</th><th data-key="strict_passed">Strict</th><th data-key="cann_passed">CANN</th><th data-key="status">状态</th><th data-key="npu_op">NPU Autofuse op</th>
              <th data-key="npu_type">类型</th><th data-key="cpu_op">CPU op</th><th data-key="shape">shape</th>
              <th data-key="dtype">dtype</th><th data-key="rtol">rtol / atol</th><th data-key="max_abs_error">max abs</th><th data-key="max_rel_error">max rel</th>
              <th data-key="rmse">rmse</th><th data-key="cann_cosine_similarity">CANN cosine</th><th data-key="mismatched">mismatch</th>
              <th data-key="max_error_index">max finite index / non-finite</th>
            </tr></thead>
            <tbody></tbody>
          </table>
          <div id="empty" class="empty" hidden>没有匹配的结果</div>
        </div>
      </div>
    </details>
  </main>
  <script id="reportData" type="application/json">__REPORT_JSON__</script>
  <script>
    const report = JSON.parse(document.getElementById('reportData').textContent);
    const rows = (report.results || []).map((row, idx) => ({...row, _idx: idx}));
    const npuGraph = report.graphs?.npu || {nodes: [], edges: [], fusions: []};
    const cpuGraph = report.graphs?.cpu || {nodes: []};
    const autofuseGraphs = report.graphs?.autofuse?.graphs || [];
    function kernelNameFromText(value) {
      const match = String(value || '').replace(/^graph\\d+\\./, '').match(/^(autofused_.+_[0-9a-f]{32})(?:__.*)?$/);
      return match ? match[1] : '';
    }
    function graphKernelName(graph) {
      if (graph?.kernel_name) return graph.kernel_name;
      const sourceMatch = String(graph?.source || '').match(/(autofused_.+?_[0-9a-f]{32})(?:\\/|$)/);
      return sourceMatch ? sourceMatch[1] : '';
    }
    function kernelDisplayName(kernel) {
      if (!kernel) return '未关联 kernel';
      const match = kernel.match(/^autofused_(.*)_([0-9a-f]{32})$/);
      return match ? `${match[1]} · ${match[2].slice(0, 8)}` : kernel;
    }
    const autofuseGroups = new Map();
    autofuseGraphs.forEach((graph, index) => {
      const kernel = graphKernelName(graph) || '__unlinked__';
      if (!autofuseGroups.has(kernel)) autofuseGroups.set(kernel, []);
      autofuseGroups.get(kernel).push(index);
    });
    const state = {sortKey: 'max_abs_error', sortDir: -1, filtered: rows.slice()};
    const selection = {name: null, mode: 'depth1', detailNode: null, explicit: null};
    const graphViews = new Map();
    const tbody = document.querySelector('#resultTable tbody');
    const empty = document.getElementById('empty');
    const search = document.getElementById('search');
    const passFilter = document.getElementById('passFilter');
    const typeFilter = document.getElementById('typeFilter');
    const detailsPanel = document.getElementById('detailsPanel');
    const mainNpuNodes = new Map((npuGraph.nodes || []).map(node => [node.name, node]));
    const cpuNodes = new Map((cpuGraph.nodes || []).map(node => [node.name, node]));
    const resultByNpu = new Map(rows.map(row => [row.npu_op, row]));
    const queueClassColors = {
      shared: {fill: '#dff3e9', stroke: '#2f8f62'},
      reused: {fill: '#fff1d6', stroke: '#b97800'},
      single: {fill: '#eaf1ff', stroke: '#2457a6'},
    };
    const queueClassLabels = {shared: 'QUE SHARED', reused: 'QUE REUSED', single: 'QUE SINGLE'};
    const broadcastCacheConditions = new Set([
      'cache_block_split_fused_brc_axis', 'cache_block_split_origin_brc_axis',
    ]);
    const resourcePalettes = {
      queue: [
        {fill: '#dce9ff', stroke: '#3e6fb6'}, {fill: '#dff1e7', stroke: '#328060'},
        {fill: '#fae8d6', stroke: '#b56c28'}, {fill: '#eee4f8', stroke: '#7956a3'},
        {fill: '#dff0f2', stroke: '#2f7f86'}, {fill: '#f8dfe4', stroke: '#b64d64'},
      ],
      buffer: [
        {fill: '#ece5f5', stroke: '#70558f'}, {fill: '#e5edf4', stroke: '#536f89'},
        {fill: '#f3e4dc', stroke: '#95604a'}, {fill: '#e4efdf', stroke: '#5c7f4d'},
        {fill: '#f2e6ee', stroke: '#8d567a'}, {fill: '#e5eeee', stroke: '#527b7d'},
      ],
    };
    const lifecycleState = {
      kind: 'queue', filter: 'all', scale: 1, graph: null, rows: [], selectedKey: null, currentNode: null,
      rowHeight: 54, height: null,
    };
    const graphContext = {
      kind: 'main',
      graph: npuGraph,
      nodes: mainNpuNodes,
      edges: (npuGraph.edges || []).filter(edge => mainNpuNodes.has(edge.src) && mainNpuNodes.has(edge.dst)),
      parent: null,
      autoIndex: null,
    };

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
    function fmt(value) {
      if (value === undefined || value === null || value === '') return '-';
      const number = Number(value);
      return Number.isFinite(number) ? number.toExponential(3) : '-';
    }
    function indexText(value) {
      return Array.isArray(value) ? '[' + value.join(', ') + ']' : (value ?? '-');
    }
    function nonfiniteCounts(counts) {
      const labels = [['neg_inf', '-Inf'], ['pos_inf', '+Inf'], ['nan', 'NaN']];
      const parts = labels.filter(([key]) => Number(counts?.[key] || 0) > 0).map(([key, label]) => `${label}×${counts[key]}`);
      return parts.length ? parts.join(', ') : 'none';
    }
    function nonfiniteBrief(row) {
      const stats = row?.nonfinite;
      if (!stats?.present) return '';
      const keys = ['neg_inf', 'pos_inf', 'nan'];
      const fullyAligned = Number(stats.different_positions || 0) === 0
        && keys.every(key => Number(stats.cpu?.[key] || 0) === Number(stats.aligned?.[key] || 0))
        && keys.every(key => Number(stats.npu?.[key] || 0) === Number(stats.aligned?.[key] || 0));
      if (fullyAligned) return `${nonfiniteCounts(stats.aligned)}（CPU/NPU 同位置）`;
      return `CPU ${nonfiniteCounts(stats.cpu)}; NPU ${nonfiniteCounts(stats.npu)}; 同类型位置 ${nonfiniteCounts(stats.aligned)}; 不同位置 ${stats.different_positions || 0}`;
    }
    function firstNonfiniteText(row) {
      const stats = row?.nonfinite;
      if (!stats?.present) return '-';
      return `${indexText(stats.first_index)}（CPU=${stats.first_cpu}, NPU=${stats.first_npu}）`;
    }
    function cannMetricText(row, key) {
      if (row?.[key] !== undefined && row?.[key] !== null && row?.[key] !== '') return fmt(row[key]);
      if (row?.cann_unavailable_reason === 'NONFINITE_INPUT') return `N/A（${nonfiniteBrief(row)}）`;
      return '-';
    }
    function finiteMetricText(row, key) {
      return row?.finite_pair_count === 0 ? 'N/A（无有限值对）' : fmt(row?.[key]);
    }
    function mismatchText(row) {
      const base = `${row?.mismatched ?? '-'}/${row?.num_elements ?? '-'}`;
      const nonfinite = nonfiniteBrief(row);
      return nonfinite ? `${base}; ${nonfinite}` : base;
    }
    function maxIndexText(row) {
      const finite = row?.finite_pair_count === 0 ? '有限值 N/A' : `有限值 ${indexText(row?.max_error_index)}`;
      return row?.nonfinite?.present ? `${finite}; 首个非有限值 ${firstNonfiniteText(row)}` : indexText(row?.max_error_index);
    }
    function shapeText(shape) {
      return Array.isArray(shape) ? '[' + shape.join('x') + ']' : '?';
    }
    function dtypeText(dtype) {
      return dtype ? String(dtype).replace('float', 'f').replace('int', 'i') : '?';
    }
    function tensorText(tensor) {
      if (!tensor) return '?';
      const format = tensor.format && tensor.format !== 'ND' ? '/' + tensor.format : '';
      return dtypeText(tensor.dtype) + shapeText(tensor.shape) + format;
    }
    function shortText(value, maxLen) {
      const text = String(value || '-');
      return text.length > maxLen ? text.slice(0, Math.max(1, maxLen - 3)) + '...' : text;
    }
    function resourceColor(kind, id) {
      const palette = resourcePalettes[kind] || resourcePalettes.queue;
      const index = Math.abs(Number(id) || 0) % palette.length;
      return palette[index];
    }
    function nodeMetric(name) {
      const row = resultByNpu.get(name);
      if (!row) return null;
      return {
        row,
        compared: row.compared !== false,
        strictPassed: !!(row.strict_passed ?? row.passed),
        cannPassed: !!(row.cann_passed ?? row.passed),
      };
    }
    function nodeState(node, contextKind = graphContext.kind) {
      if (contextKind === 'fusion') {
        if (node.origin === false) return {fill: '#fff1d6', stroke: '#a66700', label: 'INSERTED', metric: null, dashed: true};
        return {fill: '#eaf1ff', stroke: '#2457a6', label: 'INTERNAL', metric: null, dashed: false};
      }
      if (contextKind === 'autofuse') {
        const queueMemory = (node.outputs || []).find(memory => Number(memory.que_id) >= 0);
        if (queueMemory) {
          const queueClass = queueMemory.queue_class || node.queue_class || node.load_class || 'single';
          const color = resourceColor('queue', queueMemory.que_id);
          const semantic = queueClassColors[queueClass] || queueClassColors.single;
          return {
            fill: color.fill, stroke: semantic.stroke,
            label: `QUEUE ${queueMemory.que_id} · ${queueClassLabels[queueClass] || queueClassLabels.single}`,
            metric: null, dashed: queueClass === 'reused', strokeWidth: queueClass === 'shared' ? 3 : 2,
          };
        }
        const bufferMemory = (node.outputs || []).find(memory => Number(memory.buf_id) >= 0);
        const tmpBuffer = (node.tmp_buffers || [])[0];
        const bufferId = bufferMemory?.buf_id ?? tmpBuffer?.id;
        if (Number(bufferId) >= 0) {
          const color = resourceColor('buffer', bufferId);
          return {...color, label: `BUFFER ${bufferId}`, metric: null};
        }
        return {fill: '#eef1f5', stroke: '#7d8794', label: 'ASCIR OP', metric: null};
      }
      if (node.type === 'Data') return {fill: '#eaf1ff', stroke: '#2457a6', label: 'INPUT / WEIGHT', metric: null};
      const metric = nodeMetric(node.name);
      if (!metric) return {fill: '#eef1f5', stroke: '#9aa5b1', label: 'NO DATA', metric: null};
      if (!metric.compared) {
        const noDump = metric.row.status === 'NPU_OUTPUT_NOT_CAPTURED';
        return {fill: '#eef1f5', stroke: '#9aa5b1', label: noDump ? 'NO DUMP' : 'NO MAP', metric, dashed: noDump};
      }
      if (!metric.cannPassed) {
        const label = metric.strictPassed ? 'CANN FAIL' : 'BOTH FAIL';
        return {fill: '#fde2e2', stroke: '#d34a4a', label, metric};
      }
      if (!metric.strictPassed) return {fill: '#fff1d6', stroke: '#b97800', label: 'STRICT FAIL', metric};
      if (metric.row.status && metric.row.status !== 'OK') return {fill: '#fff1d6', stroke: '#b97800', label: 'WARN', metric};
      return {fill: '#dff3e9', stroke: '#2f8f62', label: 'BOTH PASS', metric};
    }
    function isFusionNode(node) {
      const origins = Array.isArray(node?.origin_ops) ? node.origin_ops : [];
      return origins.length > 1 || /fusion/i.test(String(node?.type || '') + ' ' + String(node?.name || ''));
    }
    function nodeKernelName(node) {
      return kernelNameFromText(node?.name || '');
    }
    function linkedAutofuseIndices(node) {
      const kernel = nodeKernelName(node);
      return kernel ? (autofuseGroups.get(kernel) || []) : [];
    }
    function graphEdges() { return graphContext.edges; }
    function nodeDisplayName(node) { return node.displayName || node.name; }
    function memoryText(memory) {
      if (!memory) return '?';
      const fields = [];
      if (memory.que_id >= 0) fields.push('queue_id=' + memory.que_id);
      if (memory.buf_id >= 0) fields.push('buffer_id=' + memory.buf_id);
      if (memory.reuse_id >= 0) fields.push('reuse_id=' + memory.reuse_id);
      if (memory.depth >= 0) fields.push('depth=' + memory.depth);
      return (memory.place || 'mem') + (fields.length ? '[' + fields.join(',') + ']' : '');
    }
    function hasBroadcastCache(node) {
      return node?.broadcast_cache === true || broadcastCacheConditions.has(String(node?.exec_condition || ''));
    }
    function unfusedReasons(node) {
      return Array.isArray(node?.unfused_reasons) ? node.unfused_reasons : [];
    }
    function unfusedReasonText(item) {
      if (!item) return '';
      const stage = item.category === 'lowering_fallback'
        ? 'Lowering fallback'
        : item.category === 'implicit_fallback'
        ? 'Implicit fallback'
        : 'Fusion rejected';
      const source = item.source_node ? ` · source: ${item.source_node}` : '';
      const peer = item.peer ? ` · ${item.direction || 'peer'}: ${item.peer}` : '';
      const kind = item.kind ? ` · ${item.kind}` : '';
      const attempts = Number(item.attempts) > 1 ? ` · ${item.attempts} attempts` : '';
      return `${stage}${source}${kind}${peer}${attempts}: ${item.reason || 'unknown'}`;
    }
    function graphInputText(node) {
      if (graphContext.kind === 'autofuse') return (node.inputs || []).join(', ') || '-';
      return (node.inputs || []).map(tensorText).join(', ') || '-';
    }
    function graphOutputText(node) {
      if (graphContext.kind === 'autofuse') return (node.outputs || []).map(memoryText).join(', ') || '-';
      return (node.outputs || []).map(tensorText).join(', ') || '-';
    }
    function normalizeAutofuseGraph(raw) {
      const nodes = (raw.nodes || []).map(node => ({
        ...node,
        name: String(node.id),
        displayName: node.name,
      }));
      return {
        ...raw,
        nodes,
        edges: (raw.edges || []).map(edge => ({src: String(edge.src), dst: String(edge.dst)})),
      };
    }
    function layoutGraph(graph) {
      const nodes = graph.nodes || [];
      const known = new Set(nodes.map(node => node.name));
      const graphEdges = (graph.edges || []).filter(edge => known.has(edge.src) && known.has(edge.dst));
      const indeg = new Map(nodes.map(node => [node.name, 0]));
      const outs = new Map(nodes.map(node => [node.name, []]));
      for (const edge of graphEdges) {
        indeg.set(edge.dst, (indeg.get(edge.dst) || 0) + 1);
        outs.get(edge.src)?.push(edge.dst);
      }
      const queue = nodes.filter(node => (indeg.get(node.name) || 0) === 0).map(node => node.name);
      const level = new Map(queue.map(name => [name, 0]));
      let seen = 0;
      while (queue.length) {
        const name = queue.shift();
        seen += 1;
        for (const dst of outs.get(name) || []) {
          level.set(dst, Math.max(level.get(dst) || 0, (level.get(name) || 0) + 1));
          indeg.set(dst, (indeg.get(dst) || 0) - 1);
          if ((indeg.get(dst) || 0) === 0) queue.push(dst);
        }
      }
      if (seen < nodes.length) nodes.forEach((node, index) => { if (!level.has(node.name)) level.set(node.name, index); });
      const groups = new Map();
      for (const node of nodes) {
        const current = level.get(node.name) || 0;
        if (!groups.has(current)) groups.set(current, []);
        groups.get(current).push(node);
      }
      const nodeW = graphContext.kind === 'autofuse' ? 360 : 300;
      const nodeH = 136, gapX = 36, gapY = 76;
      const positions = new Map();
      const levels = [...groups.keys()].sort((a, b) => a - b);
      let maxCols = 1;
      for (const current of levels) maxCols = Math.max(maxCols, groups.get(current).length);
      for (const current of levels) {
        groups.get(current).forEach((node, col) => positions.set(node.name, {
          x: 30 + col * (nodeW + gapX), y: 30 + current * (nodeH + gapY), w: nodeW, h: nodeH,
        }));
      }
      const maxLevel = Math.max(...levels, 0);
      return {
        nodes, edges: graphEdges, positions,
        width: Math.max(640, 60 + maxCols * nodeW + Math.max(0, maxCols - 1) * gapX),
        height: Math.max(340, 60 + (maxLevel + 1) * nodeH + maxLevel * gapY),
      };
    }
    function applyGraphView(targetId) {
      const target = document.getElementById(targetId);
      const svg = target?.querySelector('svg.graph-svg');
      const view = graphViews.get(targetId);
      if (!svg || !view) return;
      const x = -view.x / view.scale;
      const y = -view.y / view.scale;
      const width = Math.max(1, target.clientWidth) / view.scale;
      const height = Math.max(1, target.clientHeight) / view.scale;
      svg.setAttribute('viewBox', `${x} ${y} ${width} ${height}`);
      const label = document.getElementById(targetId + 'Zoom');
      if (label) label.textContent = Math.round(view.scale * 100) + '%';
    }
    function clampGraphView(targetId) {
      const target = document.getElementById(targetId);
      const view = graphViews.get(targetId);
      if (!target || !view) return;
      const edge = 36;
      const scaledWidth = view.layout.width * view.scale;
      const scaledHeight = view.layout.height * view.scale;
      view.x = Math.min(target.clientWidth - edge, Math.max(edge - scaledWidth, view.x));
      view.y = Math.min(target.clientHeight - edge, Math.max(edge - scaledHeight, view.y));
    }
    function fitGraph(targetId) {
      const target = document.getElementById(targetId);
      const view = graphViews.get(targetId);
      if (!target || !view) return;
      const padding = 28;
      const maxFitScale = graphContext.kind === 'fusion' ? 1.35 : 1;
      const scale = Math.min(maxFitScale, (target.clientWidth - padding * 2) / view.layout.width, (target.clientHeight - padding * 2) / view.layout.height);
      view.fitScale = Math.max(0.005, scale);
      view.minScale = Math.max(0.005, view.fitScale * 0.25);
      view.maxScale = Math.max(4, view.fitScale * 16);
      view.scale = view.fitScale;
      view.x = (target.clientWidth - view.layout.width * view.scale) / 2;
      view.y = (target.clientHeight - view.layout.height * view.scale) / 2;
      view.userAdjusted = false;
      applyGraphView(targetId);
    }
    function changeGraphZoom(targetId, nextScale, anchorX, anchorY) {
      const target = document.getElementById(targetId);
      const view = graphViews.get(targetId);
      if (!target || !view) return;
      const scale = Math.min(view.maxScale, Math.max(view.minScale, nextScale));
      const x = anchorX ?? target.clientWidth / 2;
      const y = anchorY ?? target.clientHeight / 2;
      const graphX = (x - view.x) / view.scale;
      const graphY = (y - view.y) / view.scale;
      view.x = x - graphX * scale;
      view.y = y - graphY * scale;
      view.scale = scale;
      view.userAdjusted = true;
      clampGraphView(targetId);
      applyGraphView(targetId);
    }
    function relationSet() {
      if (selection.explicit instanceof Set) return new Set(selection.explicit);
      const active = new Set();
      if (!selection.name || selection.mode === 'clear') return active;
      active.add(selection.name);
      const incoming = new Map(), outgoing = new Map();
      for (const edge of graphEdges()) {
        if (!incoming.has(edge.dst)) incoming.set(edge.dst, []);
        if (!outgoing.has(edge.src)) outgoing.set(edge.src, []);
        incoming.get(edge.dst).push(edge.src);
        outgoing.get(edge.src).push(edge.dst);
      }
      const maxDepth = selection.mode === 'depth1' ? 1 : selection.mode === 'depth2' ? 2 : Infinity;
      function traverse(adjacency) {
        const queue = [{name: selection.name, depth: 0}];
        const visited = new Set([selection.name]);
        while (queue.length) {
          const current = queue.shift();
          if (current.depth >= maxDepth) continue;
          for (const name of adjacency.get(current.name) || []) {
            if (visited.has(name)) continue;
            visited.add(name);
            active.add(name);
            queue.push({name, depth: current.depth + 1});
          }
        }
      }
      traverse(incoming);
      traverse(outgoing);
      return active;
    }
    function updateSelectionView() {
      const target = document.getElementById('npuGraph');
      const active = relationSet();
      const hasSelection = !!selection.name && selection.mode !== 'clear';
      for (const node of target.querySelectorAll('.graph-node')) {
        const activeNode = active.has(node.dataset.node);
        node.classList.toggle('selected', node.dataset.node === selection.name);
        node.classList.toggle('related', activeNode && node.dataset.node !== selection.name);
        node.classList.toggle('dimmed', hasSelection && !activeNode);
      }
      for (const edge of target.querySelectorAll('.graph-edge')) {
        const related = active.has(edge.dataset.src) && active.has(edge.dataset.dst);
        edge.classList.toggle('related', related && hasSelection);
        edge.classList.toggle('dimmed', hasSelection && !related);
      }
      for (const button of document.querySelectorAll('[data-highlight-mode]')) {
        button.disabled = !selection.name;
        button.classList.toggle('active', button.dataset.highlightMode === selection.mode && hasSelection);
      }
    }
    function findCpuNode(name) {
      if (!name) return null;
      if (cpuNodes.has(name)) return cpuNodes.get(name);
      const lower = String(name).toLowerCase();
      return [...cpuNodes.values()].find(node => String(node.name).toLowerCase() === lower || String(node.type).toLowerCase() === lower) || null;
    }
    function cpuFallbackSubgraph(node) {
      const origins = Array.isArray(node.origin_ops) ? node.origin_ops : [];
      const resolved = [];
      const seen = new Set();
      for (const origin of origins) {
        const cpu = findCpuNode(origin);
        if (!cpu || seen.has(cpu.name)) continue;
        resolved.push({...cpu, origin: true});
        seen.add(cpu.name);
      }
      const edges = (cpuGraph.edges || []).filter(edge => seen.has(edge.src) && seen.has(edge.dst));
      return {
        source: 'CPU FX topology (legacy report fallback)',
        nodes: resolved,
        edges,
        missing: origins.filter(origin => !findCpuNode(origin)),
      };
    }
    function fusionSubgraph(node) {
      const captured = node.origin_subgraph;
      if (captured && Array.isArray(captured.nodes) && captured.nodes.length) return captured;
      return cpuFallbackSubgraph(node);
    }
    function prepareFusionGraph(parent) {
      const raw = fusionSubgraph(parent);
      const nodes = (raw.nodes || []).map(node => ({
        ...node,
        inputs: (node.inputs || []).map(input => ({...input})),
      }));
      const byName = new Map(nodes.map(node => [node.name, node]));
      const edges = (raw.edges || []).filter(edge => byName.has(edge.src) && byName.has(edge.dst));
      for (const edge of edges) {
        const source = byName.get(edge.src);
        const destination = byName.get(edge.dst);
        const existing = destination.inputs.find(input => input.index === (edge.dst_input ?? 0));
        const summary = {
          ...(source.outputs?.[0] || {}),
          ...(existing || {}),
          src: edge.src,
          index: edge.dst_input ?? 0,
        };
        if (existing) Object.assign(existing, summary);
        else destination.inputs.push(summary);
      }
      return {...raw, nodes, edges};
    }
    function enterFusionGraph(parent) {
      const graph = prepareFusionGraph(parent);
      closeDetails();
      graphContext.kind = 'fusion';
      graphContext.graph = graph;
      graphContext.nodes = new Map(graph.nodes.map(node => [node.name, node]));
      graphContext.edges = graph.edges;
      graphContext.parent = parent;
      graphContext.autoIndex = null;
      renderGraph();
    }
    function showAutofuseGraph(index) {
      const raw = autofuseGraphs[index];
      if (!raw) return;
      const graph = normalizeAutofuseGraph(raw);
      closeDetails();
      graphContext.kind = 'autofuse';
      graphContext.graph = graph;
      graphContext.nodes = new Map(graph.nodes.map(node => [node.name, node]));
      graphContext.edges = graph.edges;
      graphContext.parent = null;
      graphContext.autoIndex = index;
      renderGraph();
    }
    function showMainGraph() {
      closeDetails();
      graphContext.kind = 'main';
      graphContext.graph = npuGraph;
      graphContext.nodes = mainNpuNodes;
      graphContext.edges = (npuGraph.edges || []).filter(edge => mainNpuNodes.has(edge.src) && mainNpuNodes.has(edge.dst));
      graphContext.parent = null;
      graphContext.autoIndex = null;
      renderGraph();
    }
    function activateNode(name) {
      selectNode(name);
    }
    function enterFusionNode(name) {
      const node = graphContext.nodes.get(name);
      if (graphContext.kind !== 'main' || !node || !isFusionNode(node)) return;
      enterFusionGraph(node);
    }
    function renderDetails() {
      const node = selection.detailNode || graphContext.nodes.get(selection.name);
      if (!node) return;
      const isFusionParent = graphContext.kind === 'fusion' && node === graphContext.parent;
      const metric = graphContext.kind === 'main' || isFusionParent ? nodeMetric(node.name) : null;
      const row = metric?.row;
      const state = nodeState(node, isFusionParent ? 'main' : graphContext.kind);
      document.getElementById('detailsKicker').textContent = isFusionParent ? 'parent fused NPU Autofuse op' : 'selected NPU Autofuse op';
      document.getElementById('detailsTitle').textContent = nodeDisplayName(node);
      document.getElementById('detailsType').textContent = node.type || '-';
      const linkedButton = document.getElementById('linkedAutofuseButton');
      const linkedGraphs = (graphContext.kind === 'main' || isFusionParent) ? linkedAutofuseIndices(node) : [];
      linkedButton.hidden = !linkedGraphs.length;
      linkedButton.dataset.graphIndex = linkedGraphs.length ? String(linkedGraphs[0]) : '';
      linkedButton.textContent = linkedGraphs.length ? `查看 AutoFuseAfterOptimize (${linkedGraphs.length})` : '';
      const grid = document.getElementById('detailsGrid');
      const mapping = row?.compared === false
        ? (row?.status === 'NPU_OUTPUT_NOT_CAPTURED' ? 'NO DUMP' : 'NO MAP')
        : (row?.cpu_op ? `${row.cpu_op}:${row.cpu_index ?? 0}` : '-');
      const strictResult = row?.compared === false ? '-' : ((row?.strict_passed ?? row?.passed) ? 'PASS' : 'FAIL');
      const cannResult = row?.compared === false ? '-' : ((row?.cann_passed ?? row?.passed) ? 'PASS' : 'FAIL');
      const detailRows = [
        ['结果', state.label], ['Strict', strictResult], ['CANN', cannResult], ['状态', row?.status || '-'],
        ['CPU 对照', mapping], ['max abs（有限值）', finiteMetricText(row, 'max_abs_error')],
        ['max rel（有限值）', finiteMetricText(row, 'max_rel_error')], ['rmse（有限值）', finiteMetricText(row, 'rmse')],
        ['CANN cosine', cannMetricText(row, 'cann_cosine_similarity')],
        ['Relative Euclidean', cannMetricText(row, 'relative_euclidean_distance')],
        ['KL divergence', cannMetricText(row, 'kl_divergence')], ['mismatch', mismatchText(row)],
        ['最大有限误差位置', row?.finite_pair_count === 0 ? 'N/A（无有限值对）' : indexText(row?.max_error_index)],
      ];
      if (row?.compared !== false) detailRows.splice(5, 0,
        ['Strict Tolerance', `${fmt(row?.rtol)} / ${fmt(row?.atol)} (${row?.tolerance_source || '-'})`],
        ['CANN thresholds', 'cos>0.99 / rel_euclid<0.05 / KL<0.005'],
      );
      if (row?.nonfinite?.present) detailRows.push(
        ['非有限值统计', nonfiniteBrief(row)],
        ['首个非有限值位置', firstNonfiniteText(row)],
      );
      if (graphContext.kind === 'autofuse') {
        const condition = node.exec_condition || '-';
        detailRows.splice(1, 0,
          ['Broadcast cache', hasBroadcastCache(node) ? 'Conditional cache enabled' : (condition === 'no_cache' ? 'Disabled' : 'Unknown')],
          ['Execution condition', condition],
        );
      }
      if (graphContext.kind === 'main') {
        unfusedReasons(node).forEach((item, index) => {
          detailRows.push([`未融合原因 ${index + 1}`, unfusedReasonText(item)]);
        });
      }
      grid.innerHTML = detailRows.map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd>`).join('');
      const io = document.getElementById('detailsIo');
      const inputItems = graphContext.kind === 'autofuse'
        ? (node.inputs || []).map((item, index) => `<div class="io-item"><strong>input ${index}</strong> ← ${escapeHtml(item)}</div>`)
        : (node.inputs || []).map(item => `<div class="io-item"><strong>input ${item.index ?? 0}</strong> ← ${escapeHtml(item.src || '?')}<small>${escapeHtml(tensorText(item))}</small></div>`);
      const outputItems = graphContext.kind === 'autofuse'
        ? (node.outputs || []).map((item, index) => `<div class="io-item"><strong>output ${index}</strong><small>${escapeHtml(memoryText(item))}</small></div>`)
        : (node.outputs || []).map((item, index) => `<div class="io-item"><strong>output ${item.index ?? index}</strong><small>${escapeHtml(tensorText(item))}</small></div>`);
      io.innerHTML = inputItems.concat(outputItems).join('') || '<div class="io-item">无输入输出描述</div>';
      for (const button of document.querySelectorAll('[data-highlight-mode]')) {
        button.classList.toggle('active', button.dataset.highlightMode === selection.mode && selection.mode !== 'clear');
      }
    }
    function selectNode(name) {
      if (!graphContext.nodes.has(name)) return;
      selection.name = name;
      selection.mode = 'depth1';
      selection.explicit = null;
      selection.detailNode = graphContext.nodes.get(name);
      detailsPanel.hidden = false;
      detailsPanel.setAttribute('aria-hidden', 'false');
      renderDetails();
      updateSelectionView();
      updateGraphTabs(true);
      if (graphContext.kind === 'autofuse') {
        lifecycleState.currentNode = name;
        lifecycleState.selectedKey = null;
        if (lifecycleState.filter === 'current') renderLifecycleTimeline();
        else updateLifecycleSelection(true);
      }
    }
    function showFusionDetails() {
      if (graphContext.kind !== 'fusion' || !graphContext.parent) return;
      selection.name = null;
      selection.mode = 'clear';
      selection.explicit = null;
      selection.detailNode = graphContext.parent;
      detailsPanel.hidden = false;
      detailsPanel.setAttribute('aria-hidden', 'false');
      renderDetails();
      updateSelectionView();
      updateGraphTabs(true);
    }
    function closeDetails() {
      selection.name = null;
      selection.mode = 'clear';
      selection.explicit = null;
      selection.detailNode = null;
      detailsPanel.hidden = true;
      detailsPanel.setAttribute('aria-hidden', 'true');
      updateSelectionView();
      updateGraphTabs();
      if (graphContext.kind === 'autofuse') {
        lifecycleState.currentNode = null;
        lifecycleState.selectedKey = null;
        if (lifecycleState.filter === 'current') renderLifecycleTimeline();
        else updateLifecycleSelection(false);
      }
    }
    function initGraphInteractions() {
      const target = document.getElementById('npuGraph');
      let pointerId = null, startX = 0, startY = 0, lastX = 0, lastY = 0;
      let pointerNode = null, pointerFusionEntry = false;
      const view = () => graphViews.get('npuGraph');
      target.addEventListener('wheel', event => {
        event.preventDefault();
        const rect = target.getBoundingClientRect();
        const delta = Math.max(-200, Math.min(200, event.deltaY));
        changeGraphZoom('npuGraph', view().scale * Math.exp(-delta * 0.002), event.clientX - rect.left, event.clientY - rect.top);
      }, {passive: false});
      target.addEventListener('pointerdown', event => {
        if (event.button !== 0) return;
        pointerId = event.pointerId;
        startX = lastX = event.clientX;
        startY = lastY = event.clientY;
        pointerNode = event.target.closest?.('.graph-node')?.dataset.node || null;
        pointerFusionEntry = !!event.target.closest?.('.fusion-mark');
        view().dragged = false;
        target.setPointerCapture(pointerId);
        target.focus({preventScroll: true});
        event.preventDefault();
      });
      target.addEventListener('pointermove', event => {
        if (event.pointerId !== pointerId) return;
        const currentView = view();
        if (!currentView.dragged && Math.hypot(event.clientX - startX, event.clientY - startY) < 6) return;
        if (!currentView.dragged) {
          currentView.dragged = true;
          target.classList.add('is-panning');
        }
        const dx = event.clientX - lastX, dy = event.clientY - lastY;
        currentView.x += dx;
        currentView.y += dy;
        currentView.userAdjusted = true;
        lastX = event.clientX;
        lastY = event.clientY;
        clampGraphView('npuGraph');
        applyGraphView('npuGraph');
      });
      target.addEventListener('pointerup', event => {
        if (event.pointerId !== pointerId) return;
        const currentView = view();
        const wasDragged = currentView.dragged;
        const selectedNode = pointerNode;
        const selectedFusionEntry = pointerFusionEntry;
        if (target.hasPointerCapture(pointerId)) target.releasePointerCapture(pointerId);
        pointerId = null;
        pointerNode = null;
        pointerFusionEntry = false;
        currentView.dragged = false;
        target.classList.remove('is-panning');
        if (wasDragged) return;
        if (selectedNode && selectedFusionEntry) enterFusionNode(selectedNode);
        else if (selectedNode) activateNode(selectedNode);
        else closeDetails();
      });
      target.addEventListener('pointercancel', event => {
        if (event.pointerId !== pointerId) return;
        pointerId = null;
        pointerNode = null;
        pointerFusionEntry = false;
        view().dragged = false;
        target.classList.remove('is-panning');
      });
      target.addEventListener('dblclick', () => fitGraph('npuGraph'));
      target.addEventListener('keydown', event => {
        const currentView = view();
        if (!currentView) return;
        const focusedNode = event.target.closest?.('.graph-node')?.dataset.node;
        const focusedFusionEntry = !!event.target.closest?.('.fusion-mark');
        if ((event.key === 'Enter' || event.key === ' ') && focusedNode && focusedFusionEntry) enterFusionNode(focusedNode);
        else if ((event.key === 'Enter' || event.key === ' ') && focusedNode) activateNode(focusedNode);
        else if (event.key === '+' || event.key === '=') changeGraphZoom('npuGraph', currentView.scale * 1.2);
        else if (event.key === '-') changeGraphZoom('npuGraph', currentView.scale / 1.2);
        else if (event.key === '0' || event.key === 'Home') fitGraph('npuGraph');
        else return;
        event.preventDefault();
      });
    }
    function initZoomControls() {
      for (const controls of document.querySelectorAll('.zoom-controls')) {
        for (const button of controls.querySelectorAll('button[data-zoom-action]')) {
          button.addEventListener('click', () => {
            const view = graphViews.get(controls.dataset.graph);
            if (!view) return;
            if (button.dataset.zoomAction === 'in') changeGraphZoom('npuGraph', view.scale * 1.2);
            else if (button.dataset.zoomAction === 'out') changeGraphZoom('npuGraph', view.scale / 1.2);
            else fitGraph('npuGraph');
          });
        }
      }
      if (window.ResizeObserver) {
        const observer = new ResizeObserver(() => {
          const view = graphViews.get('npuGraph');
          if (view?.userAdjusted) { clampGraphView('npuGraph'); applyGraphView('npuGraph'); }
          else fitGraph('npuGraph');
        });
        observer.observe(document.getElementById('npuGraph'));
      }
    }
    function uniqueNodeIds(values) {
      return [...new Set((values || []).map(Number).filter(Number.isFinite))].sort((a, b) => a - b);
    }
    function intervalNodes(instance) {
      return uniqueNodeIds([...(instance.producer_nodes || []), ...(instance.consumer_nodes || [])]);
    }
    function fallbackQueueClass(queue, graph) {
      const classes = new Set();
      for (const nodeId of queue.uses || []) {
        const node = (graph.nodes || []).find(item => Number(item.id) === Number(nodeId));
        const queueClass = node?.queue_class || node?.load_class;
        if (queueClass) classes.add(queueClass);
      }
      return ['shared', 'reused', 'single'].find(item => classes.has(item)) || 'single';
    }
    function queueTimelineRows(graph) {
      const rows = [];
      for (const queue of graph.queues || []) {
        const instances = queue.instances?.length ? queue.instances : [{
          reuse_id: -1, depth: -1, buf_num: -1, queue_class: fallbackQueueClass(queue, graph),
          producer_nodes: queue.uses || [], consumer_nodes: [],
        }];
        instances.forEach((instance, index) => {
          const producers = uniqueNodeIds(instance.producer_nodes || queue.uses || []);
          const consumers = uniqueNodeIds(instance.consumer_nodes || []);
          const nodes = uniqueNodeIds([...producers, ...consumers]);
          const status = instance.queue_class || fallbackQueueClass(queue, graph);
          const reuse = Number(instance.reuse_id);
          rows.push({
            key: `queue:${queue.id}:${Number.isFinite(reuse) ? reuse : index}`,
            kind: 'queue', resourceId: queue.id, status, nodes, producers,
            title: `Queue ${queue.id}`,
            detail: `Reuse ${reuse >= 0 ? reuse : '-'} · Depth ${instance.depth >= 0 ? instance.depth : '-'} · Buffer count ${instance.buf_num >= 0 ? instance.buf_num : '-'}`,
            segments: [{...instance, producer_nodes: producers, consumer_nodes: consumers, nodes, status}],
            color: resourceColor('queue', queue.id),
          });
        });
      }
      return rows;
    }
    function bufferTimelineRows(graph) {
      return (graph.buffers || []).map(buffer => {
        const instances = buffer.instances?.length ? buffer.instances : (buffer.uses || []).map(nodeId => ({
          kind: 'observed', source_node: nodeId, producer_nodes: [nodeId], consumer_nodes: [], start: nodeId, end: nodeId,
        }));
        const segments = instances.map(instance => {
          const producers = uniqueNodeIds(instance.producer_nodes || [instance.source_node]);
          const consumers = uniqueNodeIds(instance.consumer_nodes || []);
          return {...instance, producer_nodes: producers, consumer_nodes: consumers, nodes: uniqueNodeIds([...producers, ...consumers]), status: 'buffer'};
        });
        const nodes = uniqueNodeIds(segments.flatMap(segment => segment.nodes));
        return {
          key: `buffer:${buffer.id}`, kind: 'buffer', resourceId: buffer.id, status: 'buffer', nodes,
          producers: uniqueNodeIds(segments.flatMap(segment => segment.producer_nodes)),
          title: `Buffer ${buffer.id}`,
          detail: `${segments.length} interval${segments.length === 1 ? '' : 's'} · ${nodes.length} related nodes`,
          segments, color: resourceColor('buffer', buffer.id),
        };
      });
    }
    function lifecycleRows(graph) {
      const queues = queueTimelineRows(graph), buffers = bufferTimelineRows(graph);
      let rows = lifecycleState.kind === 'queue' ? queues : lifecycleState.kind === 'buffer' ? buffers : [...queues, ...buffers];
      if (lifecycleState.filter === 'current') {
        const current = Number(lifecycleState.currentNode);
        rows = Number.isFinite(current) ? rows.filter(row => row.nodes.includes(current)) : [];
      } else if (lifecycleState.filter !== 'all') {
        rows = rows.filter(row => row.status === lifecycleState.filter);
      }
      return rows;
    }
    function lifecycleStatusText(status) {
      return {shared: 'Shared', reused: 'Reused', single: 'Single', buffer: 'Buffer'}[status] || status;
    }
    function lifecycleTooltip(row, segment) {
      const fields = [row.title];
      if (row.kind === 'queue') fields.push(row.detail, lifecycleStatusText(row.status));
      else fields.push(segment.kind === 'tmp_buf' ? 'Temporary buffer' : 'Buffer output');
      fields.push(`Producer nodes: ${(segment.producer_nodes || []).join(', ') || '-'}`);
      fields.push(`Direct consumer nodes: ${(segment.consumer_nodes || []).join(', ') || '-'}`);
      const nodes = intervalNodes(segment);
      if (nodes.length) fields.push(`Static interval: node ${nodes[0]} to node ${nodes[nodes.length - 1]}`);
      return fields.join('\\n');
    }
    function lifecycleSummary(graph) {
      const queues = graph.queues || [], buffers = graph.buffers || [];
      const queueRows = queueTimelineRows(graph);
      const shared = queueRows.filter(row => row.status === 'shared').length;
      const reused = queueRows.filter(row => row.status === 'reused').length;
      document.getElementById('lifecycleSummaryStats').textContent = `${queues.length} Queue · ${buffers.length} Buffer`;
      document.getElementById('lifecycleStats').textContent = `${queues.length} physical Queue · ${queueRows.length} logical instances · ${shared} shared · ${reused} reused · ${buffers.length} Buffer`;
    }
    function renderLifecycleTimeline() {
      const graph = lifecycleState.graph;
      const target = document.getElementById('lifecycleTimeline');
      if (!graph || !target) return;
      const previousLeft = target.scrollLeft, previousTop = target.scrollTop;
      const nodes = (graph.nodes || []).slice().sort((a, b) => Number(a.id) - Number(b.id));
      const nodeIds = nodes.map(node => Number(node.id));
      const positions = new Map(nodeIds.map((id, index) => [id, index]));
      const rows = lifecycleRows(graph);
      lifecycleState.rows = rows;
      lifecycleSummary(graph);
      for (const button of document.querySelectorAll('[data-lifecycle-kind]')) button.classList.toggle('active', button.dataset.lifecycleKind === lifecycleState.kind);
      document.getElementById('lifecycleFilter').value = lifecycleState.filter;
      document.getElementById('lifecycleZoomLevel').textContent = `${Math.round(lifecycleState.scale * 100)}%`;
      if (!nodes.length || !rows.length) {
        target.innerHTML = `<div class="graph-empty">${lifecycleState.filter === 'current' && !lifecycleState.currentNode ? '请先选择上方拓扑节点' : '当前筛选没有可展示的资源区间'}</div>`;
        return;
      }
      const cell = Math.max(54, Math.min(150, Math.round(82 * lifecycleState.scale)));
      const labelWidth = 300;
      const rowHeight = rows.length > 10 ? 39 : 54;
      lifecycleState.rowHeight = rowHeight;
      const parts = [`<div class="lifecycle-grid" style="--cell:${cell}px;--row-height:${rowHeight}px;grid-template-columns:${labelWidth}px repeat(${nodes.length},${cell}px);grid-template-rows:48px repeat(${rows.length},${rowHeight}px)">`];
      parts.push('<div class="lifecycle-corner" style="grid-column:1;grid-row:1"><strong>Resource instance</strong><span>topological node order →</span></div>');
      nodes.forEach((node, index) => {
        const cached = hasBroadcastCache(node);
        const cacheTitle = cached ? ` · Broadcast cache: ${node.exec_condition}` : '';
        parts.push(`<div class="lifecycle-node-head${cached ? ' broadcast-cache' : ''}" data-lifecycle-node="${node.id}" title="Node ${node.id} · ${escapeHtml(nodeDisplayName(node))} · ${escapeHtml(node.type || '')}${escapeHtml(cacheTitle)}" style="grid-column:${index + 2};grid-row:1"><span>#${node.id}</span><strong>${escapeHtml(shortText(node.type || nodeDisplayName(node), 11))}</strong>${cached ? '<span class="lifecycle-cache-mini">BC</span>' : ''}</div>`);
      });
      rows.forEach((row, rowIndex) => {
        const gridRow = rowIndex + 2;
        const style = `--resource-fill:${row.color.fill};--resource-stroke:${row.color.stroke}`;
        const nodesText = row.nodes.join(',');
        parts.push(`<div class="lifecycle-row-label" data-lifecycle-key="${escapeHtml(row.key)}" data-lifecycle-row="${rowIndex}" data-lifecycle-nodes="${nodesText}" style="grid-column:1;grid-row:${gridRow};${style}" title="${escapeHtml(row.title + '\\n' + row.detail)}"><span class="lifecycle-resource-swatch"></span><span class="lifecycle-resource-copy"><strong>${escapeHtml(row.title)}</strong><small>${escapeHtml(row.detail)}</small></span><span class="lifecycle-status ${row.status}">${escapeHtml(lifecycleStatusText(row.status))}</span></div>`);
        parts.push(`<div class="lifecycle-lane" style="grid-column:2 / -1;grid-row:${gridRow};${style}">`);
        row.segments.forEach((segment, segmentIndex) => {
          const related = intervalNodes(segment).filter(id => positions.has(id));
          if (!related.length) return;
          const startIndex = Math.min(...related.map(id => positions.get(id)));
          const endIndex = Math.max(...related.map(id => positions.get(id)));
          const left = startIndex * cell + 10;
          const width = Math.max(18, (endIndex - startIndex + 1) * cell - 20);
          parts.push(`<div class="lifecycle-segment ${row.status}" data-lifecycle-key="${escapeHtml(row.key)}" data-lifecycle-row="${rowIndex}" data-lifecycle-segment="${segmentIndex}" data-lifecycle-nodes="${related.join(',')}" title="${escapeHtml(lifecycleTooltip(row, segment))}" style="left:${left}px;width:${width}px;${style}"></div>`);
          for (const producer of segment.producer_nodes || []) {
            if (positions.has(producer)) parts.push(`<span class="lifecycle-marker producer" style="left:${positions.get(producer) * cell + cell / 2}px;${style}"></span>`);
          }
          for (const consumer of segment.consumer_nodes || []) {
            if (positions.has(consumer)) parts.push(`<span class="lifecycle-marker consumer" style="left:${positions.get(consumer) * cell + cell / 2}px;${style}"></span>`);
          }
        });
        parts.push('</div>');
      });
      parts.push('</div>');
      target.innerHTML = parts.join('');
      target.scrollLeft = previousLeft;
      target.scrollTop = previousTop;
      for (const header of target.querySelectorAll('[data-lifecycle-node]')) header.addEventListener('click', () => selectNode(String(header.dataset.lifecycleNode)));
      for (const item of target.querySelectorAll('[data-lifecycle-key]')) {
        item.addEventListener('click', event => { event.stopPropagation(); selectLifecycleRow(item.dataset.lifecycleKey); });
        item.addEventListener('pointerenter', () => setLifecycleHover(item.dataset.lifecycleNodes));
        item.addEventListener('pointerleave', () => setLifecycleHover(''));
      }
      updateLifecycleSelection(false);
    }
    function renderAutofuseAnalytics(graph, reset = false) {
      lifecycleState.graph = graph;
      if (reset) {
        lifecycleState.kind = 'queue';
        lifecycleState.filter = 'all';
        lifecycleState.scale = 1;
        lifecycleState.selectedKey = null;
        lifecycleState.currentNode = null;
      }
      renderLifecycleTimeline();
    }
    function setLifecycleHover(rawNodes) {
      const active = new Set(String(rawNodes || '').split(',').filter(Boolean));
      const hasActive = graphContext.kind === 'autofuse' && active.size > 0;
      const target = document.getElementById('npuGraph');
      for (const node of target.querySelectorAll('.graph-node')) {
        const related = active.has(node.dataset.node);
        node.classList.toggle('lifecycle-related', hasActive && related);
        node.classList.toggle('lifecycle-dimmed', hasActive && !related);
      }
      for (const edge of target.querySelectorAll('.graph-edge')) {
        const related = active.has(edge.dataset.src) && active.has(edge.dataset.dst);
        edge.classList.toggle('lifecycle-related', hasActive && related);
        edge.classList.toggle('lifecycle-dimmed', hasActive && !related);
      }
    }
    function centerGraphNode(name) {
      const target = document.getElementById('npuGraph');
      const view = graphViews.get('npuGraph');
      const position = view?.layout.positions.get(String(name));
      if (!target || !view || !position) return;
      view.x = target.clientWidth / 2 - (position.x + position.w / 2) * view.scale;
      view.y = target.clientHeight / 2 - (position.y + position.h / 2) * view.scale;
      view.userAdjusted = true;
      clampGraphView('npuGraph');
      applyGraphView('npuGraph');
    }
    function updateLifecycleSelection(scroll = false) {
      const target = document.getElementById('lifecycleTimeline');
      if (!target) return;
      const current = lifecycleState.currentNode == null ? '' : String(lifecycleState.currentNode);
      const hasSelection = !!current || !!lifecycleState.selectedKey;
      let firstRelated = null, selectedResource = null;
      for (const item of target.querySelectorAll('[data-lifecycle-key]')) {
        const nodes = String(item.dataset.lifecycleNodes || '').split(',');
        const related = (!!current && nodes.includes(current)) || item.dataset.lifecycleKey === lifecycleState.selectedKey;
        item.classList.toggle('selected', item.dataset.lifecycleKey === lifecycleState.selectedKey);
        item.classList.toggle('dimmed', hasSelection && !related);
        if (related && !firstRelated) firstRelated = item;
        if (item.dataset.lifecycleKey === lifecycleState.selectedKey && !selectedResource) selectedResource = item;
      }
      let selectedHeader = null;
      for (const header of target.querySelectorAll('[data-lifecycle-node]')) {
        const selected = header.dataset.lifecycleNode === current;
        header.classList.toggle('selected', selected);
        if (selected) selectedHeader = header;
      }
      if (scroll && selectedHeader) {
        const labelWidth = 300;
        const scrollTarget = selectedResource || firstRelated;
        target.scrollLeft = Math.max(0, selectedHeader.offsetLeft - labelWidth - (target.clientWidth - labelWidth) / 2);
        if (scrollTarget) target.scrollTop = Math.max(0, Number(scrollTarget.dataset.lifecycleRow || 0) * lifecycleState.rowHeight);
      }
    }
    function selectLifecycleRow(key) {
      if (graphContext.kind !== 'autofuse') return;
      const row = lifecycleState.rows.find(item => item.key === key);
      if (!row) return;
      const names = row.nodes.map(String).filter(name => graphContext.nodes.has(name));
      const primary = row.producers.map(String).find(name => graphContext.nodes.has(name)) || names[0];
      if (!primary) return;
      lifecycleState.selectedKey = key;
      lifecycleState.currentNode = primary;
      selection.name = primary;
      selection.mode = 'resource';
      selection.explicit = new Set(names);
      selection.detailNode = graphContext.nodes.get(primary);
      detailsPanel.hidden = false;
      detailsPanel.setAttribute('aria-hidden', 'false');
      renderDetails();
      updateSelectionView();
      updateLifecycleSelection(true);
      centerGraphNode(primary);
    }
    function setLifecycleZoom(scale) {
      lifecycleState.scale = Math.max(0.65, Math.min(1.8, scale));
      renderLifecycleTimeline();
      updateLifecycleSelection(true);
    }
    function fitLifecycleTimeline() {
      const target = document.getElementById('lifecycleTimeline');
      const count = lifecycleState.graph?.nodes?.length || 0;
      if (!target || !count) return;
      setLifecycleZoom((target.clientWidth - 300) / (count * 82));
    }
    function lifecycleHeightBounds() {
      const panel = document.getElementById('autofuseAnalytics');
      const shell = panel.closest('.graph-shell');
      const tabs = document.getElementById('graphTabs');
      const head = shell.querySelector('.graph-head');
      const summary = panel.querySelector('summary');
      const min = 220;
      const reserved = (tabs?.offsetHeight || 0) + (head?.offsetHeight || 0) + (summary?.offsetHeight || 0) + 6;
      return {min, max: Math.max(min, shell.clientHeight - reserved)};
    }
    function setLifecycleHeight(value) {
      const panel = document.getElementById('autofuseAnalytics');
      const handle = document.getElementById('lifecycleResizeHandle');
      const bounds = lifecycleHeightBounds();
      const height = Math.round(Math.max(bounds.min, Math.min(bounds.max, Number(value) || bounds.min)));
      lifecycleState.height = height;
      panel.style.setProperty('--lifecycle-height', `${height}px`);
      handle.setAttribute('aria-valuemin', String(bounds.min));
      handle.setAttribute('aria-valuemax', String(Math.round(bounds.max)));
      handle.setAttribute('aria-valuenow', String(height));
      requestAnimationFrame(() => updateLifecycleSelection(false));
    }
    function expandLifecycleHeight() {
      const rows = lifecycleState.rows.length;
      const rowHeight = rows > 10 ? 39 : 54;
      setLifecycleHeight(48 + rows * rowHeight + 76);
    }
    function initLifecycleResize() {
      const handle = document.getElementById('lifecycleResizeHandle');
      const panel = document.getElementById('autofuseAnalytics');
      let pointerId = null, startY = 0, startHeight = 0;
      handle.addEventListener('pointerdown', event => {
        if (event.button !== 0 || !panel.open) return;
        pointerId = event.pointerId;
        startY = event.clientY;
        startHeight = panel.querySelector('.autofuse-analytics-content').getBoundingClientRect().height;
        handle.setPointerCapture(pointerId);
        document.body.classList.add('lifecycle-resizing');
        event.preventDefault();
        event.stopPropagation();
      });
      handle.addEventListener('pointermove', event => {
        if (event.pointerId !== pointerId) return;
        setLifecycleHeight(startHeight + startY - event.clientY);
      });
      function finishResize(event) {
        if (event.pointerId !== pointerId) return;
        if (handle.hasPointerCapture(pointerId)) handle.releasePointerCapture(pointerId);
        pointerId = null;
        document.body.classList.remove('lifecycle-resizing');
      }
      handle.addEventListener('pointerup', finishResize);
      handle.addEventListener('pointercancel', finishResize);
      handle.addEventListener('dblclick', event => { event.preventDefault(); event.stopPropagation(); expandLifecycleHeight(); });
      handle.addEventListener('keydown', event => {
        const current = panel.querySelector('.autofuse-analytics-content').getBoundingClientRect().height;
        if (event.key === 'ArrowUp') setLifecycleHeight(current + 40);
        else if (event.key === 'ArrowDown') setLifecycleHeight(current - 40);
        else if (event.key === 'Home') setLifecycleHeight(lifecycleHeightBounds().min);
        else if (event.key === 'End') expandLifecycleHeight();
        else return;
        event.preventDefault();
        event.stopPropagation();
      });
    }
    function updateGraphTabs(scrollLinked = false) {
      const linkedKernel = nodeKernelName(selection.detailNode);
      const activeKernel = graphContext.kind === 'autofuse' ? graphKernelName(autofuseGraphs[graphContext.autoIndex]) : '';
      for (const button of document.querySelectorAll('[data-graph-tab]')) {
        const tab = button.dataset.graphTab;
        const active = tab === 'main'
          ? graphContext.kind === 'main' || graphContext.kind === 'fusion'
          : graphContext.kind === 'autofuse' && Number(tab) === graphContext.autoIndex;
        button.classList.toggle('active', active);
        button.setAttribute('aria-selected', active ? 'true' : 'false');
      }
      let linkedGroup = null;
      for (const group of document.querySelectorAll('.graph-tab-group')) {
        const linked = !!linkedKernel && group.dataset.kernel === linkedKernel;
        const active = !!activeKernel && group.dataset.kernel === activeKernel;
        group.classList.toggle('linked', linked);
        group.classList.toggle('active', active);
        if (linked) linkedGroup = group;
      }
      if (scrollLinked && linkedGroup) {
        const tabs = document.getElementById('graphTabs');
        tabs.scrollTo({left: Math.max(0, linkedGroup.offsetLeft - tabs.clientWidth / 3), behavior: 'smooth'});
      }
    }
    function renderGraphTabs() {
      const tabs = document.getElementById('graphTabs');
      for (const [kernel, indices] of autofuseGroups) {
        const group = document.createElement('div');
        group.className = 'graph-tab-group';
        group.dataset.kernel = kernel;
        const kernelButton = document.createElement('button');
        kernelButton.className = 'graph-kernel-tab';
        kernelButton.type = 'button';
        kernelButton.dataset.kernelFirst = String(indices[0]);
        kernelButton.textContent = `${kernelDisplayName(kernel === '__unlinked__' ? '' : kernel)} (${indices.length})`;
        kernelButton.title = kernel === '__unlinked__' ? '无法从 dump 路径关联到融合 kernel' : kernel;
        group.appendChild(kernelButton);
        indices.forEach((index, variantIndex) => {
          const graph = autofuseGraphs[index];
          const button = document.createElement('button');
          button.className = 'graph-variant-tab';
          button.type = 'button';
          button.role = 'tab';
          button.dataset.graphTab = String(index);
          button.textContent = `Optimize ${variantIndex + 1}`;
          button.title = `${kernel === '__unlinked__' ? '未关联 kernel' : kernel}\n${graph.name || index}`;
          group.appendChild(button);
        });
        tabs.appendChild(group);
      }
      for (const button of tabs.querySelectorAll('[data-kernel-first]')) {
        button.addEventListener('click', () => showAutofuseGraph(Number(button.dataset.kernelFirst)));
      }
      for (const button of tabs.querySelectorAll('[data-graph-tab]')) {
        button.addEventListener('click', () => {
          if (button.dataset.graphTab === 'main') showMainGraph();
          else showAutofuseGraph(Number(button.dataset.graphTab));
        });
      }
    }
    function renderGraph() {
      const target = document.getElementById('npuGraph');
      const title = document.getElementById('graphTitle');
      const meta = document.getElementById('npuGraphMeta');
      const back = document.getElementById('graphBack');
      const fusionDetails = document.getElementById('fusionDetails');
      const analytics = document.getElementById('autofuseAnalytics');
      const loadLegend = document.getElementById('loadLegend');
      const graph = graphContext.graph;
      const layout = layoutGraph(graph);
      if (graphContext.kind === 'fusion') {
        const inserted = layout.nodes.filter(node => node.origin === false).length;
        title.textContent = `融合子图 · ${graphContext.parent.name}`;
        meta.textContent = `${layout.nodes.length} ops / ${layout.edges.length} edges · ${graph.source || 'unknown source'}${inserted ? ` · ${inserted} compiler inserted` : ''}`;
        back.hidden = false;
        fusionDetails.hidden = false;
        analytics.hidden = true;
        analytics.open = false;
        loadLegend.hidden = false;
        loadLegend.setAttribute('aria-label', '融合子图节点分类');
        loadLegend.innerHTML = '<span class="internal">融合来源</span><span class="inserted">编译器插入</span>';
      } else if (graphContext.kind === 'autofuse') {
        const kernel = graphKernelName(graph);
        title.textContent = `AutoFuseAfterOptimize · ${kernelDisplayName(kernel)}`;
        meta.textContent = `${layout.nodes.length} nodes / ${layout.edges.length} edges · ${graph.name || graphContext.autoIndex}`;
        back.hidden = true;
        fusionDetails.hidden = true;
        analytics.hidden = false;
        analytics.open = false;
        loadLegend.hidden = false;
        loadLegend.setAttribute('aria-label', 'Queue 节点分类');
        loadLegend.innerHTML = '<span class="resource">Queue 同一 que_id 同色；Buffer 同一 buf_id 同色</span><span class="shared">共用</span><span class="reused">复用</span><span class="single">单独</span><span class="cache">Broadcast 条件缓存</span>';
        renderAutofuseAnalytics(graph, true);
      } else {
        title.textContent = 'NPU Autofuse Dump Graph';
        meta.textContent = `${layout.nodes.length} ops / ${layout.edges.length} edges · ${npuGraph.model?.topology || 'TorchInductor Autofuse'}`;
        back.hidden = true;
        fusionDetails.hidden = true;
        analytics.hidden = true;
        analytics.open = false;
        loadLegend.hidden = false;
        loadLegend.setAttribute('aria-label', '精度结果分类');
        loadLegend.innerHTML = '<span class="pass">双标准通过</span><span class="warning">仅 Strict 失败</span><span class="fail">CANN 失败</span><span class="unmapped">未映射 / 未采集</span><span class="internal">输入 / 权重</span><span class="unfused">未融合原因</span>';
      }
      updateGraphTabs();
      if (!layout.nodes.length) { target.innerHTML = '<div class="graph-empty">当前图没有可展示的算子拓扑</div>'; return; }
      graphViews.set('npuGraph', {layout, scale: 1, fitScale: 1, minScale: 0.005, maxScale: 4, x: 0, y: 0, userAdjusted: false, dragged: false});
      const parts = [`<svg class="graph-svg" width="100%" height="100%" viewBox="0 0 ${layout.width} ${layout.height}" xmlns="http://www.w3.org/2000/svg">`];
      parts.push(`<defs><marker id="npuGraph-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#9aa5b1"/></marker></defs>`);
      for (const edge of layout.edges) {
        const source = layout.positions.get(edge.src), destination = layout.positions.get(edge.dst);
        if (!source || !destination) continue;
        const x1 = source.x + source.w / 2, y1 = source.y + source.h;
        const x2 = destination.x + destination.w / 2, y2 = destination.y;
        const mid = Math.max(y1 + 24, (y1 + y2) / 2);
        const skipsLevel = destination.y - source.y > source.h + 100;
        const route = skipsLevel && Math.abs(x1 - x2) < source.w
          ? `M${x1},${y1} C${x1 + source.w / 2 + 86},${y1 + 24} ${x2 + destination.w / 2 + 86},${y2 - 24} ${x2},${y2}`
          : `M${x1},${y1} C${x1},${mid} ${x2},${mid} ${x2},${y2}`;
        parts.push(`<path class="graph-edge" data-src="${escapeHtml(edge.src)}" data-dst="${escapeHtml(edge.dst)}" d="${route}" fill="none" stroke="#9aa5b1" stroke-width="1.3" marker-end="url(#npuGraph-arrow)"/>`);
      }
      for (const node of layout.nodes) {
        const position = layout.positions.get(node.name), state = nodeState(node), metric = state.metric;
        const broadcastCached = graphContext.kind === 'autofuse' && hasBroadcastCache(node);
        const nodeUnfusedReasons = graphContext.kind === 'main' ? unfusedReasons(node) : [];
        const detail = graphContext.kind === 'autofuse'
          ? `node id=${node.id}`
          : graphContext.kind === 'fusion'
          ? (node.origin === false ? 'compiler inserted' : 'fusion origin')
          : (metric?.compared
            ? `Strict=${metric.strictPassed ? 'PASS' : 'FAIL'} CANN=${metric.cannPassed ? 'PASS' : 'FAIL'}`
            : 'no compare result');
        const lineLength = graphContext.kind === 'autofuse' ? 56 : 44;
        const inputLine = shortText('in: ' + graphInputText(node), lineLength);
        const outputLine = shortText('out: ' + graphOutputText(node), lineLength);
        const fusionLine = graphContext.kind === 'main' && isFusionNode(node) ? `FUSED · ${node.origin_ops.length} ops` : '';
        const unfusedTooltip = nodeUnfusedReasons.map(unfusedReasonText).join('\\n');
        const tooltipDetail = [detail, broadcastCached ? `Broadcast cache: ${node.exec_condition}` : '', unfusedTooltip].filter(Boolean).join('\\n');
        parts.push(`<g class="graph-node" data-node="${escapeHtml(node.name)}" tabindex="0"><title>${escapeHtml(nodeDisplayName(node))}\n${escapeHtml(node.type || '')}\n${escapeHtml(tooltipDetail)}</title>`);
        parts.push(`<rect x="${position.x}" y="${position.y}" width="${position.w}" height="${position.h}" rx="8" fill="${state.fill}" stroke="${state.stroke}" stroke-width="${state.strokeWidth || 2}"${state.dashed ? ' stroke-dasharray="7 5"' : ''}/>`);
        parts.push(`<text x="${position.x + 10}" y="${position.y + 20}" font-size="12" font-weight="700" fill="#16202a">${escapeHtml(shortText(nodeDisplayName(node), 36))}</text>`);
        parts.push(`<text x="${position.x + 10}" y="${position.y + 39}" font-size="11" fill="#5f6b7a">${escapeHtml(shortText(node.type || '-', 36))}</text>`);
        parts.push(`<text x="${position.x + 10}" y="${position.y + 59}" font-size="10.5" fill="#3d4856">${escapeHtml(inputLine)}</text>`);
        parts.push(`<text x="${position.x + 10}" y="${position.y + 78}" font-size="10.5" fill="#3d4856">${escapeHtml(outputLine)}</text>`);
        parts.push(`<text x="${position.x + 10}" y="${position.y + 99}" font-size="11" fill="${state.stroke}">${escapeHtml(state.label)} · ${escapeHtml(detail)}</text>`);
        if (broadcastCached) {
          parts.push(`<circle class="broadcast-cache-mark" cx="${position.x + position.w - 17}" cy="${position.y + 17}" r="11"><title>Broadcast 条件缓存 · ${escapeHtml(node.exec_condition)}</title></circle>`);
          parts.push(`<text class="broadcast-cache-mark-text" x="${position.x + position.w - 17}" y="${position.y + 20}">BC</text>`);
          parts.push(`<text class="broadcast-cache-label" x="${position.x + 10}" y="${position.y + 119}">BROADCAST CACHE</text>`);
        }
        if (nodeUnfusedReasons.length) {
          const firstReason = nodeUnfusedReasons[0]?.reason || 'unknown';
          parts.push(`<circle class="unfused-mark" cx="${position.x + position.w - 17}" cy="${position.y + 17}" r="11"><title>${escapeHtml(unfusedTooltip)}</title></circle>`);
          parts.push(`<text class="unfused-mark-text" x="${position.x + position.w - 17}" y="${position.y + 20}">NF</text>`);
          parts.push(`<text class="unfused-label" x="${position.x + 10}" y="${position.y + 119}">${escapeHtml(shortText('NOT FUSED · ' + firstReason, 43))}</text>`);
        }
        if (fusionLine) parts.push(`<text x="${position.x + 10}" y="${position.y + 119}" font-size="10.5" fill="#2457a6">${escapeHtml(fusionLine)}</text><circle class="fusion-mark" data-fusion-entry="true" tabindex="0" role="button" aria-label="进入融合子图" cx="${position.x + position.w - 15}" cy="${position.y + 15}" r="8"><title>点击蓝色入口进入融合子图</title></circle>`);
        parts.push('</g>');
      }
      parts.push('</svg>');
      target.innerHTML = parts.join('');
      fitGraph('npuGraph');
      updateSelectionView();
    }
    function value(row, key) {
      const value = row[key];
      if (Array.isArray(value)) return value.join(',');
      return value === undefined || value === null ? '' : value;
    }
    function numeric(row, key) {
      if (row[key] === undefined || row[key] === null || row[key] === '') return -Infinity;
      const number = Number(row[key]);
      return Number.isFinite(number) ? number : -Infinity;
    }
    function rowText(row) {
      return [row.cpu_op, row.npu_op, row.npu_type, row.status, row.dtype, row.strict_passed, row.cann_passed, Array.isArray(row.shape) ? row.shape.join('x') : row.shape].join(' ').toLowerCase();
    }
    function statusClass(status) {
      if (!status) return 'status-bad';
      if (status === 'OK') return 'status-ok';
      if (status.startsWith('OK_') || status === 'UNMATCHED_CPU_OP') return 'status-warn';
      return 'status-bad';
    }
    function applyFilters() {
      const query = search.value.trim().toLowerCase(), pass = passFilter.value, type = typeFilter.value;
      state.filtered = rows.filter(row => {
        const strictPassed = row.strict_passed ?? row.passed;
        const cannPassed = row.cann_passed ?? row.passed;
        if (pass === 'both_passed' && (row.compared === false || !strictPassed || !cannPassed)) return false;
        if (pass === 'failed' && (row.compared === false || strictPassed)) return false;
        if (pass === 'cann_failed' && (row.compared === false || cannPassed)) return false;
        if (pass === 'disagreement' && (row.compared === false || strictPassed === cannPassed)) return false;
        if (pass === 'unmatched' && row.compared !== false) return false;
        if (pass === 'unmapped' && row.status !== 'UNMATCHED_CPU_OP') return false;
        if (pass === 'not_captured' && row.status !== 'NPU_OUTPUT_NOT_CAPTURED') return false;
        if (type !== 'all' && row.npu_type !== type) return false;
        return !query || rowText(row).includes(query);
      });
      sortRows();
      renderTable();
    }
    function sortRows() {
      const key = state.sortKey, dir = state.sortDir;
      state.filtered.sort((a, b) => {
        const an = numeric(a, key), bn = numeric(b, key);
        if (an !== -Infinity || bn !== -Infinity) return (an - bn) * dir;
        return String(value(a, key)).localeCompare(String(value(b, key))) * dir;
      });
    }
    function renderTable() {
      tbody.textContent = '';
      document.getElementById('tableSummary').textContent = `(${state.filtered.length}/${rows.length})`;
      empty.hidden = state.filtered.length !== 0;
      for (const row of state.filtered) {
        const tr = document.createElement('tr');
        const shape = Array.isArray(row.shape) ? row.shape.join('x') : (row.shape || '-');
        const strictPassed = row.strict_passed ?? row.passed;
        const cannPassed = row.cann_passed ?? row.passed;
        const badgeClass = row.compared === false ? 'warn' : (!cannPassed ? 'fail' : (!strictPassed ? 'warn' : 'pass'));
        const badgeText = row.compared === false
          ? (row.status === 'NPU_OUTPUT_NOT_CAPTURED' ? 'NO DUMP' : 'NO MAP')
          : (!cannPassed ? 'FAIL' : (!strictPassed ? 'STRICT FAIL' : 'PASS'));
        const strictText = row.compared === false ? '-' : (strictPassed ? 'PASS' : 'FAIL');
        const cannText = row.compared === false ? '-' : (cannPassed ? 'PASS' : 'FAIL');
        tr.innerHTML = `<td><span class="badge ${badgeClass}">${badgeText}</span></td><td>${strictText}</td><td>${cannText}</td><td class="${statusClass(row.status)}">${escapeHtml(row.status || '-')}</td><td class="op">${escapeHtml(row.npu_op || '-')}:${row.npu_index ?? '-'}</td><td>${escapeHtml(row.npu_type || '-')}</td><td class="op">${escapeHtml(row.cpu_op || '-')}:${row.cpu_index ?? '-'}</td><td>${escapeHtml(shape)}</td><td>${escapeHtml(row.dtype || '-')}</td><td>${fmt(row.rtol)} / ${fmt(row.atol)}</td><td>${finiteMetricText(row, 'max_abs_error')}</td><td>${finiteMetricText(row, 'max_rel_error')}</td><td>${finiteMetricText(row, 'rmse')}</td><td>${escapeHtml(cannMetricText(row, 'cann_cosine_similarity'))}</td><td>${escapeHtml(mismatchText(row))}</td><td>${escapeHtml(maxIndexText(row))}</td>`;
        tr.addEventListener('click', () => {
          if (graphContext.kind !== 'main') showMainGraph();
          selectNode(row.npu_op);
        });
        tbody.appendChild(tr);
      }
      drawChart();
    }
    function drawChart() {
      const canvas = document.getElementById('errorChart'), ctx = canvas.getContext('2d');
      const ratio = window.devicePixelRatio || 1, width = canvas.clientWidth || 800, height = canvas.clientHeight || 120;
      canvas.width = Math.floor(width * ratio); canvas.height = Math.floor(height * ratio); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
      const data = state.filtered.filter(row => row.compared !== false).slice(0, 80), pad = {l: 42, r: 14, t: 12, b: 24};
      const w = width - pad.l - pad.r, h = height - pad.t - pad.b;
      ctx.strokeStyle = '#d7dee8'; ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + h); ctx.lineTo(pad.l + w, pad.t + h); ctx.stroke();
      ctx.fillStyle = '#617083'; ctx.fillText('max_abs', 6, 17);
      if (!data.length) return;
      const max = Math.max(...data.map(row => Number(row.max_abs_error) || 0), 1e-12), bar = Math.max(2, w / data.length * 0.7);
      data.forEach((row, index) => { const value = Number(row.max_abs_error) || 0; const x = pad.l + index * (w / data.length) + 1; const bh = Math.max(1, h * value / max); const strictPassed = row.strict_passed ?? row.passed; const cannPassed = row.cann_passed ?? row.passed; ctx.fillStyle = !cannPassed ? '#d34a4a' : (!strictPassed ? '#b97800' : '#2f8f62'); ctx.fillRect(x, pad.t + h - bh, bar, bh); });
      ctx.fillStyle = '#617083'; ctx.fillText(`top ${data.length} by current sort/filter`, pad.l, height - 6);
    }
    function fillTypeFilter() {
      [...new Set(rows.map(row => row.npu_type || '').filter(Boolean))].sort().forEach(type => { const option = document.createElement('option'); option.value = type; option.textContent = type; typeFilter.appendChild(option); });
    }
    function downloadCsv() {
      const columns = ['compared','passed','strict_passed','cann_passed','status','cpu_op','cpu_index','npu_op','npu_index','npu_type','rtol','atol','tolerance_source','shape','dtype','max_abs_error','mean_abs_error','max_rel_error','rmse','cosine_similarity','cann_cosine_similarity','relative_euclidean_distance','kl_divergence','rmse_mae_ratio','cann_warning','mismatched','num_elements','finite_pair_count','max_error_index','nonfinite_summary','first_nonfinite_index','first_nonfinite_cpu','first_nonfinite_npu','cann_unavailable_reason'];
      const lines = [columns.join(',')];
      for (const row of state.filtered) lines.push(columns.map(key => { const raw = Array.isArray(row[key]) ? row[key].join('x') : (row[key] ?? ''); return '"' + String(raw).replace(/"/g, '""') + '"'; }).join(','));
      const blob = new Blob([lines.join('\\n') + '\\n'], {type: 'text/csv'}), link = document.createElement('a');
      link.href = URL.createObjectURL(blob); link.download = 'autofuse_accuracy_filtered.csv'; link.click(); URL.revokeObjectURL(link.href);
    }
    document.getElementById('metricsPanel').addEventListener('toggle', event => document.getElementById('toggleMetrics').setAttribute('aria-expanded', event.target.open ? 'true' : 'false'));
    document.getElementById('toggleMetrics').addEventListener('click', () => { const panel = document.getElementById('metricsPanel'); panel.open = !panel.open; });
    document.getElementById('dismissDetails').addEventListener('click', closeDetails);
    document.getElementById('linkedAutofuseButton').addEventListener('click', event => {
      const index = Number(event.currentTarget.dataset.graphIndex);
      if (Number.isInteger(index) && autofuseGraphs[index]) showAutofuseGraph(index);
    });
    document.getElementById('graphBack').addEventListener('click', showMainGraph);
    document.getElementById('fusionDetails').addEventListener('click', showFusionDetails);
    for (const button of document.querySelectorAll('[data-highlight-mode]')) button.addEventListener('click', () => { if (!selection.name) return; selection.explicit = null; selection.mode = button.dataset.highlightMode; updateSelectionView(); });
    for (const button of document.querySelectorAll('[data-lifecycle-kind]')) button.addEventListener('click', () => {
      lifecycleState.kind = button.dataset.lifecycleKind;
      if (lifecycleState.kind === 'buffer' && ['shared', 'reused', 'single'].includes(lifecycleState.filter)) lifecycleState.filter = 'all';
      lifecycleState.selectedKey = null;
      renderLifecycleTimeline();
    });
    document.getElementById('lifecycleFilter').addEventListener('change', event => {
      lifecycleState.filter = event.target.value;
      lifecycleState.selectedKey = null;
      renderLifecycleTimeline();
      updateLifecycleSelection(true);
    });
    for (const button of document.querySelectorAll('[data-lifecycle-zoom]')) button.addEventListener('click', () => {
      if (button.dataset.lifecycleZoom === 'in') setLifecycleZoom(lifecycleState.scale * 1.2);
      else if (button.dataset.lifecycleZoom === 'out') setLifecycleZoom(lifecycleState.scale / 1.2);
      else fitLifecycleTimeline();
    });
    document.getElementById('autofuseAnalytics').addEventListener('toggle', event => {
      event.target.closest('.graph-shell').classList.toggle('lifecycle-expanded', event.target.open);
      if (event.target.open) requestAnimationFrame(() => {
        if (lifecycleState.height == null) {
          const contentHeight = event.target.querySelector('.autofuse-analytics-content').getBoundingClientRect().height;
          setLifecycleHeight(contentHeight);
        }
        fitLifecycleTimeline();
        requestAnimationFrame(() => updateLifecycleSelection(true));
      });
    });
    search.addEventListener('input', applyFilters); passFilter.addEventListener('change', applyFilters); typeFilter.addEventListener('change', applyFilters);
    document.getElementById('exportCsv').addEventListener('click', downloadCsv);
    for (const th of document.querySelectorAll('th[data-key]')) th.addEventListener('click', () => { const key = th.dataset.key; if (state.sortKey === key) state.sortDir *= -1; else { state.sortKey = key; state.sortDir = ['max_abs_error','max_rel_error','rmse','mismatched'].includes(key) ? -1 : 1; } sortRows(); renderTable(); });
    renderGraphTabs(); renderGraph(); initGraphInteractions(); initZoomControls(); initLifecycleResize(); fillTypeFilter(); applyFilters();
    const initialParams = new URLSearchParams(window.location.search);
    const initialNode = initialParams.get('node');
    const initialAutofuseIndex = Number(initialParams.get('graph'));
    if (initialParams.get('view') === 'autofuse' && Number.isInteger(initialAutofuseIndex) && autofuseGraphs[initialAutofuseIndex]) {
      showAutofuseGraph(initialAutofuseIndex);
    } else if (initialNode && mainNpuNodes.has(initialNode)) {
      const initialMainNode = mainNpuNodes.get(initialNode);
      if (initialParams.get('view') === 'fusion' && isFusionNode(initialMainNode)) enterFusionGraph(initialMainNode);
      else selectNode(initialNode);
      const initialMode = initialParams.get('highlight');
      if (['depth1', 'depth2', 'all'].includes(initialMode)) selection.mode = initialMode;
      updateSelectionView();
    }
      if (graphContext.kind === 'autofuse') {
      const initialLifecycleKind = initialParams.get('lifecycleKind');
      const initialLifecycleFilter = initialParams.get('lifecycleFilter');
      if (['queue', 'buffer', 'all'].includes(initialLifecycleKind)) lifecycleState.kind = initialLifecycleKind;
      if (['all', 'shared', 'reused', 'single', 'current'].includes(initialLifecycleFilter)) lifecycleState.filter = initialLifecycleFilter;
      renderLifecycleTimeline();
      const initialLifecycleResource = initialParams.get('lifecycleResource');
      if (initialLifecycleResource && lifecycleState.rows.some(row => row.key === initialLifecycleResource)) selectLifecycleRow(initialLifecycleResource);
      else if (initialNode && graphContext.nodes.has(initialNode)) selectNode(initialNode);
      const initialLifecycleHeight = Number(initialParams.get('lifecycleHeight'));
      if (Number.isFinite(initialLifecycleHeight) && initialLifecycleHeight > 0) setLifecycleHeight(initialLifecycleHeight);
      if (initialParams.get('lifecycle') === 'open') document.getElementById('autofuseAnalytics').open = true;
      }
    window.addEventListener('resize', () => {
      drawChart();
      if (lifecycleState.height != null) setLifecycleHeight(lifecycleState.height);
    });
  </script>
</body>
</html>
"""
    replacements = {
        "__TITLE__": html.escape(title, quote=True),
        "__STATUS_CLASS__": status_class,
        "__STATUS_TEXT__": html.escape(status_text),
        "__PASSED__": str(passed),
        "__COMPARED__": str(compared),
        "__STRICT_PASSED__": str(strict_passed),
        "__CANN_PASSED__": str(cann_passed),
        "__COVERAGE__": f"{coverage * 100.0:.1f}",
        "__TOTAL__": str(total),
        "__PASS_RATE__": f"{pass_rate:.1f}",
        "__CANN_PASS_RATE__": f"{cann_pass_rate:.1f}",
        "__FAILED__": str(failed),
        "__STRICT_FAILED__": str(strict_failed),
        "__CANN_FAILED__": str(cann_failed),
        "__DISAGREEMENTS__": str(disagreements),
        "__UNMATCHED__": str(unmatched),
        "__UNMAPPED__": str(unmapped),
        "__NOT_CAPTURED__": str(not_captured),
        "__COVERAGE_CLASS__": "pass" if unmatched == 0 else "warn",
        "__FAILED_CLASS__": "fail" if failed else "pass",
        "__DISAGREEMENT_CLASS__": "warn" if disagreements else "pass",
        "__UNMATCHED_CLASS__": "warn" if unmatched else "pass",
        "__MAX_ABS__": _fmt(summary.get("max_abs_error")),
        "__MAX_REL__": _fmt(summary.get("max_rel_error")),
        "__REPORT_JSON__": payload,
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def write_report_html(report: dict[str, Any], path: str | Path, title: str = "Autofuse Accuracy Report") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report_html(report, title), encoding="utf-8")


def write_report_html_from_json(report_path: str | Path, html_path: str | Path, title: str = "Autofuse Accuracy Report") -> None:
    write_report_html(read_json(report_path), html_path, title)
