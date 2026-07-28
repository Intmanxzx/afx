from __future__ import annotations

import hashlib
import html
import json
import mmap
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from autofusedebugger.progress import progress_message, progress_stage


def _json_for_script(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


_DEFAULT_EMBED_TENSOR_BYTES = 64 * 1024
_DEFAULT_EMBED_TOTAL_BYTES = 8 * 1024 * 1024
_ENCODED_OBJECT_PATTERN = re.compile(rb'\{\s*"encoding"\s*:\s*"(base64|json)"')
_BASE64_OBJECT_END_PATTERN = re.compile(rb'"\s*\}')


def _env_size(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _encoded_payload_size(payload: dict[str, Any]) -> int:
    if payload.get("encoding") == "base64":
        return len(payload.get("data", ""))
    values = payload.get("values", [])
    return len(values) * 16 if isinstance(values, list) else 0


def _tensor_data_dir(target: Path) -> Path:
    return target.with_name(f"{target.stem}_data")


def _write_tensor_script(path: Path, payload_id: str, payload: dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("window.__autofuseRegisterTensor(")
        json.dump(payload_id, stream, ensure_ascii=False)
        stream.write(",")
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write(");\n")
    temporary.replace(path)
    return path.stat().st_size


class _TensorPayloadStore:
    def __init__(self, target: Path):
        self.data_dir = _tensor_data_dir(target)
        self.embed_tensor_bytes = _env_size("AUTOFUSEDEBUGGER_EMBED_TENSOR_BYTES", _DEFAULT_EMBED_TENSOR_BYTES)
        self.embed_total_bytes = _env_size("AUTOFUSEDEBUGGER_EMBED_TOTAL_BYTES", _DEFAULT_EMBED_TOTAL_BYTES)
        self.embedded_bytes = 0
        self.external_bytes = 0
        self.external_files = 0

    def store(self, payload_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload_size = _encoded_payload_size(payload)
        if (
            payload_size <= self.embed_tensor_bytes
            and self.embedded_bytes + payload_size <= self.embed_total_bytes
        ):
            self.embedded_bytes += payload_size
            return payload

        filename = f"{payload_id}.js"
        path = self.data_dir / filename
        self.external_bytes += _write_tensor_script(path, payload_id, payload)
        self.external_files += 1
        return {
            "external_script": quote(f"{self.data_dir.name}/{filename}"),
            "byte_size": payload_size,
            "dtype": payload.get("dtype"),
            "logical_dtype": payload.get("logical_dtype"),
            "shape": payload.get("shape"),
        }


def _compact_report_for_html(
    report: dict[str, Any], payload_store: _TensorPayloadStore | None = None
) -> dict[str, Any]:
    source = dict(report)
    comparisons = source.get("comparisons")
    if isinstance(comparisons, dict) and comparisons:
        source.pop("results", None)
        source["comparisons"] = {
            key: {field: value for field, value in comparison.items() if field != "graphs"}
            for key, comparison in comparisons.items()
        }

    payloads: dict[str, dict[str, Any]] = {}
    seen: dict[tuple[Any, ...], str] = {}

    def compact(value: Any) -> Any:
        if isinstance(value, dict) and value.get("encoding") in {"base64", "json"}:
            if value.get("encoding") == "base64":
                cache_key = (
                    "base64",
                    value.get("dtype"),
                    value.get("logical_dtype"),
                    tuple(value.get("shape", [])),
                    value.get("data", ""),
                )
            else:
                cache_key = ("json_identity", id(value))
            payload_id = seen.get(cache_key)
            if payload_id is None:
                payload_id = f"tensor_{len(payloads)}"
                seen[cache_key] = payload_id
                payloads[payload_id] = payload_store.store(payload_id, value) if payload_store else value
            return {"tensor_ref": payload_id}
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    compacted = compact(source)
    compacted["tensor_payloads"] = payloads
    return compacted


class _ScriptSafeWriter:
    def __init__(self, stream: Any):
        self.stream = stream
        self.pending = ""

    def write(self, value: str) -> int:
        text = self.pending + value
        self.pending = "<" if text.endswith("<") else ""
        if self.pending:
            text = text[:-1]
        return self.stream.write(text.replace("</", "<\\/"))

    def finish(self) -> None:
        if self.pending:
            self.stream.write(self.pending)
            self.pending = ""


class _ScriptSafeBinaryWriter:
    def __init__(self, stream: Any):
        self.stream = stream
        self.pending = b""

    def write(self, value: bytes) -> None:
        data = self.pending + value
        self.pending = b"<" if data.endswith(b"<") else b""
        if self.pending:
            data = data[:-1]
        self.stream.write(data.replace(b"</", b"<\\/"))

    def finish(self) -> None:
        if self.pending:
            self.stream.write(self.pending)
            self.pending = b""


def _json_object_end(data: mmap.mmap, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(data)):
        value = data[index]
        if in_string:
            if escaped:
                escaped = False
            elif value == ord("\\"):
                escaped = True
            elif value == ord('"'):
                in_string = False
            continue
        if value == ord('"'):
            in_string = True
        elif value == ord("{"):
            depth += 1
        elif value == ord("}"):
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError(f"unterminated JSON object at byte {start}")


def _encoded_object_end(data: mmap.mmap, match: re.Match[bytes]) -> int:
    if match.group(1) == b"base64":
        closing = _BASE64_OBJECT_END_PATTERN.search(data, match.end())
        if closing is None:
            raise ValueError(f"unterminated base64 tensor payload at byte {match.start()}")
        return closing.end()
    return _json_object_end(data, match.start())


def _write_mmap_range(writer: _ScriptSafeBinaryWriter, data: mmap.mmap, start: int, end: int) -> None:
    chunk_size = 1024 * 1024
    for offset in range(start, end, chunk_size):
        writer.write(data[offset : min(end, offset + chunk_size)])


def _hash_mmap_range(data: mmap.mmap, start: int, end: int) -> str:
    digest = hashlib.sha256()
    chunk_size = 1024 * 1024
    for offset in range(start, end, chunk_size):
        digest.update(data[offset : min(end, offset + chunk_size)])
    return digest.hexdigest()


def _write_raw_tensor_script(
    path: Path, payload_id: str, data: mmap.mmap, start: int, end: int
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("wb") as stream:
        stream.write(b"window.__autofuseRegisterTensor(")
        stream.write(json.dumps(payload_id).encode("ascii"))
        stream.write(b",")
        chunk_size = 1024 * 1024
        for offset in range(start, end, chunk_size):
            stream.write(data[offset : min(end, offset + chunk_size)])
        stream.write(b");\n")
    temporary.replace(path)
    return path.stat().st_size


def _stream_externalized_report_json(
    source: Path, target: Path, prefix: str, suffix: str
) -> tuple[int, int, Path]:
    data_dir = _tensor_data_dir(target)
    payloads: dict[str, dict[str, Any]] = {}
    seen: dict[tuple[str, int], str] = {}
    external_bytes = 0

    with source.open("rb") as source_stream, mmap.mmap(source_stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        root_end = len(data) - 1
        while root_end >= 0 and data[root_end] in b" \t\r\n":
            root_end -= 1
        if root_end < 0 or data[root_end] != ord("}"):
            raise ValueError(f"report JSON root must be an object: {source}")

        with target.open("wb") as stream:
            stream.write(prefix.encode("utf-8"))
            writer = _ScriptSafeBinaryWriter(stream)
            position = 0
            search_position = 0
            while match := _ENCODED_OBJECT_PATTERN.search(data, search_position, root_end):
                payload_start = match.start()
                payload_end = _encoded_object_end(data, match)
                digest = _hash_mmap_range(data, payload_start, payload_end)
                cache_key = (digest, payload_end - payload_start)
                payload_id = seen.get(cache_key)
                if payload_id is None:
                    payload_id = f"tensor_{len(payloads)}"
                    seen[cache_key] = payload_id
                    filename = f"{payload_id}.js"
                    external_bytes += _write_raw_tensor_script(
                        data_dir / filename, payload_id, data, payload_start, payload_end
                    )
                    payloads[payload_id] = {
                        "external_script": quote(f"{data_dir.name}/{filename}"),
                        "byte_size": payload_end - payload_start,
                    }
                _write_mmap_range(writer, data, position, payload_start)
                writer.write(json.dumps({"tensor_ref": payload_id}, separators=(",", ":")).encode("ascii"))
                position = payload_end
                search_position = payload_end

            _write_mmap_range(writer, data, position, root_end)
            writer.write(b',"tensor_payloads":')
            writer.write(json.dumps(payloads, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            writer.write(b"}")
            _write_mmap_range(writer, data, root_end + 1, len(data))
            writer.finish()
            stream.write(suffix.encode("utf-8"))
    return len(payloads), external_bytes, data_dir


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}e}"
    return "-"


def render_report_html(
    report: dict[str, Any], title: str = "Autofuse Accuracy Report", *, embed_report: bool = True
) -> str:
    summary = report.get("summary", {})
    failed = int(summary.get("failed", 0) or 0)
    total = int(summary.get("total", 0) or 0)
    compared = int(summary.get("compared", total) or 0)
    unmatched = int(summary.get("unmatched", max(0, total - compared)) or 0)
    unmapped = int(summary.get("unmapped", unmatched) or 0)
    not_captured = int(summary.get("not_captured", 0) or 0)
    not_observable = int(summary.get("not_observable", 0) or 0)
    observable_total = int(summary.get("observable_total", max(0, total - not_observable)) or 0)
    passed = int(summary.get("passed", 0) or 0)
    strict_passed = int(summary.get("strict_passed", passed) or 0)
    strict_failed = int(summary.get("strict_failed", failed) or 0)
    cann_passed = int(summary.get("cann_passed", strict_passed) or 0)
    cann_failed = int(summary.get("cann_failed", strict_failed) or 0)
    disagreements = int(summary.get("standard_disagreements", 0) or 0)
    pass_rate = (passed / compared * 100.0) if compared else 0.0
    cann_pass_rate = (cann_passed / compared * 100.0) if compared else 0.0
    coverage = float(summary.get("coverage", compared / observable_total if observable_total else 0.0) or 0.0)
    status_text = str(summary.get("dual_status", summary.get("status", "PASS" if failed == 0 else "FAIL")))
    status_class = "pass" if status_text == "PASS" else ("fail" if status_text == "FAIL" else "warn")
    payload = _json_for_script(_compact_report_for_html(report)) if embed_report else "__REPORT_JSON__"
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
    .comparison-label { color: var(--muted); font-size: 12px; font-weight: 700; white-space: nowrap; }
    .comparison-tabs { display: flex; gap: 4px; overflow-x: auto; padding: 7px 18px 0; border-top: 1px solid var(--line); }
    .comparison-tabs[hidden] { display: none; }
    .comparison-tab { flex: 0 0 auto; border: 0; border-bottom: 2px solid transparent; border-radius: 5px 5px 0 0; background: transparent; color: var(--muted); padding: 6px 10px 7px; white-space: nowrap; }
    .comparison-tab:hover { color: var(--text); background: var(--accent-soft); }
    .comparison-tab.active { color: var(--accent); border-bottom-color: var(--accent); background: var(--accent-soft); font-weight: 700; }
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
      min-width: 0;
      display: flex;
      flex-direction: column;
      padding: 10px 14px 12px;
      gap: 8px;
    }
    .workspace {
      flex: 1 1 auto;
      min-height: 0;
      min-width: 0;
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
    .graph-node > rect { transition: stroke 120ms ease, stroke-width 120ms ease, filter 120ms ease; }
    .graph-node.related > rect { stroke: var(--accent); stroke-width: 3; }
    .graph-node.lifecycle-related > rect { stroke: var(--accent); stroke-width: 4; filter: drop-shadow(0 0 3px rgba(36, 87, 166, 0.4)); }
    .graph-node.first-failure > rect { stroke: #8f1230; stroke-width: 4; filter: drop-shadow(0 0 3px rgba(143, 18, 48, 0.3)); }
    .graph-node.suspected-first-failure > rect { stroke: #a66700; stroke-width: 4; stroke-dasharray: 7 4; filter: drop-shadow(0 0 3px rgba(166, 103, 0, 0.25)); }
    .graph-node.selected > rect { stroke: var(--accent); stroke-width: 4; filter: drop-shadow(0 0 4px rgba(36, 87, 166, 0.35)); }
    .first-failure-mark { pointer-events: none; }
    .first-failure-mark rect { fill: #8f1230; stroke: #fff; stroke-width: 1.5; }
    .first-failure-mark.suspected rect { fill: #a66700; }
    .first-failure-mark text { fill: #fff; font-size: 8.5px; font-weight: 800; text-anchor: middle; }
    .graph-node .fusion-mark { fill: var(--accent); cursor: pointer; }
    .graph-node .fusion-mark:focus { stroke: #fff; stroke-width: 2; }
    .graph-node .broadcast-cache-mark { fill: #087f8c; stroke: #fff; stroke-width: 2; pointer-events: none; }
    .graph-node .broadcast-cache-mark-text { fill: #fff; font-size: 8px; font-weight: 800; text-anchor: middle; pointer-events: none; }
    .graph-node .broadcast-cache-label { fill: #087f8c; font-size: 10.5px; font-weight: 700; pointer-events: none; }
    .graph-node .unfused-mark { fill: #b97800; stroke: #fff; stroke-width: 2; pointer-events: none; }
    .graph-node .unfused-mark-text { fill: #fff; font-size: 8px; font-weight: 800; text-anchor: middle; pointer-events: none; }
    .graph-node .unfused-label { fill: #a66700; font-size: 10.5px; font-weight: 700; pointer-events: none; }
    .graph-node .tensor-mark { fill: #2457a6; stroke: #fff; stroke-width: 2; cursor: pointer; }
    .graph-node .tensor-mark:focus { stroke: #16202a; stroke-width: 2.5; }
    .graph-node .tensor-mark-text { fill: #fff; font-size: 8px; font-weight: 800; text-anchor: middle; pointer-events: none; }
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
    .load-legend .first-fail::before { background: #fde2e2; border: 3px double #8f1230; }
    .load-legend .suspected-first::before { background: #fff1d6; border: 2px dashed #a66700; }
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
      --details-width: 360px;
      flex: 0 0 var(--details-width);
      width: var(--details-width);
      min-height: 0;
      min-width: 0;
      max-width: min(760px, 70vw);
      position: relative;
      display: flex;
      flex-direction: column;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .details-panel[hidden] { display: none; }
    .details-resize-handle { position: absolute; z-index: 6; top: 0; bottom: 0; left: -7px; width: 14px; cursor: ew-resize; touch-action: none; }
    .details-resize-handle::after { content: '⋮'; position: absolute; top: 50%; left: 50%; width: 14px; height: 46px; color: #7d8794; background: #f6f8fb; border: 1px solid var(--line); border-radius: 5px; font-size: 18px; line-height: 41px; text-align: center; transform: translate(-50%, -50%); }
    .details-resize-handle:hover::after, .details-resize-handle:focus-visible::after { color: var(--accent); border-color: var(--accent); }
    body.details-resizing, body.details-resizing * { cursor: ew-resize !important; user-select: none !important; }
    body.tensor-viewer-open { overflow: hidden !important; }
    .details-panel.details-wide .details-grid { grid-template-columns: max-content minmax(0, 1fr); }
    .details-panel.details-wide .details-grid dd, .details-panel.details-wide .io-item { white-space: nowrap; overflow-wrap: normal; word-break: normal; }
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
    .details-body { min-width: 0; min-height: 0; overflow-x: auto; overflow-y: auto; padding: 12px; }
    .details-kicker { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
    .details-op-type { margin-top: 2px; color: var(--muted); }
    .linked-autofuse-button { width: 100%; margin-top: 10px; color: var(--accent); font-weight: 700; }
    .tensor-view-button { width: 100%; margin-top: 8px; color: var(--accent); font-weight: 700; }
    .details-grid { display: grid; grid-template-columns: auto 1fr; gap: 5px 10px; margin-top: 12px; font-size: 13px; }
    .details-grid dt { color: var(--muted); }
    .details-grid dd { min-width: 0; margin: 0; overflow-wrap: anywhere; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .detail-section { margin-top: 16px; padding-top: 12px; border-top: 1px solid #edf0f5; }
    .detail-section-title { margin-bottom: 8px; font-weight: 700; }
    .io-list { display: grid; gap: 6px; }
    .io-item {
      min-width: 0;
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
      overflow-wrap: anywhere;
    }
    .io-item small { display: block; color: var(--muted); margin-top: 2px; }
    .tensor-viewer { position: fixed; inset: 0; z-index: 40; display: grid; place-items: center; padding: 16px; }
    .tensor-viewer[hidden] { display: none; }
    .tensor-viewer-backdrop { position: absolute; inset: 0; background: rgba(22, 32, 42, 0.46); }
    .tensor-viewer-panel { position: relative; width: min(1480px, calc(100vw - 32px)); height: min(900px, calc(100vh - 32px)); min-width: 0; min-height: 0; display: flex; flex-direction: column; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 18px 48px rgba(22, 32, 42, 0.28); overflow: hidden; }
    .tensor-viewer-head { flex: 0 0 auto; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 14px; border-bottom: 1px solid var(--line); }
    .tensor-viewer-title { min-width: 0; font-size: 16px; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .tensor-viewer-meta { flex: 0 0 auto; padding: 7px 14px; color: var(--muted); font-size: 12px; border-bottom: 1px solid #edf0f5; }
    .tensor-viewer-controls { flex: 0 0 auto; display: flex; align-items: center; flex-wrap: wrap; gap: 8px; padding: 8px 14px; border-bottom: 1px solid var(--line); background: #fbfcfe; }
    .tensor-viewer-controls label { display: inline-flex; align-items: center; gap: 5px; color: var(--muted); font-size: 12px; }
    .tensor-viewer-controls input { width: 88px; }
    .tensor-window-summary { margin-left: auto; color: var(--muted); font-size: 12px; white-space: nowrap; }
    .tensor-matrix-wrap { min-width: 0; min-height: 0; flex: 1 1 auto; overflow: auto; background: #f6f8fb; scrollbar-gutter: stable both-edges; }
    .tensor-matrix { display: grid; width: max-content; min-width: 100%; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .tensor-axis, .tensor-cell { min-width: 104px; min-height: 54px; border-right: 1px solid #dfe5ed; border-bottom: 1px solid #dfe5ed; }
    .tensor-axis { position: sticky; z-index: 3; display: flex; align-items: center; justify-content: center; padding: 5px; background: #eef1f5; color: var(--muted); font-size: 10px; }
    .tensor-axis.column { top: 0; }
    .tensor-axis.row { left: 0; justify-content: flex-start; min-width: 180px; max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .tensor-axis.corner { top: 0; left: 0; z-index: 5; min-width: 180px; }
    .tensor-cell { display: flex; flex-direction: column; align-items: stretch; justify-content: center; gap: 2px; padding: 6px 7px; cursor: pointer; }
    .tensor-cell.pass { background: #e7f5ee; color: #0f6b43; }
    .tensor-cell.fail { background: #fdecec; color: #a52020; }
    .tensor-cell:hover, .tensor-cell.selected { outline: 2px solid var(--accent); outline-offset: -2px; }
    .tensor-cell strong { font-size: 11px; font-weight: 700; white-space: nowrap; }
    .tensor-cell small { color: #536172; font-size: 9.5px; white-space: nowrap; }
    .tensor-element-detail { flex: 0 0 auto; min-height: 42px; padding: 8px 14px; border-top: 1px solid var(--line); color: #3d4856; font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-x: auto; white-space: nowrap; }
    .highlight-controls { display: inline-flex; align-items: center; gap: 3px; white-space: nowrap; }
    .highlight-controls .control-label { color: var(--muted); font-size: 12px; margin-right: 2px; }
    .highlight-controls button { height: 26px; padding: 0 8px; font-size: 12px; }
    .highlight-controls button.active { color: var(--accent); background: var(--accent-soft); border-color: var(--accent); }
    .highlight-controls button:disabled { cursor: default; opacity: 0.45; }
    .table-panel {
      flex: 0 0 auto;
      min-width: 0;
      max-width: 100%;
      max-height: 46vh;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .table-panel summary { padding: 8px 12px; }
    .table-content { min-width: 0; border-top: 1px solid var(--line); overflow: hidden; }
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
    .table-wrap { min-width: 0; max-width: 100%; overflow-x: hidden; overflow-y: auto; max-height: 34vh; scrollbar-gutter: stable both-edges; }
    .table-horizontal-scrollbar { min-width: 0; max-width: 100%; height: 17px; margin: 0 12px 7px; overflow-x: auto; overflow-y: hidden; border: 1px solid var(--line); border-radius: 4px; background: #f6f8fb; scrollbar-gutter: stable; }
    .table-horizontal-scrollbar[hidden] { display: none; }
    .table-horizontal-scrollbar-content { height: 1px; min-width: 100%; }
    .table-horizontal-scrollbar::-webkit-scrollbar { height: 12px; }
    .table-horizontal-scrollbar::-webkit-scrollbar-track { background: #eef1f5; border-radius: 4px; }
    .table-horizontal-scrollbar::-webkit-scrollbar-thumb { background: #9aa5b1; border-radius: 4px; border: 2px solid #eef1f5; }
    .table-horizontal-scrollbar:hover::-webkit-scrollbar-thumb { background: #6f7d8d; }
    table { width: max-content; border-collapse: collapse; min-width: 1500px; }
    th, td { padding: 7px 9px; border-bottom: 1px solid #edf0f5; text-align: left; vertical-align: top; }
    th { background: #fbfcfe; position: sticky; top: 0; z-index: 2; font-size: 12px; color: var(--muted); user-select: none; cursor: pointer; white-space: nowrap; }
    .metric-group-row th { top: 0; z-index: 3; padding-top: 6px; padding-bottom: 6px; text-align: center; color: #3d4856; background: #f2f5f9; cursor: default; }
    .metric-group-row .elementwise-group { color: #8a5a00; background: #fff7e5; }
    .metric-group-row .tensor-group { color: #2457a6; background: #eaf1ff; }
    .metric-column-row th { top: 31px; }
    td { white-space: nowrap; }
    td.op { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    tbody tr { cursor: pointer; }
    tbody tr:hover { background: var(--accent-soft); }
    .badge { display: inline-flex; align-items: center; min-height: 22px; padding: 0 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }
    .badge.pass { color: var(--good); background: #e7f5ee; }
    .badge.fail { color: var(--bad); background: #fdecec; }
    .badge.warn { color: var(--warn); background: #fff1d6; }
    .badge.neutral { color: #2457a6; background: #eaf1ff; }
    .status-ok { color: var(--good); }
    .status-warn { color: var(--warn); }
    .status-bad { color: var(--bad); }
    .status-neutral { color: #2457a6; }
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
      .details-panel { position: absolute; inset: 8px 8px 8px auto; width: min(var(--details-width), calc(100% - 16px)); max-width: calc(100% - 16px); flex-basis: auto; z-index: 5; }
      .table-panel { max-height: none; }
      .toolbar { grid-template-columns: 1fr; }
      .table-wrap { max-height: none; }
      .tensor-viewer { padding: 8px; }
      .tensor-viewer-panel { width: calc(100vw - 16px); height: calc(100vh - 16px); }
      .tensor-window-summary { flex-basis: 100%; margin-left: 0; }
    }
  </style>
</head>
<body>
  <header class="app-header">
    <div class="title-row">
      <h1>__TITLE__</h1>
      <div class="quick-status">
        <span id="resultStatus" class="status-pill __STATUS_CLASS__">__STATUS_TEXT__</span>
        <span id="comparisonLabel" class="comparison-label">CPU ↔ NPU Autofuse</span>
        <span class="quick-stat">逐元素对比 <strong id="strictQuickValue">__STRICT_PASSED__/__COMPARED__</strong></span>
        <span class="quick-stat">整体 Tensor 对比 <strong id="cannQuickValue">__CANN_PASSED__/__COMPARED__</strong></span>
        <span class="quick-stat">coverage <strong id="coverageQuickValue">__COVERAGE__%</strong></span>
        <button id="toggleMetrics" type="button" aria-expanded="false">指标</button>
      </div>
    </div>
    <div id="comparisonTabs" class="comparison-tabs" hidden role="tablist" aria-label="三方精度对比视图"></div>
    <details id="metricsPanel" class="metrics-panel">
      <summary>报告指标</summary>
      <div class="metrics">
        <div class="metric"><div class="label">Compared</div><div id="metricCompared" class="value">__COMPARED__/__TOTAL__</div></div>
        <div class="metric"><div class="label">Coverage</div><div id="metricCoverage" class="value __COVERAGE_CLASS__">__COVERAGE__%</div></div>
        <div class="metric"><div class="label">逐元素对比通过</div><div id="metricStrict" class="value">__STRICT_PASSED__/__COMPARED__ (__PASS_RATE__%)</div></div>
        <div class="metric"><div class="label">整体 Tensor 对比通过</div><div id="metricCann" class="value">__CANN_PASSED__/__COMPARED__ (__CANN_PASS_RATE__%)</div></div>
        <div class="metric"><div class="label">逐元素 / 整体 Tensor 失败</div><div id="metricFailed" class="value __FAILED_CLASS__">__STRICT_FAILED__ / __CANN_FAILED__</div></div>
        <div class="metric"><div class="label">Standard Disagreements</div><div id="metricDisagreements" class="value __DISAGREEMENT_CLASS__">__DISAGREEMENTS__</div></div>
        <div class="metric"><div class="label">Unmapped / No dump / Internal</div><div id="metricUnmatched" class="value __UNMATCHED_CLASS__">__UNMAPPED__ / __NOT_CAPTURED__ / __NOT_OBSERVABLE__</div></div>
        <div class="metric"><div class="label">Max Abs / Max Rel</div><div id="metricMaxError" class="value">__MAX_ABS__ / __MAX_REL__</div></div>
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
              <span class="pass">逐元素 / 整体均通过</span><span class="warning">仅逐元素失败</span><span class="fail">整体 Tensor 失败</span><span class="first-fail">首次失败</span><span class="suspected-first">疑似首次失败</span><span class="unmapped">未映射 / 未采集</span><span class="internal">融合内部不可观测</span><span class="unfused">未融合原因</span>
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
        <div id="detailsResizeHandle" class="details-resize-handle" role="separator" aria-label="拖动调整详情抽屉宽度" aria-orientation="vertical" aria-valuemin="300" aria-valuemax="760" aria-valuenow="360" tabindex="0" title="左右拖动调整详情抽屉宽度"></div>
        <div class="details-head">
          <div class="details-title" id="detailsTitle">节点详情</div>
          <button id="dismissDetails" type="button" title="关闭节点详情">×</button>
        </div>
        <div class="details-body">
          <div class="details-kicker" id="detailsKicker">selected NPU Autofuse op</div>
          <div class="details-op-type" id="detailsType">点击图中节点查看</div>
          <button class="linked-autofuse-button" id="linkedAutofuseButton" type="button" hidden></button>
          <button class="tensor-view-button" id="tensorViewButton" type="button" hidden>查看输出 Tensor 逐元素精度</button>
          <dl class="details-grid" id="detailsGrid"></dl>
          <div class="detail-section" id="ioSection">
            <div class="detail-section-title">输入 / 输出</div>
            <div class="io-list" id="detailsIo"></div>
          </div>
        </div>
      </aside>
      <div id="tensorViewer" class="tensor-viewer" hidden aria-hidden="true">
        <div class="tensor-viewer-backdrop" data-tensor-close></div>
        <section class="tensor-viewer-panel" role="dialog" aria-modal="true" aria-labelledby="tensorViewerTitle">
          <div class="tensor-viewer-head">
            <div id="tensorViewerTitle" class="tensor-viewer-title">输出 Tensor 逐元素精度</div>
            <button id="closeTensorViewer" type="button" title="关闭 Tensor 查看器">×</button>
          </div>
          <div id="tensorViewerMeta" class="tensor-viewer-meta"></div>
          <div class="tensor-viewer-controls">
            <label>显示 <select id="tensorDisplayMode">
              <option value="difference">绝对误差</option>
              <option value="cpu">CPU 值</option>
              <option value="npu">NPU 值</option>
              <option value="status">逐元素状态</option>
            </select></label>
            <label>行起点 <input id="tensorRowStart" type="number" min="0" step="1" value="0"></label>
            <label>列起点 <input id="tensorColStart" type="number" min="0" step="1" value="0"></label>
            <button id="tensorPrevRows" type="button">上一组行</button>
            <button id="tensorNextRows" type="button">下一组行</button>
            <button id="tensorPrevCols" type="button">上一组列</button>
            <button id="tensorNextCols" type="button">下一组列</button>
            <span id="tensorWindowSummary" class="tensor-window-summary"></span>
          </div>
          <div id="tensorMatrixWrap" class="tensor-matrix-wrap">
            <div id="tensorMatrix" class="tensor-matrix"></div>
          </div>
          <div id="tensorElementDetail" class="tensor-element-detail">点击矩阵单元查看该位置的 CPU/NPU 值、绝对误差和相对误差。</div>
        </section>
      </div>
    </section>
    <details id="tablePanel" class="table-panel">
      <summary>逐算子精度明细 <span id="tableSummary"></span></summary>
      <div class="table-content">
        <div class="toolbar">
          <input id="search" type="search" placeholder="搜索 op / type / status">
          <select id="passFilter">
            <option value="all">全部</option>
            <option value="both_passed">逐元素 / 整体均通过</option>
            <option value="failed">逐元素对比失败</option>
            <option value="cann_failed">整体 Tensor 对比失败</option>
            <option value="disagreement">标准不一致</option>
            <option value="unmatched">全部未对比</option>
            <option value="unmapped">仅未映射</option>
            <option value="not_captured">仅未采集</option>
            <option value="not_observable">仅融合内部不可观测</option>
          </select>
          <select id="typeFilter"><option value="all">全部类型</option></select>
          <button id="exportCsv" type="button">导出 CSV</button>
        </div>
        <div class="chart-wrap"><canvas id="errorChart" height="120"></canvas></div>
        <div id="tableHorizontalScrollbar" class="table-horizontal-scrollbar" hidden tabindex="0" role="scrollbar" aria-label="逐算子精度明细横向滚动条" aria-controls="resultTable" aria-orientation="horizontal" aria-valuemin="0" aria-valuemax="0" aria-valuenow="0">
          <div id="tableHorizontalScrollbarContent" class="table-horizontal-scrollbar-content"></div>
        </div>
        <div class="table-wrap">
          <table id="resultTable">
            <thead>
            <tr class="metric-group-row">
              <th colspan="10"></th><th colspan="6" class="elementwise-group">逐元素精度对比指标</th><th colspan="3" class="tensor-group">整 Tensor 精度对比指标</th>
            </tr>
            <tr class="metric-column-row">
              <th data-key="passed">结果</th><th data-key="strict_passed">逐元素对比</th><th data-key="cann_passed">整体 Tensor 对比</th><th data-key="status">状态</th><th data-key="unmapped_reason">映射说明</th><th id="rightOpHeader" data-key="npu_op">NPU Autofuse op</th>
              <th data-key="npu_type">类型</th><th id="leftOpHeader" data-key="cpu_op">CPU op</th><th data-key="shape">shape</th>
              <th data-key="dtype">dtype</th><th data-key="rtol">rtol / atol</th><th data-key="max_abs_error">max abs</th><th data-key="max_rel_error">max rel</th>
              <th data-key="rmse">rmse</th><th data-key="mismatched">mismatch</th><th data-key="max_error_index">max finite index / non-finite</th>
              <th data-key="cann_cosine_similarity">CosineSimilarity（余弦值相似度）</th><th data-key="relative_euclidean_distance">RelativeEuclideanDistance（欧氏相对距离）</th><th data-key="kl_divergence">KullbackLeiblerDivergence（KL散度）</th>
            </tr>
            </thead>
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
    const comparisonSets = report.comparisons || {};
    const tensorPayloads = report.tensor_payloads || {};
    const loadedTensorPayloads = new Map();
    const tensorLoadPromises = new Map();
    window.__autofuseRegisterTensor = (payloadId, payload) => loadedTensorPayloads.set(payloadId, payload);
    const comparisonOrder = report.comparison_order || Object.keys(comparisonSets);
    let activeComparisonKey = comparisonOrder[0] || null;
    let rows = ((comparisonSets[activeComparisonKey]?.results || report.results || [])).map((row, idx) => ({...row, _idx: idx}));
    let npuGraph = report.graphs?.[comparisonSets[activeComparisonKey]?.right_graph_key || 'npu'] || {nodes: [], edges: [], fusions: []};
    let cpuGraph = report.graphs?.cpu || {nodes: []};
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
    let refreshTableHorizontalScrollbar = () => {};
    const selection = {name: null, mode: 'depth1', detailNode: null, explicit: null};
    const graphViews = new Map();
    const tbody = document.querySelector('#resultTable tbody');
    const empty = document.getElementById('empty');
    const search = document.getElementById('search');
    const passFilter = document.getElementById('passFilter');
    const typeFilter = document.getElementById('typeFilter');
    const detailsPanel = document.getElementById('detailsPanel');
    let mainNpuNodes = new Map((npuGraph.nodes || []).map(node => [node.name, node]));
    let cpuNodes = new Map((cpuGraph.nodes || []).map(node => [node.name, node]));
    const resultByNpu = new Map();
    const resultRowsByNpu = new Map();
    let firstFailureByNode = new Map();
    function rebuildResultIndex() {
      resultByNpu.clear();
      resultRowsByNpu.clear();
      for (const row of rows) {
        if (row.status !== 'NOT_OBSERVABLE_ON_NPU' && row.observable !== false) {
          if (!resultRowsByNpu.has(row.npu_op)) resultRowsByNpu.set(row.npu_op, []);
          resultRowsByNpu.get(row.npu_op).push(row);
        }
        const previous = resultByNpu.get(row.npu_op);
        if (!previous || (previous.status === 'NOT_OBSERVABLE_ON_NPU' && row.status !== 'NOT_OBSERVABLE_ON_NPU')) {
          resultByNpu.set(row.npu_op, row);
        }
      }
    }
    rebuildResultIndex();
    function precisionNodeStatus(node) {
      if (!node) return 'unknown';
      if (String(node.type || '').toLowerCase() === 'data') return 'source';
      const nodeRows = resultRowsByNpu.get(node.name) || [];
      if (!nodeRows.length) return 'unknown';
      let hasUnknown = false;
      for (const row of nodeRows) {
        if (row.compared === false) {
          hasUnknown = true;
          continue;
        }
        const strictPassed = row.strict_passed ?? row.passed;
        const cannPassed = row.cann_passed ?? row.passed;
        if (strictPassed === undefined && cannPassed === undefined) hasUnknown = true;
        else if (!strictPassed || !cannPassed) return 'fail';
      }
      return hasUnknown ? 'unknown' : 'pass';
    }
    function firstFailurePath(candidate, incoming, longestParent, preferUnknown) {
      if (preferUnknown) {
        const queue = [{name: candidate, path: [candidate]}];
        const visited = new Set([candidate]);
        while (queue.length) {
          const current = queue.shift();
          for (const predecessor of incoming.get(current.name) || []) {
            if (visited.has(predecessor)) continue;
            const path = current.path.concat(predecessor);
            if (precisionNodeStatus(mainNpuNodes.get(predecessor)) === 'unknown') return path.reverse();
            visited.add(predecessor);
            queue.push({name: predecessor, path});
          }
        }
      }
      const reversed = [];
      const visited = new Set();
      let current = candidate;
      while (current && !visited.has(current)) {
        visited.add(current);
        reversed.push(current);
        current = longestParent.get(current) || null;
      }
      return reversed.reverse();
    }
    function upstreamUnknownNodes(candidate, incoming) {
      const unknown = [];
      const visited = new Set([candidate]);
      const stack = [...(incoming.get(candidate) || [])];
      while (stack.length) {
        const name = stack.pop();
        if (visited.has(name)) continue;
        visited.add(name);
        if (precisionNodeStatus(mainNpuNodes.get(name)) === 'unknown') unknown.push(name);
        stack.push(...(incoming.get(name) || []));
      }
      return unknown;
    }
    function rebuildFirstFailureAnalysis() {
      firstFailureByNode = new Map();
      const nodes = npuGraph.nodes || [];
      const known = new Set(nodes.map(node => node.name));
      const incoming = new Map(nodes.map(node => [node.name, []]));
      const outgoing = new Map(nodes.map(node => [node.name, []]));
      const indegree = new Map(nodes.map(node => [node.name, 0]));
      for (const edge of npuGraph.edges || []) {
        if (!known.has(edge.src) || !known.has(edge.dst)) continue;
        incoming.get(edge.dst).push(edge.src);
        outgoing.get(edge.src).push(edge.dst);
        indegree.set(edge.dst, (indegree.get(edge.dst) || 0) + 1);
      }
      const queue = nodes.filter(node => indegree.get(node.name) === 0).map(node => node.name);
      const topological = [];
      while (queue.length) {
        const name = queue.shift();
        topological.push(name);
        for (const destination of outgoing.get(name) || []) {
          indegree.set(destination, indegree.get(destination) - 1);
          if (indegree.get(destination) === 0) queue.push(destination);
        }
      }
      const ordered = new Set(topological);
      for (const node of nodes) if (!ordered.has(node.name)) topological.push(node.name);

      const hasFailedAncestor = new Map();
      const hasUnknownAncestor = new Map();
      const depth = new Map();
      const longestParent = new Map();
      for (const name of topological) {
        const predecessors = incoming.get(name) || [];
        hasFailedAncestor.set(name, predecessors.some(predecessor =>
          precisionNodeStatus(mainNpuNodes.get(predecessor)) === 'fail' || hasFailedAncestor.get(predecessor)
        ));
        hasUnknownAncestor.set(name, predecessors.some(predecessor =>
          precisionNodeStatus(mainNpuNodes.get(predecessor)) === 'unknown' || hasUnknownAncestor.get(predecessor)
        ));
        let parent = null;
        let nodeDepth = 0;
        for (const predecessor of predecessors) {
          const candidateDepth = (depth.get(predecessor) || 0) + 1;
          if (candidateDepth > nodeDepth) {
            nodeDepth = candidateDepth;
            parent = predecessor;
          }
        }
        depth.set(name, nodeDepth);
        if (parent) longestParent.set(name, parent);
      }

      for (const name of topological) {
        const node = mainNpuNodes.get(name);
        if (precisionNodeStatus(node) !== 'fail' || hasFailedAncestor.get(name)) continue;
        const unknowns = upstreamUnknownNodes(name, incoming);
        const suspected = !!hasUnknownAncestor.get(name);
        const fusion = isFusionNode(node);
        const kind = suspected ? 'suspected' : (fusion ? 'fusion' : 'first');
        const label = suspected ? 'SUSPECTED FIRST FAIL' : (fusion ? 'FUSION FIRST FAIL' : 'FIRST FAIL');
        const reason = suspected
          ? `未发现更早的已知失败，但上游存在 ${unknowns.length} 个未比较节点`
          : (fusion ? '所有可比较上游均通过，首次失败位于融合算子出口' : '所有可比较上游均通过，当前节点首次失败');
        firstFailureByNode.set(name, {
          kind, label, reason, unknowns,
          path: firstFailurePath(name, incoming, longestParent, suspected),
        });
      }
    }
    function upstreamFirstFailures(nodeName) {
      const incoming = new Map((npuGraph.nodes || []).map(node => [node.name, []]));
      for (const edge of npuGraph.edges || []) incoming.get(edge.dst)?.push(edge.src);
      const found = new Set();
      const visited = new Set([nodeName]);
      const stack = [...(incoming.get(nodeName) || [])];
      while (stack.length) {
        const name = stack.pop();
        if (visited.has(name)) continue;
        visited.add(name);
        if (firstFailureByNode.has(name)) found.add(name);
        stack.push(...(incoming.get(name) || []));
      }
      return [...found];
    }
    rebuildFirstFailureAnalysis();
    function activeComparison() {
      return comparisonSets[activeComparisonKey] || null;
    }
    function comparisonStatus(summary) {
      return String(summary?.dual_status || summary?.status || 'PASS');
    }
    function updateComparisonSummary() {
      const comparison = activeComparison();
      if (!comparison) return;
      const summary = comparison.summary || {};
      const compared = Number(summary.compared || 0);
      const status = comparisonStatus(summary);
      const statusPill = document.getElementById('resultStatus');
      statusPill.textContent = status;
      statusPill.className = `status-pill ${status === 'PASS' ? 'pass' : status === 'FAIL' ? 'fail' : 'warn'}`;
      document.getElementById('comparisonLabel').textContent = `${comparison.left} ↔ ${comparison.right}`;
      document.getElementById('strictQuickValue').textContent = `${summary.strict_passed ?? summary.passed ?? 0}/${compared}`;
      document.getElementById('cannQuickValue').textContent = `${summary.cann_passed ?? summary.passed ?? 0}/${compared}`;
      document.getElementById('coverageQuickValue').textContent = `${(Number(summary.coverage || 0) * 100).toFixed(1)}%`;
      const strictPassed = Number(summary.strict_passed ?? summary.passed ?? 0);
      const cannPassed = Number(summary.cann_passed ?? summary.passed ?? 0);
      document.getElementById('metricCompared').textContent = `${compared}/${summary.total ?? compared}`;
      document.getElementById('metricCoverage').textContent = `${(Number(summary.coverage || 0) * 100).toFixed(1)}%`;
      document.getElementById('metricStrict').textContent = `${strictPassed}/${compared} (${compared ? (strictPassed / compared * 100).toFixed(1) : '0.0'}%)`;
      document.getElementById('metricCann').textContent = `${cannPassed}/${compared} (${compared ? (cannPassed / compared * 100).toFixed(1) : '0.0'}%)`;
      document.getElementById('metricFailed').textContent = `${summary.strict_failed ?? (compared - strictPassed)} / ${summary.cann_failed ?? (compared - cannPassed)}`;
      document.getElementById('metricDisagreements').textContent = String(summary.standard_disagreements ?? 0);
      document.getElementById('metricUnmatched').textContent = `${summary.unmapped ?? 0} / ${summary.not_captured ?? 0} / ${summary.not_observable ?? 0}`;
      document.getElementById('metricMaxError').textContent = `${fmt(summary.max_abs_error)} / ${fmt(summary.max_rel_error)}`;
      document.getElementById('leftOpHeader').textContent = `${comparison.left} op`;
      document.getElementById('rightOpHeader').textContent = `${comparison.right} op`;
      const mainTab = document.querySelector('[data-graph-tab="main"]');
      if (mainTab) mainTab.textContent = `${comparison.left} ↔ ${comparison.right} 精度对比`;
    }
    function renderComparisonTabs() {
      const target = document.getElementById('comparisonTabs');
      if (!comparisonOrder.length) return;
      target.hidden = false;
      target.textContent = '';
      for (const key of comparisonOrder) {
        const comparison = comparisonSets[key];
        if (!comparison) continue;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `comparison-tab${key === activeComparisonKey ? ' active' : ''}`;
        button.dataset.comparison = key;
        button.setAttribute('role', 'tab');
        button.setAttribute('aria-selected', key === activeComparisonKey ? 'true' : 'false');
        button.textContent = `${comparison.left} ↔ ${comparison.right}`;
        button.addEventListener('click', () => switchComparison(key));
        target.appendChild(button);
      }
    }
    function switchComparison(key) {
      const comparison = comparisonSets[key];
      if (!comparison || key === activeComparisonKey) return;
      activeComparisonKey = key;
      rows = (comparison.results || []).map((row, idx) => ({...row, _idx: idx}));
      const graphKey = comparison.right_graph_key || 'npu';
      npuGraph = report.graphs?.[graphKey] || {nodes: [], edges: [], fusions: []};
      const leftGraphKey = comparison.left === 'NPU eager' ? 'npu_eager' : 'cpu';
      cpuGraph = report.graphs?.[leftGraphKey] || {nodes: []};
      mainNpuNodes = new Map((npuGraph.nodes || []).map(node => [node.name, node]));
      cpuNodes = new Map((cpuGraph.nodes || []).map(node => [node.name, node]));
      rebuildResultIndex();
      rebuildFirstFailureAnalysis();
      graphContext.kind = 'main';
      graphContext.graph = npuGraph;
      graphContext.nodes = mainNpuNodes;
      graphContext.edges = (npuGraph.edges || []).filter(edge => mainNpuNodes.has(edge.src) && mainNpuNodes.has(edge.dst));
      graphContext.parent = null;
      graphContext.autoIndex = null;
      selection.name = null;
      selection.detailNode = null;
      detailsPanel.hidden = true;
      state.filtered = rows.slice();
      fillTypeFilter();
      applyFilters();
      renderComparisonTabs();
      updateComparisonSummary();
      renderGraph();
    }
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
      const nodeRows = resultRowsByNpu.get(name) || [];
      const failedRow = nodeRows.find(row => row.compared !== false && (
        !(row.strict_passed ?? row.passed) || !(row.cann_passed ?? row.passed)
      ));
      const row = failedRow || nodeRows.find(item => item.compared === false) || resultByNpu.get(name);
      if (!row) return null;
      const precisionStatus = precisionNodeStatus(mainNpuNodes.get(name));
      return {
        row,
        compared: precisionStatus === 'pass' || precisionStatus === 'fail',
        strictPassed: !!nodeRows.length && nodeRows.every(item => item.compared !== false && !!(item.strict_passed ?? item.passed)),
        cannPassed: !!nodeRows.length && nodeRows.every(item => item.compared !== false && !!(item.cann_passed ?? item.passed)),
      };
    }
    const tensorViewerState = {
      nodeName: null, row: null, shape: [], rows: 1, cols: 1, cpu: null, npu: null, mask: null,
      rowStart: 0, colStart: 0, rowPageSize: 32, colPageSize: 16, mode: 'difference', selected: null, loadToken: 0,
    };
    function decodeBase64Bytes(value) {
      const raw = atob(value || '');
      const bytes = new Uint8Array(raw.length);
      for (let index = 0; index < raw.length; index += 1) bytes[index] = raw.charCodeAt(index);
      return bytes;
    }
    function decodeTensorArray(payload) {
      if (!payload) return [];
      if (payload.encoding === 'json') return payload.values || [];
      const bytes = decodeBase64Bytes(payload.data);
      const buffer = bytes.buffer;
      const dtype = String(payload.dtype || 'float32');
      const constructors = {
        bool: Uint8Array, uint8: Uint8Array, int8: Int8Array, uint16: Uint16Array, int16: Int16Array,
        uint32: Uint32Array, int32: Int32Array, float32: Float32Array, float64: Float64Array,
      };
      if (dtype === 'int64' || dtype === 'uint64') {
        return dtype === 'int64' ? new BigInt64Array(buffer) : new BigUint64Array(buffer);
      }
      const Constructor = constructors[dtype] || Float32Array;
      return new Constructor(buffer);
    }
    function loadTensorPayload(payload) {
      if (!payload?.tensor_ref) return Promise.resolve(payload);
      const payloadId = payload.tensor_ref;
      const descriptor = tensorPayloads[payloadId];
      if (!descriptor) return Promise.reject(new Error(`Tensor 数据索引不存在: ${payloadId}`));
      if (!descriptor.external_script) return Promise.resolve(descriptor);
      if (tensorLoadPromises.has(payloadId)) return tensorLoadPromises.get(payloadId);
      const promise = new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.async = true;
        script.src = descriptor.external_script;
        script.onload = () => {
          script.remove();
          const loaded = loadedTensorPayloads.get(payloadId);
          loadedTensorPayloads.delete(payloadId);
          tensorLoadPromises.delete(payloadId);
          if (loaded) resolve(loaded);
          else reject(new Error(`Tensor 数据文件未注册内容: ${descriptor.external_script}`));
        };
        script.onerror = () => {
          script.remove();
          tensorLoadPromises.delete(payloadId);
          reject(new Error(`无法加载 ${descriptor.external_script}，请确认 HTML 与同名 _data 目录位于同一目录`));
        };
        document.head.appendChild(script);
      });
      tensorLoadPromises.set(payloadId, promise);
      return promise;
    }
    function tensorRowsAndCols(shape) {
      if (!Array.isArray(shape) || shape.length === 0) return {rows: 1, cols: 1};
      const cols = Math.max(1, Number(shape[shape.length - 1]) || 1);
      const rows = Math.max(1, shape.slice(0, -1).reduce((total, value) => total * Math.max(1, Number(value) || 1), 1));
      return {rows, cols};
    }
    function tensorIndexCoords(flatIndex, shape) {
      if (!Array.isArray(shape) || shape.length === 0) return [];
      let value = flatIndex;
      const coords = new Array(shape.length);
      for (let index = shape.length - 1; index >= 0; index -= 1) {
        const size = Math.max(1, Number(shape[index]) || 1);
        coords[index] = value % size;
        value = Math.floor(value / size);
      }
      return coords;
    }
    function tensorFormatValue(value) {
      if (typeof value === 'bigint') return `${value}n`;
      const number = Number(value);
      if (Number.isNaN(number)) return 'NaN';
      if (number === Infinity) return '+Inf';
      if (number === -Infinity) return '-Inf';
      if (Object.is(number, -0)) return '-0';
      if (Number.isInteger(number) && Math.abs(number) < 1e6) return String(number);
      return Number.isFinite(number) ? number.toExponential(5) : String(value);
    }
    function tensorAbsError(cpu, npu) {
      const left = Number(cpu), right = Number(npu);
      if (!Number.isFinite(left) || !Number.isFinite(right)) return '-';
      return tensorFormatValue(Math.abs(left - right));
    }
    function tensorRelError(cpu, npu) {
      const left = Number(cpu), right = Number(npu);
      if (!Number.isFinite(left) || !Number.isFinite(right)) return '-';
      return tensorFormatValue(Math.abs(left - right) / Math.max(Math.abs(left), Math.abs(right), 1e-12));
    }
    function tensorStatus(flatIndex) {
      return Number(tensorViewerState.mask?.[flatIndex] || 0) === 1;
    }
    function tensorElementTitle(flatIndex) {
      const cpu = tensorViewerState.cpu?.[flatIndex];
      const npu = tensorViewerState.npu?.[flatIndex];
      const coords = tensorIndexCoords(flatIndex, tensorViewerState.shape);
      return `${indexText(coords)}\nCPU=${tensorFormatValue(cpu)}\nNPU=${tensorFormatValue(npu)}\nabs=${tensorAbsError(cpu, npu)}\nrel=${tensorRelError(cpu, npu)}\n${tensorStatus(flatIndex) ? 'PASS' : 'FAIL'}`;
    }
    function tensorDisplayValue(flatIndex) {
      const cpu = tensorViewerState.cpu?.[flatIndex];
      const npu = tensorViewerState.npu?.[flatIndex];
      if (tensorViewerState.mode === 'cpu') return tensorFormatValue(cpu);
      if (tensorViewerState.mode === 'npu') return tensorFormatValue(npu);
      if (tensorViewerState.mode === 'status') return tensorStatus(flatIndex) ? 'PASS' : 'FAIL';
      return tensorAbsError(cpu, npu);
    }
    function renderTensorElementDetail() {
      const target = document.getElementById('tensorElementDetail');
      const flatIndex = tensorViewerState.selected;
      if (flatIndex === null || flatIndex === undefined) {
        target.textContent = '点击矩阵单元查看该位置的 CPU/NPU 值、绝对误差和相对误差。';
        return;
      }
      const cpu = tensorViewerState.cpu?.[flatIndex], npu = tensorViewerState.npu?.[flatIndex];
      target.textContent = `${indexText(tensorIndexCoords(flatIndex, tensorViewerState.shape))} · CPU=${tensorFormatValue(cpu)} · NPU=${tensorFormatValue(npu)} · abs=${tensorAbsError(cpu, npu)} · rel=${tensorRelError(cpu, npu)} · ${tensorStatus(flatIndex) ? 'PASS' : 'FAIL'}`;
    }
    function renderTensorMatrix() {
      const matrix = document.getElementById('tensorMatrix');
      const summary = document.getElementById('tensorWindowSummary');
      const rowStart = Math.max(0, Math.min(tensorViewerState.rows - 1, Number(tensorViewerState.rowStart) || 0));
      const colStart = Math.max(0, Math.min(tensorViewerState.cols - 1, Number(tensorViewerState.colStart) || 0));
      const rowEnd = Math.min(tensorViewerState.rows, rowStart + tensorViewerState.rowPageSize);
      const colEnd = Math.min(tensorViewerState.cols, colStart + tensorViewerState.colPageSize);
      tensorViewerState.rowStart = rowStart;
      tensorViewerState.colStart = colStart;
      document.getElementById('tensorRowStart').value = String(rowStart);
      document.getElementById('tensorColStart').value = String(colStart);
      summary.textContent = `行 ${rowStart}-${rowEnd - 1}/${tensorViewerState.rows - 1} · 列 ${colStart}-${colEnd - 1}/${tensorViewerState.cols - 1}`;
      matrix.style.gridTemplateColumns = `180px repeat(${Math.max(1, colEnd - colStart)}, 104px)`;
      matrix.textContent = '';
      const corner = document.createElement('div');
      corner.className = 'tensor-axis corner';
      corner.textContent = '位置 / 列';
      matrix.appendChild(corner);
      for (let column = colStart; column < colEnd; column += 1) {
        const axis = document.createElement('div');
        axis.className = 'tensor-axis column';
        axis.textContent = String(column);
        matrix.appendChild(axis);
      }
      for (let row = rowStart; row < rowEnd; row += 1) {
        const rowAxis = document.createElement('div');
        rowAxis.className = 'tensor-axis row';
        const flatIndex = row * tensorViewerState.cols + colStart;
        const prefix = tensorIndexCoords(flatIndex, tensorViewerState.shape).slice(0, -1);
        rowAxis.textContent = `行 ${row} · [${prefix.join(', ')}]`;
        matrix.appendChild(rowAxis);
        for (let column = colStart; column < colEnd; column += 1) {
          const index = row * tensorViewerState.cols + column;
          const cell = document.createElement('div');
          const passed = tensorStatus(index);
          cell.className = `tensor-cell ${passed ? 'pass' : 'fail'}${tensorViewerState.selected === index ? ' selected' : ''}`;
          cell.title = tensorElementTitle(index);
          const value = document.createElement('strong');
          value.textContent = tensorDisplayValue(index);
          const status = document.createElement('small');
          status.textContent = passed ? 'PASS' : 'FAIL';
          cell.append(value, status);
          cell.addEventListener('click', () => {
            tensorViewerState.selected = index;
            renderTensorMatrix();
            renderTensorElementDetail();
          });
          matrix.appendChild(cell);
        }
      }
      renderTensorElementDetail();
    }
    function renderTensorViewer() {
      const row = tensorViewerState.row;
      const payload = row?.elementwise_data;
      const meta = document.getElementById('tensorViewerMeta');
      document.getElementById('tensorViewerTitle').textContent = `输出 Tensor 逐元素精度 · ${row?.npu_op || tensorViewerState.nodeName || '-'}`;
      meta.textContent = `shape=${shapeText(tensorViewerState.shape)} · dtype=${row?.dtype || payload?.logical_dtype || '-'} · 元素数=${tensorViewerState.rows * tensorViewerState.cols} · mismatch=${row?.mismatched ?? '-'} · 绿色=PASS，红色=FAIL`;
      renderTensorMatrix();
    }
    async function openTensorViewer(nodeName) {
      const row = nodeMetric(nodeName)?.row;
      const payload = row?.elementwise_data;
      if (!payload) return;
      const loadToken = ++tensorViewerState.loadToken;
      const viewer = document.getElementById('tensorViewer');
      viewer.hidden = false;
      viewer.setAttribute('aria-hidden', 'false');
      document.body.classList.add('tensor-viewer-open');
      document.getElementById('tensorViewerTitle').textContent = `输出 Tensor 逐元素精度 · ${row?.npu_op || nodeName || '-'}`;
      document.getElementById('tensorViewerMeta').textContent = '正在按需加载 Tensor 数据...';
      document.getElementById('tensorWindowSummary').textContent = '';
      document.getElementById('tensorMatrix').textContent = '';
      document.getElementById('tensorElementDetail').textContent = '';
      try {
        const [cpuPayload, npuPayload, maskPayload] = await Promise.all([
          loadTensorPayload(payload.cpu), loadTensorPayload(payload.npu), loadTensorPayload(payload.strict_mask),
        ]);
        if (loadToken !== tensorViewerState.loadToken) return;
        const dimensions = tensorRowsAndCols(payload.shape || row.shape || []);
        tensorViewerState.nodeName = nodeName;
        tensorViewerState.row = row;
        tensorViewerState.shape = payload.shape || row.shape || [];
        tensorViewerState.rows = dimensions.rows;
        tensorViewerState.cols = dimensions.cols;
        tensorViewerState.cpu = decodeTensorArray(cpuPayload);
        tensorViewerState.npu = decodeTensorArray(npuPayload);
        tensorViewerState.mask = decodeTensorArray(maskPayload);
        tensorViewerState.rowStart = 0;
        tensorViewerState.colStart = 0;
        tensorViewerState.selected = null;
        renderTensorViewer();
      } catch (error) {
        console.error('failed to decode elementwise tensor data', error);
        if (loadToken !== tensorViewerState.loadToken) return;
        document.getElementById('tensorViewerMeta').textContent = `Tensor 数据加载失败: ${error?.message || error}`;
      }
    }
    function closeTensorViewer() {
      tensorViewerState.loadToken += 1;
      tensorViewerState.row = null;
      tensorViewerState.cpu = null;
      tensorViewerState.npu = null;
      tensorViewerState.mask = null;
      tensorViewerState.selected = null;
      document.getElementById('tensorMatrix').textContent = '';
      const viewer = document.getElementById('tensorViewer');
      viewer.hidden = true;
      viewer.setAttribute('aria-hidden', 'true');
      document.body.classList.remove('tensor-viewer-open');
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
        const bufferId = bufferMemory?.buf_id;
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
        if (metric.row.status === 'NOT_OBSERVABLE_ON_NPU') {
          return {fill: '#eaf1ff', stroke: '#7d9dcc', label: 'INTERNAL ONLY', metric, dashed: true};
        }
        return {fill: '#eef1f5', stroke: '#9aa5b1', label: noDump ? 'NO DUMP' : 'NO MAP', metric, dashed: noDump};
      }
      if (!metric.cannPassed) {
        const label = metric.strictPassed ? '整体 Tensor 失败' : '逐元素 + 整体均失败';
        return {fill: '#fde2e2', stroke: '#d34a4a', label, metric};
      }
      if (!metric.strictPassed) return {fill: '#fff1d6', stroke: '#b97800', label: '逐元素失败', metric};
      if (metric.row.status && metric.row.status !== 'OK') return {fill: '#fff1d6', stroke: '#b97800', label: 'WARN', metric};
      return {fill: '#dff3e9', stroke: '#2f8f62', label: '逐元素 + 整体均通过', metric};
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
      const tensorButton = document.getElementById('tensorViewButton');
      tensorButton.hidden = !row?.elementwise_data;
      tensorButton.dataset.tensorNode = row?.elementwise_data ? String(row.npu_op || node.name) : '';
      const grid = document.getElementById('detailsGrid');
      const mapping = row?.compared === false
        ? (row?.status === 'NPU_OUTPUT_NOT_CAPTURED' ? 'NO DUMP' : (row?.status === 'NOT_OBSERVABLE_ON_NPU' ? '融合内部不可观测' : 'NO MAP'))
        : (row?.cpu_op ? `${row.cpu_op}:${row.cpu_index ?? 0}` : '-');
      const strictResult = row?.compared === false ? '-' : ((row?.strict_passed ?? row?.passed) ? 'PASS' : 'FAIL');
      const cannResult = row?.compared === false ? '-' : ((row?.cann_passed ?? row?.passed) ? 'PASS' : 'FAIL');
      const firstFailure = firstFailureByNode.get(node.name);
      const detailRows = [
        ['结果', state.label], ['逐元素对比', strictResult], ['整体 Tensor 对比', cannResult], ['状态', row?.status || '-'],
        ['映射说明', row?.unmapped_reason || row?.mapping_method || '-'], ['CPU 对照', mapping], ['max abs（有限值）', finiteMetricText(row, 'max_abs_error')],
        ['max rel（有限值）', finiteMetricText(row, 'max_rel_error')], ['rmse（有限值）', finiteMetricText(row, 'rmse')],
        ['CosineSimilarity（余弦值相似度）', cannMetricText(row, 'cann_cosine_similarity')],
        ['RelativeEuclideanDistance（欧氏相对距离）', cannMetricText(row, 'relative_euclidean_distance')],
        ['KullbackLeiblerDivergence（KL散度）', cannMetricText(row, 'kl_divergence')], ['mismatch', mismatchText(row)],
        ['最大有限误差位置', row?.finite_pair_count === 0 ? 'N/A（无有限值对）' : indexText(row?.max_error_index)],
      ];
      if (firstFailure) {
        const path = firstFailure.path.map(name => nodeDisplayName(mainNpuNodes.get(name) || {name})).join(' → ');
        detailRows.splice(1, 0,
          ['首次失败判断', firstFailure.label],
          ['判断依据', firstFailure.reason],
          ['输入传播路径', path || nodeDisplayName(node)],
        );
        if (firstFailure.unknowns.length) {
          const unknownNames = firstFailure.unknowns.slice(0, 8).map(name => nodeDisplayName(mainNpuNodes.get(name) || {name}));
          const remainder = firstFailure.unknowns.length > unknownNames.length ? ` 等 ${firstFailure.unknowns.length} 个` : '';
          detailRows.splice(4, 0, ['上游不确定节点', unknownNames.join(', ') + remainder]);
        }
      } else if (metric?.compared && (!metric.strictPassed || !metric.cannPassed)) {
        const origins = upstreamFirstFailures(node.name);
        if (origins.length) {
          detailRows.splice(1, 0,
            ['首次失败判断', '否（上游已经存在精度失败）'],
            ['上游首次失败边界', origins.map(name => nodeDisplayName(mainNpuNodes.get(name) || {name})).join(', ')],
          );
        }
      }
      if (row?.compared !== false) detailRows.splice(5, 0,
        ['逐元素对比容差', `${fmt(row?.rtol)} / ${fmt(row?.atol)} (${row?.tolerance_source || '-'})`],
        ['整体 Tensor 指标阈值', 'cos>0.99 / rel_euclid<0.05 / KL<0.005'],
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
      let pointerNode = null, pointerFusionEntry = false, pointerTensorButton = false;
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
        pointerTensorButton = !!event.target.closest?.('.tensor-mark');
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
        const selectedTensorButton = pointerTensorButton;
        if (target.hasPointerCapture(pointerId)) target.releasePointerCapture(pointerId);
        pointerId = null;
        pointerNode = null;
        pointerFusionEntry = false;
        pointerTensorButton = false;
        currentView.dragged = false;
        target.classList.remove('is-panning');
        if (wasDragged) return;
        if (selectedNode && selectedFusionEntry) enterFusionNode(selectedNode);
        else if (selectedNode && selectedTensorButton) { activateNode(selectedNode); openTensorViewer(selectedNode); }
        else if (selectedNode) activateNode(selectedNode);
        else closeDetails();
      });
      target.addEventListener('pointercancel', event => {
        if (event.pointerId !== pointerId) return;
        pointerId = null;
        pointerNode = null;
        pointerFusionEntry = false;
        pointerTensorButton = false;
        view().dragged = false;
        target.classList.remove('is-panning');
      });
      target.addEventListener('dblclick', () => fitGraph('npuGraph'));
      target.addEventListener('keydown', event => {
        const currentView = view();
        if (!currentView) return;
        const focusedNode = event.target.closest?.('.graph-node')?.dataset.node;
        const focusedFusionEntry = !!event.target.closest?.('.fusion-mark');
        const focusedTensorButton = !!event.target.closest?.('.tensor-mark');
        if ((event.key === 'Enter' || event.key === ' ') && focusedNode && focusedFusionEntry) enterFusionNode(focusedNode);
        else if ((event.key === 'Enter' || event.key === ' ') && focusedNode && focusedTensorButton) { activateNode(focusedNode); openTensorViewer(focusedNode); }
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
      else fields.push('Buffer output');
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
    const detailsWidthState = {width: 360};
    function detailsWidthBounds() {
      const workspace = document.querySelector('.workspace');
      const available = workspace?.clientWidth || window.innerWidth;
      return {
        min: 300,
        max: Math.max(360, Math.min(760, Math.round(window.innerWidth * 0.7), available - 260)),
      };
    }
    function setDetailsWidth(value) {
      const panel = document.getElementById('detailsPanel');
      const handle = document.getElementById('detailsResizeHandle');
      const bounds = detailsWidthBounds();
      const width = Math.round(Math.max(bounds.min, Math.min(bounds.max, Number(value) || bounds.min)));
      detailsWidthState.width = width;
      panel.style.setProperty('--details-width', `${width}px`);
      panel.classList.toggle('details-wide', width >= 520);
      handle.setAttribute('aria-valuemin', String(bounds.min));
      handle.setAttribute('aria-valuemax', String(bounds.max));
      handle.setAttribute('aria-valuenow', String(width));
    }
    function initDetailsResize() {
      const panel = document.getElementById('detailsPanel');
      const handle = document.getElementById('detailsResizeHandle');
      let pointerId = null, startX = 0, startWidth = detailsWidthState.width;
      setDetailsWidth(detailsWidthState.width);
      handle.addEventListener('pointerdown', event => {
        if (event.button !== 0) return;
        pointerId = event.pointerId;
        startX = event.clientX;
        startWidth = panel.getBoundingClientRect().width || detailsWidthState.width;
        handle.setPointerCapture(pointerId);
        document.body.classList.add('details-resizing');
        event.preventDefault();
        event.stopPropagation();
      });
      handle.addEventListener('pointermove', event => {
        if (event.pointerId !== pointerId) return;
        setDetailsWidth(startWidth + startX - event.clientX);
      });
      function finishResize(event) {
        if (event.pointerId !== pointerId) return;
        if (handle.hasPointerCapture(pointerId)) handle.releasePointerCapture(pointerId);
        pointerId = null;
        document.body.classList.remove('details-resizing');
      }
      handle.addEventListener('pointerup', finishResize);
      handle.addEventListener('pointercancel', finishResize);
      handle.addEventListener('keydown', event => {
        if (event.key === 'ArrowLeft') setDetailsWidth(detailsWidthState.width + 40);
        else if (event.key === 'ArrowRight') setDetailsWidth(detailsWidthState.width - 40);
        else if (event.key === 'Home') setDetailsWidth(detailsWidthBounds().min);
        else if (event.key === 'End') setDetailsWidth(detailsWidthBounds().max);
        else return;
        event.preventDefault();
        event.stopPropagation();
      });
    }
    function initTableHorizontalScrollbar() {
      const wrap = document.querySelector('.table-wrap');
      const scrollbar = document.getElementById('tableHorizontalScrollbar');
      const scrollbarContent = document.getElementById('tableHorizontalScrollbarContent');
      const panel = document.getElementById('tablePanel');
      let syncing = false;
      function update() {
        const contentWidth = Math.max(wrap.scrollWidth, wrap.clientWidth);
        const max = Math.max(0, contentWidth - wrap.clientWidth);
        scrollbarContent.style.width = `${contentWidth}px`;
        scrollbar.hidden = max <= 2;
        scrollbar.setAttribute('aria-valuemax', String(max));
        scrollbar.setAttribute('aria-valuenow', String(Math.min(max, wrap.scrollLeft)));
        if (!syncing && Math.abs(scrollbar.scrollLeft - wrap.scrollLeft) > 1) {
          syncing = true;
          scrollbar.scrollLeft = Math.min(max, wrap.scrollLeft);
          syncing = false;
        }
      }
      refreshTableHorizontalScrollbar = update;
      wrap.addEventListener('scroll', () => {
        if (syncing) return;
        syncing = true;
        scrollbar.scrollLeft = wrap.scrollLeft;
        syncing = false;
        update();
      });
      scrollbar.addEventListener('scroll', () => {
        if (syncing) return;
        syncing = true;
        wrap.scrollLeft = scrollbar.scrollLeft;
        syncing = false;
        update();
      });
      panel.addEventListener('toggle', () => requestAnimationFrame(update));
      if (window.ResizeObserver) {
        const observer = new ResizeObserver(update);
        observer.observe(wrap);
        observer.observe(document.getElementById('resultTable'));
      }
      update();
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
        const comparison = activeComparison();
        const confirmedFirstFailures = [...firstFailureByNode.values()].filter(item => item.kind !== 'suspected').length;
        const suspectedFirstFailures = [...firstFailureByNode.values()].filter(item => item.kind === 'suspected').length;
        title.textContent = `${comparison?.right || 'NPU Autofuse'} Dump Graph`;
        const failureSummary = firstFailureByNode.size
          ? ` · 首次失败 ${confirmedFirstFailures} · 疑似 ${suspectedFirstFailures}`
          : '';
        meta.textContent = `${layout.nodes.length} ops / ${layout.edges.length} edges · ${npuGraph.model?.topology || comparison?.right || 'TorchInductor Autofuse'}${failureSummary}`;
        back.hidden = true;
        fusionDetails.hidden = true;
        analytics.hidden = true;
        analytics.open = false;
        loadLegend.hidden = false;
        loadLegend.setAttribute('aria-label', '精度结果分类');
        loadLegend.innerHTML = '<span class="pass">逐元素 / 整体均通过</span><span class="warning">仅逐元素失败</span><span class="fail">整体 Tensor 失败</span><span class="first-fail">首次失败</span><span class="suspected-first">疑似首次失败</span><span class="unmapped">未映射 / 未采集</span><span class="internal">融合内部不可观测</span><span class="unfused">未融合原因</span>';
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
        const firstFailure = graphContext.kind === 'main' ? firstFailureByNode.get(node.name) : null;
        const broadcastCached = graphContext.kind === 'autofuse' && hasBroadcastCache(node);
        const tensorAvailable = graphContext.kind === 'main' && !!metric?.row?.elementwise_data;
        const nodeUnfusedReasons = graphContext.kind === 'main' ? unfusedReasons(node) : [];
        const detail = graphContext.kind === 'autofuse'
          ? `node id=${node.id}`
          : graphContext.kind === 'fusion'
          ? (node.origin === false ? 'compiler inserted' : 'fusion origin')
          : (metric?.compared
            ? `逐元素=${metric.strictPassed ? 'PASS' : 'FAIL'} 整体Tensor=${metric.cannPassed ? 'PASS' : 'FAIL'}`
            : 'no compare result');
        const precisionLine = graphContext.kind === 'main'
          ? (metric?.compared
            ? `Precision: 逐元素${metric.strictPassed ? 'pass' : 'fail'}, 整Tensor ${metric.cannPassed ? 'pass' : 'fail'}`
            : `Precision: ${metric?.row?.status === 'NPU_OUTPUT_NOT_CAPTURED' ? 'no dump' : 'no map'}`)
          : `${state.label} · ${detail}`;
        const lineLength = graphContext.kind === 'autofuse' ? 56 : 44;
        const inputLine = shortText('in: ' + graphInputText(node), lineLength);
        const outputLine = shortText('out: ' + graphOutputText(node), lineLength);
        const fusionLine = graphContext.kind === 'main' && isFusionNode(node) ? `FUSED · ${node.origin_ops.length} ops` : '';
        const unfusedTooltip = nodeUnfusedReasons.map(unfusedReasonText).join('\\n');
        const firstFailureTooltip = firstFailure
          ? `${firstFailure.label}: ${firstFailure.reason}\\nPath: ${firstFailure.path.join(' -> ')}`
          : '';
        const tooltipDetail = [detail, firstFailureTooltip, broadcastCached ? `Broadcast cache: ${node.exec_condition}` : '', unfusedTooltip].filter(Boolean).join('\\n');
        const failureClasses = firstFailure
          ? ` first-failure${firstFailure.kind === 'suspected' ? ' suspected-first-failure' : ''}`
          : '';
        parts.push(`<g class="graph-node${failureClasses}" data-node="${escapeHtml(node.name)}" tabindex="0"><title>${escapeHtml(nodeDisplayName(node))}\n${escapeHtml(node.type || '')}\n${escapeHtml(tooltipDetail)}</title>`);
        parts.push(`<rect x="${position.x}" y="${position.y}" width="${position.w}" height="${position.h}" rx="8" fill="${state.fill}" stroke="${state.stroke}" stroke-width="${state.strokeWidth || 2}"${state.dashed ? ' stroke-dasharray="7 5"' : ''}/>`);
        parts.push(`<text x="${position.x + 10}" y="${position.y + 20}" font-size="12" font-weight="700" fill="#16202a">${escapeHtml(shortText(nodeDisplayName(node), 36))}</text>`);
        parts.push(`<text x="${position.x + 10}" y="${position.y + 39}" font-size="11" fill="#5f6b7a">${escapeHtml(shortText(node.type || '-', 36))}</text>`);
        parts.push(`<text x="${position.x + 10}" y="${position.y + 59}" font-size="10.5" fill="#3d4856">${escapeHtml(inputLine)}</text>`);
        parts.push(`<text x="${position.x + 10}" y="${position.y + 78}" font-size="10.5" fill="#3d4856">${escapeHtml(outputLine)}</text>`);
        parts.push(`<text x="${position.x + 10}" y="${position.y + 99}" font-size="11" fill="${state.stroke}">${escapeHtml(shortText(precisionLine, 44))}</text>`);
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
        if (firstFailure) {
          const badgeWidth = firstFailure.kind === 'suspected' ? 154 : (firstFailure.kind === 'fusion' ? 126 : 76);
          const badgeX = position.x + position.w - badgeWidth - 8;
          parts.push(`<g class="first-failure-mark${firstFailure.kind === 'suspected' ? ' suspected' : ''}"><rect x="${badgeX}" y="${position.y - 14}" width="${badgeWidth}" height="18" rx="4"/><text x="${badgeX + badgeWidth / 2}" y="${position.y - 1}">${firstFailure.label}</text></g>`);
        }
        if (fusionLine) parts.push(`<text x="${position.x + 10}" y="${position.y + 119}" font-size="10.5" fill="#2457a6">${escapeHtml(fusionLine)}</text><circle class="fusion-mark" data-fusion-entry="true" tabindex="0" role="button" aria-label="进入融合子图" cx="${position.x + position.w - 15}" cy="${position.y + 15}" r="8"><title>点击蓝色入口进入融合子图</title></circle>`);
        if (tensorAvailable) parts.push(`<circle class="tensor-mark" data-tensor-node="${escapeHtml(node.name)}" tabindex="0" role="button" aria-label="查看输出 Tensor 逐元素精度" cx="${position.x + position.w - 15}" cy="${position.y + position.h - 15}" r="9"><title>查看输出 Tensor 逐元素精度</title></circle><text class="tensor-mark-text" x="${position.x + position.w - 15}" y="${position.y + position.h - 12}">T</text>`);
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
      return [row.cpu_op, row.npu_op, row.npu_type, row.status, row.unmapped_reason, row.mapping_method, row.dtype, row.strict_passed, row.cann_passed, Array.isArray(row.shape) ? row.shape.join('x') : row.shape].join(' ').toLowerCase();
    }
    function statusClass(status) {
      if (!status) return 'status-bad';
      if (status === 'OK') return 'status-ok';
      if (status === 'NOT_OBSERVABLE_ON_NPU') return 'status-neutral';
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
        if (pass === 'not_observable' && row.status !== 'NOT_OBSERVABLE_ON_NPU') return false;
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
        const internalOnly = row.status === 'NOT_OBSERVABLE_ON_NPU';
        const badgeClass = internalOnly ? 'neutral' : (row.compared === false ? 'warn' : (!cannPassed ? 'fail' : (!strictPassed ? 'warn' : 'pass')));
        const badgeText = row.compared === false
          ? (row.status === 'NPU_OUTPUT_NOT_CAPTURED' ? 'NO DUMP' : (internalOnly ? 'INTERNAL' : 'NO MAP'))
          : (!cannPassed ? (strictPassed ? '整体 Tensor 失败' : '逐元素 + 整体均失败') : (!strictPassed ? '逐元素失败' : '逐元素 + 整体均通过'));
        const firstFailure = firstFailureByNode.get(row.npu_op);
        const resultText = firstFailure ? `${firstFailure.label} · ${badgeText}` : badgeText;
        const strictText = row.compared === false ? '-' : (strictPassed ? 'PASS' : 'FAIL');
        const cannText = row.compared === false ? '-' : (cannPassed ? 'PASS' : 'FAIL');
        tr.innerHTML = `<td><span class="badge ${badgeClass}">${escapeHtml(resultText)}</span></td><td>${strictText}</td><td>${cannText}</td><td class="${statusClass(row.status)}">${escapeHtml(row.status || '-')}</td><td>${escapeHtml(row.unmapped_reason || row.mapping_method || '-')}</td><td class="op">${escapeHtml(row.npu_op || '-')}:${row.npu_index ?? '-'}</td><td>${escapeHtml(row.npu_type || '-')}</td><td class="op">${escapeHtml(row.cpu_op || '-')}:${row.cpu_index ?? '-'}</td><td>${escapeHtml(shape)}</td><td>${escapeHtml(row.dtype || '-')}</td><td>${fmt(row.rtol)} / ${fmt(row.atol)}</td><td>${finiteMetricText(row, 'max_abs_error')}</td><td>${finiteMetricText(row, 'max_rel_error')}</td><td>${finiteMetricText(row, 'rmse')}</td><td>${escapeHtml(mismatchText(row))}</td><td>${escapeHtml(maxIndexText(row))}</td><td>${escapeHtml(cannMetricText(row, 'cann_cosine_similarity'))}</td><td>${escapeHtml(cannMetricText(row, 'relative_euclidean_distance'))}</td><td>${escapeHtml(cannMetricText(row, 'kl_divergence'))}</td>`;
        tr.addEventListener('click', () => {
          if (graphContext.kind !== 'main') showMainGraph();
          selectNode(row.npu_op);
        });
        tbody.appendChild(tr);
      }
      drawChart();
      refreshTableHorizontalScrollbar();
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
      typeFilter.innerHTML = '<option value="all">全部类型</option>';
      [...new Set(rows.map(row => row.npu_type || '').filter(Boolean))].sort().forEach(type => { const option = document.createElement('option'); option.value = type; option.textContent = type; typeFilter.appendChild(option); });
    }
    function downloadCsv() {
      const columns = ['compared','passed','strict_passed','cann_passed','status','mapping_method','unmapped_reason','observable','cpu_op','cpu_index','npu_op','npu_index','npu_type','rtol','atol','tolerance_source','shape','dtype','storage_dtype','max_abs_error','mean_abs_error','max_rel_error','rmse','cosine_similarity','cann_cosine_similarity','relative_euclidean_distance','kl_divergence','rmse_mae_ratio','cann_warning','mismatched','num_elements','finite_pair_count','max_error_index','nonfinite_summary','first_nonfinite_index','first_nonfinite_cpu','first_nonfinite_npu','cann_unavailable_reason'];
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
    document.getElementById('tensorViewButton').addEventListener('click', event => openTensorViewer(event.currentTarget.dataset.tensorNode));
    document.getElementById('closeTensorViewer').addEventListener('click', closeTensorViewer);
    for (const closeTarget of document.querySelectorAll('[data-tensor-close]')) closeTarget.addEventListener('click', closeTensorViewer);
    document.getElementById('tensorDisplayMode').addEventListener('change', event => { tensorViewerState.mode = event.target.value; renderTensorMatrix(); });
    document.getElementById('tensorRowStart').addEventListener('change', event => { tensorViewerState.rowStart = Number(event.target.value) || 0; renderTensorMatrix(); });
    document.getElementById('tensorColStart').addEventListener('change', event => { tensorViewerState.colStart = Number(event.target.value) || 0; renderTensorMatrix(); });
    document.getElementById('tensorPrevRows').addEventListener('click', () => { tensorViewerState.rowStart -= tensorViewerState.rowPageSize; renderTensorMatrix(); });
    document.getElementById('tensorNextRows').addEventListener('click', () => { tensorViewerState.rowStart += tensorViewerState.rowPageSize; renderTensorMatrix(); });
    document.getElementById('tensorPrevCols').addEventListener('click', () => { tensorViewerState.colStart -= tensorViewerState.colPageSize; renderTensorMatrix(); });
    document.getElementById('tensorNextCols').addEventListener('click', () => { tensorViewerState.colStart += tensorViewerState.colPageSize; renderTensorMatrix(); });
    document.addEventListener('keydown', event => { if (event.key === 'Escape' && !document.getElementById('tensorViewer').hidden) closeTensorViewer(); });
    for (const th of document.querySelectorAll('th[data-key]')) th.addEventListener('click', () => { const key = th.dataset.key; if (state.sortKey === key) state.sortDir *= -1; else { state.sortKey = key; state.sortDir = ['max_abs_error','max_rel_error','rmse','mismatched'].includes(key) ? -1 : 1; } sortRows(); renderTable(); });
    renderGraphTabs(); renderComparisonTabs(); updateComparisonSummary(); renderGraph(); initGraphInteractions(); initZoomControls(); initLifecycleResize(); initDetailsResize(); initTableHorizontalScrollbar(); fillTypeFilter(); applyFilters();
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
    const initialTensorNode = initialParams.get('tensor');
    if (initialTensorNode) openTensorViewer(initialTensorNode);
    window.addEventListener('resize', () => {
      drawChart();
      setDetailsWidth(detailsWidthState.width);
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
        "__NOT_OBSERVABLE__": str(not_observable),
        "__COVERAGE_CLASS__": "pass" if (unmapped == 0 and not_captured == 0) else "warn",
        "__FAILED_CLASS__": "fail" if failed else "pass",
        "__DISAGREEMENT_CLASS__": "warn" if disagreements else "pass",
        "__UNMATCHED_CLASS__": "warn" if (unmapped or not_captured) else "pass",
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
    payload_store = _TensorPayloadStore(target)
    with progress_stage("HTML template rendering", detail=f"output={target}"):
        template = render_report_html(report, title, embed_report=False)
        html_report = _compact_report_for_html(report, payload_store)
        prefix, marker, suffix = template.partition("__REPORT_JSON__")
        if not marker:
            raise RuntimeError("HTML report placeholder is missing")

    with progress_stage("HTML streaming write", detail=f"output={target}"):
        with target.open("w", encoding="utf-8") as stream:
            stream.write(prefix)
            writer = _ScriptSafeWriter(stream)
            json.dump(html_report, writer, ensure_ascii=False, separators=(",", ":"))
            writer.finish()
            stream.write(suffix)
    size_mib = target.stat().st_size / (1024.0 * 1024.0)
    progress_message("INFO", f"HTML written size={size_mib:.1f}MiB output={target}")
    if payload_store.external_files:
        external_mib = payload_store.external_bytes / (1024.0 * 1024.0)
        progress_message(
            "INFO",
            f"HTML lazy tensor data files={payload_store.external_files} size={external_mib:.1f}MiB "
            f"directory={payload_store.data_dir}",
        )


def write_report_html_from_json(report_path: str | Path, html_path: str | Path, title: str = "Autofuse Accuracy Report") -> None:
    source = Path(report_path)
    target = Path(html_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with progress_stage("HTML template rendering", detail=f"output={target}"):
        template = render_report_html(
            {"summary": {}, "graphs": {}, "results": []},
            title,
            embed_report=False,
        )
        prefix, marker, suffix = template.partition("__REPORT_JSON__")
        if not marker:
            raise RuntimeError("HTML report placeholder is missing")
    with progress_stage("HTML JSON tensor externalization", detail=f"report={source} output={target}"):
        payload_count, external_bytes, data_dir = _stream_externalized_report_json(
            source, target, prefix, suffix
        )
    size_mib = target.stat().st_size / (1024.0 * 1024.0)
    progress_message("INFO", f"HTML written size={size_mib:.1f}MiB output={target}")
    if payload_count:
        progress_message(
            "INFO",
            f"HTML lazy tensor data files={payload_count} size={external_bytes / (1024.0 * 1024.0):.1f}MiB "
            f"directory={data_dir}",
        )
