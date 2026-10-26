"""Lightweight cross-case summary index for an NNSmith fuzz session.

Deliberately separate from `autofusedebugger.html.render_report_html`: that
renderer is a single-case deep-dive view (graph viz, tensor viewer, fusion
detail) meant to be opened one case at a time. This one is a sortable/
filterable table over many cases -- pass/fail counts, op types, and a link to
the failing case's bug directory for reproduction. A failing case's own deep
report (if generated) is a separate file this index links to, not something
this module renders itself.
"""

from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any


def _json_for_script(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def render_summary_html(summary: dict[str, Any], title: str = "AF Fuzz Summary") -> str:
    cases = summary.get("cases", [])
    payload = _json_for_script(cases)
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    gen_errors = summary.get("gen_errors", 0)
    pass_rate = f"{(passed / total * 100):.1f}" if total else "0.0"

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; font: 14px/1.5 system-ui, -apple-system, sans-serif; color: #16202a; background: #f5f7fa; }}
  header {{ padding: 16px 20px; background: #fff; border-bottom: 1px solid #d7dee8; }}
  h1 {{ margin: 0 0 8px; font-size: 18px; }}
  .stats {{ display: flex; gap: 16px; color: #617083; }}
  .stats strong {{ color: #16202a; }}
  .stats .fail {{ color: #bd2b2b; }}
  .stats .pass {{ color: #147a4d; }}
  main {{ padding: 16px 20px; }}
  .toolbar {{ margin-bottom: 10px; display: flex; gap: 8px; }}
  input, select {{ height: 30px; border: 1px solid #d7dee8; border-radius: 6px; padding: 0 8px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d7dee8; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 7px 10px; border-bottom: 1px solid #edf0f5; text-align: left; font-size: 13px; }}
  th {{ background: #fbfcfe; color: #617083; cursor: pointer; white-space: nowrap; }}
  tr:hover {{ background: #eaf1ff; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
  .badge.pass {{ color: #147a4d; background: #e7f5ee; }}
  .badge.fail {{ color: #bd2b2b; background: #fdecec; }}
  .badge.gen_error {{ color: #a66700; background: #fff1d6; }}
  .ops {{ color: #617083; font-size: 12px; max-width: 360px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .shapes {{ color: #617083; font-size: 12px; max-width: 220px; white-space: pre-line; overflow-y: auto; max-height: 72px; }}
  a {{ color: #2457a6; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<header>
  <h1>{_html.escape(title)}</h1>
  <div class="stats">
    <span>Total <strong>{total}</strong></span>
    <span class="pass">Passed <strong>{passed}</strong></span>
    <span class="fail">Failed <strong>{failed}</strong></span>
    <span>Gen errors <strong>{gen_errors}</strong></span>
    <span>Pass rate <strong>{pass_rate}%</strong></span>
  </div>
</header>
<main>
  <div class="toolbar">
    <input id="filterOps" placeholder="按 op 名称筛选 (子串)">
    <select id="filterStatus">
      <option value="">全部状态</option>
      <option value="pass">pass</option>
      <option value="fail">fail</option>
      <option value="gen_error">gen_error</option>
    </select>
  </div>
  <table id="caseTable">
    <thead>
      <tr>
        <th data-key="index">#</th>
        <th data-key="status">状态</th>
        <th data-key="seed">Seed</th>
        <th data-key="n_nodes">节点数</th>
        <th data-key="duration_s">耗时(s)</th>
        <th data-key="symptom">Symptom</th>
        <th data-key="stage">Stage</th>
        <th data-key="op_types">算子集合</th>
        <th data-key="input_shapes">Input Shape</th>
        <th data-key="output_shapes">Output Shape</th>
        <th>复现路径</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>
</main>
<script>
const CASES = {payload};
const tbody = document.querySelector('#caseTable tbody');
let sortKey = 'index', sortAsc = true;

function formatShapes(shapes) {{
  const entries = Object.entries(shapes || {{}});
  if (entries.length === 0) return '-';
  return entries.map(([name, dims]) => `${{name}}: [${{dims.join(', ')}}]`).join('\\n');
}}

function render() {{
  const opsFilter = document.querySelector('#filterOps').value.toLowerCase();
  const statusFilter = document.querySelector('#filterStatus').value;
  let rows = CASES.filter(c => {{
    if (statusFilter && c.status !== statusFilter) return false;
    if (opsFilter && !(c.op_types || []).some(o => o.toLowerCase().includes(opsFilter))) return false;
    return true;
  }});
  rows.sort((a, b) => {{
    const av = a[sortKey], bv = b[sortKey];
    const cmp = (av > bv) - (av < bv);
    return sortAsc ? cmp : -cmp;
  }});
  tbody.innerHTML = rows.map(c => `
    <tr>
      <td>${{c.index}}</td>
      <td><span class="badge ${{c.status}}">${{c.status}}</span></td>
      <td>${{c.seed}}</td>
      <td>${{c.n_nodes || '-'}}</td>
      <td>${{c.duration_s.toFixed(2)}}</td>
      <td>${{c.symptom || '-'}}</td>
      <td>${{c.stage || '-'}}</td>
      <td class="ops" title="${{(c.op_types || []).join(', ')}}">${{(c.op_types || []).join(', ') || '-'}}</td>
      <td class="shapes" title="${{formatShapes(c.input_shapes)}}">${{formatShapes(c.input_shapes)}}</td>
      <td class="shapes" title="${{formatShapes(c.output_shapes)}}">${{formatShapes(c.output_shapes)}}</td>
      <td>${{c.bug_dir ? `<a href="${{c.bug_dir}}/repro.py">repro.py</a>` : '-'}}</td>
    </tr>
  `).join('');
}}

document.querySelectorAll('#caseTable th[data-key]').forEach(th => {{
  th.addEventListener('click', () => {{
    const key = th.dataset.key;
    sortAsc = (sortKey === key) ? !sortAsc : true;
    sortKey = key;
    render();
  }});
}});
document.querySelector('#filterOps').addEventListener('input', render);
document.querySelector('#filterStatus').addEventListener('change', render);
render();
</script>
</body>
</html>
"""


def write_summary_html(summary: dict[str, Any], out_path: str | Path, title: str = "AF Fuzz Summary") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_summary_html(summary, title=title), encoding="utf-8")
