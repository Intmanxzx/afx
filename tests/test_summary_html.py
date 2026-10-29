"""Tests for the cross-case summary.html renderer (stats-tab charts + table)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from autofusedebugger.nnsmith_integration.summary_html import (
    render_summary_html,
    write_summary_html,
)


def _case(index: int, status: str = "pass", **overrides) -> dict:
    base = {
        "seed": 1000 + index,
        "index": index,
        "status": status,
        "duration_s": 1.0,
        "case_id": f"{index:05d}-{1000 + index}",
        "n_nodes": 1,
        "n_vars": 2,
        "op_types": ["core.Input", "core.Add"],
        "symptom": None,
        "stage": None,
        "bug_dir": None,
        "error_summary": None,
        "input_shapes": {"v0": [4, 4]},
        "output_shapes": {"v0": [4, 4]},
        "input_dtypes": {"v0": "f32"},
        "output_dtypes": {"v0": "f32"},
        "n_kernels": 1,
    }
    base.update(overrides)
    return base


def _summary(cases: list[dict]) -> dict:
    return {
        "cases": cases,
        "total": len(cases),
        "passed": sum(c["status"] == "pass" for c in cases),
        "failed": sum(c["status"] == "fail" for c in cases),
        "gen_errors": sum(c["status"] == "gen_error" for c in cases),
    }


OP_CATS = {
    "elewise": {"core.Add", "core.Sub"},
    "reduce": {"core.ReduceMax"},
}


def test_render_basic_counts_and_payload() -> None:
    html = render_summary_html(_summary([_case(0), _case(1, "fail"), _case(2, "gen_error")]))
    assert "Total <strong>3</strong>" in html
    assert "Passed <strong>1</strong>" in html
    assert "Failed <strong>1</strong>" in html
    assert "Gen errors <strong>1</strong>" in html
    assert "Pass rate <strong>33.3%</strong>" in html
    # cases payload lands in the CASES JS constant (script-safe escaped)
    assert '"status": "pass"' in html
    assert '"status": "fail"' in html
    assert '"status": "gen_error"' in html


def test_render_zero_cases_no_div_by_zero() -> None:
    html = render_summary_html(_summary([]))
    assert "Pass rate <strong>0.0%</strong>" in html


def test_render_escapes_title() -> None:
    html = render_summary_html(_summary([]), title='<script>alert("x")</script>')
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_categories_payload_serialized() -> None:
    html = render_summary_html(_summary([_case(0)]), op_categories=OP_CATS)
    assert '"elewise"' in html
    assert '"core.Add"' in html
    # sets -> sorted plain lists for JSON
    assert html.index('"core.Add"') < html.index('"core.Sub"')


def test_table_tab_structure_unchanged() -> None:
    """用例列表表格的结构性内容（表头/筛选器/列）保持原样。"""
    html = render_summary_html(_summary([_case(0)]))
    for th in [
        'data-key="index"',
        'data-key="status"',
        'data-key="seed"',
        'data-key="n_nodes"',
        'data-key="n_kernels"',
        'data-key="duration_s"',
        'data-key="symptom"',
        'data-key="stage"',
        'data-key="op_types"',
        'data-key="input_shapes"',
        'data-key="output_shapes"',
        "复现路径",
        'placeholder="按 op 名称筛选 (子串)"',
    ]:
        assert th in html, th


def test_stats_tab_has_five_chart_cards() -> None:
    html = render_summary_html(_summary([_case(0)]))
    assert html.count('class="chart-card"') == 5
    for svg_id in ["chartOverall", "chartCategory", "chartDtype", "chartShapeDim", "chartShapeElem"]:
        assert f'id="{svg_id}"' in html, svg_id


def test_chart_cvd_safe_colors_present() -> None:
    """图表用 CVD 验证过的三色（与表格徽章色是两套）。"""
    html = render_summary_html(_summary([_case(0)]))
    assert "--viz-pass: #12805e" in html
    assert "--viz-fail: #d03b3b" in html
    assert "--viz-gen-error: #6250d6" in html
    # badge colors (table) stay on the original trio
    assert "--pass: #2f9e6e" in html
    assert "--fail: #e2635c" in html


def test_bar_geometry_caps_and_gaps_in_source() -> None:
    """柱状图规格进源码：细柱上限、段间 2px 缝、顶端圆角。"""
    html = render_summary_html(_summary([_case(0)]))
    assert "barW = Math.min(slot * 0.52, 24)" in html
    assert "segH = Math.max(h - (isTop ? 0 : 2), 1)" in html
    assert "a 4 4 0 0 1 4 -4" in html


def test_write_summary_html_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "summary.html"
    write_summary_html(_summary([_case(0), _case(1)]), out, op_categories=OP_CATS)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "AF Fuzz Summary" in text
    # the JS payload must still be valid JSON
    m = re.search(r"^const CASES = (.*);$", text, re.MULTILINE)
    assert m, "CASES payload line not found"
    cases = json.loads(m.group(1))
    assert len(cases) == 2
    assert cases[0]["status"] == "pass"
