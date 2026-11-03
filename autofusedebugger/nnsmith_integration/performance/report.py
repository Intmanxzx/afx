"""Adapt performance sessions to the shared fuzz summary report."""
import json
from pathlib import Path

from .storage import atomic_write_text
from .metrics import comparable_speedup, correctness_status, performance_status


def summary_for_html(summary: dict) -> dict:
    """Keep functional/accuracy outcomes separate from timing pass/fail."""
    accuracy_cases, performance_cases = [], []
    for case in summary.get("cases", []):
        results = case.get("backends", {})
        perf = dict(case, performance_status=performance_status(case),
                    ascendc_speedup=comparable_speedup(results, case.get("fingerprint")))
        performance_cases.append(perf)
        error = case.get("error") or "\n".join(
            f"{name}: {b.get('error') or b.get('correctness_error') or b.get('status')}"
            for name, b in results.items() if b.get("status") != "ok" or b.get("correctness") != "pass")
        accuracy_cases.append(dict(case, status=correctness_status(case),
            duration_s=sum(b.get("duration_s") or 0 for b in results.values()),
            n_kernels=results.get("ascendc", {}).get("mean_kernel_count"),
            stage=case.get("error_stage"), symptom="后端/精度异常，见性能对比页签" if error else None))
    return dict(summary, cases=accuracy_cases, performance_cases=performance_cases,
                passed=sum(c["status"] == "pass" for c in accuracy_cases),
                failed=sum(c["status"] == "fail" for c in accuracy_cases),
                gen_errors=sum(c["status"] == "gen_error" for c in accuracy_cases))


def write_reports(summary: dict, out_dir: Path):
    from ..summary_html import render_summary_html

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    page = render_summary_html(summary_for_html(summary),
        title="AF Fuzz Summary · AscendC / Triton-Ascend", op_categories=summary.get("category_ops", {}))
    atomic_write_text(out_dir / "summary.html", page)
