from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autofusedebugger.io import SCHEMA, load_array, read_json, resolve_artifact, write_json


def convert_nc1hwc0_to_nhwc(arr: np.ndarray) -> np.ndarray:
    n, c1, h, w, c0 = arr.shape
    return np.transpose(arr.reshape(n, c1, h, w, c0), axes=(0, 2, 3, 1, 4)).reshape(n, h, w, c1 * c0)


def convert_ndc1hwc0_to_ndhwc(arr: np.ndarray) -> np.ndarray:
    n, d, c1, h, w, c0 = arr.shape
    return np.transpose(arr, axes=(0, 1, 3, 4, 2, 5)).reshape(n, d, h, w, c1 * c0)


FORMAT_CONVERSIONS = {
    ("NC1HWC0", "NHWC"): convert_nc1hwc0_to_nhwc,
    ("NC1HWC0", "ND"): convert_nc1hwc0_to_nhwc,
    ("NDC1HWC0", "NDHWC"): convert_ndc1hwc0_to_ndhwc,
    ("NDC1HWC0", "ND"): convert_ndc1hwc0_to_ndhwc,
}

NON_COMPARABLE_NODE_TYPES = {"Const", "Data", "NetOutput"}
DEFAULT_DTYPE_TOLERANCES = {
    "float64": (1e-7, 1e-7),
    "float32": (1e-5, 1e-5),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1e-2, 1e-2),
    "integer": (0.0, 0.0),
    "bool": (0.0, 0.0),
}
CANN_COSINE_THRESHOLD = 0.99
CANN_RELATIVE_EUCLIDEAN_THRESHOLD = 0.05
CANN_KL_DIVERGENCE_THRESHOLD = 0.005
CANN_RMSE_MAE_RATIO_WARN = 1.1
CANN_EPSILON = float(np.finfo(np.float32).eps)


@dataclass(frozen=True)
class TensorRef:
    op_name: str
    index: int
    path: Path | None
    fmt: str | None
    node_type: str
    dtype: str | None = None


def _check_manifest(data: dict[str, Any], path: str | Path) -> None:
    if data.get("schema") != SCHEMA:
        raise ValueError(f"unsupported manifest schema in {path}: {data.get('schema')}")


def _index_manifest(data: dict[str, Any]) -> dict[tuple[str, int], TensorRef]:
    root = data.get("artifacts", {}).get("root", ".")
    index: dict[tuple[str, int], TensorRef] = {}
    for node in data.get("nodes", []):
        name = node.get("name", "")
        node_type = node.get("type", "")
        if node_type in NON_COMPARABLE_NODE_TYPES:
            continue
        for out in node.get("outputs", []):
            out_index = int(out.get("index", 0))
            raw_path = out.get("path")
            path = resolve_artifact(root, raw_path) if raw_path else None
            index[(name, out_index)] = TensorRef(
                name, out_index, path, out.get("format"), node_type, out.get("dtype")
            )
    return index


def _fusion_index(data: dict[str, Any]) -> dict[tuple[str, int], tuple[str, int]]:
    result = {}
    for item in data.get("fusions", []):
        result[(item["fused_op"], int(item["fused_index"]))] = (item["origin_op"], int(item["origin_index"]))
    return result


def _graph_summary(data: dict[str, Any]) -> dict[str, Any]:
    nodes = []
    output_by_node: dict[str, list[dict[str, Any]]] = {}
    for node in data.get("nodes", []):
        outputs = []
        for out in node.get("outputs", []):
            outputs.append(
                {
                    "index": int(out.get("index", 0)),
                    "path": out.get("path"),
                    "format": out.get("format"),
                    "dtype": out.get("dtype"),
                    "shape": out.get("shape"),
                }
            )
        output_by_node[node.get("name", "")] = outputs
        nodes.append(
            {
                "name": node.get("name", ""),
                "type": node.get("type", ""),
                "outputs": outputs,
                "inputs": [],
                "origin_ops": node.get("origin_ops", []),
                "origin_subgraph": node.get("origin_subgraph"),
                "unfused_reasons": node.get("unfused_reasons", []),
            }
        )
    node_by_name = {node["name"]: node for node in nodes}
    for edge in data.get("edges", []):
        src = edge.get("src", "")
        dst = edge.get("dst", "")
        dst_node = node_by_name.get(dst)
        if dst_node is None:
            continue
        src_outputs = output_by_node.get(src, [])
        src_output = src_outputs[0] if src_outputs else {}
        dst_node["inputs"].append(
            {
                "src": src,
                "index": int(edge.get("dst_input", 0)),
                "format": edge.get("format") or src_output.get("format"),
                "dtype": edge.get("dtype") or src_output.get("dtype"),
                "shape": edge.get("shape") if edge.get("shape") is not None else src_output.get("shape"),
            }
        )
    for node in nodes:
        node["inputs"].sort(key=lambda item: item.get("index", 0))
    return {
        "side": data.get("side", ""),
        "model": data.get("model", {}),
        "nodes": nodes,
        "edges": data.get("edges", []),
        "fusions": data.get("fusions", []),
    }


def _apply_format(golden: np.ndarray, actual: np.ndarray, golden_fmt: str | None, actual_fmt: str | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not golden_fmt or not actual_fmt or golden_fmt == actual_fmt:
        return golden, actual, []
    status: list[str] = []
    key = (actual_fmt, golden_fmt)
    reverse = (golden_fmt, actual_fmt)
    if key in FORMAT_CONVERSIONS:
        actual = FORMAT_CONVERSIONS[key](actual)
        status.append("FORMAT_CONVERTED")
    elif reverse in FORMAT_CONVERSIONS:
        golden = FORMAT_CONVERSIONS[reverse](golden)
        status.append("FORMAT_CONVERTED")
    else:
        raise ValueError(f"unsupported format conversion: cpu={golden_fmt}, npu={actual_fmt}")
    return golden, actual, status


@np.errstate(all="ignore")
def _cann_metrics(golden: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    if not np.issubdtype(golden.dtype, np.floating) and not np.issubdtype(golden.dtype, np.complexfloating):
        passed = bool(np.array_equal(golden, actual))
        return {
            "cann_cosine_similarity": None,
            "relative_euclidean_distance": None,
            "kl_divergence": None,
            "rmse_mae_ratio": None,
            "cann_passed": passed,
        }

    x = np.asarray(golden).astype(np.float64).reshape(-1)
    y = np.asarray(actual).astype(np.float64).reshape(-1)
    if x.size == 0:
        return {
            "cann_cosine_similarity": 1.0,
            "relative_euclidean_distance": 0.0,
            "kl_divergence": 0.0,
            "rmse_mae_ratio": 1.0,
            "cann_passed": True,
        }

    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if x_norm <= CANN_EPSILON and y_norm <= CANN_EPSILON:
        cosine = 1.0
    elif x_norm <= CANN_EPSILON or y_norm <= CANN_EPSILON:
        cosine = 0.0
    else:
        cosine = min(float(np.sum((x / x_norm) * (y / y_norm))), 1.0)

    relative_euclidean = (
        0.0
        if y_norm <= CANN_EPSILON
        else float(np.sqrt(np.sum((x - y) ** 2) / np.sum(y**2)))
    )

    x_range = float(np.max(x) - np.min(x))
    y_range = float(np.max(y) - np.min(y))
    x_normalized = (x - np.min(x)) / x_range if x_range > CANN_EPSILON else x
    y_normalized = (y - np.min(y)) / y_range if y_range > CANN_EPSILON else y
    x_normalized = np.maximum(x_normalized, CANN_EPSILON)
    y_normalized = np.maximum(y_normalized, CANN_EPSILON)
    x_sum = float(np.sum(x_normalized))
    y_sum = float(np.sum(y_normalized))
    if x_sum <= CANN_EPSILON or y_sum <= CANN_EPSILON:
        kl_divergence = 0.0
    else:
        probability = x_normalized / x_sum
        reference = y_normalized / y_sum
        kl_divergence = float(np.sum(probability * np.log(probability / reference)))
        if abs(kl_divergence) <= CANN_EPSILON:
            kl_divergence = 0.0

    diff = x - y
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    rmse_mae_ratio = rmse / mae if mae > CANN_EPSILON else 1.0
    passed = bool(
        np.isfinite(cosine)
        and np.isfinite(relative_euclidean)
        and np.isfinite(kl_divergence)
        and cosine > CANN_COSINE_THRESHOLD
        and relative_euclidean < CANN_RELATIVE_EUCLIDEAN_THRESHOLD
        and kl_divergence < CANN_KL_DIVERGENCE_THRESHOLD
    )
    def finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "cann_cosine_similarity": finite_or_none(cosine),
        "relative_euclidean_distance": finite_or_none(relative_euclidean),
        "kl_divergence": finite_or_none(kl_divergence),
        "rmse_mae_ratio": finite_or_none(rmse_mae_ratio),
        "cann_warning": bool(np.isfinite(rmse_mae_ratio) and rmse_mae_ratio > CANN_RMSE_MAE_RATIO_WARN),
        "cann_passed": passed,
    }


def _nonfinite_value(value: float) -> str:
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "+Inf"
    if np.isneginf(value):
        return "-Inf"
    return f"{value:.8g}"


def _nonfinite_metrics(golden: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    gf = np.asarray(golden).astype(np.float64).reshape(-1)
    af = np.asarray(actual).astype(np.float64).reshape(-1)
    cpu = {"nan": np.isnan(gf), "pos_inf": np.isposinf(gf), "neg_inf": np.isneginf(gf)}
    npu = {"nan": np.isnan(af), "pos_inf": np.isposinf(af), "neg_inf": np.isneginf(af)}
    aligned = {key: cpu[key] & npu[key] for key in cpu}
    positions = ~np.isfinite(gf) | ~np.isfinite(af)
    aligned_positions = aligned["nan"] | aligned["pos_inf"] | aligned["neg_inf"]
    flat_indices = np.flatnonzero(positions)
    first_flat = int(flat_indices[0]) if flat_indices.size else None
    first_index = (
        list(int(i) for i in np.unravel_index(first_flat, golden.shape)) if first_flat is not None else None
    )
    stats = {
        "present": bool(flat_indices.size),
        "cpu": {key: int(mask.sum()) for key, mask in cpu.items()},
        "npu": {key: int(mask.sum()) for key, mask in npu.items()},
        "aligned": {key: int(mask.sum()) for key, mask in aligned.items()},
        "different_positions": int((positions & ~aligned_positions).sum()),
        "first_index": first_index,
        "first_cpu": _nonfinite_value(float(gf[first_flat])) if first_flat is not None else None,
        "first_npu": _nonfinite_value(float(af[first_flat])) if first_flat is not None else None,
    }
    def counts(side: str) -> str:
        return ",".join(
            f"{label}={stats[side][key]}"
            for key, label in (("neg_inf", "-Inf"), ("pos_inf", "+Inf"), ("nan", "NaN"))
        )
    stats["summary"] = (
        f"CPU({counts('cpu')});NPU({counts('npu')});"
        f"aligned({counts('aligned')});different_positions={stats['different_positions']}"
        if stats["present"]
        else ""
    )
    return stats


def _metrics(golden: np.ndarray, actual: np.ndarray, *, rtol: float, atol: float) -> dict[str, Any]:
    gf = np.asarray(golden).astype(np.float64).reshape(-1)
    af = np.asarray(actual).astype(np.float64).reshape(-1)
    with np.errstate(invalid="ignore"):
        abs_err = np.abs(gf - af)
    finite = np.isfinite(gf) & np.isfinite(af)
    if finite.any():
        fe = abs_err[finite]
        denom = np.maximum(np.maximum(np.abs(gf[finite]), np.abs(af[finite])), 1e-12)
        rel_err = fe / denom
        max_abs = float(fe.max())
        mean_abs = float(fe.mean())
        max_rel = float(rel_err.max())
        rmse = float(np.sqrt(np.mean(fe * fe)))
        dot = float(np.dot(gf[finite], af[finite]))
        norm = float(np.linalg.norm(gf[finite]) * np.linalg.norm(af[finite]))
        cosine = float(dot / (norm + 1e-12))
        max_flat_index = int(np.argmax(np.where(finite, abs_err, -np.inf)))
        max_index = tuple(int(i) for i in np.unravel_index(max_flat_index, golden.shape))
    else:
        max_abs = mean_abs = max_rel = rmse = cosine = 0.0
        max_index = ()
    close = np.isclose(gf, af, rtol=rtol, atol=atol, equal_nan=False)
    strict_passed = bool(close.all())
    nonfinite = _nonfinite_metrics(golden, actual)
    cann = _cann_metrics(golden, actual)
    cann_unavailable_reason = (
        "NONFINITE_INPUT"
        if nonfinite["present"]
        and any(
            cann.get(key) is None
            for key in ("cann_cosine_similarity", "relative_euclidean_distance", "kl_divergence")
        )
        else None
    )
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "max_rel_error": max_rel,
        "rmse": rmse,
        "cosine_similarity": cosine,
        "mismatched": int((~close).sum()),
        "num_elements": int(gf.size),
        "finite_pair_count": int(finite.sum()),
        "max_error_index": list(max_index),
        "nonfinite": nonfinite,
        "nonfinite_summary": nonfinite["summary"],
        "first_nonfinite_index": nonfinite["first_index"],
        "first_nonfinite_cpu": nonfinite["first_cpu"],
        "first_nonfinite_npu": nonfinite["first_npu"],
        "cann_unavailable_reason": cann_unavailable_reason,
        "passed": strict_passed,
        "strict_passed": strict_passed,
        **cann,
    }


def _dtype_family(value: Any) -> str:
    text = str(value or "").lower().replace("torch.", "").replace("numpy.", "")
    aliases = {
        "f64": "float64",
        "f32": "float32",
        "f16": "float16",
        "bf16": "bfloat16",
        "b1": "bool",
    }
    text = aliases.get(text, text)
    if "bfloat16" in text:
        return "bfloat16"
    if "float16" in text:
        return "float16"
    if "float32" in text:
        return "float32"
    if "float64" in text or text == "double":
        return "float64"
    if text in {"bool", "bool_"}:
        return "bool"
    try:
        dtype = np.dtype(text)
    except (TypeError, ValueError):
        return "float32"
    if np.issubdtype(dtype, np.bool_):
        return "bool"
    if np.issubdtype(dtype, np.integer):
        return "integer"
    if np.issubdtype(dtype, np.floating):
        return f"float{dtype.itemsize * 8}"
    return "float32"


def resolve_tolerances(
    cpu_dtype: Any,
    npu_dtype: Any,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> tuple[float, float, str]:
    defaults = [
        DEFAULT_DTYPE_TOLERANCES.get(_dtype_family(value), DEFAULT_DTYPE_TOLERANCES["float32"])
        for value in (cpu_dtype, npu_dtype)
    ]
    default_rtol = max(item[0] for item in defaults)
    default_atol = max(item[1] for item in defaults)
    resolved_rtol = default_rtol if rtol is None else float(rtol)
    resolved_atol = default_atol if atol is None else float(atol)
    if resolved_rtol < 0 or resolved_atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    source = (
        "dtype"
        if rtol is None and atol is None
        else ("override" if rtol is not None and atol is not None else "mixed")
    )
    return resolved_rtol, resolved_atol, source


def compare_pair(
    cpu: TensorRef,
    npu: TensorRef,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cpu_op": cpu.op_name,
        "cpu_index": cpu.index,
        "npu_op": npu.op_name,
        "npu_index": npu.index,
        "npu_type": npu.node_type,
        "status": "OK",
        "passed": False,
        "compared": True,
    }
    if cpu.path is None:
        result["status"] = "CPU_FILE_NOT_FOUND"
        return result
    if npu.path is None:
        result["status"] = "NPU_FILE_NOT_FOUND"
        return result
    try:
        golden = load_array(cpu.path)
        actual = load_array(npu.path)
    except Exception as exc:
        result["status"] = f"LOAD_ERROR: {exc}"
        return result

    resolved_rtol, resolved_atol, tolerance_source = resolve_tolerances(
        cpu.dtype or golden.dtype,
        npu.dtype or actual.dtype,
        rtol=rtol,
        atol=atol,
    )
    result["rtol"] = resolved_rtol
    result["atol"] = resolved_atol
    result["tolerance_source"] = tolerance_source

    statuses: list[str] = []
    try:
        golden, actual, format_statuses = _apply_format(golden, actual, cpu.fmt, npu.fmt)
        statuses.extend(format_statuses)
    except Exception as exc:
        result["status"] = f"FORMAT_UNSUPPORTED: {exc}"
        return result

    if golden.dtype != actual.dtype:
        promoted = np.promote_types(golden.dtype, actual.dtype)
        golden = golden.astype(promoted)
        actual = actual.astype(promoted)
        statuses.append("DTYPE_CAST")
    if golden.shape != actual.shape:
        if golden.size == actual.size:
            golden = golden.reshape(-1)
            actual = actual.reshape(-1)
            statuses.append("SHAPE_FLATTENED")
        else:
            result["status"] = f"SHAPE_MISMATCH: cpu={golden.shape}, npu={actual.shape}"
            return result

    result.update(_metrics(golden, actual, rtol=resolved_rtol, atol=resolved_atol))
    result["shape"] = list(golden.shape)
    result["dtype"] = str(golden.dtype)
    if statuses:
        result["status"] = "OK_" + "_".join(statuses)
    return result


def compare_manifests(
    cpu_manifest: str | Path,
    npu_manifest: str | Path,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> dict[str, Any]:
    cpu_data = read_json(cpu_manifest)
    npu_data = read_json(npu_manifest)
    _check_manifest(cpu_data, cpu_manifest)
    _check_manifest(npu_data, npu_manifest)
    cpu_index = _index_manifest(cpu_data)
    npu_index = _index_manifest(npu_data)
    fusions = _fusion_index(npu_data)

    rows: list[dict[str, Any]] = []
    for npu_key in sorted(npu_index):
        npu_ref = npu_index[npu_key]
        if npu_ref.path is None:
            rows.append(
                {
                    "cpu_op": "",
                    "cpu_index": 0,
                    "npu_op": npu_key[0],
                    "npu_index": npu_key[1],
                    "npu_type": npu_ref.node_type,
                    "status": "NPU_OUTPUT_NOT_CAPTURED",
                    "passed": False,
                    "compared": False,
                }
            )
            continue
        cpu_key = fusions.get(npu_key)
        if cpu_key is None and npu_key in cpu_index:
            cpu_key = npu_key
        if cpu_key is None:
            rows.append(
                {
                    "cpu_op": "",
                    "cpu_index": 0,
                    "npu_op": npu_key[0],
                    "npu_index": npu_key[1],
                    "npu_type": npu_ref.node_type,
                    "status": "UNMATCHED_CPU_OP",
                    "passed": False,
                    "compared": False,
                }
            )
            continue
        cpu_ref = cpu_index.get(cpu_key)
        if cpu_ref is None:
            cpu_ref = TensorRef(cpu_key[0], cpu_key[1], None, None, "")
        rows.append(compare_pair(cpu_ref, npu_ref, rtol=rtol, atol=atol))

    compared = [r for r in rows if r.get("compared", True)]
    unmatched = [r for r in rows if not r.get("compared", True)]
    not_captured = [r for r in unmatched if r.get("status") == "NPU_OUTPUT_NOT_CAPTURED"]
    unmapped = [r for r in unmatched if r.get("status") == "UNMATCHED_CPU_OP"]
    failed = [r for r in compared if not r.get("passed")]
    passed = len(compared) - len(failed)
    cann_failed = [r for r in compared if not r.get("cann_passed", False)]
    cann_passed = len(compared) - len(cann_failed)
    dual_failed = [
        r for r in compared if not r.get("strict_passed", r.get("passed", False)) or not r.get("cann_passed", False)
    ]
    dual_passed = len(compared) - len(dual_failed)
    standard_disagreements = [
        r
        for r in compared
        if bool(r.get("strict_passed", r.get("passed", False))) != bool(r.get("cann_passed", False))
    ]
    coverage = len(compared) / len(rows) if rows else 0.0
    status = "FAIL" if failed else ("INCOMPLETE" if unmatched else "PASS")
    return {
        "schema": "autofusedebugger-report/1",
        "rtol": rtol,
        "atol": atol,
        "tolerance_mode": (
            "dtype"
            if rtol is None and atol is None
            else ("override" if rtol is not None and atol is not None else "mixed")
        ),
        "default_dtype_tolerances": {
            name: {"rtol": values[0], "atol": values[1]} for name, values in DEFAULT_DTYPE_TOLERANCES.items()
        },
        "cann_thresholds": {
            "cosine_similarity": CANN_COSINE_THRESHOLD,
            "relative_euclidean_distance": CANN_RELATIVE_EUCLIDEAN_THRESHOLD,
            "kl_divergence": CANN_KL_DIVERGENCE_THRESHOLD,
        },
        "summary": {
            "total": len(rows),
            "compared": len(compared),
            "unmatched": len(unmatched),
            "unmapped": len(unmapped),
            "not_captured": len(not_captured),
            "dumped": len(rows) - len(not_captured),
            "coverage": coverage,
            "status": status,
            "passed": passed,
            "failed": len(failed),
            "strict_status": status,
            "strict_passed": passed,
            "strict_failed": len(failed),
            "cann_status": "FAIL" if cann_failed else ("INCOMPLETE" if unmatched else "PASS"),
            "cann_passed": cann_passed,
            "cann_failed": len(cann_failed),
            "dual_status": "FAIL" if dual_failed else ("INCOMPLETE" if unmatched else "PASS"),
            "dual_passed": dual_passed,
            "dual_failed": len(dual_failed),
            "standard_disagreements": len(standard_disagreements),
            "max_abs_error": max((r.get("max_abs_error", 0.0) for r in compared), default=0.0),
            "max_rel_error": max((r.get("max_rel_error", 0.0) for r in compared), default=0.0),
        },
        "graphs": {
            "cpu": _graph_summary(cpu_data),
            "npu": _graph_summary(npu_data),
        },
        "results": rows,
    }


def write_report_csv(path: str | Path, report: dict[str, Any]) -> None:
    fieldnames = [
        "compared",
        "passed",
        "strict_passed",
        "cann_passed",
        "status",
        "cpu_op",
        "cpu_index",
        "npu_op",
        "npu_index",
        "npu_type",
        "rtol",
        "atol",
        "tolerance_source",
        "shape",
        "dtype",
        "max_abs_error",
        "mean_abs_error",
        "max_rel_error",
        "rmse",
        "cosine_similarity",
        "cann_cosine_similarity",
        "relative_euclidean_distance",
        "kl_divergence",
        "rmse_mae_ratio",
        "cann_warning",
        "mismatched",
        "num_elements",
        "finite_pair_count",
        "max_error_index",
        "nonfinite_summary",
        "first_nonfinite_index",
        "first_nonfinite_cpu",
        "first_nonfinite_npu",
        "cann_unavailable_reason",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("results", []):
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def save_report(report: dict[str, Any], out_path: str | Path | None, csv_path: str | Path | None) -> None:
    if out_path:
        write_json(out_path, report)
    if csv_path:
        write_report_csv(csv_path, report)


def format_summary(report: dict[str, Any], *, limit: int = 20) -> str:
    summary = report["summary"]
    lines = [
        "accuracy: "
        f"{summary.get('status', 'PASS' if summary['failed'] == 0 else 'FAIL')} "
        f"strict={summary['passed']}/{summary.get('compared', summary['total'])} "
        f"cann={summary.get('cann_passed', summary['passed'])}/{summary.get('compared', summary['total'])} "
        f"coverage={summary.get('compared', summary['total'])}/{summary['total']} "
        f"max_abs={summary['max_abs_error']:.3e} "
        f"max_rel={summary['max_rel_error']:.3e}"
    ]
    failed = [r for r in report.get("results", []) if r.get("compared", True) and not r.get("passed")]
    for row in failed[:limit]:
        lines.append(
            f"  [FAIL] npu={row['npu_op']}:{row['npu_index']} "
            f"cpu={row['cpu_op']}:{row['cpu_index']} status={row['status']} "
            f"max_abs={row.get('max_abs_error', '-')}"
        )
    if len(failed) > limit:
        lines.append(f"  ... {len(failed) - limit} more failed rows")
    return "\n".join(lines)
