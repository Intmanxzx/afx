from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autofusedebugger.artifact_io import SCHEMA, load_array, read_json, resolve_artifact, write_json
from autofusedebugger.progress import progress_message


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


def _provenance_conflict_index(data: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for node in data.get("nodes", []):
        name = str(node.get("name", ""))
        for item in node.get("provenance_conflicts", []):
            result[(name, int(item.get("output_index", 0)))] = item
    return result


_EAGER_ALIGNMENT_TYPE_ALIASES = {
    "_softmax": "softmax",
    "view": "reshape",
    "reshape": "reshape",
    "permute": "transpose",
    "transpose": "transpose",
    "bmm": "matmul",
    "matmul": "matmul",
}


def _alignment_type(value: Any) -> str:
    name = str(value or "").rsplit(".", 1)[-1].lower()
    return _EAGER_ALIGNMENT_TYPE_ALIASES.get(name, name)


def _alignment_outputs(
    data: dict[str, Any],
) -> list[tuple[tuple[str, int], tuple[str, tuple[Any, ...] | None, str]]]:
    result = []
    for node in data.get("nodes", []):
        if str(node.get("type", "")).lower() == "data":
            continue
        node_type = _alignment_type(node.get("type"))
        for output in node.get("outputs", []):
            raw_shape = output.get("shape")
            shape = tuple(raw_shape) if isinstance(raw_shape, (list, tuple)) else None
            key = (str(node.get("name", "")), int(output.get("index", 0)))
            result.append((key, (node_type, shape, str(output.get("dtype") or ""))))
    return result


def _compatible_alignment_shape(
    reference: tuple[Any, ...] | None, actual: tuple[Any, ...] | None
) -> bool:
    if reference is None or actual is None or reference == actual:
        return True
    return {reference, actual} == {(), (1,)}


def _compiler_origin_conflict_reason(
    cpu_key: tuple[str, int],
    npu_key: tuple[str, int],
    cpu_signatures: dict[tuple[str, int], tuple[str, tuple[Any, ...] | None, str]],
    npu_signatures: dict[tuple[str, int], tuple[str, tuple[Any, ...] | None, str]],
    cpu_index: dict[tuple[str, int], TensorRef],
    npu_index: dict[tuple[str, int], TensorRef],
) -> str | None:
    cpu_signature = cpu_signatures.get(cpu_key)
    npu_signature = npu_signatures.get(npu_key)
    if cpu_signature is None or npu_signature is None:
        return None
    cpu_shape = cpu_signature[1]
    npu_shape = npu_signature[1]
    cpu_fmt = cpu_index.get(cpu_key).fmt if cpu_key in cpu_index else None
    npu_fmt = npu_index.get(npu_key).fmt if npu_key in npu_index else None
    if cpu_fmt and npu_fmt and cpu_fmt != npu_fmt:
        if (cpu_fmt, npu_fmt) in FORMAT_CONVERSIONS or (npu_fmt, cpu_fmt) in FORMAT_CONVERSIONS:
            return None
    if _compatible_alignment_shape(cpu_shape, npu_shape):
        return None
    return (
        "compiler provenance 输出 shape 冲突："
        f"reference={list(cpu_shape) if cpu_shape is not None else '-'}, "
        f"NPU={list(npu_shape) if npu_shape is not None else '-'}；"
        "未按相同元素个数展平比较，避免误报精度失败"
    )


def _eager_topology_mapping(
    cpu_data: dict[str, Any], eager_data: dict[str, Any]
) -> dict[tuple[str, int], tuple[str, int]]:
    """Align device-specific export graphs without trusting shifted FX names."""
    cpu_outputs = _alignment_outputs(cpu_data)
    eager_outputs = _alignment_outputs(eager_data)
    cpu_count = len(cpu_outputs)
    eager_count = len(eager_outputs)
    gap_penalty = -2
    scores = np.empty((cpu_count + 1, eager_count + 1), dtype=np.int32)
    trace = np.zeros((cpu_count + 1, eager_count + 1), dtype=np.int8)
    scores[:, 0] = np.arange(cpu_count + 1) * gap_penalty
    scores[0, :] = np.arange(eager_count + 1) * gap_penalty
    trace[1:, 0] = 1
    trace[0, 1:] = 2

    for cpu_pos in range(1, cpu_count + 1):
        for eager_pos in range(1, eager_count + 1):
            cpu_key, cpu_signature = cpu_outputs[cpu_pos - 1]
            eager_key, eager_signature = eager_outputs[eager_pos - 1]
            diagonal = -100_000_000
            if cpu_signature == eager_signature:
                same_name_bonus = 4 if cpu_key == eager_key else 0
                diagonal = int(scores[cpu_pos - 1, eager_pos - 1]) + 20 + same_name_bonus
            skip_cpu = int(scores[cpu_pos - 1, eager_pos]) + gap_penalty
            skip_eager = int(scores[cpu_pos, eager_pos - 1]) + gap_penalty
            best = max(diagonal, skip_cpu, skip_eager)
            scores[cpu_pos, eager_pos] = best
            trace[cpu_pos, eager_pos] = 3 if diagonal == best else (1 if skip_cpu >= skip_eager else 2)

    result: dict[tuple[str, int], tuple[str, int]] = {}
    cpu_pos, eager_pos = cpu_count, eager_count
    while cpu_pos or eager_pos:
        direction = trace[cpu_pos, eager_pos]
        if cpu_pos and eager_pos and direction == 3:
            result[eager_outputs[eager_pos - 1][0]] = cpu_outputs[cpu_pos - 1][0]
            cpu_pos -= 1
            eager_pos -= 1
        elif cpu_pos and (not eager_pos or direction == 1):
            cpu_pos -= 1
        else:
            eager_pos -= 1
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
                "original_aten_types": node.get("original_aten_types", []),
                "origin_subgraph": node.get("origin_subgraph"),
                "unfused_reasons": node.get("unfused_reasons", []),
                "provenance_conflicts": node.get("provenance_conflicts", []),
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
def _cann_metrics(
    golden: np.ndarray,
    actual: np.ndarray,
    *,
    golden_float64: np.ndarray | None = None,
    actual_float64: np.ndarray | None = None,
    difference: np.ndarray | None = None,
) -> dict[str, Any]:
    if not np.issubdtype(golden.dtype, np.floating) and not np.issubdtype(golden.dtype, np.complexfloating):
        passed = bool(np.array_equal(golden, actual))
        return {
            "cann_cosine_similarity": None,
            "relative_euclidean_distance": None,
            "kl_divergence": None,
            "rmse_mae_ratio": None,
            "cann_passed": passed,
        }

    x = golden_float64 if golden_float64 is not None else np.asarray(golden).astype(np.float64).reshape(-1)
    y = actual_float64 if actual_float64 is not None else np.asarray(actual).astype(np.float64).reshape(-1)
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
        cosine = min(float(np.dot(x, y) / (x_norm * y_norm)), 1.0)

    diff = difference if difference is not None else x - y
    relative_euclidean = (
        0.0
        if y_norm <= CANN_EPSILON
        else float(np.sqrt(np.dot(diff, diff)) / y_norm)
    )

    x_min = float(np.min(x))
    y_min = float(np.min(y))
    x_range = float(np.max(x) - x_min)
    y_range = float(np.max(y) - y_min)
    x_normalized = (x - x_min) / x_range if x_range > CANN_EPSILON else x
    y_normalized = (y - y_min) / y_range if y_range > CANN_EPSILON else y
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

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.dot(diff, diff) / diff.size))
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


def _encode_tensor_array(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    original_dtype = str(array.dtype)
    # Browsers do not consistently expose Float16Array, so use float32 for
    # display while keeping the logical/storage dtype in the surrounding metadata.
    if array.dtype == np.dtype("float16"):
        array = array.astype(np.float32)
    if not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        return {
            "encoding": "json",
            "dtype": original_dtype,
            "shape": list(array.shape),
            "values": array.reshape(-1).tolist(),
        }
    return {
        "encoding": "base64",
        "dtype": str(array.dtype),
        "logical_dtype": original_dtype,
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _elementwise_payload(
    golden: np.ndarray,
    actual: np.ndarray,
    *,
    rtol: float,
    atol: float,
    logical_dtype: str,
    close: np.ndarray | None = None,
    encoded_golden: dict[str, Any] | None = None,
    encoded_actual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if close is None:
        close = np.isclose(
            golden.astype(np.float64), actual.astype(np.float64), rtol=rtol, atol=atol, equal_nan=False
        )
    else:
        close = close.reshape(golden.shape)
    return {
        "shape": list(golden.shape),
        "logical_dtype": logical_dtype,
        "cpu": encoded_golden or _encode_tensor_array(golden),
        "npu": encoded_actual or _encode_tensor_array(actual),
        "strict_mask": _encode_tensor_array(close.astype(np.uint8)),
    }


def _nonfinite_metrics(
    golden: np.ndarray,
    actual: np.ndarray,
    *,
    golden_float64: np.ndarray | None = None,
    actual_float64: np.ndarray | None = None,
) -> dict[str, Any]:
    gf = golden_float64 if golden_float64 is not None else np.asarray(golden).astype(np.float64).reshape(-1)
    af = actual_float64 if actual_float64 is not None else np.asarray(actual).astype(np.float64).reshape(-1)
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


def _metrics_with_close(
    golden: np.ndarray, actual: np.ndarray, *, rtol: float, atol: float
) -> tuple[dict[str, Any], np.ndarray]:
    gf = np.asarray(golden).astype(np.float64).reshape(-1)
    af = np.asarray(actual).astype(np.float64).reshape(-1)
    with np.errstate(invalid="ignore"):
        diff = gf - af
        abs_err = np.abs(diff)
    finite = np.isfinite(gf) & np.isfinite(af)
    if finite.any():
        all_finite = bool(finite.all())
        if all_finite:
            finite_golden = gf
            finite_actual = af
            fe = abs_err
        else:
            finite_golden = gf[finite]
            finite_actual = af[finite]
            fe = abs_err[finite]
        denom = np.maximum(np.maximum(np.abs(finite_golden), np.abs(finite_actual)), 1e-12)
        rel_err = fe / denom
        max_abs = float(fe.max())
        mean_abs = float(fe.mean())
        max_rel = float(rel_err.max())
        rmse = float(np.sqrt(np.dot(fe, fe) / fe.size))
        dot = float(np.dot(finite_golden, finite_actual))
        norm = float(np.linalg.norm(finite_golden) * np.linalg.norm(finite_actual))
        cosine = float(dot / (norm + 1e-12))
        max_flat_index = int(np.argmax(abs_err if all_finite else np.where(finite, abs_err, -np.inf)))
        max_index = tuple(int(i) for i in np.unravel_index(max_flat_index, golden.shape))
    else:
        max_abs = mean_abs = max_rel = rmse = cosine = 0.0
        max_index = ()
    same_nonfinite = (
        (np.isnan(gf) & np.isnan(af))
        | (np.isposinf(gf) & np.isposinf(af))
        | (np.isneginf(gf) & np.isneginf(af))
    )
    close = np.isclose(gf, af, rtol=rtol, atol=atol, equal_nan=False) | same_nonfinite
    strict_passed = bool(close.all())
    nonfinite = (
        {
            "present": False,
            "cpu": {"nan": 0, "pos_inf": 0, "neg_inf": 0},
            "npu": {"nan": 0, "pos_inf": 0, "neg_inf": 0},
            "aligned": {"nan": 0, "pos_inf": 0, "neg_inf": 0},
            "different_positions": 0,
            "first_index": None,
            "first_cpu": None,
            "first_npu": None,
            "summary": "",
        }
        if bool(finite.all())
        else _nonfinite_metrics(golden, actual, golden_float64=gf, actual_float64=af)
    )
    nonfinite_aligned = bool(nonfinite["present"] and nonfinite["different_positions"] == 0)
    if not nonfinite["present"]:
        cann = _cann_metrics(
            golden,
            actual,
            golden_float64=gf,
            actual_float64=af,
            difference=diff,
        )
        cann_unavailable_reason = None
    elif nonfinite_aligned and finite.any():
        cann = _cann_metrics(
            np.asarray(golden).reshape(-1)[finite],
            np.asarray(actual).reshape(-1)[finite],
            golden_float64=gf[finite],
            actual_float64=af[finite],
            difference=diff[finite],
        )
        cann_unavailable_reason = "FINITE_SUBSET_ONLY"
    else:
        cann = {
            "cann_cosine_similarity": None,
            "relative_euclidean_distance": None,
            "kl_divergence": None,
            "rmse_mae_ratio": None,
            "cann_warning": False,
            "cann_passed": bool(nonfinite_aligned and strict_passed),
        }
        cann_unavailable_reason = "ALIGNED_NONFINITE" if nonfinite_aligned else "NONFINITE_MISMATCH"
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
        "nonfinite_aligned": nonfinite_aligned,
        "nonfinite_summary": nonfinite["summary"],
        "first_nonfinite_index": nonfinite["first_index"],
        "first_nonfinite_cpu": nonfinite["first_cpu"],
        "first_nonfinite_npu": nonfinite["first_npu"],
        "cann_unavailable_reason": cann_unavailable_reason,
        "passed": strict_passed,
        "strict_passed": strict_passed,
        **cann,
    }, close


def _metrics(golden: np.ndarray, actual: np.ndarray, *, rtol: float, atol: float) -> dict[str, Any]:
    metrics, _ = _metrics_with_close(golden, actual, rtol=rtol, atol=atol)
    return metrics


def _cached_tensor_payload(
    ref: TensorRef,
    array: np.ndarray,
    cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] | None,
) -> dict[str, Any]:
    if cache is None or ref.path is None:
        return _encode_tensor_array(array)
    key = (str(ref.path), str(array.dtype), tuple(int(dim) for dim in array.shape))
    payload = cache.get(key)
    if payload is None:
        payload = _encode_tensor_array(array)
        cache[key] = payload
    return payload


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
    save_elementwise_data: bool = False,
    tensor_payload_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] | None = None,
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

    metrics, close = _metrics_with_close(golden, actual, rtol=resolved_rtol, atol=resolved_atol)
    result.update(metrics)
    result["shape"] = list(golden.shape)
    result["dtype"] = str(cpu.dtype or npu.dtype or golden.dtype).removeprefix("torch.")
    result["storage_dtype"] = str(golden.dtype)
    if save_elementwise_data:
        result["elementwise_data"] = _elementwise_payload(
            golden,
            actual,
            rtol=resolved_rtol,
            atol=resolved_atol,
            logical_dtype=result["dtype"],
            close=close,
            encoded_golden=_cached_tensor_payload(cpu, golden, tensor_payload_cache),
            encoded_actual=_cached_tensor_payload(npu, actual, tensor_payload_cache),
        )
    if statuses:
        result["status"] = "OK_" + "_".join(statuses)
    return result


def compare_manifests(
    cpu_manifest: str | Path,
    npu_manifest: str | Path,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    save_elementwise_data: bool = False,
    progress_label: str | None = None,
    tensor_payload_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] | None = None,
    origin_aliases: dict[tuple[str, int], tuple[str, int]] | None = None,
) -> dict[str, Any]:
    cpu_data = read_json(cpu_manifest)
    npu_data = read_json(npu_manifest)
    _check_manifest(cpu_data, cpu_manifest)
    _check_manifest(npu_data, npu_manifest)
    cpu_index = _index_manifest(cpu_data)
    npu_index = _index_manifest(npu_data)
    fusions = _fusion_index(npu_data)
    provenance_conflicts = _provenance_conflict_index(npu_data)
    cpu_signatures = dict(_alignment_outputs(cpu_data))
    npu_signatures = dict(_alignment_outputs(npu_data))
    eager_mapping = (
        _eager_topology_mapping(cpu_data, npu_data)
        if cpu_data.get("side") == "cpu" and npu_data.get("side") == "npu_eager"
        else None
    )

    rows: list[dict[str, Any]] = []
    observed_cpu_keys: set[tuple[str, int]] = set()
    sorted_npu_keys = sorted(npu_index)
    for ordinal, npu_key in enumerate(sorted_npu_keys, start=1):
        npu_ref = npu_index[npu_key]
        size_mib = 0.0
        if progress_label and npu_ref.path is not None:
            try:
                size_mib = npu_ref.path.stat().st_size / (1024.0 * 1024.0)
            except OSError:
                pass
        if progress_label and (
            ordinal == 1 or ordinal % 10 == 0 or ordinal == len(sorted_npu_keys) or size_mib >= 64.0
        ):
            progress_message(
                "PROGRESS",
                f"{progress_label} tensor={ordinal}/{len(sorted_npu_keys)} "
                f"op={npu_key[0]} output={npu_key[1]} dump={size_mib:.1f}MiB",
            )
        if npu_ref.path is None:
            rows.append(
                {
                    "cpu_op": "",
                    "cpu_index": 0,
                    "npu_op": npu_key[0],
                    "npu_index": npu_key[1],
                    "npu_type": npu_ref.node_type,
                    "status": "NPU_OUTPUT_NOT_CAPTURED",
                    "unmapped_reason": "NPU 图存在输出描述，但没有对应的 dump 文件",
                    "mapping_method": "none",
                    "observable": True,
                    "passed": False,
                    "compared": False,
                }
            )
            continue
        origin_key = fusions.get(npu_key)
        cpu_key = origin_key
        mapping_method = "compiler_origin" if origin_key is not None else None
        missing_aliased_origin = False
        if origin_key is not None and origin_aliases is not None:
            cpu_key = origin_aliases.get(origin_key)
            missing_aliased_origin = cpu_key is None
            mapping_method = "compiler_origin_via_eager" if cpu_key is not None else None
        if cpu_key is None and eager_mapping is not None and npu_key in eager_mapping:
            cpu_key = eager_mapping[npu_key]
            mapping_method = "same_name" if cpu_key == npu_key else "topology_signature"
        if (
            cpu_key is None
            and eager_mapping is None
            and origin_key is None
            and npu_key in cpu_index
            and cpu_signatures.get(npu_key) == npu_signatures.get(npu_key)
        ):
            cpu_key = npu_key
            mapping_method = "same_name"
        conflict = provenance_conflicts.get(npu_key)
        if cpu_key is not None and str(mapping_method or "").startswith("compiler_origin"):
            conflict_reason = _compiler_origin_conflict_reason(
                cpu_key, npu_key, cpu_signatures, npu_signatures, cpu_index, npu_index
            )
            if conflict_reason is not None:
                conflict = {
                    "candidate": cpu_key[0],
                    "candidate_index": cpu_key[1],
                    "reason": conflict_reason,
                }
        if conflict is not None:
            rows.append(
                {
                    "cpu_op": str(conflict.get("candidate") or (cpu_key[0] if cpu_key else "")),
                    "cpu_index": int(conflict.get("candidate_index", cpu_key[1] if cpu_key else 0)),
                    "npu_op": npu_key[0],
                    "npu_index": npu_key[1],
                    "npu_type": npu_ref.node_type,
                    "status": "PROVENANCE_CONFLICT",
                    "unmapped_reason": str(conflict.get("reason") or "compiler provenance 与输出元数据冲突"),
                    "mapping_method": "provenance_conflict",
                    "observable": True,
                    "passed": False,
                    "compared": False,
                }
            )
            continue
        if cpu_key is None:
            if missing_aliased_origin:
                rows.append(
                    {
                        "cpu_op": origin_key[0],
                        "cpu_index": origin_key[1],
                        "npu_op": npu_key[0],
                        "npu_index": npu_key[1],
                        "npu_type": npu_ref.node_type,
                        "status": "NOT_OBSERVABLE_IN_REFERENCE",
                        "unmapped_reason": "compiler origin 在 NPU eager decomposition 图中没有独立可观测输出",
                        "mapping_method": "reference_internal",
                        "observable": False,
                        "passed": False,
                        "compared": False,
                    }
                )
                continue
            rows.append(
                {
                    "cpu_op": "",
                    "cpu_index": 0,
                    "npu_op": npu_key[0],
                    "npu_index": npu_key[1],
                    "npu_type": npu_ref.node_type,
                    "status": "UNMATCHED_CPU_OP",
                    "unmapped_reason": "NPU 输出已采集，但没有 compiler origin 或语义对齐的基准输出",
                    "mapping_method": "none",
                    "observable": True,
                    "passed": False,
                    "compared": False,
                }
            )
            continue
        cpu_ref = cpu_index.get(cpu_key)
        if cpu_ref is None:
            cpu_ref = TensorRef(cpu_key[0], cpu_key[1], None, None, "")
        row = compare_pair(
            cpu_ref,
            npu_ref,
            rtol=rtol,
            atol=atol,
            save_elementwise_data=save_elementwise_data,
            tensor_payload_cache=tensor_payload_cache,
        )
        row["mapping_method"] = mapping_method
        row["unmapped_reason"] = {
            "compiler_origin": "",
            "compiler_origin_via_eager": "compiler origin 经 CPU↔NPU eager 语义映射",
            "topology_signature": "按拓扑顺序、算子类型、dtype 和完整 shape 语义对齐",
            "same_name": "按同名同输出序号映射",
        }.get(mapping_method, "")
        row["observable"] = True
        rows.append(row)
        observed_cpu_keys.add(cpu_key)

    # A fusion's origin_ops includes CPU nodes whose values are internal to the
    # NPU kernel. Only the mapped fusion output is observable in the current
    # dump workflow, so report other CPU outputs explicitly instead of marking
    # them as ordinary mapping failures.
    reported_internal: set[tuple[str, int]] = set()
    for node in npu_data.get("nodes", []):
        origin_ops = [str(name) for name in node.get("origin_ops", []) if name]
        if len(origin_ops) <= 1:
            continue
        npu_name = str(node.get("name", ""))
        if not npu_name:
            continue
        for output in node.get("outputs", []):
            npu_key = (npu_name, int(output.get("index", 0)))
            origin_key = fusions.get(npu_key)
            npu_ref = npu_index.get(npu_key)
            if origin_key is None or npu_ref is None or npu_ref.path is None:
                continue
            primary_row = next(
                (
                    row
                    for row in rows
                    if row.get("npu_op") == npu_key[0]
                    and int(row.get("npu_index", 0)) == npu_key[1]
                    and row.get("observable", True)
                ),
                None,
            )
            if (
                primary_row is None
                or not primary_row.get("compared", False)
                or not str(primary_row.get("mapping_method", "")).startswith("compiler_origin")
            ):
                continue
            for origin_name in origin_ops:
                original_key = (origin_name, 0)
                cpu_key = origin_aliases.get(original_key) if origin_aliases is not None else original_key
                if cpu_key is None:
                    continue
                if cpu_key not in cpu_index or cpu_key in observed_cpu_keys or cpu_key in reported_internal:
                    continue
                reported_internal.add(cpu_key)
                rows.append(
                    {
                        "cpu_op": origin_name,
                        "cpu_index": 0,
                        "npu_op": npu_key[0],
                        "npu_index": npu_key[1],
                        "npu_type": npu_ref.node_type,
                        "status": "NOT_OBSERVABLE_ON_NPU",
                        "unmapped_reason": "CPU 节点位于 NPU 融合内部，且不是当前 dump 的融合出口",
                        "mapping_method": "fusion_internal",
                        "observable": False,
                        "passed": False,
                        "compared": False,
                    }
                )

    compared = [r for r in rows if r.get("compared", True)]
    unmatched = [r for r in rows if not r.get("compared", True)]
    not_captured = [r for r in unmatched if r.get("status") == "NPU_OUTPUT_NOT_CAPTURED"]
    unmapped = [
        r for r in unmatched if r.get("status") in {"UNMATCHED_CPU_OP", "PROVENANCE_CONFLICT"}
    ]
    not_observable = [
        r
        for r in unmatched
        if r.get("status") in {"NOT_OBSERVABLE_ON_NPU", "NOT_OBSERVABLE_IN_REFERENCE"}
    ]
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
    observable_total = len(rows) - len(not_observable)
    coverage = len(compared) / observable_total if observable_total else 0.0
    incomplete = bool(unmapped or not_captured)
    status = "FAIL" if failed else ("INCOMPLETE" if incomplete else "PASS")
    return {
        "schema": "autofusedebugger-report/1",
        "elementwise_data_saved": save_elementwise_data,
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
            "observable_total": observable_total,
            "compared": len(compared),
            "unmatched": len(unmatched),
            "unmapped": len(unmapped),
            "not_captured": len(not_captured),
            "not_observable": len(not_observable),
            "dumped": observable_total - len(not_captured),
            "coverage": coverage,
            "status": status,
            "passed": passed,
            "failed": len(failed),
            "strict_status": status,
            "strict_passed": passed,
            "strict_failed": len(failed),
            "cann_status": "FAIL" if cann_failed else ("INCOMPLETE" if incomplete else "PASS"),
            "cann_passed": cann_passed,
            "cann_failed": len(cann_failed),
            "dual_status": "FAIL" if dual_failed else ("INCOMPLETE" if incomplete else "PASS"),
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


def compare_three_manifests(
    cpu_manifest: str | Path,
    npu_eager_manifest: str | Path,
    npu_autofuse_manifest: str | Path,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    save_elementwise_data: bool = False,
) -> dict[str, Any]:
    """Compare CPU eager/export, NPU eager, and NPU Autofuse outputs pairwise."""
    tensor_payload_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] | None = (
        {} if save_elementwise_data else None
    )
    cpu_autofuse = compare_manifests(
        cpu_manifest,
        npu_autofuse_manifest,
        rtol=rtol,
        atol=atol,
        save_elementwise_data=save_elementwise_data,
        progress_label="CPU vs NPU Autofuse",
        tensor_payload_cache=tensor_payload_cache,
    )
    cpu_eager = compare_manifests(
        cpu_manifest,
        npu_eager_manifest,
        rtol=rtol,
        atol=atol,
        save_elementwise_data=save_elementwise_data,
        progress_label="CPU vs NPU eager",
        tensor_payload_cache=tensor_payload_cache,
    )
    origin_aliases = {
        (str(row["cpu_op"]), int(row.get("cpu_index", 0))):
        (str(row["npu_op"]), int(row.get("npu_index", 0)))
        for row in cpu_eager.get("results", [])
        if row.get("compared") and row.get("cpu_op") and row.get("npu_op")
    }
    eager_autofuse = compare_manifests(
        npu_eager_manifest,
        npu_autofuse_manifest,
        rtol=rtol,
        atol=atol,
        save_elementwise_data=save_elementwise_data,
        progress_label="NPU eager vs NPU Autofuse",
        tensor_payload_cache=tensor_payload_cache,
        origin_aliases=origin_aliases,
    )
    pairs = {
        "cpu_npu_autofuse": {
            "left": "CPU", "right": "NPU Autofuse", "right_graph_key": "npu", "report": cpu_autofuse,
        },
        "cpu_npu_eager": {
            "left": "CPU", "right": "NPU eager", "right_graph_key": "npu_eager", "report": cpu_eager,
        },
        "npu_eager_npu_autofuse": {
            "left": "NPU eager", "right": "NPU Autofuse", "right_graph_key": "npu", "report": eager_autofuse,
        },
    }
    primary = pairs["cpu_npu_autofuse"]["report"]
    eager = pairs["cpu_npu_eager"]["report"]
    return {
        "schema": "autofusedebugger-report/1",
        "elementwise_data_saved": save_elementwise_data,
        "comparison_mode": "three_way",
        "comparison_order": list(pairs),
        "comparisons": {
            key: {
                "left": value["left"],
                "right": value["right"],
                "right_graph_key": value["right_graph_key"],
                "summary": value["report"]["summary"],
                "results": value["report"]["results"],
                "graphs": value["report"]["graphs"],
            }
            for key, value in pairs.items()
        },
        "rtol": rtol,
        "atol": atol,
        "tolerance_mode": primary.get("tolerance_mode"),
        "default_dtype_tolerances": primary.get("default_dtype_tolerances", {}),
        "cann_thresholds": primary.get("cann_thresholds", {}),
        "summary": primary["summary"],
        "graphs": {
            "cpu": primary["graphs"].get("cpu", {}),
            "npu_eager": eager["graphs"].get("npu", {}),
            "npu": primary["graphs"].get("npu", {}),
        },
        # Keep the existing top-level fields backward-compatible. Existing
        # report consumers continue to see CPU vs NPU Autofuse by default.
        "results": primary["results"],
    }


def write_report_csv(path: str | Path, report: dict[str, Any]) -> None:
    fieldnames = [
        "compared",
        "passed",
        "strict_passed",
        "cann_passed",
        "status",
        "mapping_method",
        "unmapped_reason",
        "observable",
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
        "storage_dtype",
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
    observable_total = summary.get("observable_total", summary["total"])
    lines = [
        "accuracy: "
        f"{summary.get('status', 'PASS' if summary['failed'] == 0 else 'FAIL')} "
        f"strict={summary['passed']}/{summary.get('compared', summary['total'])} "
        f"cann={summary.get('cann_passed', summary['passed'])}/{summary.get('compared', summary['total'])} "
        f"coverage={summary.get('compared', summary['total'])}/{observable_total} "
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
    if summary.get("not_observable", 0):
        lines.append(f"  not_observable={summary['not_observable']} (non-independent outputs excluded from coverage)")
    return "\n".join(lines)
