"""Server-authoritative failure attribution.

The client records where an exception happened; this module decides what the
failure means from the immutable evidence shipped with an execution.
"""
import json
import re

VERSION = 1

_TIMEOUT = re.compile(r"\b(?:timeout|timed ?out|TimeoutError)\b", re.I)
_ENV = re.compile(r"device busy|out of memory|no space left|npu unavailable|torch_npu unavailable|another managed test", re.I)
_TORCH = re.compile(r"UnsupportedIndexExpr|unknown SymPy node|torch\._dynamo\.exc|torch\._inductor\.exc|ConstraintViolationError|GuardOnDataDependentSymNode", re.I)
_GRAPH = re.compile(r"Check (?:dtype|shape) failed|Infer dtype failed|infer shape failed", re.I)
_SCHEDULE = re.compile(r"schedule|autoschedule|optimize module", re.I)
_LOWER = re.compile(r"lower(?:ing)?", re.I)
_TILING = re.compile(r"tiling", re.I)
_CODEGEN = re.compile(r"codegen|generate kernel", re.I)
# Errors raised by an NPU operator during the first execution are runtime
# failures.  In particular, generated Inductor source often includes a
# ``torch/_inductor/lowering.py`` warning; that filename is not evidence that
# lowering itself failed.
_NPU_RUNTIME = re.compile(
    r"NPU function error|aclnn[A-Za-z0-9_]* failed|op[_ -]?plugin|"
    r"Tensor self not implemented for DT_|AclNN_Parameter_Error", re.I)

def _detail_text(detail):
    if not detail:
        return ""
    if isinstance(detail, (bytes, bytearray)):
        try: detail = json.loads(detail)
        except Exception: return detail.decode(errors="ignore")
    if not isinstance(detail, dict):
        return str(detail)
    return json.dumps(detail, ensure_ascii=False, default=str)

def _detail_stage(detail):
    if isinstance(detail, (bytes, bytearray)):
        try: detail = json.loads(detail)
        except Exception: return None
    if not isinstance(detail, dict): return None
    for key in ("error_stage", "execution_stage"):
        if isinstance(detail.get(key), str): return detail[key]
    for key in ("attempt", "execution", "result"):
        value = detail.get(key)
        if isinstance(value, dict):
            found = _detail_stage(value)
            if found: return found
    for value in detail.get("shape_results", ()) if isinstance(detail.get("shape_results"), list) else ():
        if isinstance(value, dict) and isinstance(value.get("error_stage"), str): return value["error_stage"]
    return None

def classify(*, backend, declared_status, declared_stage=None, summary=None, log=None, detail=None):
    """Return (function_status, fail_stage, owner, version)."""
    evidence = "\n".join(str(x) for x in (summary, log, _detail_text(detail)) if x)
    execution_stage = _detail_stage(detail) or declared_stage
    if declared_status in ("ok", "fail", "pass") and not evidence:
        return "pass", None, None, VERSION
    if declared_status in ("ok", "fail", "pass"):
        # A correctness failure is represented separately by accuracy_status.
        return "pass", None, None, VERSION
    if declared_status in {"build_fail", "compile_fail", "runtime_fail", "timeout", "unsupported"} and not (
            _TIMEOUT.search(evidence) or _ENV.search(evidence) or _TORCH.search(evidence) or
            _GRAPH.search(evidence) or _LOWER.search(evidence) or _SCHEDULE.search(evidence) or
            _TILING.search(evidence) or _CODEGEN.search(evidence)):
        # Legacy uploads may not contain detail/log evidence. Preserve their
        # explicit value until a historical reclassification has evidence.
        return declared_status, declared_stage, "unknown", VERSION
    if _TIMEOUT.search(evidence):
        return "timeout", "timeout", "infrastructure", VERSION
    if _ENV.search(evidence):
        return "runtime_fail", "environment", "infrastructure", VERSION
    if _TORCH.search(evidence):
        return "build_fail", "torch", "torch", VERSION
    # Prefer the recorded execution stage and concrete NPU runtime evidence
    # over incidental warning text such as ``torch/_inductor/lowering.py``.
    if execution_stage in {"first_call_combined", "kernel", "synchronize"} and _NPU_RUNTIME.search(evidence):
        return "runtime_fail", execution_stage, "npu_runtime", VERSION
    if backend in {"af-mlir", "af-legacy"}:
        if _GRAPH.search(evidence):
            return "compile_fail", "graph_check", "autofuse_ascir", VERSION
        if _LOWER.search(evidence):
            return "compile_fail", "lowering", "autofuse", VERSION
        if _SCHEDULE.search(evidence):
            return "compile_fail", "schedule", "autofuse", VERSION
        if _TILING.search(evidence):
            return "compile_fail", "tiling", "autofuse", VERSION
        if _CODEGEN.search(evidence):
            return "compile_fail", "codegen", "autofuse", VERSION
    if execution_stage in {"first_call_combined", "kernel", "synchronize"}:
        return "runtime_fail", execution_stage, "npu_runtime", VERSION
    return "runtime_fail", execution_stage or "unknown", "unknown", VERSION
