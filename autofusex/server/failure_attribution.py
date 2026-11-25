"""Server-authoritative failure attribution.

The client records where an exception happened; this module decides what the
failure means from the immutable evidence shipped with an execution.
"""
import json
import re

VERSION = 2

_TIMEOUT = re.compile(r"\b(?:timeout|timed ?out|TimeoutError)\b", re.I)
_ENV = re.compile(r"device busy|out of memory|no space left|npu unavailable|torch_npu unavailable|another managed test", re.I)
_TORCH = re.compile(r"UnsupportedIndexExpr|unknown SymPy node|torch\._dynamo\.exc|torch\._inductor\.exc|ConstraintViolationError|GuardOnDataDependentSymNode", re.I)
_GRAPH = re.compile(r"Check (?:dtype|shape) failed|Infer dtype failed|infer shape failed", re.I)

def _detail_text(detail):
    if not detail:
        return ""
    if isinstance(detail, (bytes, bytearray)):
        try: detail = json.loads(detail)
        except Exception: return detail.decode(errors="ignore")
    if isinstance(detail, str):
        try: return _detail_text(json.loads(detail))
        except (ValueError, TypeError): return detail
    if isinstance(detail, list):
        return "\n".join(_detail_text(value) for value in detail)
    if not isinstance(detail, dict):
        return str(detail)
    return "\n".join(_detail_text(value) for value in detail.values())

def classify(*, backend, declared_status, declared_stage=None, summary=None, log=None, detail=None):
    """Return (function_status, fail_stage, owner, version)."""
    log = log.decode(errors="replace") if isinstance(log, bytes) else log
    evidence = "\n".join(str(x) for x in (summary, log, _detail_text(detail)) if x)
    if declared_status in ("ok", "fail", "pass"):
        # A correctness failure is represented separately by accuracy_status.
        return "pass", None, None, VERSION
    from autofusex.execution.module_attribution import classify_module
    # Module markers identify the actual failing frame. Otherwise only the
    # terminal summary may prove aclNN ownership; shared logs can contain
    # unrelated failures from another shape or graph.
    module = classify_module(evidence if 'AFX_MODULE_V1=' in evidence else (summary or ''), backend, declared_stage)
    if module:
        return (*module, VERSION)
    if declared_status == "unsupported":
        return "unsupported", None, None, VERSION
    if declared_status == 'timeout' or _TIMEOUT.search(summary or ""):
        return "timeout", "unknown", "infrastructure", VERSION
    if _ENV.search(summary or ""):
        return "runtime_fail", "environment", "infrastructure", VERSION
    # Concrete native errors take precedence over the outer InductorError wrapper.
    if backend in {"af-mlir", "af-legacy"}:
        from autofusex.execution.reporting.failure_origin import mlir_failed_line
        if mlir_failed_line(evidence):
            return "compile_fail", "mlir", "autofuse", VERSION
        if re.search(r"(?:ERROR|error:)[^\n]*MLIR codegen:", evidence):
            return "compile_fail", "mlir", "autofuse", VERSION
        if re.search(r"(?:Schedule|Optimize|optimize module)[^\n]*failed", summary or ""):
            return "compile_fail", "schedule", "autofuse", VERSION
        if _GRAPH.search("\n".join(x for x in (summary, log) if x)):
            # Old graph-check diagnostics do not identify whether validation was
            # in the frontend or inside schedule. Do not invent that boundary.
            return "compile_fail", "af_common", "autofuse", VERSION
    if _TORCH.search(summary or "") and 'InductorError' not in (summary or ""):
        return "gen_fail", "inductor", "torch", VERSION
    if declared_status == "build_fail":
        return "gen_fail", "inductor" if declared_stage == "torch" else "unknown", "unknown", VERSION
    if declared_status in {"gen_fail", "ascgraph_fail", "compile_fail", "runtime_fail", "timeout"}:
        stages = {"inductor", "af_fronted", "schedule", "legacy_codegen", "mlir",
                  "triton_ascend", "aclnn", "fused_kernel", "af_common", "environment", "unknown"}
        stage = declared_stage if declared_stage in stages else "unknown"
        return declared_status, stage, "unknown", VERSION
    return "runtime_fail", "unknown", "unknown", VERSION
