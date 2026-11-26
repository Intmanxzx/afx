"""Conservative triage of legacy failures whose module cannot be established.

This is an offline historical-data operation, not a client classification rule.
It preserves status and original evidence, and records reversible stage changes.
"""
import json
import re

from autofusex.execution.module_attribution import classify_module
from autofusex.execution.reporting.failure_origin import TORCH_CAUSES
from autofusex.server import aggregates
from autofusex.server.failure_attribution import frontend_failure
from autofusex.server.triton_attribution import triton_failure
from autofusex.server.native_runtime_attribution import native_runtime_failure
from autofusex.server.local_failure_attribution import local_failure
from autofusex.server.mlir_attribution import mlir_failure
from autofusex.server.reclassification import FIELDS
from autofusex.server.store import get_content

STAGE = "unknown_retest"
_FAILURES = {"build_fail", "gen_fail", "ascgraph_fail", "compile_fail", "runtime_fail"}
_ENV = re.compile(
    r"built without the MLIR bridge|Autofuse compile runtime is incomplete|"
    r"No module named|missing compile APIs|device busy|out of memory|no space left|"
    r"npu unavailable|torch_npu unavailable|another managed test|PermissionError|"
    r"NPU function error: aclInit", re.I)


def _object(value):
    if isinstance(value, (str, bytes)):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def _known_terminal(text, backend):
    """Require local failure evidence; source listings/shared logs are not proof."""
    module = classify_module(text, backend)
    if module and module[1] != 'unknown':
        return True
    if _ENV.search(text):
        return True
    if TORCH_CAUSES.search(text) or 'aot_autograd expected to have an entirely functional graph' in text:
        return True
    if "materialized Torch graph differs from declared NNSmith output signature" in text:
        return True  # Known platform validation, not an unresolved AF module.
    if "Only c10::MemoryFormat::Contiguous is supported" in text:
        return True  # Known tensor-creation runtime failure.
    if re.search(r"(?:Schedule|Optimize|optimize module)[^\n]*failed", text):
        return True
    if re.search(r"MLIR\s+Codegen\s+Result\s*[:：]\s*kFailed\b|(?:ERROR|error:)[^\n]*MLIR codegen:", text):
        return True
    if backend == "triton-npu" and re.search(
            r"Failed to run BiSheng|triton\.compiler\.errors\.CompilationError", text):
        return True
    frames = re.findall(r'File "([^"]+)", line \d+', text)
    return bool(frames and re.search(r"/torch/(?:_dynamo|_inductor|fx|_export)/", frames[-1]))


def _scoped_mlir_failure(local, shared):
    # Historical strict gates carry graph identity. Only a failed gate for the
    # graph in this shape's codegen error can establish MLIR ownership.
    if 'Codegen generate kernel failed' not in local:
        return False
    graphs = set(re.findall(r'autofused_[A-Za-z0-9_]+', local))
    for line in shared.splitlines():
        match = re.search(r'AF_MLIR_STRICT gate=(?:pipeline-session|lowering|artifact-assembly|exception) '
                          r'shortfall=(?:yes|no|unknown) graph=(autofused_[A-Za-z0-9_]+)', line)
        if match and match[1] in graphs:
            return True
    return False


def reason(row, detail):
    """Return a triage reason, or None when this record should stay untouched."""
    if row["capture_version"] >= 2 or row["function_status"] not in _FAILURES:
        return None
    if row["fail_stage"] in {STAGE, "environment"}:
        return None
    d = _object(detail)
    if frontend_failure(row['backend'], row['log_summary'], d):
        return None
    if triton_failure(row['backend'], row['log_summary'], d):
        return None
    if native_runtime_failure(d, row['backend']):
        return None
    if local_failure(d, row['backend']):
        return None
    if mlir_failure(row['backend'], row['log_summary'], d):
        return None
    contexts = _object(d.get("contexts"))
    execution = _object(contexts.get("execution") or d.get("execution"))
    diagnostics = _object(execution.get("diagnostics"))
    shapes = _object(diagnostics.get("shapes"))
    summaries = {str(s.get("shape_index")): s.get("error_summary") or ""
                 for s in d.get("shape_results", []) if isinstance(s, dict)
                 and s.get("correctness") == "error"}
    local = []
    for index in sorted(set(shapes) | set(summaries)):
        error = _object(shapes.get(index)).get("error") or ""
        text = summaries.get(index, "") + "\n" + str(error)
        if text.strip():
            local.append(text)
    if local:
        shared = str(diagnostics.get('error') or '')
        summary = row['log_summary'] or ''
        if (TORCH_CAUSES.search(summary) or 'LoweringException:' in summary):
            if not any(re.search(r'Check dtype failed|Infer dtype failed|Codegen generate kernel failed|AFX_MODULE_V1=|call aclnn', text) for text in local):
                return None
        if _ENV.search(summary):
            return None
        if all(_known_terminal(text, row["backend"]) or
               (row['backend'] == 'af-mlir' and _scoped_mlir_failure(text, shared)) for text in local):
            return None
        # Some shapes may be known, but a different unresolved shape still needs replay.
        return "legacy_shape_module_unresolved"
    summary = row["log_summary"] or ""
    # Run-level diagnostics are valid for preflight checks when no shape ran.
    preflight = summary + "\n" + str(diagnostics.get("error") or "")
    if _ENV.search(preflight) or d.get("error_stage") == "capability":
        return None
    if _known_terminal(summary, row["backend"]):
        return None
    return "legacy_failure_evidence_missing"


def plan(conn):
    """Build a reviewable manifest; latest heads are identified for replay export."""
    changes = []
    rows = conn.execute(
        "SELECT e.*, c.case_key, (h.exec_id IS NOT NULL) AS is_head "
        "FROM executions e JOIN cases c ON c.case_id=e.case_id LEFT JOIN execution_heads h "
        "ON h.case_id=e.case_id AND h.backend=e.backend AND h.npu_arch=e.npu_arch AND h.exec_id=e.exec_id "
        "WHERE e.capture_version<2 "
        "AND e.function_status IN ('build_fail','gen_fail','ascgraph_fail','compile_fail','runtime_fail') "
        "AND COALESCE(e.fail_stage,'') NOT IN ('unknown_retest','environment') ORDER BY e.exec_id")
    for row in rows:
        why = reason(row, get_content(conn, row["detail_content_id"]) if row["detail_content_id"] else None)
        if why is None:
            continue
        before = [row[k] for k in FIELDS]
        after = [row["function_status"], STAGE, "unknown", row["classifier_version"]]
        changes.append({"exec_id": row["exec_id"], "exec_uuid": row["exec_uuid"],
                        "case_key": row["case_key"], "backend": row["backend"], "npu_arch": row["npu_arch"],
                        "is_head": bool(row["is_head"]), "reason": why, "before": before, "after": after})
    return changes


def apply(store, batch_id, changes):
    """Apply exactly the reviewed manifest, refusing stale records atomically."""
    with store.write() as conn:
        if conn.execute("SELECT 1 FROM classification_changes WHERE batch_id=?", (batch_id,)).fetchone():
            raise ValueError("batch already exists")
        for change in changes:
            row = conn.execute("SELECT * FROM executions WHERE exec_id=?", (change["exec_id"],)).fetchone()
            if (row is None or row["exec_uuid"] != change["exec_uuid"]
                    or [row[k] for k in FIELDS] != change["before"]):
                raise ValueError("record changed after preview; rebuild manifest")
            if (row["capture_version"] >= 2 or row["function_status"] not in _FAILURES
                    or change["after"] != [row["function_status"], STAGE, "unknown", row["classifier_version"]]):
                raise ValueError("invalid historical retest marker")
            detail = get_content(conn, row["detail_content_id"]) if row["detail_content_id"] else None
            if reason(row, detail) != change["reason"]:
                raise ValueError("triage evidence changed after preview")
            conn.execute("INSERT INTO classification_changes VALUES(?,?,?,?)",
                         (batch_id, row["exec_id"], json.dumps(change["before"]), json.dumps(change["after"])))
            conn.execute("UPDATE executions SET function_status=?,fail_stage=?,failure_owner=?,classifier_version=? WHERE exec_id=?",
                         (*change["after"], row["exec_id"]))
        if changes:
            from autofusex.server.issues import refresh_classified_executions
            refresh_classified_executions(conn, [c['exec_id'] for c in changes])
            aggregates.rebuild(conn, store.setting(conn, "perf.tie_threshold"))
            aggregates.advance_seq(conn)
    return len(changes)
