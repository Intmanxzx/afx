"""Server-authoritative failure attribution.

The client records where an exception happened; this module decides what the
failure means from the immutable evidence shipped with an execution.
"""
import json
import re

VERSION = 10

_NNSMITH_SIGNATURE = 'ValueError: materialized Torch graph differs from declared NNSmith output signature'

_TIMEOUT = re.compile(r"\b(?:timeout|timed ?out|TimeoutError)\b", re.I)
_ENV = re.compile(r"device busy|out of memory|no space left|npu unavailable|torch_npu unavailable|another managed test", re.I)
_TORCH = re.compile(r"UnsupportedIndexExpr|unknown SymPy node|torch\._dynamo\.exc|torch\._inductor\.exc|ConstraintViolationError|GuardOnDataDependentSymNode", re.I)
_GRAPH = re.compile(r"Check (?:dtype|shape) failed|Infer dtype failed|infer shape failed", re.I)
_MLIR_RUNTIME = re.compile(r'UnsupportedBackend:\s*af-mlir runtime verification failed:')
_COMPILE_ENV = re.compile(
    r'UnsupportedBackend:\s*Autofuse compile runtime is incomplete|'
    r'ValueError:\s*missing compile APIs:\s*build_\w+|'
    r"(?:ModuleNotFoundError|ImportError):\s*No module named ['\"]"
    r"(?:autofuse(?:\.[\w.]+)?|asc_op_compile_base|tbe(?:\.[\w.]+)?)['\"]|"
    r"No module named ['\"]autofuse\.(?:pyautofuse|compile_adapter)['\"]", re.I)


def mlir_environment_failure(backend, summary, detail):
    """Runtime capability checks fail before compiling the generated graph."""
    if backend != 'af-mlir':
        return False
    return _local_failure_matches(summary, detail, _MLIR_RUNTIME.search)


def _local_failure_matches(summary, detail, matches):
    """Require the same cause in every failing shape; exclude shared logs."""
    d = _object(detail)
    execution = _object(_object(d.get('contexts')).get('execution') or d.get('execution'))
    shapes = _object(_object(execution.get('diagnostics')).get('shapes'))
    summaries = {str(s.get('shape_index')): s.get('error_summary') or ''
                 for s in d.get('shape_results', []) if isinstance(s, dict)
                 and s.get('correctness') == 'error'}
    indices = set(summaries) | {str(k) for k, v in shapes.items() if _object(v).get('error')}
    if indices:
        return all(matches(summaries.get(index, '') + '\n' +
                   str(_object(shapes.get(index)).get('error') or '')) for index in indices)
    return bool(matches(summary or '') or matches(str(d.get('error_summary') or '')))


def _object(value):
    if isinstance(value, (str, bytes, bytearray)):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def nnsmith_failure(summary=None, detail=None):
    """Attribute declared-output contract failures without using shared logs."""
    from autofusex.execution.module_attribution import classify_module
    d = _object(detail)
    contexts = _object(d.get('contexts'))
    execution = _object(contexts.get('execution') or d.get('execution'))
    diagnostics = _object(execution.get('diagnostics'))
    shapes = _object(diagnostics.get('shapes'))
    summaries = {str(s.get('shape_index')): s.get('error_summary') or ''
                 for s in d.get('shape_results', []) if isinstance(s, dict)
                 and s.get('correctness') == 'error'}
    indices = set(summaries) | {str(k) for k, v in shapes.items() if _object(v).get('error')}
    if not indices:
        return ('gen_fail', 'nnsmith', 'nnsmith') if (summary or '').strip() == _NNSMITH_SIGNATURE else None
    for index in indices:
        error = str(_object(shapes.get(index)).get('error') or '')
        terminal = summaries.get(index, '').strip()
        # A traceback includes the raise statement too; require the terminal
        # ValueError line, not a string literal or session-level mention.
        if not (terminal == _NNSMITH_SIGNATURE or
                re.search(r'^' + re.escape(_NNSMITH_SIGNATURE) + r'\s*$', error, re.M)):
            return None
        if classify_module(terminal + '\n' + error, ''):
            return None
    return 'gen_fail', 'nnsmith', 'nnsmith'


def local_aclnn_failure(detail, backend):
    """Prove aclNN failure from every failing shape, excluding shared logs.

    Native plog summaries often name an aclNN operator without including the
    actual Python exception. Retain the shape boundary when reading that cause.
    Mixed or incomplete shape evidence cannot establish execution-wide ownership.
    """
    from autofusex.execution.module_attribution import classify_module
    d = _object(detail)
    contexts = _object(d.get('contexts'))
    execution = _object(contexts.get('execution') or d.get('execution'))
    diagnostics = _object(execution.get('diagnostics'))
    shapes = _object(diagnostics.get('shapes'))
    summaries = {str(s.get('shape_index')): s.get('error_summary') or ''
                 for s in d.get('shape_results', []) if isinstance(s, dict)
                 and s.get('correctness') == 'error'}
    indices = set(summaries) | {str(k) for k, v in shapes.items() if _object(v).get('error')}
    if not indices:
        return None
    for index in indices:
        text = summaries.get(index, '') + '\n' + str(_object(shapes.get(index)).get('error') or '')
        module = classify_module(text, backend)
        if module is None or module[1] != 'aclnn':
            return None
    return 'runtime_fail', 'aclnn', 'npu_runtime'


def frontend_failure(backend, summary=None, detail=None, log=None):
    """Match PyAscIR failures to terminal errors of each shape.

    A source filename in generated code is not proof. Native ERROR records must
    identify the PyAscIR binding and the actual exception.
    """
    if backend not in {'af-legacy', 'af-mlir'}:
        return None
    from autofusex.execution.module_attribution import classify_module
    d = _object(detail)
    contexts = _object(d.get('contexts'))
    execution = _object(contexts.get('execution') or d.get('execution'))
    diagnostics = _object(execution.get('diagnostics'))
    plog = d.get('plog_errors', [])
    if not isinstance(plog, list):
        plog = []
    log = log.decode(errors='replace') if isinstance(log, bytes) else log
    native = '\n'.join(str(x) for x in [summary, log, *plog] if x)
    causes = []
    for line in native.splitlines():
        if not (re.search(r'\[ERROR\]', line) and
                re.search(r'\[[^\]\n]*\bpyascir[\w]*\.(?:cpp|h):\d+\]', line, re.I)):
            continue
        match = _GRAPH.search(line)
        if match:
            cause = line[match.start():].strip()
        else:
            # Native PY_ASSERT wrappers prefix the same error later propagated
            # to Python. Keep its payload, not the timestamp or source path.
            match = re.search(r'ErrorNo:\s*\d+\([^)]*\)\s*(?:Assert\s*)?(.+)', line)
            if not match:
                continue
            cause = match[1].strip()
        if cause.endswith(' failed'):
            cause = cause[:-7]
        if cause:
            causes.append(cause)
    shapes = _object(diagnostics.get('shapes'))
    summaries = {str(s.get('shape_index')): s.get('error_summary') or ''
                 for s in d.get('shape_results', []) if isinstance(s, dict)
                 and s.get('correctness') == 'error'}
    indices = set(summaries) | {str(k) for k, v in shapes.items() if _object(v).get('error')}
    proven = False
    for index in indices:
        text = summaries.get(index, '') + '\n' + str(_object(shapes.get(index)).get('error') or '')
        module = classify_module(text, backend)
        if module and module[1] != 'af_fronted':
            return None
        terminal = re.search(r'^(?:[\w.]*Error|[\w.]*Exception):[^\n]*\bpyascir\w*\b[^\n]*$',
                             text, re.M | re.I)
        if terminal and 'Failed to execute graph code:' in terminal[0]:
            terminal = None
        if not terminal and not any(cause in text for cause in causes):
            return None
        proven = True
    if not indices:
        # Without per-shape exceptions, only the terminal summary is sufficient.
        if not any(cause in (summary or '') for cause in causes):
            return None
        proven = True
    if not proven:
        return None
    return 'ascgraph_fail', 'af_fronted', 'autofuse'

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
    # A generated schedule frame can wrap this preflight exception. The loaded
    # CANN runtime lacks the required capability; no compiler phase failed.
    if (mlir_environment_failure(backend, summary, detail)
            or _local_failure_matches(summary, detail, _COMPILE_ENV.search)):
        return 'runtime_fail', 'environment', 'infrastructure', VERSION
    from autofusex.execution.module_attribution import torch_guard_failure
    if _local_failure_matches(summary, detail, torch_guard_failure):
        return 'gen_fail', 'inductor', 'torch', VERSION
    if declared_status not in {'unsupported', 'timeout'}:
        generated = nnsmith_failure(summary, detail)
        if generated:
            return (*generated, VERSION)
        local = local_aclnn_failure(detail, backend)
        if local:
            return (*local, VERSION)
        frontend = frontend_failure(backend, summary, detail, log)
        if frontend:
            return (*frontend, VERSION)
    from autofusex.execution.module_attribution import classify_module
    # Module markers identify the actual failing frame. Otherwise only the
    # terminal summary may prove aclNN ownership; shared logs can contain
    # unrelated failures from another shape or graph.
    module = classify_module(evidence if 'AFX_MODULE_V1=' in evidence else (summary or ''), backend)
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
    # Client stage labels are retained as reported facts only. They never
    # establish ownership, including labels emitted by newer clients.
    status = 'gen_fail' if declared_status == 'build_fail' else declared_status
    if status not in {'gen_fail', 'ascgraph_fail', 'compile_fail', 'runtime_fail'}:
        status = 'runtime_fail'
    return status, 'unknown_retest', 'unknown', VERSION
