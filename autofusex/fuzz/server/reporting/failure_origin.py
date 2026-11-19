"""Evidence-based attribution, shared by historical SQL rollups and detail rows.

An InductorError is a wrapper, not evidence of Torch ownership. Likewise an AF
import, generated source, or a successful strict gate is not an AF failure.
Missing/truncated evidence stays unknown; immutable result records are untouched.
"""
import json
import re

VERSION = 8  # Included in the disposable public-statistics cache identity.
AF_BACKENDS = {'af-mlir', 'af-legacy'}
TORCH_CAUSES = re.compile(
    r'UnsupportedIndexExpr|Unsupported index_expr|unknown SymPy node|'
    r'(?:torch\._dynamo\.exc\.(?:Unsupported|UserError|TorchRuntimeError))|'
    r'(?:torch\._inductor\.exc\.(?:LoweringException|MissingOperatorWithoutDecomp|MissingOperatorWithDecomp))|'
    r'ConstraintViolationError|GuardOnDataDependentSymNode|'
    r'(?:torch\._dynamo\.exc\.)?InternalTorchDynamoError', re.I)


# GELOGI/GELOGD prefixes are intentionally ignored. Both colon forms and whitespace
# variants occur in producer logs; numeric values follow MlirCodegenStatus.
MLIR_RESULT = re.compile(r'\bMLIR\s+Codegen\s+Result\s*[:：]\s*'
                         r'(?:(?:mlir::bridge::)?MlirCodegenStatus::)?'
                         r'(kNotSupported|kFailed|kSucceeded|[012])\b', re.I)
MLIR_RESULT_HEADER = re.compile(r'\bMLIR\s+Codegen\s+Result\s*[:：]', re.I)


def explicit_mlir_results(text):
    results = []
    for line in text.splitlines():
        if 'Failed to execute graph code:' in line:
            continue
        if MLIR_RESULT_HEADER.search(line):
            match = MLIR_RESULT.search(line)
            state = match[1].lower() if match else 'unknown'
            state = {'0': 'knotsupported', '1': 'kfailed', '2': 'ksucceeded'}.get(state, state)
            results.append((state, line))
    return results


MLIR_PIPELINE_FAILURE = re.compile(
    r'\bRunMlirCodegen\b[^\n]*\bfailed\b|'
    r'\bAF_MLIR_STRICT\s+gate=(?:pipeline-session|lowering|artifact-assembly|exception)\s+'
    r'shortfall=(?:yes|no|unknown)\b|'
    r"\berror:\s*['\"][a-zA-Z_]\w*\.\w+['\"]\s+op\b", re.I)


def mlir_failed_line(text):
    for state, line in explicit_mlir_results(text):
        if state == 'kfailed':
            return line
    for line in text.splitlines():
        if 'Failed to execute graph code:' not in line and MLIR_PIPELINE_FAILURE.search(line):
            return line
    return ''


def af_failure(evidence):
    lowered = evidence.lower()
    if mlir_failed_line(evidence):
        return True
    if MLIR_FAILED_GATE.search(evidence):
        return True
    if ('codegen generate kernel failed' in lowered or
            'no plan with an mlir-supported template' in lowered):
        return True
    return any(
        ('af_mlir_strict' in line and 'shortfall=yes' in line) or
        (any(owner in line for owner in ('autofuse', 'ascir', 'runmlircodegen')) and
         any(word in line for word in ('error:', 'failed', 'failure')))
        for line in lowered.splitlines())


def infra_failure(evidence):
    return any(word in evidence.lower() for word in (
        'device busy', 'another managed test', 'timeouterror', 'timed out',
        'out of memory', 'no space left', 'npu unavailable', 'torch_npu unavailable',
        'device-side assert'))


def torch_failure(evidence):
    lowered = evidence.lower()
    return any(word in lowered for word in (
        'unsupported', 'sympy', 'torch._', 'constraintviolationerror',
        'guardondatadependentsymnode', 'internaltorchdynamoerror')) and TORCH_CAUSES.search(evidence)


def pre_autofuse_npu_failure(evidence):
    """NPU/op-plugin execution errors before Autofuse or MLIR gets control."""
    lowered = evidence.lower()
    return (('npu function error' in lowered or 'aclnn' in lowered or 'op_plugin' in lowered)
            and not re.search(r'autofuse|ascir|runmlircodegen|af_mlir_strict', lowered))


def classify(backend, status, stage='', summary='', diagnostics=None, index=-1):
    """Return pass/gen_failed/function_failed/unknown for one selected Shape."""
    if status in {'ok', 'fail'}:
        return 'pass'
    if status not in {'error', 'unsupported'}:
        return 'unknown'
    if stage in {'generation', 'reference'}:
        return 'gen_failed'
    if isinstance(diagnostics, str):
        try:
            diagnostics = json.loads(diagnostics)
        except (ValueError, TypeError):
            diagnostics = {}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    local = diagnostics.get('shapes', {}).get(str(index), {}).get('error', '')
    # A concrete cause for this shape outranks logs shared by other shapes.
    evidence = summary + '\n' + local
    if infra_failure(evidence):
        return 'unknown'
    if torch_failure(evidence):
        return 'gen_failed'
    if pre_autofuse_npu_failure(evidence):
        return 'gen_failed'
    # Strip Inductor's source-code wrapper before looking for AF error text.
    cause = re.sub(r'Failed to execute graph code:[^\n]*', '', evidence, flags=re.I)
    if backend in AF_BACKENDS and af_failure(cause):
        return 'function_failed'
    # Generated AF code plus an actual AF graph validation failure is evidence;
    # generated code/imports on their own are deliberately not sufficient.
    if backend in AF_BACKENDS and 'autofuse.pyautofuse' in evidence and re.search(
            r'Check dtype failed|Check shape failed', evidence, re.I):
        return 'function_failed'
    # Terminal traceback frame identifies Torch ownership even when its exception
    # is wrapped as InductorError. AF python/native frames are not Torch proof.
    frames = re.findall(r'File \"([^\"]+)\", line \d+', local)
    if frames and re.search(r'/(?:torch|torch_npu)/(?:_dynamo|_inductor|fx|_export)/', frames[-1]):
        return 'gen_failed'
    shared = diagnostics.get('error', '')
    if infra_failure(shared):
        return 'unknown'
    shared_applies = (index == -1 or len(diagnostics.get('shapes', {})) <= 1 or
                      ('Failed to execute graph code' in evidence and
                       'autofuse.pyautofuse' in evidence))
    shared_cause = re.sub(r'Failed to execute graph code:[^\n]*', '', shared, flags=re.I)
    if shared_applies and backend in AF_BACKENDS and af_failure(shared_cause):
        return 'function_failed'
    if shared_applies and torch_failure(shared):
        return 'gen_failed'
    # Unambiguous eager/CPU or preparation Torch errors, not an AF code wrapper.
    if (stage == 'prepare' or backend in {'cpu'}) and re.search(
            r'(?:RuntimeError|TypeError|ValueError|NotImplementedError):', evidence) and not re.search(
                r'autofuse|autofuser|ascir', evidence, re.I):
        return 'gen_failed'
    return 'unknown'


def sql_status(backend, stage, summary, diagnostics, index):
    category = classify(backend, 'error', stage or '', summary or '', diagnostics, index)
    return 'error' if category == 'function_failed' else category


# These gates return a failed strict MLIR verdict. `none` means success and
# `cube-fusion` is outside this path, not an MLIR codegen failure.
MLIR_FAILED_GATE = re.compile(
    r'\bAF_MLIR_STRICT\s+gate=(?:soc-version|candidate-preparation|'
    r'pipeline-session|lowering|artifact-assembly|exception)\s+'
    r'shortfall=(?:yes|no|unknown)\b', re.I)



def mlir_failure_evidence(backend, status, stage='', summary='', diagnostics=None, index=-1, shape_count=None):
    """A subcategory of function_failed, never inferred from backend selection.

    A failure inside MLIR bounds an AF functional failure to MLIR even when
    stderr is shared by multiple Shapes. Scope preserves the evidence source;
    it does not claim that the log identifies a particular Shape.
    """
    if classify(backend, status, stage, summary, diagnostics, index) != 'function_failed':
        return {'path': 'not_applicable', 'scope': '', 'evidence': ''}
    if isinstance(diagnostics, str):
        try:
            diagnostics = json.loads(diagnostics)
        except (TypeError, ValueError):
            diagnostics = {}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    shapes = diagnostics.get('shapes', {})
    local = summary + '\n' + shapes.get(str(index), {}).get('error', '')
    sources = [('shape', local)]
    # An explicit AF graph-validation failure is not moved into MLIR by stderr
    # from another subgraph in the same invocation.
    if not re.search(r'Check dtype failed|Check shape failed', local, re.I):
        sources.append(('run', diagnostics.get('error', '')))
    from ...client.failure_details import redact
    # Any concrete failure inside MLIR is sufficient, regardless of later
    # successful codegen calls in the same run.
    for scope, text in sources:
        line = mlir_failed_line(text)
        if line:
            return {'path': 'mlir_failed', 'scope': scope,
                    'evidence': redact(line.strip())[:1600]}
    explicit = [(scope, state, line) for scope, text in sources
                for state, line in explicit_mlir_results(text)]
    if explicit:
        failed = next(((scope, line) for scope, state, line in explicit if state == 'kfailed'), None)
        scope, line = failed or (explicit[0][0], explicit[0][2])
        return {'path': 'mlir_failed' if failed else 'unconfirmed',
                'scope': scope, 'evidence': redact(line.strip())[:1600]}
    return {'path': 'unconfirmed', 'scope': '', 'evidence': ''}


def failure_path(backend, status, stage='', summary='', diagnostics=None, index=-1, shape_count=None):
    return mlir_failure_evidence(backend, status, stage, summary, diagnostics, index, shape_count)['path']
