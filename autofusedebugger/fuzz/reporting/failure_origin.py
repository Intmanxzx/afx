"""Evidence-based attribution, shared by historical SQL rollups and detail rows.

An InductorError is a wrapper, not evidence of Torch ownership. Likewise an AF
import, generated source, or a successful strict gate is not an AF failure.
Missing/truncated evidence stays unknown; immutable result records are untouched.
"""
import json
import re

VERSION = 3  # Included in the disposable public-statistics cache identity.
AF_BACKENDS = {'af-mlir', 'af-legacy'}
TORCH_CAUSES = re.compile(
    r'UnsupportedIndexExpr|Unsupported index_expr|unknown SymPy node|'
    r'(?:torch\._dynamo\.exc\.(?:Unsupported|UserError|TorchRuntimeError))|'
    r'(?:torch\._inductor\.exc\.(?:LoweringException|MissingOperatorWithoutDecomp|MissingOperatorWithDecomp))|'
    r'ConstraintViolationError|GuardOnDataDependentSymNode|'
    r'(?:torch\._dynamo\.exc\.)?InternalTorchDynamoError', re.I)


def af_failure(evidence):
    lowered = evidence.lower()
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
    if (stage == 'prepare' or backend in {'cpu', 'cpu-eager'}) and re.search(
            r'(?:RuntimeError|TypeError|ValueError|NotImplementedError):', evidence) and not re.search(
                r'autofuse|autofuser|ascir', evidence, re.I):
        return 'gen_failed'
    return 'unknown'


def sql_status(backend, stage, summary, diagnostics, index):
    category = classify(backend, 'error', stage or '', summary or '', diagnostics, index)
    return 'error' if category == 'function_failed' else category
