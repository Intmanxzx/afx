"""AFIR compiler failures proven by native diagnostics in the failing attempt."""
import re

from autofusex.execution.module_attribution import classify_module


# AFIR.h defines these stable verifier categories. emitOpError additionally
# prints '<operation>' op; LoopGenerate uses emitError with a category and op.
_CATEGORY = (r'UnsupportedOperation|DescriptorTypeMismatch|InvalidDescriptor|'
             r'DescriptorCountMismatch|InvalidBinding|InvalidLifetime|InvalidSchedule|DataRace')
_DIAGNOSTIC = re.compile(
    r"^\s*(?:loc\([^\n]*\):\s*|[^\s\n][^\n]*:\d+:\d+:\s*)?error:\s*(?:"
    r"'afir\.[A-Za-z_][A-Za-z0-9_]*' op\s+\S[^\n]*|"
    r"(?:" + _CATEGORY + r"):\s*'afir\.[A-Za-z_][A-Za-z0-9_]*'\s+\S[^\n]*)$", re.M)
_CONFLICT = re.compile(
    r'^(?:[\w.]*Error|[\w.]*Exception):[^\n]*(?:'
    r'out of memory|no space left|timed out|device busy|'
    r'aicpu execute failed|AICPU execution|AI CPU kernel|'
    r'materialized Torch graph differs from declared NNSmith output signature)|'
    r'^triton\.compiler\.errors\.', re.M | re.I)


def compile_evidence(text, backend):
    module = classify_module(text, backend)
    if module and module[:2] not in {('compile_fail', 'unknown'), ('compile_fail', 'mlir')}:
        return False
    return bool(_DIAGNOSTIC.search(text)) and not _CONFLICT.search(text)


def mlir_failure(backend, summary=None, detail=None):
    """Require every failed shape, or a shape-less worker failure, to prove AFIR."""
    if backend not in {'af-mlir', 'af-legacy'}:
        return None
    from autofusex.server.failure_attribution import _object
    d = _object(detail)
    execution = _object(_object(d.get('contexts')).get('execution') or d.get('execution'))
    diagnostics = _object(execution.get('diagnostics'))
    shapes = _object(diagnostics.get('shapes'))
    results = d.get('shape_results', [])
    summaries = {str(s.get('shape_index')): s.get('error_summary') or ''
                 for s in results if isinstance(s, dict) and s.get('correctness') == 'error'}
    indices = set(summaries) | {str(k) for k, v in shapes.items() if _object(v).get('error')}
    if indices:
        texts = [summaries.get(i, '') + '\n' + str(_object(shapes.get(i)).get('error') or '')
                 for i in indices]
    elif shapes or results:
        return None
    else:
        # A worker can die before producing shape results. Its saved exception
        # contains the native stderr tail; unrelated run logs remain excluded.
        text = (summary or '') + '\n' + str(d.get('error_summary') or '')
        if d.get('error_stage') == 'worker':
            text += '\n' + str(diagnostics.get('error') or '')
        texts = [text]
    if all(compile_evidence(text, backend) for text in texts):
        return 'compile_fail', 'mlir', 'autofuse'
    return None
