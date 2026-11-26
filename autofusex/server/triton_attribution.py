"""TritonNPU compile failures proven by each failing shape's exception."""
import re

from autofusex.execution.module_attribution import classify_module

_EXCEPTION = re.compile(r'^triton[.]compiler[.]errors[.](?:MLIRCompilationError|CompilationError):', re.M)
_DTYPE = re.compile(r'^operator .+ does not support dtype .+; supported dtypes on Atlas .+', re.M)
_NATIVE = re.compile(r'^(?:loc\([^\n]*\): error:|\[ERROR\])\s*(?:'
                     r'Failed to run BiSheng(?:HIR|IR) pipeline|'
                     r'Failed to compile BiShengLIR for device|'
                     r'Failed to relocate binary|External hivmc run fails)\b', re.M)
_FRAME = re.compile(r'File "[^"\n]*/triton/(?:compiler/[^"\n]+|backends/ascend/[^"\n]+)", line \d+')
_INFRA = re.compile(r'^(?:[\w.]*Error|[\w.]*Exception):[^\n]*(?:out of memory|no space left|permission denied|timed out|device busy)', re.M | re.I)
_INDUCTOR_ASSERTION = re.compile(r'^AssertionError: n=copy_, n\.args\[0\]=mul_2, placeholders=\{[^\n]*\}, graph=graph\(\):', re.M)


def compile_evidence(text):
    """Return the evidence family; a filename alone never proves ownership."""
    if classify_module(text, 'triton-npu') or _INFRA.search(text):
        return None
    if _INDUCTOR_ASSERTION.search(text):
        return 'inductor_assertion'
    if 'ValueError: materialized Torch graph differs from declared NNSmith output signature' in text:
        return None
    if _EXCEPTION.search(text):
        return 'compiler_exception'
    if _DTYPE.search(text) and re.search(r'/triton/backends/ascend/dtype_guard[.]py", line \d+', text):
        return 'dtype_guard'
    # Old tails may lose BEG and the exception header. END plus a concrete
    # compiler diagnostic still identifies a Triton error block.
    block = '[ERROR][Triton][BEG]' in text or '[ERROR][Triton][END]' in text
    if _NATIVE.search(text) and (block or _FRAME.search(text)):
        return 'native_compiler'
    return None


def triton_failure(backend, summary=None, detail=None):
    if backend != 'triton-npu':
        return None
    from autofusex.server.failure_attribution import _object
    d = _object(detail)
    execution = _object(_object(d.get('contexts')).get('execution') or d.get('execution'))
    shapes = _object(_object(execution.get('diagnostics')).get('shapes'))
    summaries = {str(s.get('shape_index')): s.get('error_summary') or ''
                 for s in d.get('shape_results', []) if isinstance(s, dict)
                 and s.get('correctness') == 'error'}
    indices = set(summaries) | {str(k) for k, v in shapes.items() if _object(v).get('error')}
    if indices:
        texts = [summaries.get(i, '') + '\n' + str(_object(shapes.get(i)).get('error') or '') for i in indices]
    else:
        # A terminal summary is sufficient; shared logs are not.
        texts = [summary or '']
    families = [compile_evidence(t) for t in texts]
    if not all(families):
        return None
    if all(f == 'inductor_assertion' for f in families):
        return 'gen_fail', 'inductor', 'torch'
    return 'compile_fail', 'triton_ascend', 'triton'
