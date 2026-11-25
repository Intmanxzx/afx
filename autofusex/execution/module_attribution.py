"""Module boundaries shared by client uploads and server reclassification."""
import json
import re

MARKER = 'AFX_MODULE_V1='

def boundary_failure(exc, sources):
    """Inspect actual generated-code frames, never source text in an error message."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        tb = exc.__traceback__
        while tb:
            source = sources.get(tb.tb_frame.f_code)
            if source is not None:
                line = source[tb.tb_lineno - 1].strip()
                stage = ('schedule' if re.search(r'\b\w+\.schedule\s*\(', line) else
                         'codegen' if re.search(r'\b\w+\.codegen(?:_bin)?\s*\(', line) else 'af_fronted')
                return {'stage': stage, 'line': tb.tb_lineno, 'source': line[:300]}
            tb = tb.tb_next
        exc = exc.__cause__ or exc.__context__
    return None

def classify_module(text, backend, declared_stage=None):
    """Return a proven (status, module, owner), or None for legacy fallback."""
    marker = re.search(r'AFX_MODULE_V1=(\{[^\n]*\})', text)
    stage = None
    if marker:
        try: stage = json.loads(marker[1]).get('stage')
        except (ValueError, TypeError): pass
    if stage == 'af_fronted': return 'ascgraph_fail', 'af_fronted', 'autofuse'
    if stage == 'schedule': return 'compile_fail', 'schedule', 'autofuse'
    if stage == 'legacy_codegen': return 'compile_fail', 'legacy_codegen', 'autofuse'
    if stage == 'mlir': return 'compile_fail', 'mlir', 'autofuse'
    if stage == 'codegen': return 'compile_fail', 'unknown', 'autofuse'
    if re.search(r'\baclnn[A-Za-z0-9_]+\s+failed\b', text):
        return 'runtime_fail', 'aclnn', 'npu_runtime'
    return None
