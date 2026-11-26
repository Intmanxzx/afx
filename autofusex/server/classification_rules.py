"""High-confidence, administrator-configured log classification rules."""
import json
import re
from .store import get_content
from .failure_attribution import VERSION

KEY = 'classification.rules'
MLIR_FALLBACK = 'mlir-fused-runtime-fallback'
FIELDS = ('function_status', 'fail_stage', 'failure_owner', 'classifier_version')
BUILTIN_RULES = [
    {'id': 'nnsmith-fake-view', 'pattern': r'torch\._dynamo\.exc\.TorchRuntimeError:[\s\S]*Dynamo failed to run FX node with fake tensors:[\s\S]*aten\.view_copy\.default[\s\S]*Cannot view a tensor with shape', 'result': {'function_status': 'gen_fail', 'fail_stage': 'nnsmith', 'failure_owner': 'nnsmith'}},
    {'id': 'mlir-runtime-timeout', 'pattern': r'(?=[\s\S]*AF_MLIR_STRICT gate=none shortfall=no graph=autofused_)[\s\S]*TimeoutError:\s*timed out', 'backend': 'af-mlir', 'result': {'function_status': 'timeout', 'fail_stage': 'timeout', 'failure_owner': 'npu_runtime'}},
    {'id': 'mlir-tiling-member', 'pattern': r'(?=[\s\S]*autofused_)(?=[\s\S]*AF_MLIR_STRICT gate=none shortfall=no graph=autofused_)[\s\S]*error:\s*no member named', 'backend': 'af-mlir', 'result': {'function_status': 'compile_fail', 'fail_stage': 'mlir', 'failure_owner': 'autofuse'}},
    {'id': 'mlir-legalize-afir', 'pattern': r"error:\s*failed to legalize operation 'afir\.", 'backend': 'af-mlir', 'result': {'function_status': 'compile_fail', 'fail_stage': 'mlir', 'failure_owner': 'autofuse'}},
    {'id': 'inductor-int-noop-keyword', 'pattern': r"torch\._dynamo\.exc\.BackendCompilerFailed:\s*backend='inductor' raised:[\s\S]*TypeError:\s*int_noop\(\) got an unexpected keyword argument", 'result': {'function_status': 'gen_fail', 'fail_stage': 'inductor', 'failure_owner': 'torch'}},
    {'id': 'inductor-symint-attribute', 'pattern': r"torch\._dynamo\.exc\.BackendCompilerFailed:\s*backend='inductor' raised:[\s\S]*AttributeError:\s*'SymInt'", 'result': {'function_status': 'gen_fail', 'fail_stage': 'inductor', 'failure_owner': 'torch'}},
    {'id': 'inductor-unsupported-index', 'pattern': r'torch\._inductor\.exc\.InductorError:\s*UnsupportedIndexExpr:\s*Unsupported', 'result': {'function_status': 'gen_fail', 'fail_stage': 'inductor', 'failure_owner': 'torch'}},
    {'id': 'inductor-symint', 'pattern': r'aten::pow.*SymInt', 'result': {'function_status': 'gen_fail', 'fail_stage': 'inductor', 'failure_owner': 'torch'}},
    {'id': 'triton-graph-assert', 'pattern': r'AssertionError: n=copy_,.*graph=graph\(\):', 'backend': 'triton-npu', 'result': {'function_status': 'gen_fail', 'fail_stage': 'inductor', 'failure_owner': 'torch'}},
    {'id': 'att-ge-autofuse', 'pattern': r'autofuse/att/.*ErrorNo:.*(?:failed|unsatisfied)|ERROR GE', 'result': {'function_status': 'compile_fail', 'fail_stage': 'af_common', 'failure_owner': 'autofuse'}},
    {'id': 'mlir-vector-core', 'pattern': r'\[Error\]: The vector core execution is abnormal\.', 'backend': 'af-mlir', 'result': {'function_status': 'runtime_fail', 'fail_stage': 'fused_kernel', 'failure_owner': 'npu_runtime'}},
    {'id': 'ascend-kernel-build', 'pattern': r'Failed to build ascend kernel, trigger bug', 'backend': 'af-mlir', 'result': {'function_status': 'compile_fail', 'fail_stage': 'fused_kernel', 'failure_owner': 'npu_runtime'}},
    {'id': 'compiler-static-assert', 'pattern': r'error: static assertion failed.*SupportType.*Adds', 'result': {'function_status': 'compile_fail', 'fail_stage': 'af_common', 'failure_owner': 'autofuse'}},
    {'id': MLIR_FALLBACK, 'pattern': r'AF_MLIR_STRICT\s+gate=none\s+shortfall=no\s+graph=autofused_\w+', 'backend': 'af-mlir', 'result': {'function_status': 'runtime_fail', 'fail_stage': 'fused_kernel', 'failure_owner': 'npu_runtime'}},
]
DEFAULT_RULES = BUILTIN_RULES

def validate(rules):
    if not isinstance(rules, list) or len(rules) > 200:
        raise ValueError('rules must be a list with at most 200 entries')
    out = []
    for rule in rules:
        if not isinstance(rule, dict): raise ValueError('each rule must be an object')
        rid = str(rule.get('id', '')).strip()
        pattern = str(rule.get('pattern', ''))
        if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,64}', rid): raise ValueError('invalid rule id')
        if not pattern or len(pattern) > 4000: raise ValueError(f'invalid pattern for {rid}')
        try: re.compile(pattern, re.I if rule.get('ignore_case', True) else 0)
        except re.error as exc: raise ValueError(f'invalid regex for {rid}: {exc}') from None
        result = rule.get('result') or {}
        if result.get('function_status') not in {'gen_fail','ascgraph_fail','compile_fail','runtime_fail','build_fail','timeout'}: raise ValueError(f'invalid status for {rid}')
        if result.get('fail_stage') is None or result.get('failure_owner') is None: raise ValueError(f'incomplete result for {rid}')
        out.append({'id': rid, 'enabled': bool(rule.get('enabled', True)), 'pattern': pattern,
                    'ignore_case': bool(rule.get('ignore_case', True)), 'backend': rule.get('backend'),
                    'result': {'function_status': result['function_status'], 'fail_stage': str(result['fail_stage']), 'failure_owner': str(result['failure_owner'])}})
    if len({r['id'] for r in out}) != len(out): raise ValueError('duplicate rule id')
    return out

def load(store):
    row = store.reader().execute('SELECT value FROM platform_settings WHERE key=?', (KEY,)).fetchone()
    if not row: return merge_builtins([])
    return merge_builtins(json.loads(row['value']))

def merge_builtins(rules):
    """Keep missing builtins; an explicit same-id override may disable one."""
    custom = validate(rules)
    ids = {r['id'] for r in custom}
    return validate(custom + [r for r in BUILTIN_RULES if r['id'] not in ids])

def history(store):
    rows = store.reader().execute(
        "SELECT rowid AS id,at,actor,detail FROM audit_log WHERE action=? ORDER BY rowid DESC LIMIT 50",
        ('classification.rules.backup',)).fetchall()
    return [dict(id=r['id'], at=r['at'], actor=r['actor'], **json.loads(r['detail'])) for r in rows]

def load_or_builtin(store):
    row = store.reader().execute('SELECT value FROM platform_settings WHERE key=?', (KEY,)).fetchone()
    if not row: return merge_builtins([])
    return load(store)

def match(rules, backend, text, *, current=None):
    # A saved editor payload can put the fallback anywhere. Always evaluate it
    # last, and never replace an established attribution with this heuristic.
    ordered = [r for r in rules if r['id'] != MLIR_FALLBACK] + [r for r in rules if r['id'] == MLIR_FALLBACK]
    for rule in ordered:
        if rule['id'] == MLIR_FALLBACK and current is not None and not (
                current[0] == 'runtime_fail' and current[1] in {'unknown', 'unknown_retest', None}):
            continue
        if not rule['enabled'] or (rule.get('backend') and rule['backend'] != backend): continue
        if re.search(rule['pattern'], text or '', re.I if rule.get('ignore_case', True) else 0):
            r = rule['result']; return r['function_status'], r['fail_stage'], r['failure_owner'], VERSION, rule['id']
    return None

def plan(store, rules, *, summary_only=True, progress=None):
    rules = merge_builtins(rules); changes = []
    with store.connect(readonly=True) as conn:
        # Rules are corrective evidence and must be able to repair records
        # already stamped with the current classifier version.  Restricting
        # this scan to old versions would preserve earlier misclassifications.
        for scanned, row in enumerate(conn.execute("SELECT e.*,c.case_key FROM executions e JOIN cases c ON c.case_id=e.case_id WHERE function_status IN ('gen_fail','ascgraph_fail','compile_fail','runtime_fail','build_fail','timeout')"), 1):
            if progress and scanned % 1000 == 0: progress(scanned, len(changes))
            text = str(row['log_summary'] or '')
            if not summary_only:
                text += '\n' + '\n'.join(str(x) for x in (get_content(conn, row['log_content_id']) if row['log_content_id'] else None, get_content(conn, row['detail_content_id']) if row['detail_content_id'] else None) if x)
            result = match(rules, row['backend'], text, current=(row['function_status'], row['fail_stage']))
            if not result: continue
            after = list(result[:4]); before = [row[k] for k in FIELDS]
            if before != after: changes.append({'exec_id': row['exec_id'], 'exec_uuid': row['exec_uuid'], 'case_key': row['case_key'], 'backend': row['backend'], 'log_summary': text, 'before': before, 'after': after, 'rule_id': result[4]})
        if progress: progress(locals().get('scanned', 0), len(changes))
    return changes
