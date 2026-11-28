"""Bounded scalar diagnostics; never persist tensors or unbounded compiler logs."""
import json
import math
import re
import traceback

FIELDS={'cpu_pass','secondary_used','secondary_pass','max_abs','mean_abs','max_rel','mismatches','elements','nonfinite','error','basis','outputs','mismatch_ratio'}


def redact(text):
    return re.sub(r'(?i)(bearer\s+|(?:token|secret|password|api_key)\s*[=:]\s*)[^\s,;]+',r'\1[REDACTED]',text)


def excerpt(exc):
    text = redact(''.join(traceback.format_exception(type(exc),exc,exc.__traceback__)))
    current, seen = exc, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        fact = getattr(current, '_afx_module_failure', None)
        if fact:
            marker = 'AFX_MODULE_V1=' + json.dumps(fact, ensure_ascii=True) + '\n'
            return marker + text[-(1600-len(marker)):]
        current = current.__cause__ or current.__context__
    return text[-1600:]


def error_summary(exc):
    """Extract the cause before bounding a compiler exception's generated source."""
    text = f'{type(exc).__name__}: {exc}'
    return diagnostic_view({'shapes': {'0': {'error': text}}}, 0)['summary'][:500]


def validate(raw):
    if not isinstance(raw,str) or len(raw)>128*1024:raise ValueError('diagnostics size exceeded')
    data=json.loads(raw)
    if not isinstance(data,dict) or set(data)-{'version','error','shapes','autofuse_coverage','accuracy_evidence','baseline_status','baseline_exec_uuid'} or data.get('version')!=1:raise ValueError('invalid diagnostics')
    if 'baseline_status' in data and data['baseline_status'] != 'unavailable':raise ValueError('invalid baseline status')
    if 'baseline_exec_uuid' in data and (not isinstance(data['baseline_exec_uuid'], str) or len(data['baseline_exec_uuid']) > 64):raise ValueError('invalid baseline execution')
    if 'error' in data and (not isinstance(data['error'],str) or len(data['error'])>1600):raise ValueError('invalid core log')
    coverage = data.get('autofuse_coverage')
    if coverage is not None:
        counters = {'candidates', 'scheduler_codegen_entered', 'scheduler_codegen_succeeded', 'codegen_succeeded'}
        if not isinstance(coverage, dict) or set(coverage) != counters | {'events', 'installed', 'no_coverage'}:
            raise ValueError('invalid Autofuse coverage')
        if any(type(coverage[k]) is not int or not 0 <= coverage[k] < 2**31 for k in counters):
            raise ValueError('invalid Autofuse coverage counter')
        if any(type(coverage[k]) is not bool for k in ('installed', 'no_coverage')):
            raise ValueError('invalid Autofuse coverage flag')
        events = coverage['events']
        allowed = {'AF_CANDIDATE', 'AF_CODEGEN_OK', 'SCHEDULER_CODEGEN_ENTER', 'SCHEDULER_CODEGEN_OK', 'NO_AF_COVERAGE'}
        if not isinstance(events, list) or len(events) > 64 or any(not isinstance(e, str) or e not in allowed for e in events):
            raise ValueError('invalid Autofuse coverage events')
    shapes=data.get('shapes',{})
    if not isinstance(shapes,dict) or len(shapes)>16:raise ValueError('diagnostic shape limit')
    for index,value in shapes.items():
        if not index.isdecimal() or not isinstance(value,dict) or set(value)-FIELDS:raise ValueError('invalid diagnostic fields')
        for key,v in value.items():
            if key == 'outputs':
                if not isinstance(v, list) or len(v) > 256: raise ValueError('invalid accuracy outputs')
            elif v is None and key in {'max_abs','max_rel','mean_abs'}:
                continue
            elif key in {'error','basis'}:
                if not isinstance(v,str) or len(v)>1600:raise ValueError('invalid diagnostic text')
            elif not isinstance(v,(int,float,bool)) or not math.isfinite(v):raise ValueError('invalid diagnostic metric')
    def finite_json(value, depth=0):
        if depth > 12: raise ValueError('diagnostic nesting exceeded')
        if isinstance(value, float) and not math.isfinite(value): raise ValueError('nonfinite diagnostic')
        if isinstance(value, dict):
            for item in value.values(): finite_json(item, depth+1)
        elif isinstance(value, list):
            for item in value: finite_json(item, depth+1)
    finite_json(data)
    evidence = data.get('accuracy_evidence')
    if evidence is not None:
        if not isinstance(evidence, dict) or evidence.get('version') != 1 or evidence.get('final_basis') not in {'cpu','eager-npu'}:
            raise ValueError('invalid accuracy evidence')
        comparisons = evidence.get('comparisons')
        if not isinstance(comparisons, list) or not 1 <= len(comparisons) <= 2:
            raise ValueError('invalid comparisons')
        for comparison in comparisons:
            if comparison.get('reference') not in {'cpu','eager-npu'} or type(comparison.get('passed')) is not bool:
                raise ValueError('invalid comparison')
            if not isinstance(comparison.get('shapes'), list) or len(comparison['shapes']) > 16:
                raise ValueError('invalid comparison shapes')
    return json.dumps(data,sort_keys=True,separators=(',',':'))


def core_error(diagnostics, index, fallback=''):
    """Use the same shape-first summary as the detail page."""
    return diagnostic_view(diagnostics, index, fallback)['summary']


def diagnostic_view(diagnostics, index, fallback=''):
    """Separate shape evidence from run-wide stderr; never infer attribution."""
    shape = diagnostics.get('shapes', {}).get(str(index), {})
    local = shape.get('error', '') or fallback or ''
    shared = diagnostics.get('error', '')

    def useful(text):
        lines = []
        for raw in text.splitlines():
            line = redact(raw.strip())
            if len(line) < 12 or line.lower() in {"error:", "error: '"}:
                continue
            if any(word in line.lower() for word in (
                'unsupportedoperation:', 'fatal:', 'error:', 'runtimeerror:',
                'codegen generate kernel failed', 'af_mlir_strict',
                'check dtype failed', 'infer dtype failed', 'check shape failed',
            )) and line not in lines:
                # Wrapper exceptions often append generated source on the same line.
                lines.append(line.split('Failed to execute graph code:')[0] +
                             'Failed to execute graph code' if 'Failed to execute graph code:' in line else line)
        return lines

    current = useful(local)
    if not current:
        current = useful(fallback)
    # Prefer concrete codegen/native diagnostics over wrapper exceptions.
    current.sort(key=lambda line: (
        'failed to execute graph code' in line.lower(),
        'codegen generate kernel failed' in line.lower() or 'af_mlir_strict' in line.lower()))
    shared_lines = [line for line in useful(shared) if line not in current]
    summary = current[0] if current else (redact(local).strip().splitlines() or [''])[0]
    if not summary and shared_lines:
        summary = '运行级日志：' + shared_lines[0]
    return {'summary': summary[:1600], 'shape_lines': current,
            'shared_lines': shared_lines, 'shape_log': redact(local),
            'run_log': redact(shared),
            'shared_log': redact(shared) if shared.strip() != local.strip() else '',
            'explanation': ('Kernel代码生成失败或中止；具体不支持的算子/原因未在已保存日志中给出。'
                            if 'Codegen generate kernel failed' in local else ''),
            'shared_explanation': ('本次运行触发AF-MLIR严格模式的lowering检查（shortfall=yes）。'
                                   '这是运行级线索，不能仅凭该日志确定各shape的具体失败原因。'
                                   if 'AF_MLIR_STRICT' in shared and 'shortfall=yes' in shared else '')}
