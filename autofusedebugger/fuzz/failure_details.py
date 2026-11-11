"""Bounded scalar diagnostics; never persist tensors or unbounded compiler logs."""
import json
import math
import re
import traceback

FIELDS={'cpu_pass','secondary_used','secondary_pass','max_abs','mean_abs','max_rel','mismatches','elements','nonfinite','error','basis'}


def redact(text):
    return re.sub(r'(?i)(bearer\s+|(?:token|secret|password|api_key)\s*[=:]\s*)[^\s,;]+',r'\1[REDACTED]',text)


def excerpt(exc):
    return redact(''.join(traceback.format_exception(type(exc),exc,exc.__traceback__)))[-1600:]


def validate(raw):
    if not isinstance(raw,str) or len(raw)>8000:raise ValueError('diagnostics size exceeded')
    data=json.loads(raw)
    if not isinstance(data,dict) or set(data)-{'version','error','shapes'} or data.get('version')!=1:raise ValueError('invalid diagnostics')
    if 'error' in data and (not isinstance(data['error'],str) or len(data['error'])>1600):raise ValueError('invalid core log')
    shapes=data.get('shapes',{})
    if not isinstance(shapes,dict) or len(shapes)>16:raise ValueError('diagnostic shape limit')
    for index,value in shapes.items():
        if not index.isdecimal() or not isinstance(value,dict) or set(value)-FIELDS:raise ValueError('invalid diagnostic fields')
        for key,v in value.items():
            if key in {'error','basis'}:
                if not isinstance(v,str) or len(v)>1600:raise ValueError('invalid diagnostic text')
            elif not isinstance(v,(int,float,bool)) or not math.isfinite(v):raise ValueError('invalid diagnostic metric')
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
            )) and line not in lines:
                # Wrapper exceptions often append generated source on the same line.
                lines.append(line.split('Failed to execute graph code:')[0] +
                             'Failed to execute graph code' if 'Failed to execute graph code:' in line else line)
        return lines

    current = useful(local)
    if not current:
        current = useful(fallback)
    # Prefer concrete codegen/native diagnostics over wrapper exceptions.
    current.sort(key=lambda line: 'failed to execute graph code' in line.lower())
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
