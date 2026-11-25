"""Read saved error evidence and graph source without rewriting immutable uploads."""
import json
import re

from autofusex.execution.failure_details import diagnostic_view, redact
from autofusex.execution.ascir import construction_source, saved_construction_source

_ERROR = re.compile(r'\b(?:[\w.]*Error|[\w.]*Exception):|\[(?:ERROR|FATAL)\]|\berror:|\bfatal:', re.I)
_WRAPPER = re.compile(r'Failed to execute graph code|BackendCompilerFailed|^torch\._dynamo\.exc\.|^InductorError:|Traceback \(most recent call last\)', re.I)


def obj(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def sections(detail=None, log=None, fallback=None):
    d = obj(detail)
    diag = obj(obj(obj(d.get('contexts')).get('execution')).get('diagnostics'))
    result = []
    seen = set()

    def add(name, text):
        if not isinstance(text, str) or not text.strip():
            return
        text = redact(text.strip())
        if text not in seen and not any(text in section['text'] for section in result):
            result.append({'name': name, 'text': text})
            seen.add(text)

    for index, shape in obj(diag.get('shapes')).items():
        add('Shape ' + index, obj(shape).get('error'))
    for index, view in obj(d.get('shape_views')).items():
        add('Shape ' + index, obj(view).get('shape_log'))
    for row in d.get('shape_results') or []:
        add('Shape ' + str(row.get('shape_index')), row.get('error_summary'))
    add('执行错误', d.get('error_summary'))
    add('执行错误', d.get('error'))
    add('运行日志', diag.get('error'))
    view = obj(d.get('diagnostic_view'))
    add('执行日志', view.get('shape_log'))
    add('运行日志', view.get('run_log') or view.get('shared_log'))
    plog = d.get('plog_errors')
    if isinstance(plog, list):
        add('CANN plog', '\n'.join(x for x in plog if isinstance(x, str)))
    add('保存日志', log)
    add('上报摘要', fallback)
    return result


def summary(detail=None, log=None, fallback=None):
    entries = sections(detail, log, fallback)
    # Prefer the concrete exception/native failure over outer compiler wrappers.
    candidates = []
    for section in entries:
        for line in section['text'].splitlines():
            line = line.strip()
            if _ERROR.search(line) and not _WRAPPER.search(line):
                candidates.append(line)
    if candidates:
        return candidates[0][:500]
    for section in entries:
        for line in section['text'].splitlines():
            if _ERROR.search(line):
                return line.strip().split('Failed to execute graph code:')[0][:500] or 'Failed to execute graph code'
    return redact(fallback).strip()[:500] if isinstance(fallback, str) and fallback.strip() else None


def presentation(detail=None, log=None, fallback=None):
    d = obj(detail)
    execution = obj(obj(d.get('contexts')).get('execution'))
    diag = obj(execution.get('diagnostics'))
    views = dict(obj(d.get('shape_views')))
    for index, shape in obj(diag.get('shapes')).items():
        views[index] = diagnostic_view(diag, index, obj(shape).get('error') or '')
    sources = execution.get('ascir_sources')
    if isinstance(sources, str):
        try:
            sources = json.loads(sources)
        except ValueError:
            sources = []
    graphs = []
    for row in sources if isinstance(sources, list) else []:
        if not isinstance(row, dict):
            continue
        source = construction_source(row.get('source'))
        if source:
            graphs.append({'shape_index': row.get('shape_index'), 'source': source})
    # Legacy logs qualify only when the complete generated construction block survived.
    for index, view in views.items():
        if not str(index).isdecimal() or any(g['shape_index'] == int(index) for g in graphs):
            continue
        source = saved_construction_source(obj(view).get('shape_log'))
        if source:
            graphs.append({'shape_index': int(index), 'source': source})
    return {'error_logs': sections(d, log, fallback), 'ascir_graphs': graphs,
            'shape_views': views}
