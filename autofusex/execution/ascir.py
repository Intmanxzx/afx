"""Bounded ASCIR construction source, observed without enabling compiler dumps."""
import ast
from contextlib import contextmanager
from contextvars import ContextVar
import json

from autofusex.execution.failure_details import redact

SHAPE = ContextVar('afx_ascir_shape', default=-1)
MAX_SOURCE = 64 * 1024
MAX_SOURCES = 128 * 1024


@contextmanager
def shape_scope(index):
    token = SHAPE.set(index)
    try:
        yield
    finally:
        SHAPE.reset(token)


def construction_source(source):
    """Keep complete AST statements, excluding scheduler comments and execution calls."""
    if not isinstance(source, str) or len(source) > 512 * 1024:
        return ''
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return ''
    if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and isinstance(n.func.value, ast.Name) and n.func.value.id == 'ascir'
               and n.func.attr in {'HintGraph', 'FusedGraph'} for n in ast.walk(tree)):
        return ''
    lines = source.splitlines()
    parts = []
    for stmt in tree.body:
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr in {'schedule', 'codegen', 'codegen_bin'} for n in ast.walk(stmt)):
            break
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        parts.append('\n'.join(lines[stmt.lineno - 1:stmt.end_lineno]))
    text = redact('\n'.join(parts))
    return text if len(text) <= MAX_SOURCE else ''


def validate_sources(raw):
    if not isinstance(raw, str) or len(raw) > MAX_SOURCES:
        raise ValueError('ASCIR sources size exceeded')
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) > 16:
        raise ValueError('invalid ASCIR sources')
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'shape_index', 'graph', 'source'}
                or type(row['shape_index']) is not int or not -1 <= row['shape_index'] < 1024
                or not isinstance(row['graph'], str) or len(row['graph']) > 100
                or not isinstance(row['source'], str) or not construction_source(row['source'])):
            raise ValueError('invalid ASCIR source')
    return json.dumps(rows, ensure_ascii=True)


def saved_construction_source(text):
    """Recover only a complete generated construction block from an old exception."""
    if not isinstance(text, str) or 'Failed to execute graph code:' not in text:
        return ''
    source = text.split('Failed to execute graph code:', 1)[1]
    lines = source.splitlines()
    import re
    for i, line in enumerate(lines):
        if re.match(r'^\w+\s*=\s*\w+\.schedule\(', line):
            return construction_source('\n'.join(lines[:i]))
    return ''
