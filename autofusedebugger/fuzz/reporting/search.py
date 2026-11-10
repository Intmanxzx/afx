"""Bounded SQL-style WHERE expressions, compiled only against a case projection.

No user SQL reaches SQLite: identifiers/operators are allowlisted and all values
are bound parameters. No functions, statements, comments or subqueries.
"""
import re

FIELDS = ('seed', 'shape_index', 'status', 'backend', 'function_status',
          'accuracy_status', 'user', 'op', 'input_shape', 'input_dtype', 'mode',
          'cann', 'torch', 'torch_npu', 'autofuse_commit', 'created_at')
TOKEN = re.compile(r"\s*(?:(?P<str>'(?:[^']|'')*')|(?P<num>-?\d+)|(?P<op><=|>=|!=|<>|=|<|>|\(|\))|(?P<word>[A-Za-z_][A-Za-z_0-9]*))")


def compile_where(text):
    if len(text) > 2048:
        raise ValueError('SQL condition is too long')
    text = re.sub(r'^\s*where\s+', '', text, flags=re.I).strip()
    tokens = []
    pos = 0
    while pos < len(text):
        match = TOKEN.match(text, pos)
        if not match:
            raise ValueError('unsupported SQL syntax; use field comparisons with AND/OR')
        tokens.append((match.lastgroup, match.group(match.lastgroup)))
        pos = match.end()
    if not tokens or len(tokens) > 128:
        raise ValueError('empty or overly complex SQL condition')
    index = 0
    values = []

    def peek(word):
        return index < len(tokens) and tokens[index][1].upper() == word

    def atom(depth):
        nonlocal index
        if depth > 12:
            raise ValueError('SQL nesting limit exceeded')
        if peek('('):
            index += 1
            result = expression(depth + 1)
            if not peek(')'):
                raise ValueError('missing closing parenthesis')
            index += 1
            return '(' + result + ')'
        if index + 2 >= len(tokens):
            raise ValueError('expected field operator value')
        kind, field = tokens[index]
        op = tokens[index + 1][1].upper()
        value_kind, value = tokens[index + 2]
        if kind != 'word' or field.lower() not in FIELDS:
            raise ValueError('unknown query field: ' + field)
        if op not in {'=', '!=', '<>', '<', '>', '<=', '>=', 'LIKE'}:
            raise ValueError('unsupported query operator')
        if value_kind not in {'str', 'num'}:
            raise ValueError('quote text values with single quotes')
        value = int(value) if value_kind == 'num' else value[1:-1].replace("''", "'")
        if field.lower() == 'seed' and op in {'=', '!=', '<>'}:
            from ..contracts import Seed
            value = str(Seed.parse(value))
        values.append(value)
        index += 3
        return 'p."' + field.lower() + '" ' + op + ' ? COLLATE NOCASE'

    def conjunction(depth):
        nonlocal index
        result = atom(depth)
        while peek('AND'):
            index += 1
            result += ' AND ' + atom(depth)
        return result

    def expression(depth):
        nonlocal index
        result = conjunction(depth)
        while peek('OR'):
            index += 1
            result += ' OR ' + conjunction(depth)
        return result

    result = expression(0)
    if index != len(tokens):
        raise ValueError('unsupported trailing SQL syntax')
    return result, values


def condition(q):
    if not q:
        return '1', []
    if not isinstance(q, str) or len(q) > 2048:
        raise ValueError('invalid search')
    q = q.strip()
    if re.match(r'^where\s', q, re.I):
        return compile_where(q)
    states = {'pass': 'Pass', 'failed': None, '功能failed': 'function_failed',
              'function_failed': 'function_failed', '精度failed': 'accuracy_failed',
              'accuracy_failed': 'accuracy_failed', 'unknown': 'Unknown'}
    if q.lower() in states:
        state = states[q.lower()]
        if state is None:
            return "p.status IN ('function_failed','accuracy_failed')", []
        return 'p.status=? COLLATE NOCASE', [state]
    if re.fullmatch(r'[AB][0-9A-Z]{9}', q):
        from ..contracts import Seed
        return 'p.seed=?', [str(Seed.parse(q))]
    return 'instr(lower(p.search),lower(?))>0', [q]
