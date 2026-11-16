"""Stable operator categories for reporting and filtering."""

_OP_CATEGORY = {
    'add': 'Elementwise', 'sub': 'Elementwise', 'mul': 'Elementwise',
    'max': 'Elementwise', 'min': 'Elementwise', 'div': 'Elementwise',
    'pow': 'Elementwise', 'abs': 'Elementwise', 'neg': 'Elementwise',
    'sigmoid': 'Elementwise', 'clip': 'Elementwise', 'equal': 'Elementwise',
    'and': 'Elementwise', 'or': 'Elementwise', 'where': 'Elementwise',
    'greater': 'Elementwise', 'less': 'Elementwise', 'ceil': 'Elementwise',
    'floor': 'Elementwise', 'round': 'Elementwise', 'sin': 'Elementwise',
    'cos': 'Elementwise', 'tan': 'Elementwise', 'atan': 'Elementwise',
    'asin': 'Elementwise', 'acos': 'Elementwise', 'gelu': 'Elementwise',
    'leaky_relu': 'Elementwise', 'sqrt': 'Elementwise', 'relu': 'Elementwise',
    'sum': 'Reduce', 'mean': 'Reduce', 'amin': 'Reduce', 'amax': 'Reduce',
    'transpose': 'Transpose', 'reshape': 'View', 'squeeze': 'View',
    'concat': 'View', 'cast': 'View', 'slice': 'View',
    'expandLast1': 'View', 'expandLast2': 'View', 'expandLast3': 'View',
    'expandLast4': 'View', 'tril': 'View', 'triu': 'View',
    'index_select': 'Gather', 'gather': 'Gather', 'layer_norm': 'Norm',
    'matmul': 'MatMul', 'linear': 'MatMul',
}
# Import-light metadata: never consult device/backend support for categories.
from autofusex.fuzz.client.nnsmith.aten_catalog import RULES
for _rule in RULES:
    _OP_CATEGORY[_rule.target] = _rule.category

OP_CATEGORY = _OP_CATEGORY
SUPPORTED_CATEGORIES = tuple(sorted(set(_OP_CATEGORY.values())))


def categories(spec):
    import json
    graph = json.loads(spec.graph_json)
    result = {_OP_CATEGORY.get(node['op'], 'Other') for node in graph['nodes']}
    if graph.get('reduce_axis') is not None:
        result.add('Reduce')
    return tuple(sorted(result))
