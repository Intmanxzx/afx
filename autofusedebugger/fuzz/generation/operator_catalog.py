"""Operator names and NNSmith class discovery.

C/D are frozen protocols and retain their historical nine-operation catalog.
New G/H cases deliberately discover every registered NNSmith Torch dialect
operator at runtime.  The generator is not allowed to use backend support as
an eligibility filter: an operation that cannot be materialized or executed is
an ordinary skipped case, while ``--ops`` filters the already materialized
canonical graph.
"""

# Short NNSmith class names corresponding to the old qualified allowlist.
A3_OPS = (
    'Add', 'Sub', 'Mul', 'Max', 'Min', 'Div', 'Pow', 'Abs', 'Neg', 'Sigmoid',
    'Clip', 'Equal', 'And', 'Or', 'Where', 'ReduceSum', 'ReduceMean',
    'ReduceMin', 'ReduceMax', 'Reshape', 'Transpose', 'Squeeze', 'Concat',
    'Cast', 'Slice', 'ExpandLast1', 'ExpandLast2', 'ExpandLast3',
    'ExpandLast4', 'Sqrt',
)
A5_ONLY_OPS = (
    'Greater', 'Less', 'Ceil', 'Floor', 'Round', 'Sin', 'Cos', 'Tan', 'Atan',
    'Asin', 'Acos', 'GELU', 'LeakyReLU', 'Tril', 'Triu', 'IndexSelect',
    'Gather', 'LayerNorm', 'MatMul', 'Linear',
)
# C/D are immutable replay protocols and retain their original nine-op
# vocabulary. E/F keep the historical catalog; G/H use the full dialect.
LEGACY_OPS = ('Add', 'Mul', 'Neg', 'ReLU', 'Transpose', 'ReduceSum',
              'ReduceMean', 'ReduceMin', 'ReduceMax')

HISTORICAL_OPS = A3_OPS
# Compatibility export for reports and old callers.  E/F does not use this
# tuple to choose G/H operators; see full_operator_classes().
ALL_OPS = A3_OPS + A5_ONLY_OPS

# Complete concrete core dialect observed in the pinned NNSmith Torch module.
# This is metadata for --ops validation only; generation discovers the module
# at runtime so the seed does not depend on a host-side allowlist.
FULL_DIALECT_OPS = (
    'Abs', 'Acos', 'Add', 'And', 'ArgMax', 'ArgMin', 'Asin', 'Atan',
    'AvgPool2d', 'BatchNorm2d', 'BicubicInterp', 'BilinearInterp',
    'CastBool', 'CastF32', 'CastF64', 'CastI32', 'CastI64', 'Ceil', 'Clip',
    'Concat1', 'Concat2', 'Concat3', 'Concat4', 'Concat5', 'ConstPad', 'Conv1d',
    'Cos', 'Div', 'Equal', 'ExpandLast1', 'ExpandLast2', 'ExpandLast3',
    'ExpandLast4', 'Floor', 'GELU', 'Gather', 'Greater', 'IndexSelect',
    'LayerNorm', 'LeakyReLU', 'Less', 'LinearInterp', 'Log2', 'Max', 'MaxPool2d',
    'Min', 'Mul', 'NCHWConv2d', 'NearestInterp', 'Neg', 'Or', 'PReLU', 'Pow',
    'ReLU', 'ReduceMax', 'ReduceMean', 'ReduceMin', 'ReduceProd', 'ReduceSum',
    'ReflectPad', 'ReplicatePad', 'Reshape', 'Round', 'Sigmoid', 'Sin', 'Slice',
    'Softmax', 'Sqrt', 'Squeeze', 'Sub', 'Tan', 'Transpose', 'Tril',
    'TrilinearInterp', 'Triu', 'Unsqueeze', 'Where', 'Xor',
    'Flatten', 'Linear', 'PTMatMul', 'TorchReduceSum',
)

A3_ALLOWLIST = tuple(
    'torch.TorchReduceSum' if name == 'ReduceSum' else 'core.' + name
    for name in A3_OPS
)
A5_ALLOWLIST = tuple('core.' + name for name in A5_ONLY_OPS[:-2]) + (
    'torch.MatMul', 'core.Linear',
)
ALL_ALLOWLIST = A3_ALLOWLIST + A5_ALLOWLIST

# Stable graph names. ReLU is retained for C/D replay.
NAMES = {name: name[0].lower() + name[1:] for name in ALL_OPS}
NAMES.update(ReduceSum='sum', ReduceMean='mean', ReduceMin='amin', ReduceMax='amax',
             GELU='gelu', LeakyReLU='leaky_relu', MatMul='matmul',
             LayerNorm='layer_norm', IndexSelect='index_select')
NAMES['ReLU'] = 'relu'
LEGACY_NAMES = {**NAMES, 'ReLU': 'relu'}

# CLI and persisted graph names are accepted interchangeably.  The canonical
# spelling used in graph JSON and --ops payloads is the lower-case NAMES value.
OP_ALIASES = {value: value for value in NAMES.values()}
OP_ALIASES['ReLU'] = OP_ALIASES['relu'] = 'relu'
OP_ALIASES.update({name: NAMES[name] for name in ALL_OPS})
OP_ALIASES.update({qualified: NAMES[short] for qualified, short in
                   zip(ALL_ALLOWLIST, ALL_OPS)})

def _snake_name(name):
    import re
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name)).lower()
    return value.replace("__", "_")


def canonical_class_name(name):
    if name.startswith('Cast'):
        return 'Cast'
    if name.startswith('Concat'):
        return 'Concat'
    if name == 'TorchReduceSum':
        return 'ReduceSum'
    return name


def graph_name(name):
    """Return the stable snake_case graph spelling for a dialect class."""
    raw = str(name)
    canonical = canonical_class_name(raw)
    special = {
        'ReduceSum': 'sum', 'ReduceMean': 'mean', 'ReduceMin': 'amin',
        'ReduceMax': 'amax', 'ReduceProd': 'reduce_prod', 'ReLU': 'relu',
        'GELU': 'gelu', 'LeakyReLU': 'leaky_relu', 'IndexSelect': 'index_select',
        'PTMatMul': 'matmul',
    }
    if raw.startswith('Cast') and raw != 'Cast':
        return 'cast_' + raw[4:].lower()
    if raw.startswith('Concat') and raw != 'Concat':
        return 'concat_' + raw[6:]
    if canonical == 'Cast':
        return 'cast'
    if canonical == 'Concat':
        return 'concat'
    return special.get(canonical, _snake_name(canonical))


# CLI normalization is intentionally import-light; register all pinned
# dialect spellings without importing NNSmith.
for _name in FULL_DIALECT_OPS:
    _canonical_graph = graph_name(canonical_class_name(_name))
    OP_ALIASES.setdefault(_name, _canonical_graph)
    OP_ALIASES.setdefault('core.' + _name, _canonical_graph)
    OP_ALIASES.setdefault(_canonical_graph, _canonical_graph)


def discover_operator_names(operators):
    """Discover all concrete Torch dialect classes in a pinned NNSmith.

    Only classes declaring ``dialect='core'`` in their own class dictionary
    participate.  This excludes abstract bases and helper classes inherited
    from the dialect marker.  Sorted class names make the candidate catalog
    independent of module/import order and therefore reproducible for a seed.
    """
    result = []
    for name, value in vars(operators).items():
        if not isinstance(value, type) or value.__module__ != operators.__name__:
            continue
        if value.__dict__.get('dialect') != 'core':
            continue
        if name.endswith('Base') or name in {'AbsOpBase', 'ConcreteOp'}:
            continue
        if getattr(value, '__abstractmethods__', frozenset()):
            continue
        # Input/Constant are graph terminals, not selectable operators.  The
        # guard also keeps future NNSmith terminal classes out of the catalog.
        if name in {'Input', 'Constant'}:
            continue
        result.append(name)
    return tuple(sorted(set(result)))


def full_operator_classes(operators):
    """Resolve the complete deterministic NNSmith Torch operator catalog."""
    from .torch_dialect import classes
    result = classes()
    if not result:
        raise RuntimeError('NNSmith exposes no registered Torch operators')
    return result


from autofusedebugger.nnsmith_integration.aten_catalog import RULES as ATEN_RULES
OP_ALIASES.update({rule.target: rule.target for rule in ATEN_RULES})
for _rule in ATEN_RULES:
    _packet = _rule.target.split('.')[1]
    OP_ALIASES.setdefault(_packet, _packet)

def normalize_ops(values):
    """Canonicalize a post-generation operator whitelist.

    The catalog is import-light but explicit: names are validated against the
    pinned NNSmith Torch dialect metadata, while generation itself discovers
    the actual module on the worker.  Filtering never alters the seed or
    triggers a backend-dependent generation path.
    """
    if values is None:
        return None
    if isinstance(values, str):
        values = values.split(',')
    result = []
    for value in values:
        value = str(value).strip()
        if not value:
            continue
        canonical = OP_ALIASES.get(value)
        if canonical is None:
            raise ValueError(f"unknown op {value!r}; use a pinned Torch/NNSmith op name")
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise ValueError('--ops requires at least one operator')
    return tuple(result)


# Abstract family entries in the old allowlist need one concrete arity/dtype
# class for symbolic generation.  The serialized factor remains Cast/Concat.
CONCRETE_ALIASES = {'Cast': 'CastF32', 'Concat': 'Concat2'}

def operator_classes(operators, names=ALL_OPS, *, prefer_torch_reduce=False):
    """Resolve the catalog against a patched NNSmith installation."""
    result = []
    missing = []
    for name in names:
        op = None
        lookup_name = CONCRETE_ALIASES.get(name, name)
        if name == 'ReduceSum' and prefer_torch_reduce:
            try:
                from nnsmith.materialize.torch.dialect import TorchReduceSum
                op = TorchReduceSum
            except ImportError:
                op = getattr(operators, name, None)
        elif lookup_name != name:
            op = getattr(operators, lookup_name, None)
        else:
            op = getattr(operators, name, None)
        if op is None:
            # Frozen protocols fail closed: omission changes random selection
            # and must never silently produce a different graph for an old seed.
            missing.append(name)
        else:
            result.append(op)
    if missing:
        raise RuntimeError('NNSmith operator catalog is missing: ' + ', '.join(missing))
    return result


def matches_ops(op, accepted):
    """Exact ATen overloads or legacy family names; no generation filtering."""
    if op in accepted:
        return True
    if op.startswith('aten.'):
        name = op.split('.')[1]
        aliases = {'maximum':'max', 'minimum':'min', 'reduce_sum':'sum',
                   'argmax':'arg_max', 'argmin':'arg_min'}
        return aliases.get(name, name) in accepted
    return False
