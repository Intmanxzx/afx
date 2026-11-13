"""Historical NNSmith operator factor catalog.

Before the fuzz-generation refactor the generator received the probed
AscendC allowlist, not a nine-operation hand-written subset.  The allowlist
has 30 A3 operations and 20 additional A5 operations.  Keep this data
independent of NNSmith imports so tests can audit it on a light-weight host.
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
# vocabulary. E/F use the historical hardware catalog below.
LEGACY_OPS = ('Add', 'Mul', 'Neg', 'ReLU', 'Transpose', 'ReduceSum',
              'ReduceMean', 'ReduceMin', 'ReduceMax')
HISTORICAL_OPS = A3_OPS
ALL_OPS = A3_OPS + A5_ONLY_OPS

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

def normalize_ops(values):
    """Validate and canonicalize a --ops whitelist."""
    if values is None:
        return None
    if isinstance(values, str):
        values = values.split(',')
    result = []
    for value in values:
        value = str(value).strip()
        if not value:
            continue
        try:
            canonical = OP_ALIASES[value]
        except KeyError:
            raise ValueError(f"unknown op {value!r}; use graph names from NAMES") from None
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise ValueError('--ops requires at least one operator')
    return tuple(result)


# Abstract family entries in the old allowlist need one concrete arity/dtype
# class for symbolic generation.  The serialized factor remains Cast/Concat.
CONCRETE_ALIASES = {'Cast': 'CastF32', 'Concat': 'Concat2'}

def canonical_class_name(name):
    if name.startswith('Cast'):
        return 'Cast'
    if name.startswith('Concat'):
        return 'Concat'
    if name == 'TorchReduceSum':
        return 'ReduceSum'
    return name


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
            missing.append(name)
        else:
            result.append(op)
    if missing:
        raise RuntimeError('NNSmith operator catalog is missing: ' + ', '.join(missing))
    return result
