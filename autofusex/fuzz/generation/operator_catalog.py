"""Import-light current ATen names; whitelist filtering never changes generation."""
from autofusex.fuzz.nnsmith.aten_catalog import RULES

OP_ALIASES = {rule.target: rule.target for rule in RULES}
OP_ALIASES.update({rule.target.split('.')[1]: rule.target.split('.')[1] for rule in RULES})
OP_ALIASES.update({'Add':'add', 'Sub':'sub', 'Mul':'mul', 'Div':'div', 'ReLU':'relu',
                  'ReduceSum':'sum', 'ReduceMean':'mean', 'ReduceMin':'amin', 'ReduceMax':'amax',
                  'Transpose':'transpose', 'Reshape':'reshape', 'Clip':'clamp', 'Equal':'eq'})
OP_ALIASES.update({'core.'+k:v for k,v in tuple(OP_ALIASES.items()) if k[0].isupper()})


def normalize_ops(values):
    if values is None:
        return None
    if isinstance(values, str):
        values = values.split(',')
    result = []
    for value in values:
        value = str(value).strip()
        if not value:
            continue
        if value not in OP_ALIASES:
            raise ValueError(f'unknown op {value!r}; use a registered ATen target or packet')
        canonical = OP_ALIASES[value]
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise ValueError('--ops requires at least one operator')
    return tuple(result)


def matches_ops(op, accepted):
    return op in accepted or (op.startswith('aten.') and op.split('.')[1] in accepted)
