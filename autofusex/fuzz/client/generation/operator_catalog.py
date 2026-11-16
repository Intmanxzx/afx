"""Import-light current ATen names; whitelist filtering never changes generation."""
from autofusex.fuzz.client.nnsmith.aten_catalog import RULES

OP_ALIASES = {rule.target: rule.target for rule in RULES}
OP_ALIASES.update({rule.target.split('.')[1]: rule.target.split('.')[1] for rule in RULES})
OP_ALIASES.update({'Add':'add', 'Sub':'sub', 'Mul':'mul', 'Div':'div', 'ReLU':'relu',
                  'ReduceSum':'sum', 'ReduceMean':'mean', 'ReduceMin':'amin', 'ReduceMax':'amax',
                  'Transpose':'transpose', 'Reshape':'reshape', 'Clip':'clamp', 'Equal':'eq'})
OP_ALIASES.update({'core.'+k:v for k,v in tuple(OP_ALIASES.items()) if k[0].isupper()})
DISPLAY_NAMES = {}
for _rule in RULES:
    DISPLAY_NAMES.setdefault(_rule.display_name, []).append(_rule.target)
for _name in DISPLAY_NAMES:
    OP_ALIASES.setdefault(_name, _name)

CATEGORY_ALIASES = {rule.category.lower(): rule.category for rule in RULES}


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
        category = CATEGORY_ALIASES.get(value.lower())
        if value not in OP_ALIASES and category is None:
            raise ValueError(f'unknown op {value!r}; use a registered ATen target, packet, display name or category')
        canonical = category or OP_ALIASES[value]
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise ValueError('--ops requires at least one operator')
    return tuple(result)


def matches_ops(op, accepted):
    if op in accepted or (op.startswith('aten.') and op.split('.')[1] in accepted):
        return True
    from autofusex.fuzz.client.nnsmith.aten_catalog import BY_TARGET
    rule = BY_TARGET.get(op)
    return bool(rule and (rule.display_name in accepted or
                         any(value.lower() == rule.category.lower() for value in accepted)))
