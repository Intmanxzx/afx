"""Configuration dispatch for control and workload-specific validation."""
from pathlib import Path
from autofusex.contracts.configuration import MAX_BYTES, parse


def validate(value):
    if not isinstance(value, dict):
        raise ValueError('configuration must be an object')
    kind = value.get('kind')
    if kind in {'client_policy', 'environment'}:
        from autofusex.contracts.control_configuration import validate as check
    elif kind == 'fuzz_task':
        from autofusex.workloads.fuzz.configuration import validate as check
    elif kind in {'imported_task', 'model_task'}:
        from autofusex.workloads.imported.configuration import validate as check
    else:
        raise ValueError('unsupported configuration kind')
    return check(value)


def load(path):
    with Path(path).open() as stream:
        return validate(parse(stream.read(MAX_BYTES + 1)))
