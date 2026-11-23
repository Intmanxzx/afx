"""Fixed-source task configuration; no seed or NNSmith fields."""
from jsonschema import Draft202012Validator
from autofusex.contracts.configuration import canonical, merge, MAX_BYTES, obj, enum, TEXT, NAME, BOOL, DURATION, RESOURCE, DEVICES

WORKLOAD = obj(dict(case_key=dict(type='string', pattern=r'^[IM]:[0-9a-f]{64}$'),
                    revision=dict(type='string', pattern=r'^[0-9a-f]{64}$'),
                    script_args=dict(type='array', items=TEXT, maxItems=256),
                    fullgraph=BOOL, dynamic=BOOL, with_cpu=BOOL,
                    rtol=dict(type='number', minimum=0), atol=dict(type='number', minimum=0)),
               ('case_key', 'revision'))
SCHEMA = obj(dict(schema_version=enum(1), kind=enum('imported_task', 'model_task'), name=TEXT,
    environment=NAME, workload=WORKLOAD,
    execution=obj(dict(backends=dict(type='array', items=enum('af-legacy'), minItems=1, maxItems=1),
                       devices=DEVICES, timeout=DURATION, resources=RESOURCE)),
    results=obj(dict(save=BOOL, upload=BOOL))), ('schema_version','kind','environment','workload'))
DEFAULTS = dict(execution=dict(backends=['af-legacy'], devices=['all'], timeout='1h',
                               resources=dict(policy='manual', workers=1, compile_threads=1, worker_max_tasks=1)),
                results=dict(save=True, upload=True))


def validate(value):
    if len(canonical(value).encode()) > MAX_BYTES: raise ValueError('configuration exceeds 64 KiB')
    for error in Draft202012Validator(SCHEMA).iter_errors(value):
        raise ValueError('%s: %s' % ('.'.join(map(str,error.absolute_path)), error.message[:500]))
    result = merge(DEFAULTS, value)
    expected = 'M:' if result['kind'] == 'model_task' else 'I:'
    if not result['workload']['case_key'].startswith(expected): raise ValueError('case workload mismatch')
    if result['workload']['case_key'][2:] != result['workload']['revision']: raise ValueError('case revision mismatch')
    if result['results']['upload'] and not result['results']['save']: raise ValueError('upload requires saved results')
    resources = result['execution']['resources']
    if resources['policy'] == 'auto' and any(key in value.get('execution',{}).get('resources',{}) for key in ('workers','compile_threads')):
        raise ValueError('auto resources conflict with manual worker/thread values')
    return result
