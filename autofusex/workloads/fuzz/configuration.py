"""Existing fuzz task and execution-environment configuration contract.

Wire format is unchanged during migration. Seed semantics remain fuzz-specific.
"""
from pathlib import Path
from jsonschema import Draft202012Validator
from autofusex.contracts.configuration import (
    MAX_BYTES, canonical, fingerprint, parse, merge, quantity,
    obj, enum, TEXT, NAME, POSITIVE, BOOL, DURATION, MEMORY, SHA,
    DEVICES, RESOURCE, PACKAGE,
)

CANN = dict(oneOf=[
    obj(dict(source=enum('existing'), path=TEXT), ('source', 'path')),
    obj(dict(source=enum('download'), install_root=TEXT,
             packages=dict(type='array', items=PACKAGE, minItems=1, maxItems=8)), ('source', 'packages')),
    obj(dict(source=enum('build'), install_root=TEXT,
             build=obj(dict(source_dir=TEXT, profile=NAME, mode=enum('incremental', 'snapshot'), jobs=POSITIVE),
                       ('source_dir', 'profile'))), ('source', 'build'))])
SCHEMAS = {
    'client_policy': obj(dict(schema_version=enum(1), kind=enum('client_policy'),
        resources=obj(dict(allowed_devices=DEVICES, max_workers=POSITIVE, max_compile_threads=POSITIVE, memory_budget=MEMORY)),
        sync=obj(dict(interval=DURATION)), updates=obj(dict(tool_policy=enum('manual', 'when_idle')))), ('schema_version', 'kind')),
    'environment': obj(dict(schema_version=enum(1), kind=enum('environment'), name=NAME, python=dict(type='string', pattern=r'^3\.[0-9]+$'),
        compatibility_profile=NAME, cann=CANN,
        packages=obj(dict(torch=TEXT, torch_npu=TEXT, nnsmith=TEXT), ('torch', 'torch_npu', 'nnsmith'))),
        ('schema_version', 'kind', 'name', 'python', 'compatibility_profile', 'cann', 'packages')),
    'fuzz_task': obj(dict(schema_version=enum(1), kind=enum('fuzz_task'), name=TEXT, environment=NAME,
        workload=dict(oneOf=[obj(dict(mode=enum('replay'), seeds=dict(type='array', minItems=1, maxItems=1000, items=TEXT)), ('mode', 'seeds')),
            obj(dict(mode=enum('generate'), generation=obj(dict(tensor_scale=enum('small', 'medium', 'large', '16k', '32k', '64k', '128k', 'default'),
                max_nodes=dict(type='integer',minimum=1,maximum=64), max_elements_per_tensor=POSITIVE, timeout=DURATION,
                shape_mode=enum('static','dynamic','mixed'), dim_min=POSITIVE, dim_max=POSITIVE,
                dim_dist_mode=enum('uniform','half_normal','log_normal'), value_dist_mode=enum('log_scale','llm','recommendation'),
                dtype_mode=enum('float32','float16','mixed','all'), dynamic_variants=dict(type='integer',minimum=2,maximum=8),
                method=enum('symbolic-cinit','symbolic','concolic'), ops=dict(type='array',items=TEXT),
                exclude_ops=dict(type='array', items=enum('transpose'), uniqueItems=True)))), ('mode',))]),
        execution=obj(dict(backends=dict(type='array', items=enum('cpu', 'af-mlir', 'af-legacy', 'eager-npu', 'triton-npu'), minItems=1, uniqueItems=True),
            devices=DEVICES, scheduling=enum('staged'), static_only=BOOL, timeout=DURATION, resources=RESOURCE)),
        evaluation=obj(dict(performance=obj(dict(enabled=BOOL, warmup_steps=dict(type='integer',minimum=0,maximum=10000), measured_steps=POSITIVE)))),
        termination=dict(oneOf=[obj(dict(mode=enum('finite'), max_cases=POSITIVE), ('mode', 'max_cases')),
                               obj(dict(mode=enum('continuous')), ('mode',))]),
        results=obj(dict(save=BOOL, upload=BOOL))), ('schema_version', 'kind', 'environment', 'workload'))}


DEFAULTS = {
    'client_policy': dict(resources=dict(allowed_devices=['npu:0'], max_workers=1, max_compile_threads=1),
                          sync=dict(interval='30s'), updates=dict(tool_policy='manual')),
    'environment': {},
    'fuzz_task': dict(workload=dict(mode='generate', generation=dict(tensor_scale='128k', dtype_mode='all', max_nodes=10)),
                      execution=dict(backends=['cpu', 'af-legacy', 'af-mlir'], devices=['all'], scheduling='staged', static_only=False,
                      timeout='120s', resources=dict(policy='auto', worker_max_tasks=100)),
                      evaluation=dict(performance=dict(enabled=False, warmup_steps=3, measured_steps=50)),
                      results=dict(save=True, upload=True), termination=dict(mode='continuous'))}


def validate(value):
    if not isinstance(value, dict) or value.get('kind') not in SCHEMAS:
        raise ValueError('unsupported configuration kind')
    if len(canonical(value).encode()) > MAX_BYTES:
        raise ValueError('configuration exceeds 64 KiB')
    kind = value['kind']
    for error in Draft202012Validator(SCHEMAS[kind]).iter_errors(value):
        raise ValueError('%s: %s' % ('.'.join(map(str,error.absolute_path)) or kind, error.message[:500]))
    result = merge(DEFAULTS[kind], value)
    if kind == 'fuzz_task':
        ex = result['execution']; resources = ex['resources']
        if resources['policy'] == 'auto' and any(k in resources for k in ('workers', 'compile_threads')):
            raise ValueError('auto resources conflict with manual worker/thread values')
        if result['results']['upload'] and not result['results']['save']:
            raise ValueError('upload requires saved results')
        from autofusex.workloads.fuzz.contracts import Seed
        if result['workload']['mode'] == 'replay':
            result['workload'].pop('generation', None)
            result.pop('termination', None)
            for seed in result['workload']['seeds']: Seed.parse(seed)
        else:
            from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
            generation = result['workload'].get('generation', {})
            seconds = quantity(generation['timeout']) if generation.get('timeout') else None
            if seconds is not None and (seconds != int(seconds) or seconds > 511):
                raise ValueError('generation timeout must be whole seconds <= 511')
            from autofusex.workloads.fuzz.generation.operator_catalog import normalize_ops
            if generation.get('ops'): normalize_ops(generation['ops'])
            SamplingConfig.profile(generation.get('tensor_scale', 'default'),
                **{k:generation[k] for k in ('dim_min','dim_max','dim_dist_mode','value_dist_mode','dtype_mode','dynamic_variants','method') if k in generation},
                max_nodes=generation.get('max_nodes'),max_elem_per_tensor=generation.get('max_elements_per_tensor'),
                timeout_seconds=int(seconds) if seconds is not None else None)
        if result['workload']['mode'] == 'replay' and 'termination' in value:
            raise ValueError('replay terminates after the seed list; omit termination')
        if result['workload']['mode'] == 'generate' and 'termination' not in result:
            result['termination'] = dict(mode='finite', max_cases=100)
    if kind == 'client_policy' and not 1 <= quantity(result['sync']['interval']) <= 86400:
        raise ValueError('sync interval must be between 1s and 24h')
    return result


def load(path):
    with Path(path).open() as stream:
        return validate(parse(stream.read(MAX_BYTES + 1)))
