"""One strict configuration contract for CLI, browser and device control."""
import copy
import hashlib
import json
import re
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

# Configuration parsing is shared by the daemon CLI and client-facing tests.
# Keep the implementation in the daemon contract while exposing the small
# client configuration API from its new module.

MAX_BYTES = 65536


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Loader(yaml.SafeLoader):
    yaml_implicit_resolvers = copy.deepcopy(yaml.SafeLoader.yaml_implicit_resolvers)

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ValueError('YAML aliases are not supported')
        self.depth = getattr(self, 'depth', 0) + 1
        if self.depth > 24:
            raise ValueError('configuration nesting exceeds 24')
        try:
            return super().compose_node(parent, index)
        finally:
            self.depth -= 1

    def construct_mapping(self, node, deep=False):
        result = {}
        for key, value in node.value:
            key = self.construct_object(key, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError('configuration keys must be unique strings')
            result[key] = self.construct_object(value, deep=deep)
        return result


for key, resolvers in Loader.yaml_implicit_resolvers.items():
    Loader.yaml_implicit_resolvers[key] = [(tag, regex) for tag, regex in resolvers
                                         if tag not in {'tag:yaml.org,2002:bool', 'tag:yaml.org,2002:timestamp'}]
Loader.add_implicit_resolver('tag:yaml.org,2002:bool', re.compile(r'^(true|false)$'), list('tf'))


def parse(text):
    if not isinstance(text, str) or len(text.encode()) > MAX_BYTES:
        raise ValueError('configuration exceeds 64 KiB')
    try:
        value = yaml.load(text, Loader=Loader)
        canonical(value)
    except (yaml.YAMLError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError('invalid YAML/JSON configuration') from exc
    return value


def obj(properties, required=()):
    return dict(type='object', properties=properties, required=list(required), additionalProperties=False)


def enum(*values):
    return dict(enum=list(values))


TEXT = dict(type='string', minLength=1, maxLength=4096)
NAME = dict(type='string', pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')
POSITIVE = dict(type='integer', minimum=1, maximum=2147483647)
BOOL = dict(type='boolean')
DURATION = dict(type='string', pattern=r'^[1-9][0-9]{0,7}(ms|s|m|h)$')
MEMORY = dict(type='string', pattern=r'^[1-9][0-9]{0,9}(B|KiB|MiB|GiB)$')
SHA = dict(type='string', pattern=r'^[0-9a-f]{64}$')
DEVICES = dict(type='array', uniqueItems=True, minItems=1, maxItems=64,
               items=dict(oneOf=[dict(type='string', enum=['all']), dict(type='string', pattern=r'^npu:[0-9]+$')]))
RESOURCE = obj(dict(policy=enum('auto', 'manual'), workers=POSITIVE,
                    compile_threads=POSITIVE, worker_max_tasks=POSITIVE, memory_budget=MEMORY))
PACKAGE = obj(dict(kind=enum('toolkit', 'kernels'), url=TEXT, sha256=SHA, adapter=NAME),
              ('kind', 'url', 'sha256', 'adapter'))
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
                dim_dist_mode=enum('uniform','half_normal','log_normal'), value_dist_mode=enum('uniform','log_scale','llm','recommendation'),
                dtype_mode=enum('float32','float16','mixed','all'), dynamic_variants=dict(type='integer',minimum=2,maximum=8),
                method=enum('symbolic-cinit','symbolic','concolic'), ops=dict(type='array',items=TEXT),
                exclude_ops=dict(type='array', items=enum('transpose'), uniqueItems=True)))), ('mode',))]),
        execution=obj(dict(backends=dict(type='array', items=enum('cpu', 'af-mlir', 'af-legacy', 'eager-npu', 'triton-npu'), minItems=1, uniqueItems=True),
            devices=DEVICES, scheduling=enum('staged'), static_only=BOOL, timeout=DURATION, resources=RESOURCE)),
        evaluation=obj(dict(performance=obj(dict(enabled=BOOL, warmup_steps=dict(type='integer',minimum=0,maximum=10000), measured_steps=POSITIVE)))),
        termination=dict(oneOf=[obj(dict(mode=enum('finite'), max_cases=POSITIVE), ('mode', 'max_cases')),
                               obj(dict(mode=enum('continuous')), ('mode',))]),
        results=obj(dict(save=BOOL, upload=BOOL))), ('schema_version', 'kind', 'environment', 'workload'))}


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


DEFAULTS = {
    'client_policy': dict(resources=dict(allowed_devices=['npu:0'], max_workers=1, max_compile_threads=1),
                          sync=dict(interval='30s'), updates=dict(tool_policy='manual')),
    'environment': {},
    'fuzz_task': dict(workload=dict(mode='generate', generation=dict(tensor_scale='128k', dtype_mode='all', max_nodes=10)),
                      execution=dict(backends=['cpu', 'af-legacy', 'af-mlir'], devices=['all'], scheduling='staged', static_only=False,
                      timeout='120s', resources=dict(policy='auto', worker_max_tasks=100)),
                      evaluation=dict(performance=dict(enabled=False, warmup_steps=3, measured_steps=50)),
                      results=dict(save=True, upload=True), termination=dict(mode='continuous'))}


def quantity(value):
    match = re.fullmatch(r'([1-9][0-9]*)(ms|s|m|h|B|KiB|MiB|GiB)', value)
    if not match:
        raise ValueError('invalid quantity')
    n, unit = match.groups()
    return int(n) * {'ms':.001, 's':1, 'm':60, 'h':3600, 'B':1, 'KiB':1024, 'MiB':1024**2, 'GiB':1024**3}[unit]


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
        from ...client.contracts import Seed
        if result['workload']['mode'] == 'replay':
            for seed in result['workload']['seeds']: Seed.parse(seed)
        else:
            from ...client.generation.sampling import SamplingConfig
            generation = result['workload'].get('generation', {})
            seconds = quantity(generation['timeout']) if generation.get('timeout') else None
            if seconds is not None and (seconds != int(seconds) or seconds > 511):
                raise ValueError('generation timeout must be whole seconds <= 511')
            from ...client.generation.operator_catalog import normalize_ops
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
