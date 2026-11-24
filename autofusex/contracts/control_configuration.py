"""Client policy and execution-environment configuration contract.

Wire format is unchanged during migration. No workload runtime is imported.
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
        packages=obj(dict(torch=TEXT, torch_npu=TEXT, nnsmith=TEXT), ('torch', 'torch_npu'))),
        ('schema_version', 'kind', 'name', 'python', 'compatibility_profile', 'cann', 'packages'))}

DEFAULTS = {
    'client_policy': dict(resources=dict(allowed_devices=['npu:0'], max_workers=1, max_compile_threads=1),
                          sync=dict(interval='30s'), updates=dict(tool_policy='manual')),
    'environment': {}}


def validate(value):
    if not isinstance(value, dict) or value.get('kind') not in SCHEMAS:
        raise ValueError('unsupported configuration kind')
    if len(canonical(value).encode()) > MAX_BYTES:
        raise ValueError('configuration exceeds 64 KiB')
    kind = value['kind']
    for error in Draft202012Validator(SCHEMAS[kind]).iter_errors(value):
        raise ValueError('%s: %s' % ('.'.join(map(str,error.absolute_path)) or kind, error.message[:500]))
    result = merge(DEFAULTS[kind], value)
    if kind == 'client_policy' and not 1 <= quantity(result['sync']['interval']) <= 86400:
        raise ValueError('sync interval must be between 1s and 24h')
    return result


def load(path):
    with Path(path).open() as stream:
        return validate(parse(stream.read(MAX_BYTES + 1)))
