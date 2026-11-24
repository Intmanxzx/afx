"""One strict configuration contract for CLI, browser and device control."""
import copy
import hashlib
import json
import re
from pathlib import Path

import yaml



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
MEMORY = dict(type='string', pattern=r'^[1-9][0-9]{0,18}(B|KiB|MiB|GiB)$')
SHA = dict(type='string', pattern=r'^[0-9a-f]{64}$')
DEVICES = dict(type='array', uniqueItems=True, minItems=1, maxItems=64,
               items=dict(oneOf=[dict(type='string', enum=['all']), dict(type='string', pattern=r'^npu:[0-9]+$')]))
RESOURCE = obj(dict(policy=enum('auto', 'manual'), workers=POSITIVE,
                    compile_threads=POSITIVE, worker_max_tasks=POSITIVE, memory_budget=MEMORY))
PACKAGE = obj(dict(kind=enum('toolkit', 'kernels'), url=TEXT, sha256=SHA, adapter=NAME),
              ('kind', 'url', 'sha256', 'adapter'))
def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def quantity(value):
    match = re.fullmatch(r'([1-9][0-9]*)(ms|s|m|h|B|KiB|MiB|GiB)', value)
    if not match:
        raise ValueError('invalid quantity')
    n, unit = match.groups()
    return int(n) * {'ms':.001, 's':1, 'm':60, 'h':3600, 'B':1, 'KiB':1024, 'MiB':1024**2, 'GiB':1024**3}[unit]


