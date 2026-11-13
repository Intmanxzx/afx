"""Bounded public case description. No tensors, executable code or dependencies."""
from dataclasses import asdict
import json
from .contracts import CaseSpec, Seed, TensorSpec, canonical_json


def encode(spec):
    value = asdict(spec)
    value['seed'] = str(spec.seed)
    return canonical_json(value)


def decode(raw):
    if not isinstance(raw, str) or len(raw.encode()) > 12000:
        raise ValueError('case specification exceeds limit')
    value = json.loads(raw)
    if set(value) != {'seed','graph_json','inputs','weights','shape_samples','output_samples'}:
        raise ValueError('invalid case specification fields')
    def tensors(rows):
        if not isinstance(rows,list) or len(rows)>64: raise ValueError('too many tensors')
        return tuple(TensorSpec(**dict(row, shape=tuple(row['shape']), stride=tuple(row['stride']))) for row in rows)
    spec = CaseSpec(Seed.parse(value['seed']), value['graph_json'], tensors(value['inputs']), tensors(value['weights']),
                    tuple(tensors(row) for row in value['shape_samples']), tuple(tensors(row) for row in value['output_samples']))
    if len(spec.shape_samples)>16 or not spec.output_samples:
        raise ValueError('invalid shape samples')
    graph=json.loads(spec.graph_json)
    expected = ('nnsmith-sampled/9' if spec.seed.version in {9,10} else
                'nnsmith-sampled/7' if spec.seed.version in {7,8} else
                'nnsmith-sampled/5' if spec.seed.version in {5,6} else 'nnsmith-ert/3')
    if graph.get('dialect')!=expected or len(graph.get('nodes',[]))>64:
        raise ValueError('invalid NNSmith graph metadata')
    if spec.seed.version in {5,6,7,8,9,10}:
        from .generation.sampling import decode as decode_sampling
        config,_=decode_sampling(spec.seed)
        if graph.get('sampling')!=asdict(config):raise ValueError('sampling metadata differs from seed')
    import re
    if spec.seed.version in {3, 4}:
        from .generation.operator_catalog import LEGACY_NAMES
        allowed = set(LEGACY_NAMES.values())
        if any(node.get('op') not in allowed for node in graph['nodes']):
            raise ValueError('invalid op metadata')
    elif spec.seed.version in {5, 6}:
        from .generation.operator_catalog import NAMES
        if any(node.get('op') not in set(NAMES.values()) for node in graph['nodes']):
            raise ValueError('invalid op metadata')
    else:
        # G/H graphs are generated from the complete pinned Torch dialect;
        # backend support is decided by CPU/target execution, not this gate.
        if any(not isinstance(node.get('op'), str) or
               not re.fullmatch(r'(?:aten\.[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*|[a-z][a-z0-9_]{0,63})', node['op'])
               for node in graph['nodes']):
            raise ValueError('invalid op metadata')
    if spec.seed.version in {9, 10}:
        from autofusedebugger.nnsmith_integration.aten_catalog import BY_TARGET
        known = {t.name for t in (*spec.inputs, *spec.weights)}
        if graph.get('aten_rules') != 'elementwise-comparison-reduce-view/v1' or graph.get('torch') != '2.10.0+cpu':
            raise ValueError('ATen protocol metadata mismatch')
        for node in graph['nodes']:
            rule = BY_TARGET.get(node.get('op'))
            if rule is None or len(node.get('inputs', [])) != rule.arity or any(n not in known for n in node['inputs']):
                raise ValueError('invalid ATen inputs/target')
            outputs = node.get('outputs')
            if (not isinstance(outputs, list) or len(outputs) != rule.outputs
                    or node.get('output') != outputs[0]
                    or any(not isinstance(n, str) or n in known for n in outputs)
                    or len(set(outputs)) != len(outputs)):
                raise ValueError('invalid ATen outputs')
            data = node.get('materializer', {})
            expected_class = 'autofusedebugger.nnsmith_integration.aten_dialect.Aten_' + rule.target[5:].replace('.', '_')
            if data.get('class') != expected_class or data.get('state', {}).get('extra_attrs') != node.get('attrs'):
                raise ValueError('ATen materializer metadata mismatch')
            known.update(outputs)
        if any(n not in known for n in graph['outputs']):
            raise ValueError('undefined ATen graph output')
    return spec
