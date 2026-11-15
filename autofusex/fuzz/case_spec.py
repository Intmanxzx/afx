"""Bounded public case description. No tensors, executable code or dependencies."""
from dataclasses import asdict
import json
from .limits import MAX_CASE_BYTES, MAX_NODES, MAX_TENSORS, MAX_SHAPES
from .contracts import CaseSpec, Seed, TensorSpec, canonical_json


def encode(spec):
    if (len(json.loads(spec.graph_json).get('nodes', [])) > MAX_NODES or
            len(spec.shape_samples) > (MAX_SHAPES if spec.seed.current else 16) or
            any(len(group) > MAX_TENSORS for group in
                (spec.inputs, spec.weights, *spec.shape_samples, *spec.output_samples))):
        raise ValueError('case specification exceeds structural limits')
    value = asdict(spec)
    value['seed'] = str(spec.seed)
    raw = canonical_json(value)
    if len(raw.encode()) > MAX_CASE_BYTES:
        raise ValueError('case specification exceeds limit')
    return raw


def decode(raw):
    if not isinstance(raw, str) or len(raw.encode()) > MAX_CASE_BYTES:
        raise ValueError('case specification exceeds limit')
    value = json.loads(raw)
    if set(value) != {'seed','graph_json','inputs','weights','shape_samples','output_samples'}:
        raise ValueError('invalid case specification fields')
    def tensors(rows):
        if not isinstance(rows,list) or len(rows)>MAX_TENSORS: raise ValueError('too many tensors')
        return tuple(TensorSpec(**dict(row, shape=tuple(row['shape']), stride=tuple(row['stride']))) for row in rows)
    spec = CaseSpec(Seed.parse(value['seed']), value['graph_json'], tensors(value['inputs']), tensors(value['weights']),
                    tuple(tensors(row) for row in value['shape_samples']), tuple(tensors(row) for row in value['output_samples']))
    if len(spec.shape_samples)>(MAX_SHAPES if spec.seed.current else 16) or not spec.output_samples:
        raise ValueError('invalid shape samples')
    graph=json.loads(spec.graph_json)
    if len(graph.get('nodes', [])) > MAX_NODES:
        raise ValueError('too many graph nodes')
    if spec.seed.current:
        from .generation.sampling import decode as decode_sampling
        config, _ = decode_sampling(spec.seed)
        if graph.get('dialect') != 'nnsmith-aten/2' or graph.get('sampling') != asdict(config):
            raise ValueError('current graph metadata differs from seed')
        expected_samples = config.dynamic_variants if spec.seed.dynamic else 1
        if len(spec.shape_samples) != expected_samples:
            raise ValueError('shape count differs from seed')
        from autofusex.fuzz.nnsmith.aten_catalog import BY_TARGET
        known = {t.name for t in (*spec.inputs, *spec.weights)}
        if graph.get('aten_rules') != 'elementwise-comparison-reduce-view/v1' or graph.get('torch') != '2.10.0+cpu':
            raise ValueError('ATen protocol metadata mismatch')
        if len(graph['nodes']) > config.max_nodes:
            raise ValueError('node count exceeds seed budget')
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
            expected_class = 'autofusex.fuzz.nnsmith.aten_dialect.Aten_' + rule.target[5:].replace('.', '_')
            if data.get('class') != expected_class or data.get('state', {}).get('extra_attrs') != node.get('attrs'):
                raise ValueError('ATen materializer metadata mismatch')
            known.update(outputs)
        if any(n not in known for n in graph['outputs']):
            raise ValueError('undefined ATen graph output')
    return spec
