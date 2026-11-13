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
    expected='nnsmith-sampled/5' if spec.seed.version in {5,6} else 'nnsmith-ert/3'
    if graph.get('dialect')!=expected or len(graph.get('nodes',[]))>64:
        raise ValueError('invalid NNSmith graph metadata')
    if spec.seed.version in {5,6}:
        from .generation.sampling import decode as decode_sampling
        config,_=decode_sampling(spec.seed)
        if graph.get('sampling')!=asdict(config):raise ValueError('sampling metadata differs from seed')
    from .generation.operator_catalog import LEGACY_NAMES
    allowed=set(LEGACY_NAMES.values())
    if any(node.get('op') not in allowed for node in graph['nodes']):
        raise ValueError('invalid op metadata')
    return spec
