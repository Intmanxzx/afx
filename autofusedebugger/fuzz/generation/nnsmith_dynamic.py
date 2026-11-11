"""D seeds: the frozen C graph with three deterministic, legal shape samples.

C seeds remain byte-for-byte unchanged. D fails explicitly when the fixed graph
has no bounded varying input extent; it never retries a different graph.
"""
from functools import lru_cache
from dataclasses import replace
import json
import math

from ..contracts import Seed
from . import nnsmith_v3

ORACLE_ID = 'nnsmith-cpu/4'


def _reshape(tensor, shape):
    shape = tuple(shape)
    return replace(tensor, shape=shape,
                   stride=tuple(math.prod(shape[i+1:]) for i in range(len(shape))))


def output_shapes(graph, inputs, weights, max_elements=4096):
    values = {t.name: t.shape for t in (*inputs, *weights)}
    for node in graph['nodes']:
        shapes = [values[name] for name in node['inputs']]
        shape = list(shapes[0]); op = node['op']; attrs = node['attrs']
        if op in {'add', 'mul'}:
            rank = max(map(len, shapes)); shape = []
            for i in range(rank):
                dims = {s[i-rank] if len(s) >= rank-i else 1 for s in shapes} - {1}
                if len(dims) > 1:
                    raise ValueError('incompatible broadcast')
                shape.append(next(iter(dims), 1))
        elif op == 'transpose':
            a, b = attrs['dim0'], attrs['dim1']
            shape[a], shape[b] = shape[b], shape[a]
        elif op in {'sum', 'mean', 'amin', 'amax'}:
            dims = attrs['reduce_dim']
            dims = list(range(len(shape))) if dims is None else dims if isinstance(dims, list) else [dims]
            dims = {d % len(shape) for d in dims} if shape else set()
            shape = [s for i, s in enumerate(shape) if i not in dims]
        elif op not in {'neg', 'relu'}:
            raise ValueError('unsupported shape inference')
        if math.prod(shape) > max_elements:
            raise ValueError('dynamic intermediate exceeds element budget')
        values[node['output']] = tuple(shape)
    return [values[name] for name in graph['outputs']]


def variants(spec, max_elements=4096, dim_range=None, rng=None):
    graph = json.loads(spec.graph_json)
    # Changing every occurrence preserves input/input broadcast relationships.
    # Never resize a captured constant/weight or specialize singleton dimensions.
    fixed = {d for t in spec.weights for d in t.shape}
    extents = sorted({d for t in spec.inputs for d in t.shape if d > 1} - fixed)
    for extent in extents:
        samples = [spec.inputs]; outputs = [spec.output_samples[0]]
        sizes = sorted({2, 3, 4, max(2, extent-1), extent+1} - {extent})
        if rng is not None:
            sizes = sorted({2, 3, 4, 8, 16, max(2, extent//2), max(2, extent-1),
                            extent+1, extent+2} - {extent})
            rng.shuffle(sizes)
        for size in sizes:
            if dim_range is not None and not dim_range[0] <= size <= dim_range[1]:
                continue
            inputs = tuple(_reshape(t, [size if d == extent else d for d in t.shape]) for t in spec.inputs)
            if any(math.prod(t.shape) > max_elements for t in inputs):
                continue
            try:
                shapes = output_shapes(graph, inputs, spec.weights, max_elements)
            except (ValueError, IndexError):
                continue
            samples.append(inputs)
            outputs.append(tuple(_reshape(t, shape) for t, shape in zip(spec.output_samples[0], shapes)))
            if len(samples) == 3:
                return replace(spec, shape_samples=tuple(samples), output_samples=tuple(outputs))
    raise ValueError('D seed fixed graph has no three legal bounded shapes; no static fallback')


@lru_cache(maxsize=8)
def plan(token):
    seed = Seed.parse(token)
    if seed.version != 4:
        raise ValueError('D seed required')
    base = nnsmith_v3.plan(Seed(2*2**128 + seed.entropy))
    return variants(replace(base, seed=seed))


def materialize(token):
    return nnsmith_v3.materialize_spec(plan(token))


def reference(bundle, index):
    raise ValueError('NNSmith requires CPU eager oracle')
