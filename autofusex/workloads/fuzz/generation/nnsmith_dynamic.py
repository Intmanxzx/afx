"""Bounded 2–8 shape variants of the same graph; CPU validation happens once later."""
from dataclasses import replace
import json
import math


def _reshape(tensor, shape):
    shape = tuple(shape)
    return replace(tensor, shape=shape,
                   stride=tuple(math.prod(shape[i+1:]) for i in range(len(shape))))


def output_shapes(graph, inputs, weights, max_elements):
    """Reuse the new NNSmith model, including tuple outputs, for variants."""
    from autofusex.workloads.fuzz.generation.torch_dialect import classes, key, decode
    from nnsmith.abstract.tensor import AbsTensor
    from nnsmith.abstract.dtype import DType
    import z3
    registry = {key(cls): cls for cls in classes()}
    values = {t.name: AbsTensor(list(t.shape), DType.from_str(t.dtype)) for t in (*inputs, *weights)}
    for node in graph['nodes']:
        data = node['materializer']
        cls = registry[data['class']]
        op = cls.__new__(cls)
        op.__dict__.update(decode(data['state']))
        tensors = [values[n] for n in node['inputs']]
        solver = z3.Solver()
        solver.add(*op.checked_requires(tensors))
        verdict=solver.check()
        if verdict==z3.unknown:
            raise TimeoutError('dynamic constraint solver returned unknown; no alternate shapes')
        if verdict!=z3.sat:
            raise ValueError('dynamic ATen constraints unsatisfied')
        outputs = op.checked_type_transfer(tensors)
        for name, tensor in zip(node['outputs'], outputs):
            if any(type(d) is not int or d < 0 for d in tensor.shape) or math.prod(tensor.shape) > max_elements:
                raise ValueError('dynamic ATen output exceeds bounds')
            values[name] = tensor
    return [tuple(values[name].shape) for name in graph['outputs']]


def variants(spec, max_elements, count, rng):
    graph = json.loads(spec.graph_json)
    # Changing every occurrence preserves input/input broadcast relationships.
    # Never resize a captured constant/weight or specialize singleton dimensions.
    extents = sorted({d for t in spec.inputs for d in t.shape if d > 1})
    rng.shuffle(extents)
    for extent in extents:
        samples = [spec.inputs]; outputs = [spec.output_samples[0]]
        sizes = sorted({2, 3, 4, 8, 16, max(2, extent//2), max(2, extent-1),
                        extent+1, extent+2} - {extent})
        rng.shuffle(sizes)
        for size in sizes:
            inputs = tuple(_reshape(t, [size if d == extent else d for d in t.shape]) for t in spec.inputs)
            if any(math.prod(t.shape) > max_elements for t in inputs):
                continue
            try:
                shapes = output_shapes(graph, inputs, spec.weights, max_elements)
            except (ValueError, IndexError):
                continue
            samples.append(inputs)
            outputs.append(tuple(_reshape(t, shape) for t, shape in zip(spec.output_samples[0], shapes)))
            if len(samples) == count:
                return replace(spec, shape_samples=tuple(samples), output_samples=tuple(outputs))
    raise ValueError(f'fixed graph has no {count} legal bounded shapes; no static fallback')
