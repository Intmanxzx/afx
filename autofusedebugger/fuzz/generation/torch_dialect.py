"""Data-only, faithful serialization of trusted NNSmith Torch operators.

Never load pickle, eval a target, or import a module named by a case. Operators
are resolved exclusively from the installed, source-pinned Torch registry.
"""
import hashlib
import json
import math
from functools import lru_cache


@lru_cache(maxsize=4)
def classes(protocol=7):
    if protocol in {9, 10}:
        from autofusedebugger.nnsmith_integration.aten_dialect import classes as aten_classes
        return aten_classes()
    from nnsmith.materialize.torch.forward import ALL_TORCH_OPS
    return sorted({op for op in ALL_TORCH_OPS
                   if op.__name__ not in {'ConcreteOp', 'Input', 'Constant'}},
                  key=lambda op: (op.__module__, op.__name__))


def key(cls):
    return cls.__module__ + '.' + cls.__name__


def encode(value):
    from nnsmith.abstract.dtype import DType
    from nnsmith.abstract.tensor import AbsTensor
    if isinstance(value, DType):
        return {'$dtype': str(value)}
    if isinstance(value, AbsTensor):
        return {'$tensor': {'shape': encode(value.shape), 'dtype': encode(value.dtype)}}
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: encode(v) for k, v in value.items()}
    import z3
    if isinstance(value, z3.IntNumRef):
        return value.as_long()
    raise ValueError(f'operator metadata cannot be faithfully serialized: {type(value).__name__}')


def decode(value):
    from nnsmith.abstract.dtype import DType
    from nnsmith.abstract.tensor import AbsTensor
    if isinstance(value, dict):
        if set(value) == {'$dtype'}:
            return DType.from_str(value['$dtype'])
        if set(value) == {'$tensor'}:
            tensor = value['$tensor']
            return AbsTensor(decode(tensor['shape']), decode(tensor['dtype']))
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def capture(op, protocol=7):
    if type(op) not in classes(protocol):
        raise ValueError('unregistered Torch operator')
    return {'class': key(type(op)), 'state': encode(vars(op))}


def execute_graph(torch, graph, weights, names, device):
    from nnsmith.materialize.torch.forward import forward_fn
    protocol = 9 if graph.get('dialect') == 'nnsmith-sampled/9' else 7
    registry = {key(cls): cls for cls in classes(protocol)}
    functions = []
    for node in graph['nodes']:
        data = node['materializer']
        cls = registry.get(data['class'])
        if cls is None:
            raise ValueError('unknown registered Torch operator')
        state = data['state']
        if not isinstance(state, dict) or any(k.startswith('__') for k in state):
            raise ValueError('invalid operator state')
        op = cls.__new__(cls)
        op.__dict__.update(decode(state))
        # Module initializers must be identical on every device/worker and
        # must not perturb the reusable worker's global Torch RNG stream.
        seed = int.from_bytes(hashlib.sha256(json.dumps(node, sort_keys=True, separators=(',', ':')).encode()).digest()[:8], 'little')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            if protocol == 9:
                from autofusedebugger.nnsmith_integration.aten_dialect import materialize
                function = materialize(op)
            else:
                function = forward_fn(op)
        if isinstance(function, torch.nn.Module):
            function = function.to(device).eval()
        functions.append(function)

    def forward(*inputs):
        values = dict(weights)
        values.update(zip(names, inputs))
        for node, function in zip(graph['nodes'], functions):
            output = function(*(values[name] for name in node['inputs']))
            if protocol == 9:
                outputs = node['outputs']
                actual = (output,) if len(outputs) == 1 else output
                if not isinstance(actual, (tuple, list)) or len(actual) != len(outputs):
                    raise ValueError('ATen return arity differs from NNSmith model')
                values.update(zip(outputs, actual))
            else:
                values[node['output']] = output
        return tuple(values[name] for name in graph['outputs'])
    return forward
