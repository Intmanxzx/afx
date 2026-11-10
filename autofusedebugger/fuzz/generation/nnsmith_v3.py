"""NNSmith E/R/T protocol: actual symbolic graph search, never template selection.

Static single-shape protocol; pinned source/solver gate fails rather than changing
cases. Cross-platform solver determinism is not yet a qualified guarantee.
"""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import struct

from ..contracts import Seed, TensorSpec, CaseSpec, CaseBundle, canonical_json
from .fingerprint import content_manifest, workload_id

ORACLE_ID = 'nnsmith-cpu/3'
SOURCE_SHA256 = 'a1a524f235f8b6299c397fd12c383d4b15a80f027230ece2d7d18259f92c872e'
Z3_VERSION = '4.12.2'
OPS = ('Add', 'Mul', 'Neg', 'ReLU', 'Transpose', 'ReduceSum', 'ReduceMean', 'ReduceMin', 'ReduceMax')
NAMES = dict(zip(OPS, ('add', 'mul', 'neg', 'relu', 'transpose', 'sum', 'mean', 'amin', 'amax')))


def source_evidence():
    import nnsmith
    import z3
    root = Path(nnsmith.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes())
    result = {'generator': 'nnsmith.graph_gen.model_gen', 'source_sha256': digest.hexdigest(),
              'z3': z3.get_version_string(), 'source_path': str(root)}
    if result['source_sha256'] != SOURCE_SHA256 or result['z3'] != Z3_VERSION:
        raise ValueError('NNSmith v3 requires its pinned isolated source and Z3; no fallback')
    return result


def _tensor(name, abstract, alias=0):
    shape = tuple(int(v) for v in abstract.shape)
    stride = tuple(math.prod(shape[i+1:]) for i in range(len(shape)))
    return TensorSpec(name, str(abstract.dtype), shape, stride, alias_group=alias)


@lru_cache(maxsize=8)
def plan(token):
    seed = Seed.parse(token)
    if seed.version != 3:
        raise ValueError('NNSmith v3 seed required')
    evidence = source_evidence()
    from nnsmith.abstract import op as operators
    from nnsmith.abstract.dtype import DType
    from nnsmith.graph_gen import model_gen
    import random
    import z3
    from z3 import z3 as impl
    # All selection is fixed, not filtered by device/backend capabilities.
    ops = [getattr(operators, name) for name in OPS]
    original = [(op, op.in_dtypes, op.out_dtypes) for op in ops]
    state = random.getstate()
    context = z3.main_ctx()
    try:
        impl._main_ctx = z3.Context()
        random.seed(seed.entropy)
        for op in ops:
            pairs = [(a,b) for a,b in zip(op.in_dtypes,op.out_dtypes)
                     if all(t == DType.float32 for t in (*a,*b))]
            op.in_dtypes, op.out_dtypes = [a for a,b in pairs], [b for a,b in pairs]
        from nnsmith.graph_gen import set_z3_state
        set_z3_state(seed.entropy % (2**31-1))
        ir = model_gen(ops, method='symbolic-cinit', seed=seed.entropy % (2**31-1),
                       init_fp=True, max_nodes=8, timeout_ms=10000,
                       concr_ph_dim_rng=(1,8), max_elem_per_tensor=4096).make_concrete()
        nodes, inputs, weights = [], [], []
        for inst in ir.insts:
            op = inst.iexpr.op
            name = type(op).__name__
            output = inst.retval()
            if name == 'Input':
                inputs.append(_tensor(output, ir.vars[output], len(inputs)+len(weights)))
            elif name == 'Constant':
                weights.append(_tensor(output, ir.vars[output], len(inputs)+len(weights)))
            else:
                nodes.append({'op': NAMES[name], 'inputs': list(inst.iexpr.args), 'output': output,
                              'attrs': dict(op.extra_attrs)})
        if len(nodes) != 8:
            raise ValueError('NNSmith generation ended before its fixed node budget; no partial case accepted')
        outputs = tuple(_tensor(name, ir.vars[name]) for name in ir.leaf_var())
        graph = {'dialect': 'nnsmith-ert/3', 'nodes': nodes, 'outputs': [t.name for t in outputs],
                 'reduce_axis': None, 'generator': evidence['generator'],
                 'source_sha256': SOURCE_SHA256, 'z3': Z3_VERSION}
        return CaseSpec(seed, canonical_json(graph), tuple(inputs), tuple(weights), (tuple(inputs),), (outputs,))
    finally:
        for op, ins, outs in original:
            op.in_dtypes, op.out_dtypes = ins, outs
        random.setstate(state)
        impl._main_ctx = context


def materialize(token):
    return materialize_spec(plan(token))


def materialize_spec(spec):
    """Rebuild canonical bytes from captured metadata without another solver run."""
    storages = [None] * (len(spec.inputs)+len(spec.weights))
    for tensor in (*spec.inputs, *spec.weights):
        # Independent canonical values: no Torch/NumPy RNG or numeric-search state.
        values = []
        size=max(math.prod(t.shape) for t in (*spec.inputs,*spec.weights,*(t for sample in spec.shape_samples for t in sample)) if t.alias_group==tensor.alias_group)
        for index in range(size):
            raw = hashlib.sha256(f'nnsmith-v3/{spec.seed}/{tensor.name}/{index}'.encode()).digest()
            values.append((int.from_bytes(raw[:4], 'little') % 65 - 32) / 16)
        storages[tensor.alias_group] = struct.pack('<'+'f'*len(values), *values)
    return CaseBundle(spec, tuple(storages))


def reference(bundle, index):
    raise ValueError('NNSmith v3 requires CPU eager oracle')


def execute_graph(torch, graph, weights, names):
    """Interpret only the fixed op vocabulary; never eval uploaded source."""
    def forward(*inputs):
        values = dict(weights)
        values.update(zip(names, inputs))
        for node in graph['nodes']:
            xs = [values[name] for name in node['inputs']]
            op, attrs = node['op'], node['attrs']
            if op == 'add': out = xs[0] + xs[1]
            elif op == 'mul': out = xs[0] * xs[1]
            elif op == 'neg': out = -xs[0]
            elif op == 'relu': out = torch.relu(xs[0])
            elif op == 'transpose': out = xs[0].transpose(attrs['dim0'], attrs['dim1'])
            else:
                fn = {'sum': torch.sum, 'mean': torch.mean, 'amin': torch.amin, 'amax': torch.amax}[op]
                dims = attrs['reduce_dim']
                out = fn(xs[0], dim=tuple(dims) if isinstance(dims, list) else dims)
            values[node['output']] = out
        return tuple(values[name] for name in graph['outputs'])
    return forward
