"""E/F: restored historical sampling with configuration embedded in the seed.

C/D generation and storage semantics remain frozen in their existing modules.
"""
from functools import lru_cache
import hashlib
import math
import struct

from ..contracts import Seed, TensorSpec, CaseSpec, CaseBundle, canonical_json

ORACLE_ID = 'nnsmith-cpu/5'
SOURCE_SHA256 = 'c47e5611db39361963be90786c7ea208647df4b6eb9546ce911122b81721cba5'
Z3_VERSION = '4.12.2'
OPS = ('Add', 'Mul', 'Neg', 'ReLU', 'Transpose', 'ReduceSum', 'ReduceMean', 'ReduceMin', 'ReduceMax')
NAMES = dict(zip(OPS, ('add', 'mul', 'neg', 'relu', 'transpose', 'sum', 'mean', 'amin', 'amax')))


from .nnsmith_v3 import source_evidence
from .sampling import decode
from dataclasses import asdict


def _tensor(name, abstract, alias=0):
    shape = tuple(int(v) for v in abstract.shape)
    stride = tuple(math.prod(shape[i+1:]) for i in range(len(shape)))
    return TensorSpec(name, str(abstract.dtype), shape, stride, alias_group=alias)


@lru_cache(maxsize=8)
def plan(token):
    seed = Seed.parse(token)
    if seed.version not in {5, 6}:
        raise ValueError('E/F seed required')
    config, random_seed = decode(seed)
    evidence = source_evidence()
    from nnsmith.abstract import op as pinned_ops
    if pinned_ops.__MAX_RANK__ != 5:
        raise ValueError('E/F requires default NNSMITH_MAX_RANK=5; environment must not alter replay')
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
        random.seed(random_seed)
        node_count = random.randint(2, config.max_nodes) if config.max_nodes >= 2 else 1
        for op in ops:
            pairs = [(a,b) for a,b in zip(op.in_dtypes,op.out_dtypes)
                     if all(t in ({DType.float16, DType.float32} if config.dtype_mode == 'mixed'
                                  else {getattr(DType, config.dtype_mode)}) for t in (*a,*b))]
            op.in_dtypes, op.out_dtypes = [a for a,b in pairs], [b for a,b in pairs]
        from nnsmith.graph_gen import set_z3_state
        set_z3_state(random_seed % (2**31-1))
        ir = model_gen(ops, method='symbolic-cinit', seed=random_seed % (2**31-1),
                       init_fp=True, max_nodes=node_count, timeout_ms=config.timeout_seconds*1000,
                       concr_ph_dim_rng=(config.dim_min, config.dim_max),
                       dim_dist_mode=config.dim_dist_mode,
                       max_elem_per_tensor=config.max_elem_per_tensor).make_concrete()
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
        if len(nodes) != node_count:
            raise ValueError('NNSmith generation ended before its fixed node budget; no partial case accepted')
        outputs = tuple(_tensor(name, ir.vars[name]) for name in ir.leaf_var())
        graph = {'dialect': 'nnsmith-sampled/5', 'sampling': asdict(config), 'nodes': nodes, 'outputs': [t.name for t in outputs],
                 'reduce_axis': None, 'generator': evidence['generator'],
                 'source_sha256': SOURCE_SHA256, 'z3': Z3_VERSION}
        spec = CaseSpec(seed, canonical_json(graph), tuple(inputs), tuple(weights), (tuple(inputs),), (outputs,))
        if seed.version == 6:
            from .nnsmith_dynamic import variants
            spec = variants(spec, max_elements=config.max_elem_per_tensor,
                            dim_range=(config.dim_min, config.dim_max), rng=random.Random(random_seed))
        return spec
    finally:
        for op, ins, outs in original:
            op.in_dtypes, op.out_dtypes = ins, outs
        random.setstate(state)
        impl._main_ctx = context


def materialize(token):
    return materialize_spec(plan(token))


def materialize_spec(spec):
    """Role-aware historical distributions, with seed-local canonical RNG streams.

    Unlike C/D's discrete [-2,2] lattice, uniform is historical [3,7).
    Gaussian/log-scale formulas match the legacy value-distribution patch.
    """
    import random
    from array import array
    import sys
    config, _ = decode(spec.seed)
    storages = [None] * (len(spec.inputs) + len(spec.weights))
    for tensor in (*spec.inputs, *spec.weights):
        size = max(math.prod(t.shape) for t in (*spec.inputs, *spec.weights,
                   *(t for sample in spec.shape_samples for t in sample))
                   if t.alias_group == tensor.alias_group)
        rng = random.Random(int.from_bytes(hashlib.sha256(
            f'nnsmith-sampled/5/{spec.seed}/{tensor.name}'.encode()).digest(), 'little'))
        weight = tensor in spec.weights
        mode = config.value_dist_mode
        def value():
            if mode == 'uniform':
                return 3 + 4 * rng.random()
            if mode == 'log_scale':
                sign = -1 if rng.random() < 0.5 else 1
                return sign * 10 ** (8 * rng.random() - 4)
            if mode == 'llm':
                if weight:
                    return rng.gauss(0, 0.02)
                base = rng.gauss(0, 0.2)
                outlier = rng.random() < 0.05
                amplification = 100 + 9900 * rng.random()
                return base * (amplification if outlier else 1)
            return rng.gauss(0, 1 / math.sqrt(size)) if weight else rng.random()
        if tensor.dtype == 'float16':
            # Rounding may overflow in future value modes; fail, never silently clamp.
            storages[tensor.alias_group] = b''.join(struct.pack('<e', value()) for _ in range(size))
            continue
        data = array('f', (value() for _ in range(size)))
        if data.itemsize != 4:
            raise ValueError('canonical float32 storage required')
        if sys.byteorder != 'little':
            data.byteswap()
        storages[tensor.alias_group] = data.tobytes()
    return CaseBundle(spec, tuple(storages))


def reference(bundle, index):
    raise ValueError('NNSmith requires CPU eager oracle')
