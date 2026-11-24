"""Current E/F NNSmith ATen generation and canonical values."""
from functools import lru_cache
import hashlib
import math
import struct

from autofusex.workloads.fuzz.contracts import Seed, TensorSpec, CaseSpec, CaseBundle, canonical_json

from autofusex.workloads.fuzz.generation.runtime import source_evidence, SOURCE_SHA256, Z3_VERSION
from autofusex.workloads.fuzz.generation.sampling import decode
from dataclasses import asdict


def _tensor(name, abstract, alias=0):
    shape = tuple(int(v) for v in abstract.shape)
    stride = tuple(math.prod(shape[i+1:]) for i in range(len(shape)))
    return TensorSpec(name, str(abstract.dtype), shape, stride, alias_group=alias)


@lru_cache(maxsize=8)
def plan(token):
    seed = Seed.parse(token)
    config, random_seed = decode(seed)
    evidence = source_evidence()
    from nnsmith.abstract import op as pinned_ops
    if pinned_ops.__MAX_RANK__ != 5:
        raise ValueError('E/F requires default NNSMITH_MAX_RANK=5; environment must not alter replay')
    from nnsmith.abstract.dtype import DType
    from nnsmith.graph_gen import model_gen
    import random
    import z3
    from z3 import z3 as impl
    # All selection is fixed, not filtered by device/backend capabilities.
    from autofusex.workloads.fuzz.nnsmith.aten_dialect import classes, validate_runtime
    validate_runtime()
    ops = classes()
    original = [(op, op.in_dtypes, op.out_dtypes) for op in ops]
    state = random.getstate()
    context = z3.main_ctx()
    try:
        impl._main_ctx = z3.Context()
        random.seed(random_seed)
        node_count = random.randint(2, config.max_nodes) if config.max_nodes >= 2 else 1
        # Before the protocol refactor dtype selection was not restricted to
        # fp16/fp32.  NNSmith's DTYPE_GEN_ALL is the source of truth and the
        # pinned operator signatures provide the final intersection, not a
        # machine-dependent hardware probe (which would change seed replay).
        if config.dtype_mode == 'mixed':
            allowed = {DType.float16, DType.float32}
        elif config.dtype_mode == 'all':
            from nnsmith.abstract.dtype import DTYPE_GEN_ALL
            allowed = set(DTYPE_GEN_ALL)
        else:
            dtype = getattr(DType, config.dtype_mode, None)
            if dtype is None:
                raise ValueError(f'unknown NNSmith dtype: {config.dtype_mode}')
            allowed = {dtype}
        eligible_ops = []
        for op in ops:
            # A selectable model must declare a nonempty input/output signature.
            if not op.in_dtypes or not op.out_dtypes:
                continue
            pairs = [(a, b) for a, b in zip(op.in_dtypes, op.out_dtypes)
                     if all(t in allowed for t in a)]
            # Do not leave an operator with an empty signature list: remove it
            # from this mode's candidate catalog instead. This is important for
            # explicit integer/complex modes where many floating-point-only
            # operators have no legal signature.
            if pairs:
                op.in_dtypes, op.out_dtypes = [a for a, b in pairs], [b for a, b in pairs]
                eligible_ops.append(op)
        if not eligible_ops:
            raise ValueError(f'no NNSmith operators support dtype mode {config.dtype_mode!r}')
        from nnsmith.graph_gen import set_z3_state
        set_z3_state(random_seed % (2**31-1))
        ir = model_gen(eligible_ops, method=config.method, seed=random_seed % (2**31-1),
                       dtype_choices=sorted(allowed, key=str),
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
                from autofusex.workloads.fuzz.generation.torch_dialect import capture, encode
                target = getattr(getattr(op, 'rule', None), 'target', None) or op.name()
                nodes.append({'op': target, 'inputs': list(inst.iexpr.args),
                              'output': output, 'outputs': list(inst.retvals()),
                              'attrs': encode(dict(op.extra_attrs)), 'materializer': capture(op)})
        if len(nodes) != node_count:
            raise ValueError('NNSmith generation ended before its fixed node budget; no partial case accepted')
        outputs = tuple(_tensor(name, ir.vars[name]) for name in ir.leaf_var())
        graph = {'dialect': 'nnsmith-aten/2', 'sampling': asdict(config), 'nodes': nodes,
                 'outputs': [t.name for t in outputs], 'reduce_axis': None,
                 'generator': evidence['generator'], 'source_sha256': SOURCE_SHA256,
                 'z3': Z3_VERSION, 'aten_rules': 'elementwise-comparison-reduce-view/v1',
                 # Historical E/F wire marker, not the executing Torch version.
                 # Keep it frozen so existing case identities remain unchanged;
                 # actual software versions belong to execution.environment.
                 'torch': '2.10.0+cpu'}
        spec = CaseSpec(seed, canonical_json(graph), tuple(inputs), tuple(weights), (tuple(inputs),), (outputs,))
        if seed.dynamic:
            from autofusex.workloads.fuzz.generation.nnsmith_dynamic import variants
            spec = variants(spec, max_elements=config.max_elem_per_tensor,
                            count=config.dynamic_variants, rng=random.Random(random_seed ^ 0xD1A5E))
        from autofusex.workloads.fuzz.case_spec import encode
        encode(spec)  # Reject oversized graphs before CPU/NPU execution.
        return spec
    finally:
        for op, ins, outs in original:
            op.in_dtypes, op.out_dtypes = ins, outs
        random.setstate(state)
        impl._main_ctx = context


def materialize(token):
    return materialize_spec(plan(token))


def _round_to_bfloat16(value):
    """Encode a Python float as canonical little-endian bfloat16 bytes."""
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    # Round to nearest-even before dropping the low 16 bits.
    lower = bits & 0xFFFF
    upper = bits >> 16
    if lower > 0x8000 or (lower == 0x8000 and (upper & 1)):
        upper = (upper + 1) & 0xFFFF
    return struct.pack("<H", upper)


def _canonical_storage(dtype, values):
    """Pack generated scalar values for every NNSmith generation dtype."""
    if dtype == 'bfloat16':
        return b''.join(_round_to_bfloat16(v) for v in values)
    if dtype in {'complex64', 'complex128'}:
        scalar = 'f' if dtype == 'complex64' else 'd'
        flat = [component for value in values
                for component in (complex(value).real, complex(value).imag)]
        return struct.pack('<' + scalar * len(flat), *flat)
    formats = {
        'bool': '?', 'uint8': 'B', 'uint16': 'H', 'uint32': 'I',
        'int8': 'b', 'int16': 'h', 'int32': 'i', 'int64': 'q', 'float16': 'e',
        'float32': 'f', 'float64': 'd',
    }
    try:
        fmt = formats[dtype]
    except KeyError as exc:
        raise ValueError(f'unsupported canonical storage dtype: {dtype}') from exc
    if dtype == 'bool':
        values = [bool(v) for v in values]
    elif dtype.startswith(('int', 'uint')):
        bits = int(dtype.removeprefix('uint').removeprefix('int'))
        if dtype.startswith('uint'):
            low, high = 0, 2 ** bits - 1
        else:
            low, high = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        values = [max(low, min(high, round(v))) for v in values]
    return struct.pack('<' + fmt * len(values), *values)


def materialize_spec(spec):
    """Role-aware historical distributions with complete NNSmith dtype coverage."""
    import random
    config, _ = decode(spec.seed)
    storages = [None] * (len(spec.inputs) + len(spec.weights))
    for tensor in (*spec.inputs, *spec.weights):
        size = max(math.prod(t.shape) for t in (*spec.inputs, *spec.weights,
                   *(t for sample in spec.shape_samples for t in sample))
                   if t.alias_group == tensor.alias_group)
        rng = random.Random(int.from_bytes(hashlib.sha256(
            f'nnsmith-aten/2/{spec.seed}/{tensor.name}'.encode()).digest(), 'little'))
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
        values = [complex(value(), value()) if tensor.dtype.startswith('complex') else value() for _ in range(size)]
        storages[tensor.alias_group] = _canonical_storage(tensor.dtype, values)
    return CaseBundle(spec, tuple(storages))


def reference(bundle, index):
    raise ValueError('NNSmith requires CPU eager oracle')
