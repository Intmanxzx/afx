"""Frozen minimal v1 protocol, preview pending cross-architecture acceptance.

No Torch, NumPy, solver, device inspection, wall clock, global RNG, files or cache.
Keep this implementation for numeric version 1; changed semantics need a new version.
"""
from dataclasses import asdict
import hashlib
import math
import struct

from .contracts import CaseBundle, CaseSpec, Seed, TensorSpec, canonical_json, fingerprint

DTYPES = {"float16": ("e", 2), "float32": ("f", 4)}
ORACLE_ID = "af-mini-reference-v1-exact-dyadic"


def draw(seed, field, count):
    """Field-addressed SHA256 rejection sampler; never Python hash/random.

    Preimage: ASCII 'autofusedbg/numeric/v1\0' + canonical token + '\0' + ASCII field
    + '\0' + unsigned big-endian uint64 attempt. Digest is big-endian uint256.
    """
    if not 1 <= count <= 2**256:
        raise ValueError("invalid choice count")
    prefix = b"autofusedbg/numeric/v1\0" + str(seed).encode("ascii") + b"\0" + field.encode("ascii") + b"\0"
    limit = 2**256 - (2**256 % count)
    for attempt in range(2**64):
        number = int.from_bytes(hashlib.sha256(prefix + attempt.to_bytes(8, "big")).digest(), "big")
        if number < limit:
            return number % count
    raise RuntimeError("field sampler exhausted")


def resolve_seed(value):
    return Seed.parse(value)

def plan(token):
    seed = resolve_seed(token)
    dtype = ("float16", "float32")[draw(seed, "dtype", 2)]
    broadcast_axis = draw(seed, "broadcast_axis", 2)
    fixed_size = 2 + draw(seed, "fixed_size", 3)
    base_size = 2 + draw(seed, "base_size", 3)
    sizes = (base_size, base_size + 1, base_size + 2)
    transposed = bool(draw(seed, "input_layout", 2))
    operation = ("relu", "neg", "mul")[draw(seed, "operation", 3)]
    reduction = draw(seed, "reduction", 3)  # 0:none, 1:axis0, 2:axis1
    weight_shape = [fixed_size, fixed_size]
    weight_shape[broadcast_axis] = 1
    weight = TensorSpec("weight", dtype, tuple(weight_shape), (weight_shape[1], 1), alias_group=0)
    samples, outputs = [], []
    for index, size in enumerate(sizes):
        shape = [fixed_size, fixed_size]
        shape[broadcast_axis] = size
        stride = (1, shape[0]) if transposed else (shape[1], 1)
        inputs = (TensorSpec("x", dtype, tuple(shape), stride, alias_group=index + 1),)
        samples.append(inputs)
        output_shape = tuple(shape) if reduction == 0 else (shape[2 - reduction],)
        # Output alias IDs are not bound to input storage: outputs are independently produced.
        outputs.append((TensorSpec("output", dtype, output_shape,
                                  (output_shape[1], 1) if len(output_shape) == 2 else (1,)),))
    graph = {"dialect": "af-mini/1", "inputs": ["x"], "weights": ["weight"],
             "nodes": [{"op": "add", "inputs": ["x", "weight"], "output": "sum",
                        "weight_broadcast_axes": [broadcast_axis]},
                       {"op": operation, "inputs": ["sum", "weight"] if operation == "mul" else ["sum"],
                        "output": "activated"}],
             "reduce_axis": None if reduction == 0 else reduction - 1,
             "output_tree": {"tuple": ["output"]}, "dynamic_axis": broadcast_axis if len(sizes) > 1 else None}
    return CaseSpec(seed, canonical_json(graph), samples[0], (weight,), tuple(samples), tuple(outputs))


def materialize(token):
    spec = plan(token)
    tensors = (*spec.weights, *(s[0] for s in spec.shape_samples))
    storages = []
    for tensor in tensors:
        # Each value is k/8, k in [-8,8]. Exact binary fractions for both dtypes;
        # bounded add/mul/sum also remain exactly representable for this profile.
        fmt, _ = DTYPES[tensor.dtype]
        count = math.prod(tensor.shape)
        values = [(draw(spec.seed, f"storage/{tensor.alias_group}/element/{i}", 17) - 8) / 8
                  for i in range(count)]
        storages.append(struct.pack("<" + fmt * count, *values))
    return CaseBundle(spec, tuple(storages))


def content_manifest(bundle):
    spec = asdict(bundle.spec)
    spec.pop("seed")  # Content identity, not provenance.
    return {"schema": "af-mini-workload/1", "spec": spec,
            "storages": [{"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for b in bundle.storages]}


def workload_id(bundle):
    return fingerprint(content_manifest(bundle))


def storage_values(bundle, tensor):
    fmt, width = DTYPES[tensor.dtype]
    raw = bundle.storages[tensor.alias_group]
    return struct.unpack("<" + fmt * (len(raw) // width), raw)


def reference(bundle, sample_index):
    """Independent scalar oracle using exact dyadics, no Torch eager dependency.

    Return ordered (TensorSpec, logical little-endian bytes) pairs. Not a cache.
    """
    import json
    graph = json.loads(bundle.spec.graph_json)
    x = bundle.spec.shape_samples[sample_index][0]
    weight = bundle.spec.weights[0]
    xv, wv = storage_values(bundle, x), storage_values(bundle, weight)
    rows, cols = x.shape
    values = []
    for row in range(rows):
        for col in range(cols):
            a = xv[x.storage_offset + row * x.stride[0] + col * x.stride[1]]
            wi = (row % weight.shape[0]) * weight.stride[0] + (col % weight.shape[1]) * weight.stride[1]
            b = wv[weight.storage_offset + wi]
            value = a + b
            op = graph["nodes"][1]["op"]
            values.append(max(0.0, value) if op == "relu" else -value if op == "neg" else value * b)
    axis = graph["reduce_axis"]
    if axis == 0:
        values = [sum(values[row * cols + col] for row in range(rows)) for col in range(cols)]
    elif axis == 1:
        values = [sum(values[row * cols:(row + 1) * cols]) for row in range(rows)]
    output = bundle.spec.output_samples[sample_index][0]
    fmt, _ = DTYPES[output.dtype]
    return ((output, struct.pack("<" + fmt * len(values), *values)),)
