"""Candidate snapshots and case identity, never serialized models or tensors."""
from enum import Enum
import hashlib
import json


def snapshot_opset(opset) -> list[dict]:
    return [{"name": op.name(), "in_dtypes": [[dt.name for dt in c] for c in op.in_dtypes],
             "out_dtypes": [[dt.name for dt in c] for c in op.out_dtypes]} for op in opset]


def restore_opset(snapshot, model_cls):
    from nnsmith.abstract.dtype import DType
    available = {op.name(): op for op in model_cls.operators()}
    ops = []
    for item in snapshot:
        op = available[item["name"]]
        op.in_dtypes = [tuple(DType[n] for n in c) for c in item["in_dtypes"]]
        op.out_dtypes = [tuple(DType[n] for n in c) for c in item["out_dtypes"]]
        ops.append(op)
    if not ops:
        raise ValueError("Empty performance op pool")
    return ops


def _normal(value):
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {str(k): _normal(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (list, tuple)):
        return [_normal(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_normal(v) for v in value), key=repr)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    raise TypeError(f"Unsupported graph attribute: {type(value).__name__}")


def case_fingerprint(case) -> str:
    import torch
    ir = case.model.native_model.ir
    h = hashlib.sha256()
    graph = [{"id": i.identifier, "op": i.iexpr.op.name(), "args": list(i.iexpr.args),
              "attributes": _normal(vars(i.iexpr.op))} for i in ir.insts]
    h.update(json.dumps(graph, sort_keys=True).encode())
    for group, values in [("weights", case.model.native_model.state_dict()),
                          ("inputs", case.oracle.input), ("oracle", case.oracle.output)]:
        for name, value in sorted(values.items()):
            header = [group, name, str(value.dtype), list(value.shape)]
            h.update(json.dumps(header).encode())
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            h.update(hashlib.sha256(value.tobytes()).digest())
    return h.hexdigest()


def describe_case(case) -> dict:
    model = case.model
    ir = model.native_model.ir
    return {"n_nodes": ir.n_compute_inst(),
            "op_types": sorted({i.iexpr.op.name() for i in ir.insts}),
            "input_shapes": {n: list(t.shape) for n, t in model.input_like.items()},
            "input_dtypes": {n: t.dtype.short() for n, t in model.input_like.items()},
            "output_shapes": {n: list(t.shape) for n, t in model.output_like.items()},
            "output_dtypes": {n: t.dtype.short() for n, t in model.output_like.items()}}
