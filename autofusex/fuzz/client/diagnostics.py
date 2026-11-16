"""Opt-in D0 trace data, never a replay cache or default result archive."""
import hashlib
from .contracts import canonical_json, fingerprint


def first_divergence(expected, actual):
    """Return first differing stage and JSON path, preserving all sequence order."""
    for trace in (expected, actual):
        if not isinstance(trace, list) or not trace or any(
            not isinstance(entry, dict) or not isinstance(entry.get("stage"), str)
            or "payload" not in entry for entry in trace
        ):
            raise ValueError("expected a nonempty list of stage/payload records")
    def walk(a, b, path):
        if type(a) is not type(b):
            return {"path": path, "expected": a, "actual": b}
        if isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    return {"path": f"{path}.{key}", "expected": a.get(key), "actual": b.get(key), "missing": True}
                diff = walk(a[key], b[key], f"{path}.{key}")
                if diff:
                    return diff
        elif isinstance(a, list):
            for i in range(max(len(a), len(b))):
                if i >= len(a) or i >= len(b):
                    return {"path": f"{path}[{i}]", "expected": a[i:] , "actual": b[i:]}
                diff = walk(a[i], b[i], f"{path}[{i}]")
                if diff:
                    return diff
        elif a != b:
            return {"path": path, "expected": a, "actual": b}
        return None
    for i in range(max(len(expected), len(actual))):
        if i >= len(expected) or i >= len(actual):
            return {"stage": "trace_length", "expected": len(expected), "actual": len(actual)}
        diff = walk(expected[i], actual[i], "$")
        if diff:
            return {"stage": expected[i]["stage"], **diff}
    return None


def graph_snapshot(ir):
    # Legacy IR text omits some extra_attrs (e.g. Transpose swap axes).
    # Capture those explicitly instead of treating text equality as graph equality.
    return {"ir": str(ir), "variables": [
        {"name": n, "dtype": str(t.dtype), "shape": list(t.shape)} for n, t in ir.vars.items()]}


def tensor_signature(items):
    """Ordered NumPy/Torch signatures and logical bytes, including layout/alias.

    D0 only: captures native byte representation; not the D2 cross-arch materializer.
    """
    rows, aliases = [], {}
    for name, value in items:
        if hasattr(value, "detach"):
            tensor = value.detach().cpu()
            storage = tensor.untyped_storage()
            key = ("torch", storage._cdata)
            shape, stride = list(tensor.shape), list(tensor.stride())
            offset = tensor.storage_offset()
            # uint8 view supports bfloat16 without a NumPy dtype conversion.
            import torch
            raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            dtype = str(tensor.dtype)
        else:
            root = value
            while getattr(root, "base", None) is not None:
                root = root.base
            key = ("numpy", id(root))
            shape = list(value.shape)
            stride = [int(s // value.itemsize) for s in value.strides]
            address = value.__array_interface__["data"][0]
            base_address = getattr(root, "__array_interface__", value.__array_interface__)["data"][0]
            offset = (address - base_address) // value.itemsize
            raw, dtype = value.tobytes(order="C"), str(value.dtype)
        alias = aliases.setdefault(key, len(aliases))
        rows.append({"name": name, "dtype": dtype, "shape": shape, "stride": stride,
                     "offset": offset, "alias_group": alias,
                     "sha256": hashlib.sha256(raw).hexdigest()})
    return rows


class GenerationTrace:
    def __init__(self):
        self.stages = []

    def record(self, stage, payload):
        # Round-trip detaches mutable containers at the observation boundary.
        import json
        self.stages.append({"stage": stage, "payload": json.loads(canonical_json(payload))})

    def graph(self, ir):
        self.record("graph", graph_snapshot(ir))
        self.record("graph_extra_attrs", [
            {"instruction": inst.identifier, "args": list(inst.iexpr.args),
             "extra_attrs": dict(getattr(inst.iexpr.op, "extra_attrs", {}))}
            for inst in ir.insts
        ])

    def weights(self, stage, model):
        self.record(stage, tensor_signature(model.native_model.state_dict().items()))

    def inputs(self, stage, inputs):
        self.record(stage, tensor_signature(inputs.items()))

    @property
    def digest(self):
        return fingerprint(self.stages)


class InputSignatureError(ValueError):
    pass


def validate_input_signature(expected, actual, *, stage):
    """Check ordered names, dtype and size before backend binding, without repair.

    expected: ordered records with name/dtype/shape; actual: ordered NumPy inputs.
    Layout/alias remain visible in the trace, not guessed from NNSmith's AbsTensor.
    """
    observed = [{"name": n, "dtype": str(v.dtype), "shape": list(v.shape)}
                for n, v in actual.items()]
    difference = first_divergence([{"stage": stage, "payload": expected}],
                                 [{"stage": stage, "payload": observed}])
    if difference:
        raise InputSignatureError(canonical_json(difference))
