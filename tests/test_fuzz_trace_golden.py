"""Hand-constructed diagnostic fixtures, NOT v1 generator golden acceptance."""
import hashlib
import struct

import numpy as np
import pytest

from autofusedebugger.fuzz.diagnostics import GenerationTrace, tensor_signature


@pytest.mark.parametrize("dtype,fmt", [("float32", "f"), ("float16", "e"), ("int32", "i")])
def test_broadcast_reduce_layout_and_dynamic_input_bytes(dtype, fmt):
    # Fixed literals, no library RNG or device capability choices.
    weight = np.array([[1, 2, 3]], dtype=dtype)
    trace = GenerationTrace()
    graph = {"ops": [{"op": "Add", "broadcast_axes": [0]}, {"op": "Sum", "axes": [1]}]}
    trace.record("graph", graph)
    for rows in (2, 4):
        x = np.arange(rows * 3).astype(dtype).reshape(rows, 3)
        expected = struct.pack("<" + fmt * (rows * 3), *range(rows * 3))
        signature = tensor_signature([("x", x), ("weight", weight), ("view", x.T)])
        assert signature[0]["sha256"] == hashlib.sha256(expected).hexdigest()
        assert signature[0]["alias_group"] == signature[2]["alias_group"]
        assert signature[2]["stride"] == [1, 3]
        trace.record(f"shape_{rows}", signature)
        assert np.sum(x + weight, axis=1).shape == (rows,)
    graph["ops"][0]["broadcast_axes"] = [1]
    assert trace.stages[0]["payload"]["ops"][0]["broadcast_axes"] == [0]
