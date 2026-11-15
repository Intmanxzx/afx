"""Seeded graph generation must survive process and candidate-order changes.

These use real NNSmith/Z3 on CPU; no NPU compilation or hardware probe is needed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("nnsmith")
pytest.importorskip("z3")
# Tests require an explicitly prepared compatible NNSmith copy; imports never patch it.
import autofusex  # noqa: F401

pytest.importorskip("nnsmith.graph_gen")

from nnsmith.abstract.op import Add, Mul, Neg, ReLU
from nnsmith.graph_gen import SymbolicGen, check_sat, model_gen
import z3


def _graph(seed, ops):
    # Match the runtime boundary: each generation uses an independent Z3 context.
    from z3 import z3 as impl
    previous = z3.main_ctx()
    try:
        impl._main_ctx = z3.Context()
        ir = model_gen(
            ops, method="symbolic-cinit", seed=seed, init_fp=True,
            max_nodes=4, timeout_ms=10000, concr_ph_dim_rng=(1, 8),
        ).make_concrete()
        return {
            "instructions": str(ir),
            "tensors": [(name, str(t.dtype), list(t.shape)) for name, t in ir.vars.items()],
        }
    finally:
        impl._main_ctx = previous


@pytest.mark.parametrize("seed", [0, 42, 389609433])
def test_seed_survives_candidate_order_and_previous_cases(seed):
    ops = [Add, Mul, Neg, ReLU]
    original = _graph(seed, ops)
    _graph(seed + 1, ops)
    assert _graph(seed, list(reversed(ops))) == original


def test_seed_survives_fresh_process_and_hash_randomization():
    worker = '''
import json
import autofusex
from nnsmith.abstract.op import Add, Mul, Neg, ReLU
from nnsmith.graph_gen import model_gen
ops = list({Add, Mul, Neg, ReLU})
results = []
for seed in [0, 42, 389609433]:
    ir = model_gen(ops, method="symbolic-cinit", seed=seed, init_fp=True,
                   max_nodes=4, timeout_ms=10000, concr_ph_dim_rng=(1, 8)).make_concrete()
    results.append([str(ir), [(n, str(t.dtype), list(t.shape)) for n, t in ir.vars.items()]])
print(json.dumps(results))
'''
    results = []
    for hash_seed in ("0", "1", "123"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run(
            [sys.executable, "-c", worker], env=env,
            check=True, capture_output=True, text=True, timeout=60,
        )
        results.append(json.loads(result.stdout))
    assert results[0] == results[1] == results[2]


def test_generation_timeout_rejects_partial_graph(monkeypatch):
    import nnsmith.graph_gen as graph_gen

    gen = SymbolicGen([Add], seed=42, symbolic_init=False, init_fp=True)
    # Simulate a machine that exhausts the deadline before the first insertion.
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(graph_gen.time, "time", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="seed=42, nodes=0/4"):
        gen.abstract_gen(max_node_size=4, max_gen_millisec=100)


def test_solver_unknown_is_not_skipped_as_unsatisfiable():
    solver = SimpleNamespace(check=lambda *args: z3.unknown,
                             reason_unknown=lambda: "timeout")
    with pytest.raises(RuntimeError, match="SMT solver returned unknown: timeout"):
        check_sat(solver)
