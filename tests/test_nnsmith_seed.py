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
import autofusedebugger  # noqa: F401

pytest.importorskip("nnsmith.graph_gen")

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import Add, Mul, Neg, ReLU
from nnsmith.graph_gen import SymbolicGen, check_sat, model_gen
import z3


def _graph(seed, ops):
    ir = model_gen(
        ops, method="symbolic-cinit", seed=seed, init_fp=True,
        max_nodes=4, timeout_ms=10000, concr_ph_dim_rng=(1, 8),
    ).make_concrete()
    return {
        "instructions": str(ir),
        "tensors": [(name, str(t.dtype), list(t.shape)) for name, t in ir.vars.items()],
    }


@pytest.mark.parametrize("seed", [0, 42, 389609433])
def test_seed_survives_candidate_order_and_previous_cases(seed):
    ops = [Add, Mul, Neg, ReLU]
    original = _graph(seed, ops)
    _graph(seed + 1, ops)
    assert _graph(seed, list(reversed(ops))) == original


def test_seed_survives_fresh_process_and_hash_randomization():
    worker = '''
import json
import autofusedebugger
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


def test_probe_canonicalizes_ops_and_paired_dtype_signatures(tmp_path, monkeypatch):
    import nnsmith.narrow_spec as narrow
    from autofusedebugger.nnsmith_integration import ascendc_opset

    # This is a cached hardware-probe result, deliberately in unstable order.
    cache = {
        op.name(): SimpleNamespace(
            in_dtypes=[(DType.float32,) * arity, (DType.float16,) * arity],
            out_dtypes=[(DType.float32,), (DType.float16,)],
        )
        for op, arity in [(Mul, 2), (Neg, 1)]
    }
    monkeypatch.setattr(narrow, "NNSMITH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(narrow, "get_cache_name", lambda *args: "seed")
    monkeypatch.setattr(narrow, "load_topset", lambda path: cache)
    (tmp_path / "seed_ascendc_A3.yaml").touch()
    monkeypatch.setattr(
        ascendc_opset, "_filter_probed_dtypes_by_lowering_guard",
        lambda name, ins, outs, tier: (ins, outs),
    )
    # Restore shared op class state when the test ends.
    for op in (Mul, Neg):
        monkeypatch.setattr(op, "in_dtypes", op.in_dtypes)
        monkeypatch.setattr(op, "out_dtypes", op.out_dtypes)
    model_cls = SimpleNamespace(operators=lambda: [Neg, Mul])
    ops = ascendc_opset.probe_ascendc_opset(model_cls, object())
    assert [op.name() for op in ops] == ["core.Mul", "core.Neg"]
    for op in ops:
        assert op.in_dtypes[0][0] == op.out_dtypes[0][0] == DType.float16
        assert op.in_dtypes[1][0] == op.out_dtypes[1][0] == DType.float32


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


def test_graph_timeout_becomes_case_generation_error(monkeypatch):
    import nnsmith.graph_gen as graph_gen
    from autofusedebugger.nnsmith_integration.fuzz_loop import run_single_case

    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(graph_gen, "time", SimpleNamespace(time=lambda: next(ticks)))
    result = run_single_case(
        factory=object(), opset=[Add], model_cls=object(), case_seed=42,
        index=0, max_nodes=4, timeout_ms=100, tensor_scale="small",
    )
    assert result.status == "gen_error"
    assert result.n_nodes == 0
    assert "TimeoutError" in result.error_summary


def _materialized_case(seed, mode, *, slow=False, out_dir=None, complex_graph=False):
    """Fingerprint every graph instruction and every byte of numeric case data."""
    import hashlib
    import random
    import numpy as np
    import torch
    from nnsmith.materialize.torch import TorchModelCPU
    from nnsmith.materialize.torch import input_gen
    from autofusedebugger.nnsmith_integration import fuzz_loop

    captured = {}
    ops = [Add, Mul, Neg, ReLU]
    if complex_graph:
        from nnsmith.abstract.op import (
            Sqrt, Transpose, Reshape, ReduceMin, Slice, MatMul, ExpandLast1,
        )
        from nnsmith.materialize.torch.dialect import TorchReduceSum
        ops += [Sqrt, Transpose, Reshape, ReduceMin, Slice, MatMul,
                ExpandLast1, TorchReduceSum]

    def array(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous().numpy()
        return [str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]

    def verify(case, **kwargs):
        ir = case.model.native_model.ir
        captured.update(
            graph=[str(ir), [(n, str(t.dtype), list(t.shape)) for n, t in ir.vars.items()]],
            weights={n: array(t) for n, t in case.model.native_model.state_dict().items()},
            inputs={n: array(t) for n, t in case.oracle.input.items()},
            outputs={n: array(t) for n, t in case.oracle.output.items()},
        )
        return None

    with pytest.MonkeyPatch.context() as patch:
        for op in ops:
            patch.setattr(op, "in_dtypes", [
                combo for combo in op.in_dtypes if all(dt == DType.float32 for dt in combo)
            ])
            patch.setattr(op, "out_dtypes", [
                combo for combo in op.out_dtypes if all(dt == DType.float32 for dt in combo)
            ])
        patch.setattr(fuzz_loop, "install_autofuse_intermediate_hooks", lambda: None)
        patch.setattr(fuzz_loop, "reset_autofuse_diagnostics", lambda: None)
        patch.setattr(fuzz_loop, "autofuse_diagnostics", lambda: {})
        original_search = input_gen.PracticalHybridSearch.search_one
        calls = []

        def search_one(self, *args, **kwargs):
            calls.append(1)
            if slow:
                import time
                time.sleep(0.03)  # Exceeds the old 20 ms search budget.
            # Force a retry so the test covers both sample attempts.
            if len(calls) == 1:
                return None
            return original_search(self, *args, **kwargs)

        patch.setattr(input_gen.PracticalHybridSearch, "search_one", search_one)
        # Prior cases/probes/backend code are allowed to consume global RNGs.
        random.seed(seed + 1)
        np.random.seed((seed + 1) % 2**32)
        torch.manual_seed(seed + 1)
        torch.rand(19)
        result = fuzz_loop.run_single_case(
            factory=SimpleNamespace(verify_testcase=verify),
            opset=list(set(ops)), model_cls=TorchModelCPU,
            case_seed=seed, index=13 if slow else 0,
            max_nodes=5, tensor_scale="small", concr_ph_dim_rng=(1, 4),
            max_elem_per_tensor=256, out_dir=out_dir, debug=False,
            value_dist_mode=mode,
        )
    assert result.status == "pass", result.error_summary
    assert len(calls) == 2
    assert captured["inputs"]
    if seed == 42:
        assert captured["weights"], "Regression must exercise actual weights"
    return captured


@pytest.mark.parametrize("mode", ["uniform", "log_scale", "llm", "recommendation"])
def test_full_case_seed_ignores_search_speed_threads_and_prior_rng(mode, tmp_path, monkeypatch):
    import torch

    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        monkeypatch.setenv("NNSMITH_GRAD_TIME", "0.000001")
        original = _materialized_case(42, mode, out_dir=tmp_path)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True, warn_only=True)
        monkeypatch.setenv("NNSMITH_GRAD_TIME", "100")
        replay = _materialized_case(42, mode, slow=True, out_dir=tmp_path)
        assert original == replay
        assert torch.get_num_threads() == 2
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert not list(tmp_path.iterdir())  # No artifacts needed or produced.
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)


def test_full_case_seed_survives_fresh_processes():
    worker = '''
import json, runpy, sys
ns = runpy.run_path(sys.argv[1])
cases = [ns["_materialized_case"](42, mode, slow=bool(int(sys.argv[2])))
         for mode in ["uniform", "log_scale", "llm", "recommendation"]]
cases.append(ns["_materialized_case"](18, "uniform", complex_graph=True))
print(json.dumps(cases))
'''
    results = []
    for hash_seed, slow in [("0", "0"), ("1", "1"), ("123", "0")]:
        result = subprocess.run(
            [sys.executable, "-c", worker, __file__, slow],
            env=dict(os.environ, PYTHONHASHSEED=hash_seed),
            check=True, capture_output=True, text=True, timeout=60,
        )
        results.append(json.loads(result.stdout))
    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize("initial, expected_success", [(-2.0, True), (-100.0, False)])
def test_gradient_refinement_uses_steps_without_reading_clock(initial, expected_success, monkeypatch):
    import torch
    import nnsmith.materialize.torch.symbolnet as symbolnet
    from nnsmith.abstract.op import Placeholder, Sqrt
    from nnsmith.abstract.tensor import AbsTensor
    from nnsmith.gir import GraphIR, InstExpr
    from autofusedebugger.nnsmith_integration.fuzz_loop import _seeded_materialization

    ir = GraphIR()
    t = AbsTensor([4], DType.float32)
    inp = ir.add_inst(InstExpr(Placeholder(t).input(), []))
    sqrt = Sqrt()
    sqrt.bind_input_like([t])
    sqrt.bind_output_like([t])
    ir.add_inst(InstExpr(sqrt, [inp.retval()]))
    snapshots = []

    def no_clock():
        raise AssertionError("Seeded gradient search must not read the clock")

    monkeypatch.setattr(symbolnet, "time", SimpleNamespace(time=no_clock))
    for _ in range(2):
        with _seeded_materialization(42):
            net = symbolnet.SymbolNet(ir).to("cpu")
            inputs = net.grad_input_gen(
                {inp.retval(): torch.full((4,), initial)}, max_steps=16,
            )
            assert (inputs is not None) == expected_success
            assert 1 < net.n_iter <= 16
            if not expected_success:
                assert net.n_iter == 16
            snapshots.append((net.n_iter, None if inputs is None else inputs[inp.retval()].tolist()))
    assert snapshots[0] == snapshots[1]


def test_complex_graph_seed_ignores_solver_history():
    from nnsmith.abstract.op import (
        Sqrt, Transpose, Reshape, ReduceMin, Slice, MatMul, ExpandLast1,
    )
    from nnsmith.materialize.torch.dialect import TorchReduceSum
    from autofusedebugger.nnsmith_integration.fuzz_loop import _seeded_solver_context

    ops = [Add, Mul, Neg, ReLU, Sqrt, Transpose, Reshape, ReduceMin,
           Slice, MatMul, ExpandLast1, TorchReduceSum]

    def graph(seed):
        previous = z3.main_ctx()
        with _seeded_solver_context(seed):
            assert z3.main_ctx() is not previous
            ir = model_gen(
                ops, method="symbolic-cinit", seed=seed, init_fp=True,
                max_nodes=5, timeout_ms=10000, concr_ph_dim_rng=(1, 8),
                max_elem_per_tensor=4096, dtype_choices=[DType.float32],
            ).make_concrete()
            fingerprint = [str(ir), [(n, str(t.dtype), list(t.shape)) for n, t in ir.vars.items()]]
        assert z3.main_ctx() is previous
        return fingerprint

    expected = [graph(seed) for seed in range(20)]
    for seed in reversed(range(20)):
        assert graph(seed) == expected[seed]


def test_generation_contexts_restore_settings_after_exception():
    import torch
    from autofusedebugger.nnsmith_integration.fuzz_loop import (
        _seeded_materialization, _seeded_solver_context,
    )

    ctx = z3.main_ctx()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with pytest.raises(RuntimeError, match="test failure"):
        with _seeded_solver_context(42), _seeded_materialization(42):
            raise RuntimeError("test failure")
    assert z3.main_ctx() is ctx
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only


def test_repro_seed_without_debug_does_not_dump_failed_case(tmp_path, monkeypatch):
    from autofusedebugger.nnsmith_integration import fuzz_loop

    def no_dump(path):
        raise AssertionError("Seed-only replay must not dump model artifacts")

    bug = SimpleNamespace(
        dump=no_dump, symptom=SimpleNamespace(value="inconsistency"),
        stage=SimpleNamespace(value="verification"), log="test mismatch",
    )
    factory = SimpleNamespace(verify_testcase=lambda *args, **kwargs: bug)
    monkeypatch.setattr(fuzz_loop, "AscendC", lambda **kwargs: factory)
    monkeypatch.setattr(fuzz_loop, "probe_ascendc_opset", lambda *args, **kwargs: [Add])
    monkeypatch.setattr(fuzz_loop, "install_autofuse_intermediate_hooks", lambda: None)
    monkeypatch.setattr(fuzz_loop, "reset_autofuse_diagnostics", lambda: None)
    monkeypatch.setattr(fuzz_loop, "autofuse_diagnostics", lambda: {})
    result = fuzz_loop.run_repro_case(
        out_dir=tmp_path, repro_seed=42, max_nodes=4, tensor_scale="small",
        concr_ph_dim_rng=(1, 4), max_elem_per_tensor=256, debug=False,
    )
    assert result.status == "fail", result.error_summary
    assert result.bug_dir is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("seed", [0, 42, 389609433])
def test_scoped_solver_context_prevents_observed_history_drift(seed):
    from autofusedebugger.nnsmith_integration.generation import _seeded_solver_context
    ops = [Add, Mul, Neg, ReLU]
    previous = z3.main_ctx()
    with _seeded_solver_context(seed):
        first = _graph(seed, ops)
    assert z3.main_ctx() is previous
    _graph(seed + 1, ops)
    with _seeded_solver_context(seed):
        second = _graph(seed, list(reversed(ops)))
    assert first == second
    assert z3.main_ctx() is previous


def test_production_solver_scope_isolates_candidate_order_and_history():
    """Production scope, unlike raw model_gen, resets the context AST history."""
    from autofusedebugger.nnsmith_integration.generation import _seeded_solver_context
    ops = [Add, Mul, Neg, ReLU]
    for seed in (0, 42, 389609433):
        with _seeded_solver_context(seed):
            first = _graph(seed, ops)
        _graph(seed + 1, ops)
        with _seeded_solver_context(seed):
            replay = _graph(seed, list(reversed(ops)))
        assert replay == first
