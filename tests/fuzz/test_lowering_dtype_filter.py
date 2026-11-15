"""Tests for lowering_dtype_filter.py's dtype-combo filtering logic.

`dtype_support`/`to_torch_dtypes` are injected fakes throughout (real
`_LoweringGuard` state requires torch_npu/ascendc, not available off real
NPU hardware) -- these tests exercise the pure filtering logic, not the
live registration data. See the module docstring for why the filtering
exists and `LOWERING_GUARD_TARGETS`' own comment for how its entries were
verified against real `_LoweringGuard` data on A5 hardware.
"""

from __future__ import annotations

import torch

from autofusex.fuzz.nnsmith.lowering_dtype_filter import (
    LOWERING_GUARD_TARGETS,
    SOC_GATED_OPS,
    filter_probed_dtypes,
)

aten = torch.ops.aten

IDENTITY = lambda combo: combo  # noqa: E731 -- combos already real torch.dtype in these tests


def _fake_support(table):
    def dtype_support(op, soc_tier):
        return table.get((op, soc_tier))

    return dtype_support


def test_registered_and_covered_combo_survives():
    support = _fake_support(
        {(aten.add.Tensor, "A3"): ((torch.float16, torch.int32), (torch.float16, torch.int32))}
    )
    kept_in, kept_out = filter_probed_dtypes(
        "core.Add",
        [(torch.float16,), (torch.int32,)],
        [(torch.float16,), (torch.int32,)],
        "A3",
        to_torch_dtypes=IDENTITY,
        dtype_support=support,
    )
    assert kept_in == [(torch.float16,), (torch.int32,)]
    assert kept_out == [(torch.float16,), (torch.int32,)]


def test_uncovered_dtype_combo_dropped_pairing_preserved():
    """The TorchReduceSum scenario this filter exists for: aten.sum.dim_IntList
    only registers float16/float32, so int64/complex64 probe results that
    merely "compile and run" (via eager fallback) must be dropped."""
    support = _fake_support(
        {(aten.sum.dim_IntList, "A3"): ((torch.float16, torch.float32), (torch.float16, torch.float32))}
    )
    kept_in, kept_out = filter_probed_dtypes(
        "torch.TorchReduceSum",
        [(torch.float32,), (torch.int64,), (torch.complex64,)],
        [(torch.float32,), (torch.int64,), (torch.complex64,)],
        "A3",
        to_torch_dtypes=IDENTITY,
        dtype_support=support,
    )
    assert kept_in == [(torch.float32,)]
    assert kept_out == [(torch.float32,)]


def test_matmul_excluded_on_a3_soc_gate():
    """aten.mm/addmm/bmm/baddbmm register in _LoweringGuard with no since=
    gate, but mm_lowering.py's _is_matmul_supported_on_soc() still requires
    current_soc >= Soc.A5 -- a constraint _LoweringGuard can't see at all.
    SOC_GATED_OPS must apply even when the (fake) dtype_support would
    otherwise say "registered"."""
    support = _fake_support(
        {(aten.mm.default, "A3"): ((torch.float16,), (torch.float16,))}
    )
    kept_in, _ = filter_probed_dtypes(
        "torch.MatMul",
        [(torch.float16,)],
        [(torch.float16,)],
        "A3",
        to_torch_dtypes=IDENTITY,
        dtype_support=support,
    )
    assert kept_in == []


def test_matmul_survives_on_a5():
    support = _fake_support(
        {(aten.mm.default, "A5"): ((torch.float16,), (torch.float16,))}
    )
    kept_in, _ = filter_probed_dtypes(
        "torch.MatMul",
        [(torch.float16,)],
        [(torch.float16,)],
        "A5",
        to_torch_dtypes=IDENTITY,
        dtype_support=support,
    )
    assert kept_in == [(torch.float16,)]


def test_decomposition_only_op_passes_through_unfiltered():
    """core.LayerNorm/GELU/LeakyReLU have no _LoweringGuard entry of their
    own (only reachable via decomposition.py) -- nothing to filter against,
    so the probe result is trusted as-is, unchanged object identity."""
    in_dtypes = [(torch.float64,), (torch.complex128,)]
    out_dtypes = [(torch.float64,), (torch.complex128,)]
    kept_in, kept_out = filter_probed_dtypes(
        "core.LayerNorm",
        in_dtypes,
        out_dtypes,
        "A3",
        to_torch_dtypes=IDENTITY,
        dtype_support=_fake_support({}),
    )
    assert kept_in is in_dtypes
    assert kept_out is out_dtypes


def test_op_unregistered_for_soc_tier_drops_every_combo():
    kept_in, _ = filter_probed_dtypes(
        "torch.TorchReduceSum",
        [(torch.float32,)],
        [(torch.float32,)],
        "A3",
        to_torch_dtypes=IDENTITY,
        dtype_support=_fake_support({}),  # no entry at all -> None for every lookup
    )
    assert kept_in == []


def test_lowering_guard_targets_resolve_to_real_overloads():
    """Every mapped target must be a real OpOverload -- a typo'd attribute
    access would otherwise silently construct nothing useful or raise, not
    something a type checker catches for dynamically-accessed aten/prims
    attributes."""
    for name, targets in LOWERING_GUARD_TARGETS.items():
        for target in targets:
            assert isinstance(target, torch._ops.OpOverload), f"{name}: {target!r}"


def test_soc_gated_ops_values_are_valid_tier_strings():
    for name, tiers in SOC_GATED_OPS.items():
        assert tiers <= {"A3", "A5"}, f"{name}: {tiers!r}"
