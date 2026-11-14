"""NNSmith op name -> ascendc `_LoweringGuard` target(s), plus the filtering
logic that narrows `probe_ascendc_opset()`'s real compile+run probe results
down to dtype combos that actually reach ascendc's registered fused
lowering.

Why this filtering matters: `probe_ascendc_opset()` determines "does this op
compile+run at all" by actually calling `torch.compile(...)` -- which proves
the op *executes correctly*, but not that it reaches ascendc's fused
lowering. A dtype combo ascendc's `_LoweringGuard` never registered for a
given op still compiles+runs fine, it just silently falls back to
eager/extern codegen -- same outcome (a `pass` case) as a combo that
genuinely fuses, but useless for testing fusion, which is the whole point of
`ascendc_opset.py`'s allowlist. Concretely confirmed via the live A5 probe
cache: `torch.TorchReduceSum` (-> `aten.sum.dim_IntList`, registered only for
`float_dtypes() + byte_dtypes()`, no int/complex at all) still probed
successfully for `int32`/`int64`/`int16`/`complex64`/`complex128` inputs --
every one of those is a pure fallback case that would otherwise silently
dilute every batch's dtype distribution with "passes but never fuses" cases.

The mapping below was built the same way as `ascendc_opset.py`'s own
allowlist: by reading, line by line, every `_LoweringGuard.support(...)` call
in the installed `torch_npu._inductor.ascendc.lowering.{aten_lowering,
cat_lowering,gather_lowering,mm_lowering,npu_lowering}` and matching each
NNSmith op's `forward_fn` to the aten/prims overload it actually traces to --
not from documentation or the aten op names in isolation. Every target in
`LOWERING_GUARD_TARGETS`/`SOC_GATED_OPS` below was cross-checked against the
real `_LoweringGuard._data` dict on live A5 hardware (both `has()`/
`dtypes_support()` resolving sensibly, and the matmul SoC-gate reproduced
exactly as `mm_lowering.py`'s own `_is_matmul_supported_on_soc()` describes).

Deliberately excludes ops that only ever reach fusion through
`decomposition.py` with no `_LoweringGuard` registration of their own
(`core.LayerNorm`/`core.GELU`/`core.LeakyReLU`): there is no single
registration to narrow against (support is determined transitively by the
decomposition chain's own sub-ops), and the real compile+run probe is
already the authoritative signal there -- second-guessing it with a
hand-built sub-op intersection would add complexity and a real chance of
being wrong, not more correctness.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

aten = torch.ops.aten
prims = torch.ops.prims

# NNSmith op name -> tuple of aten/prims OpOverload(s) whose registered
# dtype sets must ALL cover a probed combo for it to survive. More than one
# target means the op only fuses via a multi-op decomposition where every
# sub-op *is* directly registered (e.g. core.Clip -> clamp_min + clamp_max) --
# not the LayerNorm/GELU/LeakyReLU situation above, where no sub-op mapping
# is tracked at all.
LOWERING_GUARD_TARGETS: dict[str, tuple] = {
    # elementwise binary
    "core.Add": (aten.add.Tensor,),
    "core.Sub": (aten.sub.Tensor,),
    "core.Mul": (aten.mul.Tensor,),
    "core.Max": (aten.maximum.default,),  # binary torch.max(a, b) -> aten.maximum
    "core.Min": (aten.minimum.default,),  # binary torch.min(a, b) -> aten.minimum
    "core.Div": (aten.div.Tensor,),
    "core.Pow": (aten.pow.Tensor_Tensor,),
    # elementwise unary
    "core.Abs": (aten.abs.default,),
    "core.Neg": (aten.neg.default,),
    "core.Sigmoid": (aten.sigmoid.default,),
    "core.Sqrt": (aten.sqrt.default,),
    # core.Clip -> torch.clip(x, lo, hi), decomposes to clamp_min + clamp_max
    # (both directly registered, no since= gate) -- not a decomposition-only
    # op in the LayerNorm/GELU sense, both halves are checked here.
    "core.Clip": (aten.clamp_min.default, aten.clamp_max.default),
    # comparators / logical
    "core.Equal": (aten.eq.Tensor,),
    "core.And": (aten.logical_and.default,),
    "core.Or": (aten.logical_or.default,),
    "core.Greater": (aten.gt.Tensor,),
    "core.Less": (aten.lt.Tensor,),
    "core.Where": (aten.where.self,),
    # reductions
    "torch.TorchReduceSum": (aten.sum.dim_IntList,),
    "core.ReduceMean": (aten.mean.dim,),
    # ReduceMin/ReduceMax's forward_fn is rewritten by
    # ascendc_reduce_min_max_amin_amax.patch to call amin/amax instead of
    # min/max.dim -- see that patch and the NNSmith fork.
    "core.ReduceMin": (aten.amin.default,),
    "core.ReduceMax": (aten.amax.default,),
    # view / layout
    "core.Reshape": (aten.reshape.default,),
    "core.Transpose": (aten.transpose.int,),
    "core.Squeeze": (aten.squeeze.dim,),
    "core.Concat": (aten.cat.default,),
    "core.Cast": (prims.convert_element_type.default,),
    "core.Slice": (aten.slice.Tensor,),
    "core.ExpandLast1": (aten.expand.default,),
    "core.ExpandLast2": (aten.expand.default,),
    "core.ExpandLast3": (aten.expand.default,),
    "core.ExpandLast4": (aten.expand.default,),
    # A5-only additions
    "core.Ceil": (aten.ceil.default,),
    "core.Floor": (aten.floor.default,),
    "core.Round": (aten.round.default,),
    "core.Sin": (aten.sin.default,),
    "core.Cos": (aten.cos.default,),
    "core.Tan": (aten.tan.default,),
    "core.Atan": (aten.atan.default,),
    "core.Asin": (aten.asin.default,),
    "core.Acos": (aten.acos.default,),
    "core.Tril": (aten.tril.default,),
    "core.Triu": (aten.triu.default,),
    # gather family: the registered dtype tuple describes the *data*
    # operand; the index operand is always int32/int64 (see
    # ascendc_index_select.patch's in_dtypes), which happens to already be
    # included in every gather-family registration below, so checking
    # "every dtype in the probed combo is a subset of the registered set"
    # (this module's general rule, not a per-operand-position model) gives
    # the right answer here without needing special-casing -- a convenient
    # coincidence of the current registrations, not a logical guarantee if
    # a future registration ever narrowed the data dtype set to exclude
    # int32/int64.
    "core.IndexSelect": (aten.index_select.default,),
    "core.Gather": (aten.gather.default,),
}

# NNSmith op names registered in `_LoweringGuard` with no (or an
# already-satisfied) `since=` gate, but additionally gated to a *later* SoC
# tier by a check inside the lowering function body itself --
# `_LoweringGuard.has()`/`dtypes_support()` can't see this at all (confirmed:
# `aten.mm`/`addmm`/`bmm`/`baddbmm` register for `float_dtypes()` with no
# since= in `mm_lowering.py`, but every one of their lowering functions
# calls `_is_matmul_supported_on_soc()`, which requires `current_soc >=
# Soc.A5` and otherwise falls back regardless of dtype). Maps op name -> set
# of soc_tier strings where it's actually usable.
SOC_GATED_OPS: dict[str, frozenset[str]] = {
    "torch.MatMul": frozenset({"A5"}),
    "core.Linear": frozenset({"A5"}),
}


def _combo_survives(
    op_name: str,
    in_combo_torch: tuple,
    out_combo_torch: tuple,
    soc_tier: str,
    dtype_support: Callable[[object, str], Optional[tuple]],
) -> bool:
    """True iff every `torch.dtype` in `in_combo_torch`/`out_combo_torch` is
    covered by `_LoweringGuard`'s registered dtype sets for `op_name`'s real
    aten/prims target(s). `op_name` not in `LOWERING_GUARD_TARGETS` (e.g. a
    decomposition-only op) means "nothing to check against" -- trust the
    probe as-is, see the module docstring.

    `dtype_support(op_overload, soc_tier)` must return `(support_in_dtypes,
    support_out_dtypes)` (both iterables of `torch.dtype`) or `None` if
    `op_overload` isn't registered for `soc_tier` at all -- injected so this
    function has no `inductor_npu_ext` import of its own and is testable
    with a fake.
    """
    if op_name in SOC_GATED_OPS and soc_tier not in SOC_GATED_OPS[op_name]:
        # Registered in _LoweringGuard with no (or an already-satisfied)
        # since= gate, but gated to a later SoC tier by a check *inside* the
        # lowering function body itself (matmul family -- see
        # `_is_matmul_supported_on_soc()` in mm_lowering.py) that
        # `_LoweringGuard.has()`/`dtypes_support()` can't see at all.
        return False

    targets = LOWERING_GUARD_TARGETS.get(op_name)
    if targets is None:
        return True

    in_set = set(in_combo_torch)
    out_set = set(out_combo_torch)
    for target in targets:
        support = dtype_support(target, soc_tier)
        if support is None:
            # Registered with since=Soc.A5 and we're checking A3 (or not
            # registered at all) -- _LoweringGuard.support()'s since= gate
            # is a no-op in that case, so _data never gets this entry.
            return False
        support_in, support_out = support
        if not in_set.issubset(support_in) or not out_set.issubset(support_out):
            return False
    return True


def filter_probed_dtypes(
    op_name: str,
    in_dtypes: list,
    out_dtypes: list,
    soc_tier: str,
    *,
    to_torch_dtypes: Callable[[tuple], tuple],
    dtype_support: Callable[[object, str], Optional[tuple]],
) -> tuple:
    """Narrow a probed op's `(in_dtypes, out_dtypes)` combo lists (whatever
    dtype representation the caller's probe cache uses -- NNSmith `DType` in
    `ascendc_opset.py`'s real usage) down to the combos that actually reach
    ascendc's registered fused lowering, preserving combo pairing
    (`in_dtypes[i]` still corresponds to `out_dtypes[i]` after filtering).
    Returns the original lists unchanged (not copies) if `op_name` isn't in
    `LOWERING_GUARD_TARGETS`/`SOC_GATED_OPS` at all -- the common case for
    ops this module has no opinion on.

    `to_torch_dtypes(combo)` converts one `in_dtypes[i]`/`out_dtypes[i]`
    tuple to real `torch.dtype`s; `dtype_support` is forwarded to
    `_combo_survives()`. Both injected so this stays import-light and
    testable -- see that function's docstring.
    """
    if op_name not in LOWERING_GUARD_TARGETS and op_name not in SOC_GATED_OPS:
        return in_dtypes, out_dtypes

    kept_in, kept_out = [], []
    for in_combo, out_combo in zip(in_dtypes, out_dtypes):
        in_torch = to_torch_dtypes(tuple(in_combo))
        out_torch = to_torch_dtypes(tuple(out_combo))
        if _combo_survives(op_name, in_torch, out_torch, soc_tier, dtype_support):
            kept_in.append(in_combo)
            kept_out.append(out_combo)
    return kept_in, kept_out
