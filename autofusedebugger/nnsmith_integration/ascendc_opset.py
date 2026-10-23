"""Restricts NNSmith's op catalog to the subset that `torch_npu`'s ascendc
Inductor backend actually registers a fused lowering for.

Why this exists: `auto_opset()` (NNSmith's own narrowing) only proves an op
*compiles and runs correctly* through `AscendC.verify_testcase` -- including
ops that silently fall back to plain, non-fused Inductor/eager codegen. Most
of NNSmith's default op catalog (`tan`, `sin`, `cos`, `gelu`, `leaky_relu`,
`argmax`/`argmin`, `conv`, pooling, ...) is *not* registered in ascendc's
`_LoweringGuard` at all on either SoC generation checked here, so a randomly
generated graph built from the full catalog almost always contains at least
one such op, which severs the graph into isolated fallback nodes before the
scheduler ever gets to ask "can these two fuse?" -- confirmed empirically:
batches of NNSmith-default graphs produced zero fusion decisions, while a
hand-written model built only from the ops below fused all 8 of its nodes
into a single ascendc kernel (`autofused_...` codegen artifact, zero
fallbacks).

Source of truth: this allowlist was built by reading, line by line, every
`_LoweringGuard.support(...)` call actually present in the installed
`torch_npu._inductor.ascendc.lowering.{aten_lowering,cat_lowering,
gather_lowering,mm_lowering,npu_lowering}` modules (plus `decomposition.py`
for ops that only reach a registered lowering by first being decomposed into
registered primitives), not from documentation or the aten op names in
isolation -- the same op can be registered under different dtype sets, or
only `since=Soc.A5`, and some ops (`aten.mm`/`addmm`/`bmm`/`baddbmm`) have a
registration call with no `since=` guard but gate themselves to A5+ inside
the lowering function body itself (`_is_matmul_supported_on_soc()` in
`mm_lowering.py`), so `_LoweringGuard.has()` alone is not sufficient evidence
for matmul-family ops specifically.

This module does not hardcode a SoC choice: it exposes both tiers
(`ASCENDC_OPSET_A3`, `ASCENDC_OPSET_A5`) so callers filter against whichever
hardware they're actually running on. `current_soc_tier()` reads it from the
live `torch_npu` installation when available.
"""

from __future__ import annotations

import os
from typing import List, Type

from nnsmith.abstract.op import AbsOpBase

# NNSmith op class names (as returned by `AbsOpBase.name()`, e.g. "core.Add")
# that map to an aten/prims target registered in ascendc's `_LoweringGuard`
# with no `since=` gate (or a gate already satisfied on A3) -- i.e. these are
# expected to be fusable on A3 *and* A5.
#
# Columns below (as a mapping comment, not enforced in code): NNSmith class ->
# aten target actually hit by its `forward_fn` -> dtype note.
ASCENDC_OPSET_A3: List[str] = [
    # elementwise binary (BcastBinaryOp1: +, -, *, max, min) -> aten.add/sub/
    # mul/maximum/minimum. All four registered for float_dtypes()+int32(+int16
    # for add/sub/maximum/minimum) with no since= gate.
    "core.Add",
    "core.Sub",
    "core.Mul",
    "core.Max",  # torch.max(a, b) (binary form) -> aten.maximum
    "core.Min",  # torch.min(a, b) (binary form) -> aten.minimum
    # core.Div -> aten.div/aten.true_divide, registered for
    # float_dtypes()+int_dtypes() with no since= gate (narrows to float_dtypes()
    # only starting Soc.A5 -- still fp32-fusable on both tiers).
    "core.Div",
    "core.Pow",  # aten.pow, float_dtypes()+int_dtypes(), no since= gate.
    # elementwise unary
    "core.Abs",  # aten.abs
    "core.Neg",  # aten.neg (int16 explicitly carved out pre-A5, fp32 unaffected)
    "core.Sigmoid",  # aten.sigmoid
    # core.Clip -> torch.clip(x, lo, hi) decomposes to clamp_min+clamp_max,
    # both registered float_dtypes()+int_dtypes()(+int16) with no since= gate.
    "core.Clip",
    # comparators / logical (Comparator/Logical subclasses) -- out dtype bool
    # explicitly whitelisted via support_out_dtypes=byte_dtypes() (which is
    # (uint8, bool), matching PyTorch's real bool output for these ops).
    # NOTE: core.Greater/core.Less deliberately excluded here -- see
    # _ASCENDC_A5_ONLY_ADDITIONS for why aten.gt/lt specifically do NOT belong
    # in this tier despite looking identical in shape to aten.eq.
    "core.Equal",  # aten.eq, support_out_dtypes=byte_dtypes() includes bool: matches.
    "core.And",  # aten.logical_and, bool-only pre-A5 (NNSmith's And is bool x bool)
    "core.Or",  # aten.logical_or, bool-only pre-A5
    "core.Where",  # aten.where
    # reductions. NOTE: core.ReduceMin/core.ReduceMax deliberately excluded --
    # see the "excluded from both tiers" note below.
    "torch.TorchReduceSum",  # aten.sum / aten.sum.dim_IntList, float_dtypes()
    "core.ReduceMean",  # x.mean(dim) -> aten.mean, float_dtypes()
    # view / layout ops (needed as connective tissue between compute ops;
    # NNSmith generates these constantly to reconcile shapes)
    "core.Reshape",  # aten.reshape, float_dtypes() (no since= gate; A5 widens to +int32/int64)
    "core.Transpose",  # aten.transpose, float_dtypes()+int16/uint16/int32/uint32
    "core.Squeeze",  # aten.squeeze, float_dtypes()
    "core.Concat",  # aten.cat, float_dtypes() (custom NPUConcatBuffer lowering)
    "core.Cast",  # -> prims.convert_element_type, float_dtypes()+bool+int32+uint8
]

# Additional NNSmith op class names that map to an aten target registered
# *only* `since=Soc.A5` (confirmed absent from A3's `_LoweringGuard._data` at
# runtime), or whose custom lowering function gates itself to A5+ internally
# regardless of `_LoweringGuard.has()` (matmul family -- see module docstring).
# Use `ASCENDC_OPSET_A5` (which already includes everything in
# `ASCENDC_OPSET_A3`) when targeting A5 hardware.
_ASCENDC_A5_ONLY_ADDITIONS: List[str] = [
    # aten.gt/aten.lt (NNSmith's Greater/Less) are registered on A3 too, but
    # with support_out_dtypes=(torch.uint8,) only -- real-world torch.gt/lt
    # always produce torch.bool output (confirmed: `(x > y).dtype ==
    # torch.bool` at runtime), which never matches that A3 registration, so
    # the op always falls back on A3 despite _LoweringGuard.has() == True.
    # The A5 registration widens support_out_dtypes to byte_dtypes() =
    # (uint8, bool), which does match -- confirmed fusable from A5 onward
    # only. core.Equal (aten.eq) does not have this problem: its A3
    # registration already used byte_dtypes() for the output, not a
    # uint8-only tuple.
    "core.Greater",
    "core.Less",
    "core.Ceil",  # aten.ceil, since=Soc.A5
    "core.Floor",  # aten.floor, since=Soc.A5
    "core.Round",  # aten.round, since=Soc.A5
    "core.Sin",  # aten.sin, since=Soc.A5
    "core.Cos",  # aten.cos, since=Soc.A5
    "core.Tan",  # aten.tan, since=Soc.A5
    "core.Atan",  # aten.atan, since=Soc.A5
    "core.Asin",  # aten.asin, since=Soc.A5 (also limit_domain=True -- needs vulops=True)
    "core.Acos",  # aten.acos, since=Soc.A5 (also limit_domain=True -- needs vulops=True)
    "core.GELU",  # aten.gelu has no direct _LoweringGuard entry; only reachable
    # via decomposition, and decomposition.py only keeps aten.gelu's
    # decomposition in the whitelist `if not (current_soc and current_soc <
    # Soc.A5)` -- i.e. GELU decomposes into registered primitives on A5+
    # only; on A3 it's a sealed single fallback op either way.
    "core.LeakyReLU",  # same situation as GELU: decomposition only kept for A5+
    # (`decomposition.py`'s A5-only block appends aten.leaky_relu).
    "torch.MatMul",  # aten.mm/addmm/bmm/baddbmm: _LoweringGuard.support() call
    # itself has no since= gate, but mm_lowering.py's custom lowering
    # functions call _is_matmul_supported_on_soc(), which requires
    # current_soc >= Soc.A5 and otherwise falls back -- so matmul never
    # actually fuses pre-A5 despite _LoweringGuard.has() returning True.
    "core.Linear",  # torch.nn.Linear -> addmm, same A5 gating as MatMul above.
]

ASCENDC_OPSET_A5: List[str] = ASCENDC_OPSET_A3 + _ASCENDC_A5_ONLY_ADDITIONS

# Ops NNSmith's default catalog contains that are intentionally excluded from
# *both* tiers above:
#
# - core.ReduceMin / core.ReduceMax: NNSmith's forward_fn calls `x.min(dim)` /
#   `x.max(dim)`, which traces to `aten.min.dim` / `aten.max.dim` -- the
#   2-tuple-returning overload (values, indices), not the reduce-to-scalar
#   `aten.min`/`aten.max` overload. `_LoweringGuard.has()` returns True for
#   `.dim` too (registered via the whole `OpOverloadPacket`, which covers
#   every overload with the same dtype tuple), but the dtype check runs
#   against the *node's full output type* before the `.values`/`.indices`
#   split: the indices half is always int64, which is never in the
#   registered (fp16, fp32, bf16) output set, so the `.dim` node always
#   fails the check and falls back -- confirmed via the live
#   fusion-diagnostics fallback log (`'target': 'aten.max.dim', 'reason':
#   'output dtype torch.int64 not in (...)'`), not just static reading of
#   aten_lowering.py. This is a structural limitation, not a dtype-tier gap,
#   so it applies identically on A5 and isn't added as an A5-only addition.
#
# - core.Log2, core.Xor, core.ArgMin, core.ArgMax, core.PReLU, core.Softmax
#   (not registered directly; only a `_softmax` decomposition exists, gated
#   dynamic-shape-only -- doesn't widen the registered-primitive set),
#   pooling (MaxPool2d/AvgPool2d), conv (Conv1d/NCHWConv2d), BatchNorm2d,
#   interpolation (NearestInterp/LinearInterp/BilinearInterp/BicubicInterp/
#   TrilinearInterp), Pad, Expand, Tril/Triu (A5-only: aten.tril/triu
#   since=Soc.A5, so these belong in neither tier below A5 and weren't
#   independently re-verified for the "registered dtype tuple actually
#   matches real output dtype" pitfall above A5 split, the way gt/lt were)
#   -- no entry found anywhere across aten_lowering.py / cat_lowering.py /
#   gather_lowering.py / mm_lowering.py / npu_lowering.py / decomposition.py.


def filter_ascendc_opset(
    opset: List[Type[AbsOpBase]], *, soc_tier: str = "A3"
) -> List[Type[AbsOpBase]]:
    """Narrow an `auto_opset()` result down to ops ascendc actually fuses on
    the given SoC tier ("A3" or "A5"). Raises if the allowlist and the probed
    opset have zero overlap (almost certainly a stale NNSmith/ascendc version
    mismatch, not a legitimate empty result).
    """
    if soc_tier == "A3":
        allow = set(ASCENDC_OPSET_A3)
    elif soc_tier == "A5":
        allow = set(ASCENDC_OPSET_A5)
    else:
        raise ValueError(f"Unknown soc_tier: {soc_tier!r}; expected 'A3' or 'A5'")

    filtered = [op for op in opset if op.name() in allow]
    if not filtered:
        raise RuntimeError(
            f"filter_ascendc_opset(soc_tier={soc_tier!r}) matched zero ops against "
            f"the probed opset ({len(opset)} candidates) -- the allowlist in "
            "ascendc_opset.py is likely stale against the installed torch_npu "
            "version; re-derive it from the live "
            "torch_npu._inductor.ascendc.lowering source files."
        )
    return filtered


def probe_ascendc_opset(
    model_cls, factory, *, soc_tier: str = "A3", grad: bool = False
) -> List[Type[AbsOpBase]]:
    """The actual entry point `fuzz_loop.py` should call instead of
    `auto_opset()` + `filter_ascendc_opset()`.

    Does NOT call `nnsmith.narrow_spec.auto_opset()` / `auto_opconfig()`:
    those probe every op in `model_cls.operators()` (NNSmith's full ~67-op
    catalog, most of which we're discarding anyway -- see module docstring)
    in one `infer_topset_from_scratch()` loop, and that loop has no
    per-op fault isolation. A single op whose `type_transfer()` raises
    `nnsmith.error.InternalError` during the solver-driven single-op-IR
    construction (confirmed hit in practice: `core.Reshape`'s `type_transfer`
    has a latent bug where a z3 model that assigns `-1` to more than one
    `target_shape` slot trips `SanityCheck.eq(auto_dim, -1)` on the second
    occurrence -- a real NNSmith bug, not anything in this codebase, and not
    something we're allowed to patch in-place per this project's "NNSmith is
    a dependency, not modified" rule) aborts the whole probe loop immediately,
    silently losing every op queued after it -- which is exactly how the
    on-disk topset cache ended up with *zero* fp32 combinations recorded for
    `Add`/`Sub`/`Mul`/etc despite them genuinely working on real hardware: the
    probe never got that far before `auto_opconfig()` cached the (incomplete)
    partial result.

    This function sidesteps both problems: it only probes
    `ASCENDC_OPSET_A3`/`ASCENDC_OPSET_A5` (not the full catalog, so the
    expensive compile+run probe is ~20 ops instead of ~67), and it probes
    each op in its own `infer_topset_from_scratch()` call so one op's
    `InternalError` only drops that op, not everything after it in iteration
    order. Caches results itself (keyed by soc_tier, separate from NNSmith's
    own `auto_opconfig()` cache file, to avoid the two caching paths
    clobbering each other) under NNSmith's own cache directory.
    """
    from nnsmith.error import InternalError
    from nnsmith.narrow_spec import (
        NNSMITH_CACHE_DIR,
        dump_topset,
        get_cache_name,
        infer_topset_from_scratch,
        load_topset,
    )

    allow = set(ASCENDC_OPSET_A5 if soc_tier == "A5" else ASCENDC_OPSET_A3)
    if soc_tier not in ("A3", "A5"):
        raise ValueError(f"Unknown soc_tier: {soc_tier!r}; expected 'A3' or 'A5'")

    all_ops = model_cls.operators()
    candidate_ops = [op for op in all_ops if op.name() in allow]
    if not candidate_ops:
        raise RuntimeError(
            f"probe_ascendc_opset(soc_tier={soc_tier!r}) found zero ops in "
            f"model_cls.operators() ({len(all_ops)} total) matching the "
            "ascendc_opset.py allowlist -- the allowlist is likely stale "
            "against the installed NNSmith/torch_npu versions."
        )

    cache_path = os.path.join(
        NNSMITH_CACHE_DIR,
        get_cache_name(model_cls, factory, grad) + f"_ascendc_{soc_tier}.yaml",
    )
    if os.path.exists(cache_path):
        topset_config = load_topset(cache_path)
    else:
        topset_config: dict = {}
        for op in candidate_ops:
            try:
                probed = infer_topset_from_scratch(
                    model_cls, factory, op_types=[op], grad=grad
                )
            except InternalError:
                # A single op's probe hit an NNSmith-internal assertion
                # (observed for core.Reshape -- see docstring above). Skip
                # just this op rather than losing every op queued after it.
                continue
            topset_config.update(probed)
        os.makedirs(NNSMITH_CACHE_DIR, exist_ok=True)
        dump_topset(topset_config, cache_path)

    opset = []
    for op in candidate_ops:
        if op.name() not in topset_config:
            continue
        op.in_dtypes = topset_config[op.name()].in_dtypes
        op.out_dtypes = topset_config[op.name()].out_dtypes
        opset.append(op)

    if not opset:
        raise RuntimeError(
            f"probe_ascendc_opset(soc_tier={soc_tier!r}) probed "
            f"{len(candidate_ops)} candidate ops but none survived "
            f"(compiled+ran successfully against {factory}) -- check "
            f"{cache_path} for per-op failure details, or delete it to "
            "force a fresh probe."
        )
    return opset
