"""Seeded case generation shared by correctness checks and performance workers."""
from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from nnsmith.abstract.dtype import DType
from nnsmith.graph_gen import model_gen, set_z3_state
from nnsmith.materialize import TestCase
from nnsmith.util import set_seed

_NUMERIC_SEARCH_STEPS = 16

def _tolerance_for_ir(ir) -> tuple[float, float]:
    """Loosen rtol/atol when any variable in this graph is a low-precision
    float dtype. PyTorch's own torch.testing.assert_close defaults widen by
    roughly 10x per halving of mantissa bits going fp32 -> fp16/bf16 (both
    have a 7-8 bit mantissa vs fp32's 23), so rtol=1e-1/atol=1e-2 (10x
    nnsmith's fp32 defaults) is the right order of magnitude, not a number
    pulled from nowhere. Pure-fp32/fp64/int graphs are unaffected (same
    constants nnsmith always used), so this is a no-op for the regression
    batches already validated under the old fp32-only lock.
    """
    # TODO(bf16 task): this nnsmith snapshot's DType enum has no `bfloat16`
    # member yet (verified: hasattr(DType, "bfloat16") is False here) -- bf16
    # support is a separate follow-up task that patches nnsmith's DType enum
    # to add it. Once that patch lands, add DType.bfloat16 to this set too.
    low_precision = {DType.float16}
    dtypes_used = {var.dtype for var in ir.vars.values()}
    if dtypes_used & low_precision:
        return 1e-1, 1e-2
    return 1e-2, 1e-3


TENSOR_SCALE_PROFILES: dict[str, dict[str, object]] = {
    # small/medium/large: unchanged from the original hardcoded defaults
    # (including dim_dist_mode="uniform", nnsmith's own pre-existing
    # behavior) -- kept as the regression baseline so existing --repro-seed
    # batches stay reproducible unless --tensor-scale is explicitly set to
    # something else.
    "small": {
        "max_elem_per_tensor": 2**16,
        "concr_ph_dim_rng": (1, 64),
        "timeout_ms": 10000,
        "dim_dist_mode": "uniform",
    },
    "medium": {
        "max_elem_per_tensor": 2**20,
        "concr_ph_dim_rng": (8, 4096),
        "timeout_ms": 30000,
        "dim_dist_mode": "uniform",
    },
    "large": {
        "max_elem_per_tensor": 2**24,
        "concr_ph_dim_rng": (8, 8192),
        "timeout_ms": 60000,
        "dim_dist_mode": "uniform",
    },
    # 16k/32k/64k/128k: concr_ph_dim_rng's upper bound is the actual target
    # (seq_len-style long-context scenarios, see docs/af_fuzz_randomization.md's
    # "业界典型数值参考" -- large-model seq_len commonly reaches 4K~128K).
    # max_elem_per_tensor is set to 4x that upper bound: enough for ~2 dims to
    # both land near the upper bound at once (their product ~= upper_bound^2
    # would blow the budget immediately, so this is deliberately NOT upper_bound
    # squared) while every other dim in the same tensor gets squeezed down to 1
    # by graph_gen.py's existing total-nelement clamp -- not a guess at "how
    # large a tensor should be allowed to get" from scratch, since the dim
    # upper bound is the thing actually being stress-tested here, not the
    # overall tensor size.
    #
    # dim_dist_mode="log_normal" (unlike the three profiles above, which keep
    # nnsmith's original uniform sampling): at these upper bounds, sampling
    # every size in [8, upper_bound] with equal probability means most
    # generated dims land nowhere near the business-relevant small/medium
    # range -- log_normal instead skews each dim toward smaller values while
    # still letting a (deliberately small) fraction of draws reach the full
    # upper bound, matching the request that "large shapes should be rarer,
    # not equally likely". See _sample_dim_size()'s docstring in the
    # ascendc_dim_size_distribution.patch for the exact mu/sigma and why.
    "16k": {
        "max_elem_per_tensor": 16384 * 4,
        "concr_ph_dim_rng": (8, 16384),
        "timeout_ms": 90000,
        "dim_dist_mode": "log_normal",
    },
    "32k": {
        "max_elem_per_tensor": 32768 * 4,
        "concr_ph_dim_rng": (8, 32768),
        "timeout_ms": 120000,
        "dim_dist_mode": "log_normal",
    },
    "64k": {
        "max_elem_per_tensor": 65536 * 4,
        "concr_ph_dim_rng": (8, 65536),
        "timeout_ms": 180000,
        "dim_dist_mode": "log_normal",
    },
    "128k": {
        "max_elem_per_tensor": 131072 * 4,
        "concr_ph_dim_rng": (8, 131072),
        "timeout_ms": 300000,
        "dim_dist_mode": "log_normal",
    },
    # default: the new out-of-the-box tier (--tensor-scale's default, see
    # cli.py) -- range is [1, 128k] rather than 16k/32k/64k/128k's [8, ...]
    # since this tier isn't targeting a specific seq_len-style scenario, just
    # giving every batch broad shape coverage by default. max_elem_per_tensor
    # and timeout_ms are reused from "128k" as-is (same upper bound, so its
    # already-vetted safety margins apply unchanged) rather than re-derived.
    # dim_dist_mode="half_normal" (unlike 16k/32k/64k/128k's "log_normal",
    # which peaks at an interior value): density is highest at the range's
    # minimum and strictly decreases toward the maximum, so small shapes are
    # the most common outcome and large shapes get rarer the larger they
    # get, with no interior mode. See _sample_dim_size()'s docstring in
    # ascendc_dim_size_distribution.patch for the exact sigma and why.
    "default": {
        "max_elem_per_tensor": 131072 * 4,
        "concr_ph_dim_rng": (1, 131072),
        "timeout_ms": 300000,
        "dim_dist_mode": "half_normal",
    },
}


@contextmanager
def _seeded_solver_context(case_seed: int):
    """Give each graph a fresh Z3 context, independent of earlier searches.

    NNSmith constructs expressions using Z3's implicit default context. Z3
    has no public setter for that context; scope the swap to generation in
    this worker (cases run sequentially per process), and always restore it.
    Resetting solver seeds alone does not reset context-internal AST history.
    """
    import z3
    from z3 import z3 as z3_impl

    previous = z3.main_ctx()
    set_z3_state(case_seed)
    try:
        z3_impl._main_ctx = z3.Context()
        yield
    finally:
        z3_impl._main_ctx = previous


@contextmanager
def _seeded_materialization(case_seed: int):
    """Pin arithmetic as well as RNG state, then restore caller settings."""
    import torch

    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        set_seed(case_seed)
        # run_single_case is also a public entrypoint; do not rely on a batch
        # caller having registered TorchModel.add_seed_setter() beforehand.
        torch.manual_seed(case_seed)
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)


@dataclass(frozen=True)
class GenerationConfig:
    max_nodes: int = 5
    timeout_ms: Optional[int] = None
    method: str = "symbolic-cinit"
    max_elem_per_tensor: Optional[int] = None
    concr_ph_dim_rng: Optional[tuple[int, int]] = None
    tensor_scale: str = "default"
    value_dist_mode: str = "uniform"
    mixed_shapes: bool = False
    dynamic_shapes: bool = False
    dynamic_variants: int = 3

    def __post_init__(self):
        if self.mixed_shapes and self.dynamic_shapes:
            raise ValueError("mixed_shapes and dynamic_shapes are mutually exclusive")
        if self.tensor_scale not in TENSOR_SCALE_PROFILES:
            raise ValueError(f"Unknown tensor_scale {self.tensor_scale!r}")
        if not 2 <= self.dynamic_variants <= 8:
            raise ValueError("dynamic_variants must be between 2 and 8")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")


def generate_case(*, opset, model_cls, case_seed: int, config: GenerationConfig):
    profile = TENSOR_SCALE_PROFILES[config.tensor_scale]
    set_seed(case_seed)
    max_nodes = random.randint(2, config.max_nodes) if config.max_nodes >= 2 else config.max_nodes
    with _seeded_solver_context(case_seed):
        gen = model_gen(
            opset=opset, method=config.method, seed=case_seed,
            max_elem_per_tensor=config.max_elem_per_tensor if config.max_elem_per_tensor is not None else profile["max_elem_per_tensor"],
            concr_ph_dim_rng=config.concr_ph_dim_rng if config.concr_ph_dim_rng is not None else profile["concr_ph_dim_rng"],
            dim_dist_mode=profile["dim_dist_mode"], max_nodes=max_nodes,
            timeout_ms=config.timeout_ms if config.timeout_ms is not None else profile["timeout_ms"],
            init_fp=True,
        )
        ir = gen.make_concrete()
        del gen
    with _seeded_materialization(case_seed):
        model = model_cls.from_gir(ir)
        model.refine_weights(value_dist_mode=config.value_dist_mode, max_search_steps=_NUMERIC_SEARCH_STEPS)
        oracle = model.make_oracle(mode=config.value_dist_mode)
    case = TestCase(model, oracle)
    if config.dynamic_shapes:
        from .dynamic_shapes import build_workload, workload_metadata, workload_fingerprint
        with _seeded_materialization(case_seed):
            samples, axes = build_workload(case, seed=case_seed, variants=config.dynamic_variants,
                max_elements=config.max_elem_per_tensor or profile["max_elem_per_tensor"])
        case.dynamic_samples = samples
        case.dynamic_axes = axes
        case.dynamic_metadata = workload_metadata(samples, axes)
        case.dynamic_fingerprint = workload_fingerprint(samples, axes)
    return case
