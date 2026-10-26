"""Thin NNSmith fuzz loop targeting the autofuse (ascendc) backend.

Deliberately bypasses `nnsmith.cli.fuzz.FuzzingLoop` / hydra config: that
machinery is wired to `BackendFactory.init()`'s hardcoded name table, which we
are not touching (see `ascendc_backend.py`). This loop re-implements only the
generate -> end-to-end-verify -> record steps of that machinery, directly
against `AscendC`.

Phase 2 scope (see conversation history for the full rationale):
- dtype is not locked to fp32 -- `BaseGen.__init__` intersects whatever
  dtype_choices we pass (here, nnsmith's own default) with the probed
  opset's real dtype union, so every dtype this ascendc opset snapshot
  actually supports can appear.
- Op catalog is restricted to `ascendc_opset.probe_ascendc_opset()`'s
  allowlist: ops ascendc's Inductor backend actually registers a *fused*
  lowering for on the target SoC tier (`--soc-tier`), with each op's real
  compile+run dtype support probed directly (not via NNSmith's own
  `auto_opset()`/`auto_opconfig()`, whose fault-intolerant probe loop and
  shared cache file turned out to silently produce an empty/wrong topset --
  see `probe_ascendc_opset()`'s docstring). This restriction is required, not
  optional: most of NNSmith's default catalog has no ascendc lowering at all,
  so graphs built from the unfiltered catalog essentially never produce a
  fusable node pair -- see ascendc_opset.py.
- Dtype-aware difftest tolerance (`_tolerance_for_ir`): fp32/fp64/int graphs
  keep NNSmith's original fixed tolerance (rtol=1e-2, atol=1e-3); graphs
  using a low-precision float dtype get a looser one. fp16 is wired in now;
  bf16 will be added to the low-precision set once its own nnsmith patch
  lands (see `_tolerance_for_ir`'s TODO).
- Passing cases are *not* dumped to disk (only a summary row is kept) to avoid
  unbounded disk growth; only failing cases get the full NNSmith
  `BugReport.dump()` artifact set (model.pth / gir.pkl / oracle.pkl / err.log /
  meta.json). Exception: `--debug` mode (single-case repro) always dumps,
  pass or fail, since the point there is inspecting one targeted case.

Case identity: every case is fully determined by `case_seed` (an NNSmith
`model_gen` seed) plus the generation knobs (`method`, `max_nodes`,
`timeout_ms`, `max_elem_per_tensor`, `target`) -- `case_id` below is just
`f"{index:05d}-{case_seed}"`, matching the `bug-<index>-<case_seed>` dump
folder naming. Passing `--repro-seed <case_seed>` (with the same generation
knobs used originally) regenerates the exact same graph deterministically,
without needing any dumped artifacts -- see `run_single_case`.

Concurrency: generation and verification run one case at a time within a
single process/device context (see `run_fuzz_session`). Parallelism across
cases is done by sharding into multiple OS *processes*
(`run_fuzz_session_parallel`, `--jobs`), not threads: `activate_autofuse_backend`
and the autofuse Inductor hooks mutate process-global state (env vars, a
monkeypatched class attribute, module-level diagnostic lists guarded only by a
`threading.Lock`), so sharing one process across threads would race on that
state. Separate processes each get their own copy and their own NPU device
stream/compile-cache context, which is also safer for concurrent NPU use than
in-process threads.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import random
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from nnsmith.abstract.dtype import DType
from nnsmith.error import InternalError
from nnsmith.graph_gen import model_gen
from nnsmith.materialize import Model, TestCase
from nnsmith.util import set_seed

# Phase 2: dtype is no longer locked to fp32. BaseGen.__init__ (graph_gen.py)
# already intersects whatever dtype_choices we pass with the opset's own
# dtype union, and probe_ascendc_opset() rewrites each op's in_dtypes/
# out_dtypes to the dtype combos it confirmed actually compile+run on real
# A5 hardware -- so passing dtype_choices=None (nnsmith's own default,
# DTYPE_GEN_ALL) and letting that intersection do the work automatically
# yields "every dtype this ascendc opset snapshot really supports", with no
# separate per-op mapping table to maintain.

from autofusedebugger.inductor_autofuse import (
    autofuse_diagnostics,
    install_autofuse_intermediate_hooks,
    reset_autofuse_diagnostics,
)
from autofusedebugger.nnsmith_integration.ascendc_backend import AscendC
from autofusedebugger.nnsmith_integration.ascendc_opset import probe_ascendc_opset
from autofusedebugger.nnsmith_integration.repro import write_repro_script
from autofusedebugger.nnsmith_integration.summary_html import write_summary_html


@dataclass
class CaseResult:
    seed: int
    index: int
    status: str  # "pass" | "fail" | "gen_error"
    duration_s: float
    case_id: str = ""
    n_nodes: int = 0
    n_vars: int = 0
    op_types: list[str] = field(default_factory=list)
    symptom: Optional[str] = None
    stage: Optional[str] = None
    bug_dir: Optional[str] = None
    error_summary: Optional[str] = None
    input_shapes: dict[str, list[int]] = field(default_factory=dict)
    output_shapes: dict[str, list[int]] = field(default_factory=dict)
    # Number of kernels inductor/autofuse actually generated and ran on NPU
    # for this case (len(generated_kernels) from autofuse_diagnostics()).
    # Left at the default 0 for "gen_error" cases, which never reach
    # compilation.
    n_kernels: int = 0

    def __post_init__(self) -> None:
        if not self.case_id:
            self.case_id = f"{self.index:05d}-{self.seed}"


@dataclass
class FuzzSummary:
    total: int = 0
    passed: int = 0
    failed: int = 0
    gen_errors: int = 0
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "gen_errors": self.gen_errors,
            "cases": [asdict(c) for c in self.cases],
        }

    def merge(self, other: "FuzzSummary") -> None:
        self.total += other.total
        self.passed += other.passed
        self.failed += other.failed
        self.gen_errors += other.gen_errors
        self.cases.extend(other.cases)


def _truncate(text: str, limit: int = 500) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


class _DebugDumpEnv:
    """Context manager enabling torch/inductor AND autofuse-side debug-dump
    env vars for the duration of a single case's compile+run, scoped to that
    case's own directory so concurrent/sequential cases never clobber each
    other's dumps.

    TORCH_COMPILE_DEBUG/TORCH_LOGS/TORCHINDUCTOR_CACHE_DIR cover TorchInductor's
    own scheduling trace. They say nothing about whether autofuse actually
    produced a fused ascendc kernel for this case, vs. every op falling back to
    per-op codegen or eager -- that's a separate, autofuse-specific dump
    controlled by AUTOFUSE_DFX_FLAGS (see autofuse_graph.enable_autofuse_graph_dump,
    used by the hand-written-case path in torch_model.py). When set, autofuse
    writes `ascgraph_*AutoFuseAfterOptimize*.txt` under the given debug_dir for
    every graph it actually fuses -- presence/absence of that file is the
    direct way to confirm a case reached the real autofuse codegen path rather
    than silently bypassing it.
    """

    _ENV_KEYS = ("TORCH_COMPILE_DEBUG", "TORCH_LOGS", "TORCHINDUCTOR_CACHE_DIR", "AUTOFUSE_DFX_FLAGS")

    def __init__(self, case_dir: Path):
        self.case_dir = case_dir
        self._saved: dict[str, Optional[str]] = {}

    def __enter__(self) -> "_DebugDumpEnv":
        from autofusedebugger.autofuse_graph import enable_autofuse_graph_dump

        self.case_dir.mkdir(parents=True, exist_ok=True)
        debug_dir = self.case_dir / "torch_compile_debug"
        inductor_cache_dir = self.case_dir / "inductor_cache"
        autofuse_dump_dir = self.case_dir / "autofuse_dump"
        autofuse_dump_dir.mkdir(parents=True, exist_ok=True)

        for key in self._ENV_KEYS:
            self._saved[key] = os.environ.get(key)
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCH_LOGS"] = "+dynamo,+inductor,+graph_breaks"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache_dir)
        os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(debug_dir)
        enable_autofuse_graph_dump(autofuse_dump_dir)
        return self

    def __exit__(self, *exc_info) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


# "scale" controls max_elem_per_tensor (total tensor size), concr_ph_dim_rng
# (per-dim range), and timeout_ms together: these all answer the same
# underlying question ("how big/slow are generated cases allowed to be"),
# so one --tensor-scale flag drives all three rather than three
# independently-settable, easily-inconsistent flags (e.g. a large
# max_elem_per_tensor paired with a tiny concr_ph_dim_rng just gets spent on
# more dims of size 1, not genuinely larger tensors).
TENSOR_SCALE_PROFILES: dict[str, dict[str, object]] = {
    # small: unchanged from the original hardcoded defaults -- kept as the
    # regression baseline so existing --repro-seed batches stay reproducible
    # unless --tensor-scale is explicitly set to something else.
    "small": {
        "max_elem_per_tensor": 2**16,
        "concr_ph_dim_rng": (1, 64),
        "timeout_ms": 10000,
    },
    "medium": {
        "max_elem_per_tensor": 2**20,
        "concr_ph_dim_rng": (8, 4096),
        "timeout_ms": 30000,
    },
    "large": {
        "max_elem_per_tensor": 2**24,
        "concr_ph_dim_rng": (8, 8192),
        "timeout_ms": 60000,
    },
}


def run_single_case(
    *,
    factory: AscendC,
    opset,
    model_cls,
    case_seed: int,
    index: int = 0,
    max_nodes: int = 5,
    timeout_ms: Optional[int] = None,
    method: str = "symbolic-cinit",
    max_elem_per_tensor: Optional[int] = None,
    concr_ph_dim_rng: Optional[tuple[int, int]] = None,
    tensor_scale: str = "small",
    out_dir: Optional[str | Path] = None,
    debug: bool = False,
    value_dist_mode: str = "uniform",
) -> CaseResult:
    """Generate and verify exactly one case, identified solely by `case_seed`
    (plus the generation knobs) -- deterministic and independent of any prior
    batch run or dumped file. Used both by `run_fuzz_session`'s loop body and
    by the `--repro-seed` CLI path.
    """
    if tensor_scale not in TENSOR_SCALE_PROFILES:
        raise ValueError(
            f"Unknown tensor_scale {tensor_scale!r}; expected one of "
            f"{sorted(TENSOR_SCALE_PROFILES.keys())}"
        )
    profile = TENSOR_SCALE_PROFILES[tensor_scale]
    if timeout_ms is None:
        timeout_ms = profile["timeout_ms"]
    if max_elem_per_tensor is None:
        max_elem_per_tensor = profile["max_elem_per_tensor"]
    if concr_ph_dim_rng is None:
        concr_ph_dim_rng = profile["concr_ph_dim_rng"]

    t0 = time.time()
    case_id = f"{index:05d}-{case_seed}"
    out_dir = Path(out_dir) if out_dir is not None else None

    # Randomize the actual node-count ceiling within [2, max_nodes] so
    # generated graphs vary in size instead of every case landing on
    # exactly max_nodes nodes (abstract_gen's loop condition is
    # `num_op() < max_node_size`, so it always inserts up to the ceiling
    # it's given -- the ceiling itself needs to vary, not the loop).
    # set_seed(case_seed) here, before consuming this random draw, keeps
    # it reproducible under --repro-seed: model_gen(seed=case_seed, ...)
    # below does its own set_seed(case_seed) again (a full, idempotent
    # reset, not an incremental draw), so this draw doesn't perturb the
    # RNG state the graph-structure search consumes afterwards.
    set_seed(case_seed)
    actual_max_nodes = random.randint(2, max_nodes) if max_nodes >= 2 else max_nodes

    try:
        # model_gen -> BaseGen.__init__ already calls nnsmith.util.set_seed(seed)
        # itself, so graph structure (op/shape/dtype) is already fully
        # determined by case_seed. But graph *search* then consumes further
        # steps from the same random/numpy/torch global RNGs it just reset,
        # leaving them at a state that depends on how long the search ran --
        # not purely on case_seed. SymbolNet's default weight init,
        # refine_weights's PracticalHybridSearch, and make_oracle's
        # get_random_inps all read from those same global RNGs with no seed
        # argument of their own. Re-pinning them here, right before weight
        # init, decouples weight/input generation from the graph search's
        # random-draw count, so the same case_seed reproduces the same
        # weights and inputs too -- not just the same graph.
        gen = model_gen(
            opset=opset,
            method=method,
            seed=case_seed,
            max_elem_per_tensor=max_elem_per_tensor,
            concr_ph_dim_rng=concr_ph_dim_rng,
            max_nodes=actual_max_nodes,
            timeout_ms=timeout_ms,
            # dtype_choices intentionally omitted: passing dtype_choices=None
            # lets BaseGen intersect nnsmith's full dtype set (DTYPE_GEN_ALL)
            # with the opset's own supported dtypes, so every dtype this
            # ascendc opset snapshot actually supports can appear, instead of
            # locking generation to fp32 only.
            init_fp=True,
            # dtype_choices alone only constrains *later* nodes' dtype pool --
            # SymbolicGen's very first placeholder ignores it and is fp32 only
            # when init_fp=True is passed explicitly (default False lets the
            # entry node roll any dtype, which then propagates downstream).
        )
        ir = gen.make_concrete()
        set_seed(case_seed)
        model = model_cls.from_gir(ir)
        model.refine_weights(value_dist_mode=value_dist_mode)
        oracle = model.make_oracle(mode=value_dist_mode)
        testcase = TestCase(model, oracle)
    except InternalError:
        raise
    except Exception:
        return CaseResult(
            seed=case_seed,
            index=index,
            case_id=case_id,
            status="gen_error",
            duration_s=time.time() - t0,
            error_summary=_truncate(traceback.format_exc()),
        )

    op_types = sorted({inst.iexpr.op.name() for inst in ir.insts})

    case_dir = (out_dir / f"case-{case_id}") if (out_dir is not None and debug) else None
    dump_dir = (out_dir / f"bug-{index}-{case_seed}") if out_dir is not None else None

    rtol, atol = _tolerance_for_ir(ir)

    # Lightweight fusion-outcome bookkeeping, independent of the heavy
    # per-case `debug`/`_DebugDumpEnv` dump path above: install_autofuse_
    # intermediate_hooks() is idempotent (a no-op after its first call), so
    # calling it unconditionally here is safe and self-contained -- it also
    # self-activates the ascendc backend if needed (its own first line is
    # `package_root = package_root or activate_autofuse_backend()`), so no
    # separate activate_autofuse_backend() call is needed here. reset before
    # the compile+run so only this case's kernels are counted; collect after,
    # regardless of pass/fail, since a failed case's fusion outcome is
    # diagnostic too.
    install_autofuse_intermediate_hooks()
    reset_autofuse_diagnostics()

    if debug and case_dir is not None:
        with _DebugDumpEnv(case_dir):
            bug = factory.verify_testcase(testcase, equal_nan=True, rtol=rtol, atol=atol)
    else:
        bug = factory.verify_testcase(testcase, equal_nan=True, rtol=rtol, atol=atol)

    diagnostics = autofuse_diagnostics()
    n_kernels = len(diagnostics.get("generated_kernels", []))

    duration = time.time() - t0

    if bug is None:
        if debug and case_dir is not None:
            testcase.dump(str(case_dir))
            write_repro_script(case_dir, target=factory.target, rtol=rtol, atol=atol)
        return CaseResult(
            seed=case_seed,
            index=index,
            case_id=case_id,
            status="pass",
            duration_s=duration,
            n_nodes=ir.n_compute_inst(),
            n_vars=ir.n_var(),
            op_types=op_types,
            bug_dir=str(case_dir) if (debug and case_dir is not None) else None,
            input_shapes={name: list(t.shape) for name, t in model.input_like.items()},
            output_shapes={name: list(t.shape) for name, t in model.output_like.items()},
            n_kernels=n_kernels,
        )

    target_dir = case_dir if (debug and case_dir is not None) else dump_dir
    if target_dir is not None:
        bug.dump(str(target_dir))
        write_repro_script(target_dir, target=factory.target, rtol=rtol, atol=atol)
    return CaseResult(
        seed=case_seed,
        index=index,
        case_id=case_id,
        status="fail",
        duration_s=duration,
        n_nodes=ir.n_compute_inst(),
        n_vars=ir.n_var(),
        op_types=op_types,
        symptom=bug.symptom.value,
        stage=bug.stage.value,
        bug_dir=str(target_dir) if target_dir is not None else None,
        error_summary=_truncate(bug.log or ""),
        input_shapes={name: list(t.shape) for name, t in model.input_like.items()},
        output_shapes={name: list(t.shape) for name, t in model.output_like.items()},
        n_kernels=n_kernels,
    )


def run_repro_case(
    *,
    out_dir: str | Path,
    repro_seed: int,
    max_nodes: int = 5,
    timeout_ms: Optional[int] = None,
    method: str = "symbolic-cinit",
    max_elem_per_tensor: Optional[int] = None,
    concr_ph_dim_rng: Optional[tuple[int, int]] = None,
    tensor_scale: str = "small",
    target: str = "npu",
    debug: bool = True,
    soc_tier: str = "A3",
    value_dist_mode: str = "uniform",
) -> CaseResult:
    """Re-run exactly one case identified by its original `case_seed`
    (`--repro-seed`), with debug dumping on by default -- the "give me this
    one case's id, reproduce it, and dump debug artifacts" entrypoint.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    factory = AscendC(target=target)
    model_cls = Model.init("torch", backend_target=target)
    model_cls.add_seed_setter()
    opset = probe_ascendc_opset(model_cls, factory, soc_tier=soc_tier, grad=False)

    return run_single_case(
        factory=factory,
        opset=opset,
        model_cls=model_cls,
        case_seed=repro_seed,
        index=0,
        max_nodes=max_nodes,
        timeout_ms=timeout_ms,
        method=method,
        max_elem_per_tensor=max_elem_per_tensor,
        concr_ph_dim_rng=concr_ph_dim_rng,
        tensor_scale=tensor_scale,
        out_dir=out_dir,
        debug=debug,
        value_dist_mode=value_dist_mode,
    )


def run_fuzz_session(
    *,
    out_dir: str | Path,
    n_cases: int = 100,
    max_nodes: int = 5,
    timeout_ms: Optional[int] = None,
    method: str = "symbolic-cinit",
    max_elem_per_tensor: Optional[int] = None,
    concr_ph_dim_rng: Optional[tuple[int, int]] = None,
    tensor_scale: str = "small",
    seed: Optional[int] = None,
    target: str = "npu",
    index_offset: int = 0,
    write_outputs: bool = True,
    soc_tier: str = "A3",
    value_dist_mode: str = "uniform",
) -> FuzzSummary:
    """Generate `n_cases` NNSmith torch models, run each end-to-end against the
    autofuse backend, verify against the CPU/eager oracle NNSmith itself
    generated, and record results. Returns the run summary (also written to
    `<out_dir>/summary.json` unless `write_outputs=False`, which
    `run_fuzz_session_parallel` uses for per-shard runs).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    factory = AscendC(target=target)
    model_cls = Model.init("torch", backend_target=target)
    model_cls.add_seed_setter()

    # Narrow NNSmith's op catalog to the ones ascendc's `_LoweringGuard`
    # actually registers a fused lowering for on `soc_tier`, probing each op's
    # real compile+run dtype support directly (does NOT go through NNSmith's
    # own auto_opset()/auto_opconfig() -- see probe_ascendc_opset()'s
    # docstring: that path probes the full ~67-op catalog in one fault-
    # intolerant loop, and a single op's probe-time crash silently discards
    # every op queued after it, which is how an earlier run ended up with a
    # topset cache containing zero fp32 combinations for Add/Sub/Mul/etc).
    opset = probe_ascendc_opset(model_cls, factory, soc_tier=soc_tier, grad=False)

    summary = FuzzSummary()

    for offset in range(n_cases):
        index = index_offset + offset
        case_seed = rng.getrandbits(32)
        result = run_single_case(
            factory=factory,
            opset=opset,
            model_cls=model_cls,
            case_seed=case_seed,
            index=index,
            max_nodes=max_nodes,
            timeout_ms=timeout_ms,
            method=method,
            max_elem_per_tensor=max_elem_per_tensor,
            concr_ph_dim_rng=concr_ph_dim_rng,
            tensor_scale=tensor_scale,
            out_dir=out_dir,
            debug=False,
            value_dist_mode=value_dist_mode,
        )
        # Cheap per-case safety net; it is NOT what bounds memory in a batch.
        # Measured on A5 across 20 real cases (run_single_case in a loop):
        # torch.npu.memory_allocated() stayed at 0 and memory_reserved() at
        # 0-2MB, so there is almost nothing here for empty_cache() to give
        # back. The cold-compile-cache OOM seen on a 32GB cgroup (exit 137
        # at ~116s of a 20-case batch) came from inductor's parallel-compile
        # worker pool instead: TORCHINDUCTOR_COMPILE_THREADS defaults to
        # min(32, host cpu count) and a container reports the host's cores,
        # so ~33 python processes (0.7-1.5GB each, ~24GB total) were spawned.
        # They are child processes, so they are invisible to this process's
        # own RSS and only show up in the cgroup. Bounded in
        # autofusedebugger/__init__.py (default 4) and tunable via
        # `af-fuzz --compile-threads` / `--jobs`; see docs/af_fuzz_npu_env.md.
        try:
            import torch

            torch.npu.empty_cache()
        except Exception:
            pass
        summary.cases.append(result)
        summary.total += 1
        if result.status == "pass":
            summary.passed += 1
        elif result.status == "fail":
            summary.failed += 1
        else:
            summary.gen_errors += 1

    if write_outputs:
        summary_dict = summary.to_dict()
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)
        write_summary_html(summary_dict, out_dir / "summary.html")

    return summary


def _run_shard(args: tuple) -> FuzzSummary:
    (
        shard_out_dir,
        shard_n_cases,
        max_nodes,
        timeout_ms,
        method,
        max_elem_per_tensor,
        shard_seed,
        target,
        index_offset,
        soc_tier,
        concr_ph_dim_rng,
        tensor_scale,
        value_dist_mode,
    ) = args
    return run_fuzz_session(
        out_dir=shard_out_dir,
        n_cases=shard_n_cases,
        max_nodes=max_nodes,
        timeout_ms=timeout_ms,
        method=method,
        max_elem_per_tensor=max_elem_per_tensor,
        concr_ph_dim_rng=concr_ph_dim_rng,
        tensor_scale=tensor_scale,
        seed=shard_seed,
        target=target,
        index_offset=index_offset,
        write_outputs=False,
        soc_tier=soc_tier,
        value_dist_mode=value_dist_mode,
    )


def run_fuzz_session_parallel(
    *,
    out_dir: str | Path,
    n_cases: int = 100,
    max_nodes: int = 5,
    timeout_ms: Optional[int] = None,
    method: str = "symbolic-cinit",
    max_elem_per_tensor: Optional[int] = None,
    concr_ph_dim_rng: Optional[tuple[int, int]] = None,
    tensor_scale: str = "small",
    seed: Optional[int] = None,
    target: str = "npu",
    jobs: int = 1,
    soc_tier: str = "A3",
    value_dist_mode: str = "uniform",
) -> FuzzSummary:
    """Shard `n_cases` across `jobs` worker processes, each with its own
    disjoint seed sub-stream (derived from the top-level `seed` so the whole
    run stays reproducible) and own case-index range, so dumped `bug-<index>-*`
    directories never collide. Results are merged into one summary written to
    `<out_dir>/summary.json` / `summary.html`, matching the single-process
    path's output shape exactly.
    """
    if jobs <= 1:
        return run_fuzz_session(
            out_dir=out_dir,
            n_cases=n_cases,
            max_nodes=max_nodes,
            timeout_ms=timeout_ms,
            method=method,
            max_elem_per_tensor=max_elem_per_tensor,
            concr_ph_dim_rng=concr_ph_dim_rng,
            tensor_scale=tensor_scale,
            seed=seed,
            target=target,
            soc_tier=soc_tier,
            value_dist_mode=value_dist_mode,
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive one independent seed per shard from the top-level seed so the
    # overall run is still reproducible as a whole, without every shard
    # replaying the same case_seed sequence.
    master_rng = random.Random(seed)
    shard_seeds = [master_rng.getrandbits(32) for _ in range(jobs)]

    base = n_cases // jobs
    remainder = n_cases % jobs
    shard_args = []
    index_offset = 0
    for shard_idx in range(jobs):
        shard_n_cases = base + (1 if shard_idx < remainder else 0)
        if shard_n_cases == 0:
            continue
        shard_args.append(
            (
                out_dir,
                shard_n_cases,
                max_nodes,
                timeout_ms,
                method,
                max_elem_per_tensor,
                shard_seeds[shard_idx],
                target,
                index_offset,
                soc_tier,
                concr_ph_dim_rng,
                tensor_scale,
                value_dist_mode,
            )
        )
        index_offset += shard_n_cases

    ctx = multiprocessing.get_context("spawn")
    merged = FuzzSummary()
    with ctx.Pool(processes=len(shard_args)) as pool:
        for shard_summary in pool.map(_run_shard, shard_args):
            merged.merge(shard_summary)

    merged.cases.sort(key=lambda c: c.index)

    summary_dict = merged.to_dict()
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)
    write_summary_html(summary_dict, out_dir / "summary.html")

    return merged
