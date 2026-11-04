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

Case identity: `case_id` is `f"{index:05d}-{case_seed}"`. Under fixed
software, generation configuration and op/dtype candidates, `case_seed`
regenerates the graph, weights, inputs and CPU oracle without saved artifacts.
Candidate ordering is canonical, numeric search uses fixed sample/gradient
step counts, and CPU materialization uses one thread and deterministic Torch
algorithms. Graph/SMT timeouts fail generation instead of returning a different
case. See docs/af_fuzz_guide.md for the replay command and version boundary.

Concurrency: generation and verification run one case at a time within a
single process/device context (see `run_fuzz_session`). Parallelism across
cases uses a shared task pool with multiple OS *processes*
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
from typing import Any, Optional, Sequence

from nnsmith.error import InternalError
from nnsmith.materialize import Model

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
from autofusedebugger.nnsmith_integration.ascendc_opset import (
    OP_CATEGORIES,
    probe_ascendc_opset,
)
from autofusedebugger.nnsmith_integration.generation import (
    GenerationConfig, generate_case, TENSOR_SCALE_PROFILES,
    _seeded_solver_context, _seeded_materialization, _tolerance_for_ir,
)
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
    # DType.short() string (e.g. "f32", "bf16") per tensor name, same keys as
    # input_shapes/output_shapes -- kept as separate dicts rather than
    # merging into the shape dicts so existing readers of input_shapes/
    # output_shapes (e.g. any external script parsing summary.json) don't
    # need to change how they index into those two fields.
    input_dtypes: dict[str, str] = field(default_factory=dict)
    output_dtypes: dict[str, str] = field(default_factory=dict)
    # Number of kernels inductor/autofuse actually generated and ran on NPU
    # for this case (len(generated_kernels) from autofuse_diagnostics()).
    # Left at the default 0 for "gen_error" cases, which never reach
    # compilation.
    n_kernels: int = 0
    shape_mode: str = "static"
    dynamic_axes: dict = field(default_factory=dict)
    shape_variants: list = field(default_factory=list)
    dynamic_fingerprint: Optional[str] = None

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
    if len(text) <= limit:
        return text
    # Keep the exception type/message at the end even when call paths grow.
    head = limit // 3
    return text[:head] + " ...[truncated]... " + text[-(limit - head):]


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
    tensor_scale: str = "default",
    out_dir: Optional[str | Path] = None,
    debug: bool = False,
    value_dist_mode: str = "uniform",
    dynamic_shapes: bool = False,
    dynamic_variants: int = 3,
) -> CaseResult:
    """Generate and verify one case using `case_seed` under a fixed generation
    configuration, op/dtype pool and software environment. Used both by
    `run_fuzz_session`'s loop body and by the `--repro-seed` CLI path.
    """
    config = GenerationConfig(
        max_nodes=max_nodes, timeout_ms=timeout_ms, method=method,
        max_elem_per_tensor=max_elem_per_tensor, concr_ph_dim_rng=concr_ph_dim_rng,
        tensor_scale=tensor_scale, value_dist_mode=value_dist_mode,
        dynamic_shapes=dynamic_shapes, dynamic_variants=dynamic_variants,
    )
    t0 = time.time()
    case_id = f"{index:05d}-{case_seed}"
    out_dir = Path(out_dir) if out_dir is not None else None
    try:
        testcase = generate_case(opset=opset, model_cls=model_cls, case_seed=case_seed, config=config)
        model = testcase.model
        ir = model.native_model.ir
    except Exception as exc:
        # A graph search can exhaust its budget without finding a model.
        # Record that case and continue, but preserve other internal failures.
        if isinstance(exc, InternalError) and "Run check_sat first! | expr is None" not in str(exc):
            raise
        return CaseResult(
            seed=case_seed,
            index=index,
            case_id=case_id,
            status="gen_error",
            shape_mode="dynamic" if dynamic_shapes else "static",
            duration_s=time.time() - t0,
            error_summary=_truncate(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"),
        )

    if dynamic_shapes:
        from .dynamic_shapes import verify_workload
        metadata = dict(testcase.dynamic_metadata)
        try:
            rtol,atol = _tolerance_for_ir(ir)
            verify_workload(testcase,target=factory.target,seed=case_seed,variants=dynamic_variants,
                max_elements=max_elem_per_tensor or TENSOR_SCALE_PROFILES[tensor_scale]['max_elem_per_tensor'],
                rtol=rtol,atol=atol)
            status, error = "pass", None
        except Exception:
            status,error = "fail",_truncate(traceback.format_exc())
        return CaseResult(seed=case_seed,index=index,status=status,duration_s=time.time()-t0,
            n_nodes=ir.n_compute_inst(),op_types=sorted({i.iexpr.op.name() for i in ir.insts}),
            input_shapes={n:list(v.shape) for n,v in testcase.oracle.input.items()},
            output_shapes={n:list(v.shape) for n,v in testcase.oracle.output.items()},
            input_dtypes={n:t.dtype.short() for n,t in model.input_like.items()},
            output_dtypes={n:t.dtype.short() for n,t in model.output_like.items()},
            dynamic_fingerprint=testcase.dynamic_fingerprint,
            error_summary=error,stage="dynamic_compile_execute" if error else None,**metadata)

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
            input_dtypes={name: t.dtype.short() for name, t in model.input_like.items()},
            output_dtypes={name: t.dtype.short() for name, t in model.output_like.items()},
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
        input_dtypes={name: t.dtype.short() for name, t in model.input_like.items()},
        output_dtypes={name: t.dtype.short() for name, t in model.output_like.items()},
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
    tensor_scale: str = "default",
    target: str = "npu",
    debug: bool = True,
    soc_tier: str = "A3",
    value_dist_mode: str = "uniform",
    dynamic_shapes: bool = False,
    dynamic_variants: int = 3,
    op_categories: Optional[Sequence[str]] = None,
) -> CaseResult:
    """Re-run exactly one case identified by its original `case_seed`
    (`--repro-seed`), with debug dumping on by default -- the "give me this
    one case's id, reproduce it, and dump debug artifacts" entrypoint.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    factory = AscendC(target=target)
    # Generate numeric data and the reference oracle on CPU for every target.
    # The backend moves the resulting model/inputs to its execution device.
    model_cls = Model.init("torch", backend_target="cpu")
    model_cls.add_seed_setter()
    opset = probe_ascendc_opset(
        model_cls, factory, soc_tier=soc_tier, grad=False, op_categories=op_categories
    )

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
        # --no-debug is a seed-only replay, including when it fails.
        out_dir=out_dir if debug else None,
        debug=debug,
        value_dist_mode=value_dist_mode,
        dynamic_shapes=dynamic_shapes, dynamic_variants=dynamic_variants,
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
    tensor_scale: str = "default",
    seed: Optional[int] = None,
    target: str = "npu",
    index_offset: int = 0,
    write_outputs: bool = True,
    soc_tier: str = "A3",
    value_dist_mode: str = "uniform",
    dynamic_shapes: bool = False,
    dynamic_variants: int = 3,
    op_categories: Optional[Sequence[str]] = None,
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
    # Generate numeric data and the reference oracle on CPU for every target.
    # The backend moves the resulting model/inputs to its execution device.
    model_cls = Model.init("torch", backend_target="cpu")
    model_cls.add_seed_setter()

    # Narrow NNSmith's op catalog to the ones ascendc's `_LoweringGuard`
    # actually registers a fused lowering for on `soc_tier`, probing each op's
    # real compile+run dtype support directly (does NOT go through NNSmith's
    # own auto_opset()/auto_opconfig() -- see probe_ascendc_opset()'s
    # docstring: that path probes the full ~67-op catalog in one fault-
    # intolerant loop, and a single op's probe-time crash silently discards
    # every op queued after it, which is how an earlier run ended up with a
    # topset cache containing zero fp32 combinations for Add/Sub/Mul/etc).
    opset = probe_ascendc_opset(
        model_cls, factory, soc_tier=soc_tier, grad=False, op_categories=op_categories
    )

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
            dynamic_shapes=dynamic_shapes, dynamic_variants=dynamic_variants,
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
        write_summary_html(summary_dict, out_dir / "summary.html", op_categories=OP_CATEGORIES)

    return summary


def _shard_target(target: str, devices: Optional[Sequence[int]], shard_idx: int) -> str:
    """Resolve the `target` a given shard index should use: plain `target`
    if `devices` is unset or `target` isn't `"npu"`, else `"npu:<index>"`
    with `devices[shard_idx % len(devices)]` round-robin so `--jobs` can
    exceed the card count (e.g. `--jobs 4 --devices 0,1` puts shards 0/2 on
    npu:0 and shards 1/3 on npu:1).
    """
    if not devices or target.split(":", 1)[0] != "npu":
        return target
    return f"npu:{devices[shard_idx % len(devices)]}"


def _initialize_pool_worker(worker_index, config):
    import os
    worker_target = _shard_target(config['target'], config['devices'], worker_index)
    if worker_target.split(':', 1)[0] == 'npu':
        import torch
        torch.npu.set_device(worker_target)
    cache = Path(config['out_dir']).resolve() / 'worker_cache' / str(worker_index)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache)
    os.environ['TRITON_CACHE_DIR'] = str(cache / 'triton')
    factory = AscendC(target=worker_target)
    model_cls = Model.init('torch', backend_target='cpu')
    model_cls.add_seed_setter()
    opset = probe_ascendc_opset(model_cls, factory, soc_tier=config['soc_tier'],
        grad=False, op_categories=config['op_categories'])
    return factory, model_cls, opset


def _execute_pool_case(runtime, task, config):
    factory, model_cls, opset = runtime
    index, case_seed = task
    result = run_single_case(factory=factory, model_cls=model_cls, opset=opset,
        index=index, case_seed=case_seed, out_dir=Path(config['out_dir']), debug=False,
        **config['generation'])
    try:
        import torch
        torch.npu.empty_cache()
    except Exception:
        pass
    return result


def run_fuzz_session_parallel(
    *,
    out_dir: str | Path,
    n_cases: int = 100,
    max_nodes: int = 5,
    timeout_ms: Optional[int] = None,
    method: str = "symbolic-cinit",
    max_elem_per_tensor: Optional[int] = None,
    concr_ph_dim_rng: Optional[tuple[int, int]] = None,
    tensor_scale: str = "default",
    seed: Optional[int] = None,
    target: str = "npu",
    jobs: int = 1,
    devices: Optional[Sequence[int]] = None,
    soc_tier: str = "A3",
    value_dist_mode: str = "uniform",
    dynamic_shapes: bool = False,
    dynamic_variants: int = 3,
    op_categories: Optional[Sequence[str]] = None,
) -> FuzzSummary:
    """Workers bound to devices dynamically claim individual cases from one pool.

    All workers consume one global index/seed sequence. No worker owns a fixed
    fraction of the batch. Only the parent writes the merged session reports.
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
            target=_shard_target(target, devices, 0),
            soc_tier=soc_tier,
            value_dist_mode=value_dist_mode,
            dynamic_shapes=dynamic_shapes, dynamic_variants=dynamic_variants,
            op_categories=op_categories,
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_cases < 1:
        raise ValueError('n_cases must be positive')
    from .task_pool import dynamic_results
    from .performance.storage import atomic_write_text

    # One global seed sequence, independent of worker count and completion order.
    rng = random.Random(seed)
    tasks = ((index, rng.getrandbits(32)) for index in range(n_cases))
    config = dict(out_dir=str(out_dir), target=target, devices=devices,
        soc_tier=soc_tier, op_categories=op_categories, generation=dict(
            max_nodes=max_nodes, timeout_ms=timeout_ms, method=method,
            max_elem_per_tensor=max_elem_per_tensor, concr_ph_dim_rng=concr_ph_dim_rng,
            tensor_scale=tensor_scale, value_dist_mode=value_dist_mode,
            dynamic_shapes=dynamic_shapes, dynamic_variants=dynamic_variants))
    merged = FuzzSummary()

    def save():
        merged.cases.sort(key=lambda c: c.index)
        data = merged.to_dict()
        atomic_write_text(out_dir / 'summary.json', json.dumps(data, indent=2, ensure_ascii=False))
        from .summary_html import render_summary_html
        atomic_write_text(out_dir / 'summary.html', render_summary_html(data, op_categories=OP_CATEGORIES))

    stream = dynamic_results(tasks, workers=min(jobs, n_cases), config=config,
        initialize=_initialize_pool_worker, execute=_execute_pool_case)
    try:
        save()
        for result in stream:
            merged.cases.append(result)
            merged.total += 1
            if result.status == 'pass':
                merged.passed += 1
            elif result.status == 'fail':
                merged.failed += 1
            else:
                merged.gen_errors += 1
            save()
    finally:
        stream.close()
        save()
    return merged
