"""Optional af-fuzz performance mode; argparse setup stays lightweight."""

def add_arguments(parser):
    group = parser.add_argument_group("backend performance comparison")
    group.add_argument("--compare-performance", action="store_true",
                       help="compare AscendC and default Triton-Ascend kernel time using isolated NPU profiler workers; dynamic workers across distinct --devices; each case benchmarks backends serially")
    group.add_argument("--keep-perf-artifacts", action="store_true",
                       help="retain raw profiling and per-run compile caches; default keeps only CSV/results/logs/reports")
    group.add_argument("--resume", action="store_true",
                       help="resume a performance session from --out-dir; keep generation and measurement settings unchanged")
    group.add_argument("--warmup-steps", type=int, default=3,
                       help="warmup iterations outside profiling, after compilation (default 3)")
    group.add_argument("--profile-steps", type=int, default=10,
                       help="measured iterations; sum op_statistic Total Time / this count (default 10)")
    group.add_argument("--perf-timeout-s", type=float, default=600,
                       help="timeout for each backend worker including compilation and profiling (default 600s)")


def run(args, *, op_categories, devices):
    import os
    from ..generation import GenerationConfig
    from .schema import ProfileConfig
    from .session import run_performance_session

    if args.compile_threads is not None:
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(args.compile_threads)
    generation = GenerationConfig(
        max_nodes=args.max_nodes, timeout_ms=args.timeout_ms, method=args.method,
        max_elem_per_tensor=args.max_elem_per_tensor, tensor_scale=args.tensor_scale,
        value_dist_mode=args.value_dist_mode,
    )
    profile = ProfileConfig(warmup_steps=args.warmup_steps, steps=args.profile_steps,
                            timeout_s=args.perf_timeout_s, keep_artifacts=args.keep_perf_artifacts)
    summary = run_performance_session(
        out_dir=args.out_dir, generation=generation, profile=profile,
        n_cases=args.n_cases, seed=args.seed, repro_seed=args.repro_seed,
        target=args.target, soc_tier=args.soc_tier, op_categories=op_categories,
        devices=devices, resume=args.resume, jobs=args.jobs,
    )
    print(f"total={summary['total']} comparable={summary['comparable']} perf_pass={summary['performance_passed']} perf_fail={summary['performance_failed']} errors={summary['errors']}; summary: {args.out_dir}/summary.html")
    if summary.get("stop_signal"):
        return 128 + summary["stop_signal"]
    return 0 if summary["errors"] == 0 else 1
