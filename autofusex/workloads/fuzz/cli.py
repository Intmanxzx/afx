"""Independent stdlib-only front door for fuzz use cases."""
import argparse
import importlib.metadata
import json
import platform
import sqlite3
import sys

from autofusex.client.paths import data_home, setup, load_config, identity_status
from autofusex.execution.resources import resource_inventory, budget_warnings
from autofusex.execution.backends import BACKEND_NAMES


def doctor(home):
    versions = {}
    for name in ("torch", "torch-npu", "nnsmith", "triton-ascend"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    inventory = resource_inventory(data_home(home))
    config = load_config(home)
    return {"resources": inventory,
            "configuration": "configured" if config else "not_configured",
            "identity": identity_status(config or {}),
            "budget_warnings": budget_warnings(config or {}, inventory),
            "python": platform.python_version(), "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version, "home": str(data_home(home)),
            "packages": versions, "device_probe": "not_run", "split_execution": "unverified",
            "generation": "Single NNSmith ATen E/F protocol; pinned source/Z3/catalog; generation and CPU skips are excluded"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "replay":
        return replay(argv[1:])
    from autofusex.workloads.fuzz.help import ScenarioParser
    parser = ScenarioParser(prog="afx fuzz")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=argparse.ArgumentParser)
    sub.add_parser("replay", help="replay a self-contained short seed")
    maintenance = sub
    seed = maintenance.add_parser("seed", help="inspect a frozen short-seed case without execution")
    seed.add_argument("token")
    seed.add_argument("--show-spec", action="store_true")
    compare = maintenance.add_parser("diagnose", help="compare two opt-in generation traces")
    compare.add_argument("expected")
    compare.add_argument("actual")
    for child, command in [(seed, 'seed'), (compare, 'diagnose')]:
        child.set_defaults(command=command)
    from autofusex.workloads.fuzz.scheduling.cli import register
    register(sub, maintenance)
    args = parser.parse_args(argv)
    try:
        if args.command in {"run", "status", "stop", "resume"}:
            from autofusex.workloads.fuzz.scheduling.cli import dispatch
            result = dispatch(args)
        elif args.command == "seed":
            from autofusex.workloads.fuzz.generation import materialize, content_manifest, workload_id
            import time
            generation_start=time.perf_counter_ns()
            bundle = materialize(args.token)
            generation_ns=time.perf_counter_ns()-generation_start
            result = {"seed": bundle.spec.seed.short, "seed_numeric": str(bundle.spec.seed), "workload_id": workload_id(bundle),
                      "preview": True, "shape_count": len(bundle.spec.shape_samples)}
            if args.show_spec:
                result["manifest"] = content_manifest(bundle)
        elif args.command == "diagnose":
            from pathlib import Path
            from autofusex.workloads.fuzz.diagnostics import first_divergence
            result = first_divergence(json.loads(Path(args.expected).read_text()),
                                      json.loads(Path(args.actual).read_text()))
            print(json.dumps(result, indent=2))
            return 1 if result else 0
        else:
            parser.error(f"{args.command} is not implemented yet; no service was started")
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; active managed worker cleanup attempted. Resume interrupted tasks explicitly.", file=sys.stderr)
        return 130
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def replay(argv):
    parser = replay_parser()
    args = parser.parse_args(argv)
    try:
        from autofusex.workloads.fuzz.contracts import Seed
        Seed.parse(args.seed)
    except (ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if getattr(args,'scheduling','combined')=='staged':
        from autofusex.workloads.fuzz.scheduling.staged import replay as staged_replay
        return staged_replay(args)
    if args.backend is None:
        # Normal replay includes both mandatory standards. Explicit --backend
        # remains useful for focused debugging of an already identified failure.
        import copy
        codes = []
        for backend in ('cpu', 'af-mlir'):
            selected = copy.copy(args)
            selected.backend = backend
            if backend == 'cpu':
                selected.device = 'cpu'
            else:
                from autofusex.execution.scheduling.devices import select_replay_device
                selected.device = select_replay_device(args.device)
            codes.append(execute_args(selected))
        return max(codes)
    return execute_args(args)


def replay_parser():
    parser = argparse.ArgumentParser(prog="afx fuzz replay")
    parser.add_argument("seed", help="self-contained short seed")
    parser.add_argument("--scheduling",choices=("combined","staged"),default="staged")
    parser.add_argument("--dump", action="store_true", help="enable compiler debug dumps (default: off); independent of --save")
    parser.add_argument("--static-only", action="store_true", help="execute base sample only; does not change case or workload ID")
    parser.add_argument("--backend", choices=BACKEND_NAMES, help="focused backend diagnostic; default replays CPU and AF-MLIR")
    parser.add_argument("--device")
    parser.add_argument("--database", help=argparse.SUPPRESS)
    parser.add_argument("--save", action="store_true", help="save results to the local database; default: neither")
    parser.add_argument("--compile-threads", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--perf", dest="performance", action="store_true", help="request precise metrics where supported; unavailable metrics remain empty")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=10)
    return parser


def execute_args(args):
    import os
    import tempfile
    from autofusex.client.cleanup import debug_worker
    from autofusex.workloads.fuzz.scheduling.supervisor import _debug_env
    dump = getattr(args,'dump',False)
    home = data_home(); home.mkdir(parents=True,exist_ok=True)
    directory, lease = debug_worker(home) if dump else (None,None)
    with tempfile.TemporaryDirectory(prefix='.replay-cache-',dir=home) as scratch:
        before = dict(os.environ)
        try:
            configured = _debug_env(before,scratch,directory,args.backend,dump)
            os.environ.clear(); os.environ.update(configured)
            return _execute_args(args)
        finally:
            os.environ.clear(); os.environ.update(before)
            if lease: lease.close()


def _execute_args(args):
    try:
        from autofusex.contracts.execution import ExecutionConfig, EvaluationConfig
        from autofusex.workloads.fuzz.contracts import Seed
        from autofusex.workloads.fuzz.generation import materialize
        from autofusex.workloads.fuzz.execution import execute
        from autofusex.execution.backends import create_backend
        from autofusex.execution.resources import adaptive_defaults
        threads = args.compile_threads or adaptive_defaults(data_home())["compile_threads"]
        config = ExecutionConfig(args.backend, args.device or ("npu:0" if args.backend in {"eager-npu", "triton-npu", "af-legacy", "af-mlir"} else "cpu"), threads, static_only=args.static_only or not Seed.parse(args.seed).dynamic, save_results=args.save, dump=getattr(args,'dump',False))
        evaluation = EvaluationConfig(performance=args.performance, warmup_steps=args.warmup_steps, measured_steps=args.measured_steps)
        import time
        generation_start = time.perf_counter_ns()
        bundle = materialize(args.seed)
        generation_ns = time.perf_counter_ns() - generation_start
        result = execute(bundle, create_backend(config), evaluation)
        telemetry=json.loads(result['execution']['telemetry'])
        telemetry['stages']['generation']=generation_ns
        result['execution']['telemetry']=json.dumps(telemetry)
        if args.save:
            from autofusex.execution.storage import ResultRepository
            database = args.database or data_home() / "results/results.sqlite"
            from autofusex.execution.environment import collect
            environment=collect()
            repo = ResultRepository(database)
            result["attempt_id"] = repo.save(result, environment=environment)
            result["database"] = str(repo.path)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "ok" else 1
    except (ValueError, TypeError, OSError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
