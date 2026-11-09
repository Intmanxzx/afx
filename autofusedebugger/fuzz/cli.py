"""Stdlib-only front door. Legacy execution is forwarded, not duplicated."""
import argparse
import importlib.metadata
import json
import platform
import sqlite3
import sys

from .config import data_home, setup, load_config, identity_status
from .resources import resource_inventory, budget_warnings


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
            "v1_generation": "numeric version 1; cross-architecture acceptance pending", "sync": "not_implemented"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "replay":
        return replay(argv[1:])
    parser = argparse.ArgumentParser(prog="autofusedbg fuzz")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("replay", help="replay a complete numeric seed; use replay --help")
    for command in ("setup", "doctor"):
        p = sub.add_parser(command)
        p.add_argument("--data-dir")
        if command == "setup":
            p.add_argument("--display-name")
            p.add_argument("--user-id")
            p.add_argument("--key-file")
            p.add_argument("--cpu-workers", type=int)
            p.add_argument("--compile-threads", type=int)
            p.add_argument("--memory-budget-bytes", type=int)
            p.add_argument("--cache-budget-bytes", type=int)
    seed = sub.add_parser("seed", help="materialize a preview v1 seed without execution dependencies")
    seed.add_argument("token")
    seed.add_argument("--show-spec", action="store_true")
    compare = sub.add_parser("diagnose", help="compare two opt-in generation traces")
    compare.add_argument("expected")
    compare.add_argument("actual")
    for command in ("view", "snapshot", "status", "stop", "resume"):
        p = sub.add_parser(command, help="reserved; not implemented in D1")
        p.add_argument("target", nargs="?")
    args = parser.parse_args(argv)
    try:
        if args.command == "setup":
            result = setup(args.data_dir, display_name=args.display_name, user_id=args.user_id,
                           key_file=args.key_file, cpu_workers=args.cpu_workers,
                           compile_threads=args.compile_threads, memory_budget_bytes=args.memory_budget_bytes,
                           cache_budget_bytes=args.cache_budget_bytes)
        elif args.command == "seed":
            from .protocol_v1 import materialize, content_manifest, workload_id
            bundle = materialize(args.token)
            result = {"seed": str(bundle.spec.seed), "workload_id": workload_id(bundle),
                      "preview": True, "shape_count": len(bundle.spec.shape_samples)}
            if args.show_spec:
                result["manifest"] = content_manifest(bundle)
        elif args.command == "doctor":
            result = doctor(args.data_dir)
        elif args.command == "diagnose":
            from pathlib import Path
            from .diagnostics import first_divergence
            result = first_divergence(json.loads(Path(args.expected).read_text()),
                                      json.loads(Path(args.actual).read_text()))
            print(json.dumps(result, indent=2))
            return 1 if result else 0
        else:
            parser.error(f"{args.command} is not implemented yet; no service was started")
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def replay(argv):
    parser = replay_parser()
    args = parser.parse_args(argv)
    return execute_args(args)


def replay_parser():
    parser = argparse.ArgumentParser(prog="autofusedbg fuzz replay")
    parser.add_argument("seed", help="self-contained nonnegative decimal seed")
    parser.add_argument("--static-only", action="store_true", help="execute base sample only; does not change case or workload ID")
    parser.add_argument("--backend", choices=("cpu-eager", "cpu-compile", "ascendc", "triton"), default="cpu-eager")
    parser.add_argument("--device")
    parser.add_argument("--compile-threads", type=int, default=1)
    parser.add_argument("--performance", action="store_true", help="optional CPU wall-time measurement; NPU measurement needs D4 leases")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=10)
    return parser


def execute_args(args):
    try:
        from .contracts import ExecutionConfig, EvaluationConfig
        from .protocol_v1 import materialize
        from .execution import TorchAdapter, execute
        config = ExecutionConfig(args.backend, args.device or ("npu:0" if args.backend in {"ascendc", "triton"} else "cpu"), args.compile_threads, static_only=args.static_only)
        evaluation = EvaluationConfig(performance=args.performance, warmup_steps=args.warmup_steps, measured_steps=args.measured_steps)
        bundle = materialize(args.seed)
        result = execute(bundle, TorchAdapter(config), evaluation)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "ok" else 1
    except (ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
