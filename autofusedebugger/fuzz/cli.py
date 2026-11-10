"""Independent stdlib-only front door for fuzz use cases."""
import argparse
import importlib.metadata
import json
import platform
import sqlite3
import sys

from .config import data_home, setup, load_config, identity_status
from .resources import resource_inventory, budget_warnings
from .backends import BACKEND_NAMES


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
            "v1_generation": "frozen numeric v1/v2; aarch64 and x86_64 goldens verified", "sync": "independent_outbox_daemon"}


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
            p.add_argument("--sync-endpoint")
            p.add_argument("--sync-interval", type=float, default=30)
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
    pair = sub.add_parser("compare", help="compare two result attempts with explicit comparability checks")
    pair.add_argument("database")
    pair.add_argument("left")
    pair.add_argument("right")
    snapshot = sub.add_parser("snapshot", help="export a consistent lightweight SQLite snapshot")
    snapshot.add_argument("source")
    snapshot.add_argument("destination")
    for command in ("restore", "import"):
        transfer = sub.add_parser(command, help="restore to a new database" if command == "restore" else "merge into an existing database atomically")
        transfer.add_argument("source")
        transfer.add_argument("destination")
    check = sub.add_parser("check-db", help="audit a trusted result database without modification")
    check.add_argument("database")
    results = sub.add_parser("results", help="read lightweight results without model dependencies")
    results.add_argument("database")
    results.add_argument("--limit", type=int, default=50)
    results.add_argument("--after", type=int)
    results.add_argument("--seed")
    results.add_argument("--attempt")
    view = sub.add_parser("view", help="open an offline read-only SQLite viewer on this computer")
    view.add_argument("database")
    view.add_argument("--port", type=int, default=0, help="loopback port; default chooses a free port")
    view.add_argument("--no-browser", action="store_true")
    central = sub.add_parser("serve", help="central service behind a TLS proxy")
    central.add_argument("database")
    central.add_argument("--port", type=int, default=8090)
    indexing = sub.add_parser('index-results', help='build additive SQLite reporting index after backup')
    indexing.add_argument('database')
    backup = sub.add_parser("server-backup", help="private complete central backup including identities and ACKs")
    backup.add_argument("database")
    backup.add_argument("destination")
    admin = sub.add_parser("admin-user", help="offline administrator creates an upload identity")
    admin.add_argument("database")
    admin.add_argument("display_name")
    admin.add_argument("key_file")
    rotate = sub.add_parser("admin-rotate-key", help="offline administrator rotates an existing upload key")
    rotate.add_argument("database")
    rotate.add_argument("user_id")
    rotate.add_argument("key_file")
    sync = sub.add_parser("sync", help="configure/run independent automatic upload")
    sync.add_argument("action", choices=("configure", "once", "run", "start", "status", "reset-epoch", "verify"))
    sync.add_argument("--data-dir", required=True)
    sync.add_argument("--database")
    sync.add_argument("--endpoint")
    sync.add_argument("--key-file")
    sync.add_argument("--interval", type=float, default=30)
    from .scheduling.cli import register
    register(sub)
    args = parser.parse_args(argv)
    try:
        if args.command == 'index-results':
            from .reporting.index import rebuild
            result = rebuild(args.database)
        elif args.command == "serve":
            from .central.server import serve
            return serve(args.database, args.port)
        elif args.command == "server-backup":
            from .central.backup import backup
            result = backup(args.database, args.destination)
        elif args.command == "admin-user":
            from .central.repository import create_user
            result = create_user(args.database, args.display_name, args.key_file)
        elif args.command == "sync":
            from .sync.client import configure, once, run, start, status, reset, verify_identity
            home = data_home(args.data_dir)
            if args.action == "configure":
                if not args.endpoint or not args.key_file:
                    raise ValueError("configure requires endpoint and administrator key file")
                result = configure(home, args.endpoint, args.key_file, args.interval)
            elif args.action == "verify":
                result = verify_identity(home)
            elif args.action == "status":
                result = status(home, args.database or home / "results/results.sqlite")
            else:
                database = args.database or home / "results/results.sqlite"
                result = {"once": once, "run": run, "start": start, "reset-epoch": reset}[args.action](home, database)
        elif args.command in {"run", "status", "stop", "resume", "upgrade-queue"}:
            from .scheduling.cli import dispatch
            result = dispatch(args)
        elif args.command == "admin-rotate-key":
            from .central.repository import rotate_key
            result = rotate_key(args.database, args.user_id, args.key_file)
        elif args.command == "setup":
            result = setup(args.data_dir, display_name=args.display_name, user_id=args.user_id,
                           key_file=args.key_file, cpu_workers=args.cpu_workers,
                           compile_threads=args.compile_threads, memory_budget_bytes=args.memory_budget_bytes,
                           cache_budget_bytes=args.cache_budget_bytes)
            if args.sync_endpoint:
                from .sync.client import configure, start, verify_identity
                config = load_config(args.data_dir)
                key_file = (config or {}).get("identity", {}).get("key_file")
                if not key_file:
                    raise ValueError("automatic sync requires administrator key-file")
                home = data_home(args.data_dir)
                result["sync"] = configure(home, args.sync_endpoint, key_file, args.sync_interval)
                try:
                    result["server_identity"] = verify_identity(home)
                except (OSError, ValueError):
                    result["server_identity"] = {"state": "unverified_offline_or_rejected"}
                result["sync_daemon"] = start(home, home / "results/results.sqlite")
        elif args.command == "seed":
            from .generation import materialize, content_manifest, workload_id
            import time
            generation_start=time.perf_counter_ns()
            bundle = materialize(args.token)
            generation_ns=time.perf_counter_ns()-generation_start
            result = {"seed": bundle.spec.seed.short, "seed_numeric": str(bundle.spec.seed), "workload_id": workload_id(bundle),
                      "preview": True, "shape_count": len(bundle.spec.shape_samples)}
            if args.show_spec:
                result["manifest"] = content_manifest(bundle)
        elif args.command == "compare":
            from .storage.comparison import compare
            result = compare(args.database, args.left, args.right)
        elif args.command == "snapshot":
            from .storage import ResultRepository
            result = {"snapshot": ResultRepository(args.source).snapshot(args.destination)}
        elif args.command in {"restore", "import", "check-db"}:
            from .storage.transfer import restore, merge, check_database
            if args.command == "check-db":
                result = check_database(args.database)
            else:
                result = (restore if args.command == "restore" else merge)(args.source, args.destination)
        elif args.command == "view":
            from .view.server import serve
            return serve(args.database, port=args.port, open_browser=not args.no_browser)
        elif args.command == "results":
            from .storage import ResultRepository
            repo = ResultRepository(args.database)
            result = repo.detail(args.attempt) if args.attempt else repo.list(limit=args.limit, after=args.after, seed=args.seed)
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
    except KeyboardInterrupt:
        print("Interrupted; active managed worker cleanup attempted. Resume interrupted tasks explicitly.", file=sys.stderr)
        return 130
    except (ValueError, OSError, sqlite3.Error) as exc:
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
    parser.add_argument("--backend", choices=BACKEND_NAMES, default="cpu")
    parser.add_argument("--device")
    parser.add_argument("--database", help="result SQLite path; defaults to data home/results/results.sqlite")
    parser.add_argument("--no-save", action="store_true", help="explicit ephemeral execution")
    parser.add_argument("--compile-threads", type=int, default=1)
    parser.add_argument("--performance", action="store_true", help="optional CPU wall-time measurement; NPU measurement needs D4 leases")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=10)
    return parser


def execute_args(args):
    try:
        from .contracts import ExecutionConfig, EvaluationConfig
        from .generation import materialize
        from .execution import execute
        from .backends import create_backend
        config = ExecutionConfig(args.backend, args.device or ("npu:0" if args.backend in {"ascendc", "triton", "npu-eager", "triton-npu", "af-legacy", "af-mlir"} else "cpu"), args.compile_threads, static_only=args.static_only)
        evaluation = EvaluationConfig(performance=args.performance, warmup_steps=args.warmup_steps, measured_steps=args.measured_steps)
        import time
        generation_start = time.perf_counter_ns()
        bundle = materialize(args.seed)
        generation_ns = time.perf_counter_ns() - generation_start
        result = execute(bundle, create_backend(config), evaluation)
        telemetry=json.loads(result['execution']['telemetry'])
        telemetry['stages']['generation']=generation_ns
        result['execution']['telemetry']=json.dumps(telemetry)
        if not args.no_save:
            from .storage import ResultRepository
            database = args.database or data_home() / "results/results.sqlite"
            from .environment import collect
            environment=collect()
            repo = ResultRepository(database)
            result["attempt_id"] = repo.save(result, environment=environment)
            result["database"] = str(repo.path)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "ok" else 1
    except (ValueError, TypeError, OSError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
