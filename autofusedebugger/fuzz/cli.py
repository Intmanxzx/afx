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
            "generation": "NNSmith C protocol; pinned source/Z3; cross-platform determinism pending", "sync": "independent_outbox_daemon"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "replay":
        return replay(argv[1:])
    from .help import ScenarioParser
    parser = ScenarioParser(prog="autofusedbg fuzz")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=argparse.ArgumentParser)
    sub.add_parser("replay", help="replay a self-contained short seed")
    maintenance = sub.add_parser('maintenance', help='diagnostics, sync and data maintenance').add_subparsers(dest='maintenance_command', required=True)
    administration = sub.add_parser('admin', help='offline identities and server operations').add_subparsers(dest='admin_command', required=True)
    for command in ("setup", "doctor"):
        p = (sub if command == "setup" else maintenance).add_parser(command)
        p.set_defaults(command=command)
        p.add_argument("--data-dir", help=argparse.SUPPRESS if command=="setup" else None)
        if command == "setup":
            p.add_argument('--token', required=True, help='administrator-issued secret; never printed')
            p.add_argument('--sync-interval', type=float, default=30)
    seed = maintenance.add_parser("seed", help="inspect a frozen short-seed case without execution")
    seed.add_argument("token")
    seed.add_argument("--show-spec", action="store_true")
    compare = maintenance.add_parser("diagnose", help="compare two opt-in generation traces")
    compare.add_argument("expected")
    compare.add_argument("actual")
    pair = maintenance.add_parser("compare", help="compare two result attempts with explicit comparability checks")
    pair.add_argument("database")
    pair.add_argument("left")
    pair.add_argument("right")
    snapshot = maintenance.add_parser("snapshot", help="export a consistent lightweight SQLite snapshot")
    snapshot.add_argument("source")
    snapshot.add_argument("destination")
    for command in ("restore", "import"):
        transfer = maintenance.add_parser(command, help="restore to a new database" if command == "restore" else "merge into an existing database atomically")
        transfer.set_defaults(command=command)
        transfer.add_argument("source")
        transfer.add_argument("destination")
    check = maintenance.add_parser("check-db", help="audit a trusted result database without modification")
    check.add_argument("database")
    results = maintenance.add_parser("results", help="read lightweight results without model dependencies")
    results.add_argument("database")
    results.add_argument("--limit", type=int, default=50)
    results.add_argument("--after", type=int)
    results.add_argument("--seed")
    results.add_argument("--attempt")
    view = sub.add_parser("view", help="open an offline read-only SQLite viewer on this computer")
    view.add_argument("database")
    view.add_argument("--port", type=int, default=0, help="loopback port; default chooses a free port")
    view.add_argument("--no-browser", action="store_true")
    central = administration.add_parser("serve", help="central service behind a TLS proxy")
    central.add_argument("database")
    central.add_argument("--port", type=int, default=8090)
    indexing = maintenance.add_parser('index-results', help='build additive SQLite reporting index after backup')
    indexing.add_argument('database')
    backup = administration.add_parser("backup", help="private complete central backup including identities and ACKs")
    backup.add_argument("database")
    backup.add_argument("destination")
    admin = administration.add_parser("user", help="offline administrator creates an upload identity")
    admin.add_argument("database")
    admin.add_argument("display_name")
    rotate = administration.add_parser("rotate-token", help="offline administrator rotates an existing upload key")
    rotate.add_argument("database")
    rotate.add_argument("user")
    sync = maintenance.add_parser("sync", help="configure/run independent automatic upload")
    sync.add_argument("action", choices=("configure", "once", "run", "start", "status", "reset-epoch", "verify"))
    sync.add_argument("--data-dir", help="override the remembered/default data directory")
    sync.add_argument("--database")
    sync.add_argument("--endpoint")
    sync.add_argument("--key-file")
    sync.add_argument("--interval", type=float, default=30)
    for child, command in [(seed,'seed'),(compare,'diagnose'),(pair,'compare'),(snapshot,'snapshot'),(check,'check-db'),(results,'results'),(indexing,'index-results'),(sync,'sync'),(central,'serve'),(backup,'server-backup'),(admin,'admin-user'),(rotate,'admin-rotate-key')]:
        child.set_defaults(command=command)
    from .scheduling.cli import register
    register(sub, maintenance)
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
            result = create_user(args.database, args.display_name)
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
            from .storage.queries import reading
            with reading(args.database) as conn:
                row=conn.execute('SELECT id FROM users WHERE display_name=?',(args.user,)).fetchone()
            if row is None:raise ValueError('unknown user')
            result = rotate_key(args.database, row[0])
        elif args.command == "setup":
            from .onboarding import initialize_user
            result = initialize_user(args.data_dir, args.token, args.sync_interval)
            from .config import remember_home
            remember_home(data_home(args.data_dir))
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
    if getattr(args,'scheduling','combined')=='staged':
        from .scheduling.staged import replay as staged_replay
        return staged_replay(args)
    if args.backend is None:
        # Normal replay includes both mandatory standards. Explicit --backend
        # remains useful for focused debugging of an already identified failure.
        import copy
        codes = []
        for backend in ('cpu', 'af-mlir'):
            selected = copy.copy(args)
            selected.backend = backend
            selected.device = 'cpu' if backend == 'cpu' else (args.device or 'npu:0')
            codes.append(execute_args(selected))
        return max(codes)
    return execute_args(args)


def replay_parser():
    parser = argparse.ArgumentParser(prog="autofusedbg fuzz replay")
    parser.add_argument("seed", help="self-contained short seed")
    parser.add_argument("--scheduling",choices=("combined","staged"),default="staged")
    parser.add_argument("--static-only", action="store_true", help="execute base sample only; does not change case or workload ID")
    parser.add_argument("--backend", choices=BACKEND_NAMES, help="focused backend diagnostic; default replays CPU and AF-MLIR")
    parser.add_argument("--device")
    parser.add_argument("--database", help=argparse.SUPPRESS)
    parser.add_argument("--save", action="store_true", help="save results and enqueue automatic upload; default: neither")
    parser.add_argument("--compile-threads", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--perf", dest="performance", action="store_true", help="request precise metrics where supported; unavailable metrics remain empty")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=10)
    return parser


def execute_args(args):
    try:
        from .contracts import ExecutionConfig, EvaluationConfig, Seed
        from .generation import materialize
        from .execution import execute
        from .backends import create_backend
        from .resources import adaptive_defaults
        threads = args.compile_threads or adaptive_defaults(data_home())["compile_threads"]
        config = ExecutionConfig(args.backend, args.device or ("npu:0" if args.backend in {"ascendc", "triton", "eager-npu", "triton-npu", "af-legacy", "af-mlir"} else "cpu"), threads, static_only=args.static_only or Seed.parse(args.seed).version==3, save_results=args.save)
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
