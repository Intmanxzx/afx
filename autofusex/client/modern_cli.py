"""Small user-facing command facade over the managed daemon.

The legacy management tree remains available for compatibility; this module
owns the intentionally small ``afx init/start/status/stop/replay`` surface.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from autofusex.contracts.configuration import canonical
from autofusex.client.configuration import validate
from autofusex.client.paths import data_home


def _home(args):
    return data_home(getattr(args, "data_dir", None))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    from autofusex.client.artifacts import atomic
    atomic(path, value)


def _default_task():
    return validate({
        "schema_version": 1, "kind": "fuzz_task",
        "name": "local-default", "environment": "local",
        "workload": {"mode": "generate"},
    })


def _autofuse_path():
    try:
        import importlib.util
        spec = importlib.util.find_spec("autofuse")
        return str(Path(spec.origin).resolve()) if spec and spec.origin else None
    except (ImportError, ValueError, OSError):
        return None


def _legacy(argv):
    from autofusex.client.cli import parser, _run
    return _run(parser().parse_args(argv))


def parser():
    from importlib.metadata import PackageNotFoundError, version
    try:
        release = version("autofusex")
    except PackageNotFoundError:
        release = "unknown"
    root = argparse.ArgumentParser(prog="afx", description="AutofuseX client")
    root.add_argument("-v", "--version", action="version", version=f"%(prog)s {release}")
    root.add_argument("--data-dir", dest="data_dir")
    sub = root.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="initialize local or managed client")
    init.add_argument("--endpoint", default="https://autofuse.online")
    init.add_argument("--token", default=None)
    init.add_argument("--local", action="store_true")
    init.add_argument("--device", action="append", default=None, help="NPU device (repeatable, e.g. npu:0)")
    for name in ("start", "status", "stop", "restart"):
        sub.add_parser(name, help={"start": "start the configured client", "status": "show client status",
                                   "stop": "stop the client", "restart": "restart after environment/package changes"}[name])

    cfg = sub.add_parser("config")
    cfg.add_argument("action", choices=("show", "edit", "reset", "validate", "export"), nargs="?", default="show")
    cfg.add_argument("file", nargs="?")
    replay = sub.add_parser("replay")
    replay.add_argument("seeds", nargs="+")
    env = sub.add_parser("env")
    env.add_argument("action", choices=("list", "status", "inspect", "prepare", "apply"))
    env.add_argument("value", nargs="?")
    return root


def run(argv=None):
    args = parser().parse_args(argv)
    home = _home(args)
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.command == "init":
        if (home / "client.json").exists():
            raise ValueError("client already initialized; use its existing configuration")
        managed = bool(args.token) and not args.local
        # Preserve existing identities; each newly initialized directory is distinct.
        if not (home / "config.json").exists():
            import uuid
            _write(home / "config.json", dict(version=1, source_id=str(uuid.uuid4()),
                                               cpu_workers=1, resource_policy="auto"))
        if args.token and not args.endpoint:
            raise ValueError("--token requires --endpoint")
        # These are implementation details. Keep stable defaults in the
        # persisted configuration so the normal CLI never asks users for
        # database paths or polling knobs.
        results = (home / "results" / "results.sqlite").expanduser()
        from autofusex.client.paths import sync_directory
        state = sync_directory(home) / "platform-sync.sqlite"
        client = {"version": 1, "mode": "managed" if managed else "local"}
        if args.device:
            if any(d != "all" and not __import__('re').fullmatch(r"npu:[0-9]+", d) for d in args.device):
                raise ValueError("--device must be npu:<number>")
            client["devices"] = list(dict.fromkeys(args.device))
        if managed:
            client.update(endpoint=args.endpoint, token=args.token)
        elif args.endpoint:
            client["endpoint"] = args.endpoint
        client["platform"] = {"results": str(results), "state": str(state),
                                  "retest": True, "interval": 30.0}
        _write(home / "client.json", client)
        if not (home / "task.json").exists():
            task = _default_task()
            if not managed:
                task['results']['upload'] = False
            _write(home / "task.json", task)
        if client.get("devices"):
            task = json.loads((home / "task.json").read_text())
            task.setdefault("execution", {})["devices"] = client["devices"]
            _write(home / "task.json", task)
        return {"mode": client["mode"], "home": str(home), "task": str(home / "task.json"),
                **({"devices": client["devices"]} if "devices" in client else {})}
    if args.command == "config":
        client_path=home/'client.json'
        client=json.loads(client_path.read_text()) if client_path.exists() else {}
        from autofusex.contracts.configuration import parse, MAX_BYTES
        from autofusex.client.configuration import load
        spec = None
        if args.action in {'edit', 'validate'}:
            if not args.file:
                raise ValueError('config edit/validate requires a JSON/YAML file or -')
            if args.file == '-':
                import sys
                spec = validate(parse(sys.stdin.read(MAX_BYTES + 1)))
            else:
                spec = load(args.file)
            if spec['kind'] != 'fuzz_task':
                raise ValueError('config requires a task configuration')
            if client.get('mode') == 'managed':
                if spec['workload']['mode'] != 'generate':
                    raise ValueError('managed task profile must describe generation')
            elif spec['results']['upload']:
                raise ValueError('local mode cannot upload results')
            if args.action == 'validate':
                return spec
        elif args.file:
            raise ValueError('show/reset/export do not accept a file')
        if client.get('mode')=='managed':
            from autofusex.client.platform_transport import API
            api=API(client['endpoint'],client['token'])
            if args.action in {'show', 'export'}:
                result = api.call('GET', '/api/config/me')
            else:
                result = api.call('POST', '/api/config/me', {'config': spec})
            return result['config'] if args.action == 'export' else result
        task = home / "task.json"
        if args.action == "reset":
            spec = _default_task()
            spec['results']['upload'] = False
            _write(task, spec)
        elif args.action == "edit":
            if not args.file:
                raise ValueError("config edit requires a JSON/YAML file path")
            from autofusex.client.configuration import load
            _write(task, spec)
        elif not task.exists():
            spec = _default_task()
            spec['results']['upload'] = False
            return spec if args.action == 'export' else {'config': spec, 'source': 'local'}
        spec = load(task)
        return spec if args.action == 'export' else {'config': spec, 'source': 'local'}
    if args.command == "start":
        if not (home / "client.json").exists():
            raise ValueError("run afx init first")
        result = _legacy(["daemon", "start", "--data-dir", str(home)])
        client_path = home / "client.json"
        client = json.loads(client_path.read_text())
        from autofusex.client.daemon import request
        if client['mode'] == 'managed':
            from autofusex.client.platform_transport import API
            from autofusex.client.artifacts import atomic
            effective = API(client['endpoint'], client['token']).call('GET','/api/config/me')
            config = effective['config']
            if client.get('devices'):
                config.setdefault('execution', {})['devices'] = client['devices']
            atomic(home/'task.json', config)
        return request(home, {"action": "start_session"})
    if args.command == "status":
        result = _legacy(["daemon", "status", "--data-dir", str(home)])
        client = json.loads((home / "client.json").read_text()) if (home / "client.json").exists() else {}
        from autofusex.execution.environment import collect
        env = collect(home)
        result["environment"] = {"cann_version": env.get("cann"),
                                     "pyautofuse_sha256": env.get("autofuse_binary"),
                                     "autofuse_path": _autofuse_path()}
        result["platform_client"] = client.get("platform", {})
        marker = home / "platform-client.json"
        if marker.exists():
            try: result["platform_client"].update(json.loads(marker.read_text()))
            except (OSError, ValueError): pass
        return result
    if args.command == "stop":
        from autofusex.client.daemon import request
        return request(home, {"action": "stop_session"})
    if args.command == "restart":
        if not (home / "client.json").exists():
            raise ValueError("run afx init first")
        from autofusex.client.daemon import request
        request(home, {"action": "stop_session"})
        # Stop the supervisor as well so a restarted execution process cannot
        # retain modules loaded from the previous CANN installation.
        _legacy(["daemon", "stop", "--data-dir", str(home)])
        return run(["--data-dir", str(home), "start"])
    if args.command == "replay":
        _legacy(["daemon", "start", "--data-dir", str(home)])
        from autofusex.client.daemon import request
        return request(home, {"action": "replay", "seeds": args.seeds})
    if args.command == "env":
        mapping = {"inspect": ["environment", "inspect"], "list": ["environment", "list"], "status": ["environment", "status", args.value or "local"],
                   "prepare": ["environment", "plan", args.value or "-"], "apply": ["environment", "apply", args.value or ""]}
        return _legacy([*mapping[args.action], "--data-dir", str(home)])
    raise ValueError("unsupported command")


def main(argv=None):
    try:
        result = run(argv)
        args = parser().parse_args(argv)
        if args.command == 'config' and args.action == 'export':
            print(canonical(result))
        else:
            print(canonical({"schema_version": 1, "ok": True, "data": result}))
        return 0
    except Exception as exc:
        print(canonical({"schema_version": 1, "ok": False, "error": {"code": type(exc).__name__, "message": str(exc)[:400]}}))
        return 2
