"""Small user-facing command facade over the managed daemon.

The legacy management tree remains available for compatibility; this module
owns the intentionally small ``afx init/start/status/stop/replay`` surface.
"""
from __future__ import annotations

import argparse
import json

from .config import canonical, validate
from ...client.config import data_home


def _home(args):
    return data_home(getattr(args, "data_dir", None))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    from .artifacts import atomic
    atomic(path, value)


def _default_task():
    return validate({
        "schema_version": 1, "kind": "fuzz_task",
        "name": "local-default", "environment": "local",
        "workload": {"mode": "generate"},
    })


def _legacy(argv):
    from .cli import parser, _run
    return _run(parser().parse_args(argv))


def parser():
    root = argparse.ArgumentParser(prog="afx", description="AutofuseX client")
    root.add_argument("--data-dir", dest="data_dir")
    sub = root.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="initialize local or managed client")
    init.add_argument("--endpoint", default="https://autofuse.online")
    init.add_argument("--token", default=None)
    init.add_argument("--local", action="store_true")
    for name in ("start", "status", "stop"):
        p = sub.add_parser(name)

    cfg = sub.add_parser("config")
    cfg.add_argument("action", choices=("show", "edit", "reset"), nargs="?", default="show")
    cfg.add_argument("file", nargs="?")
    replay = sub.add_parser("replay")
    replay.add_argument("seeds", nargs="+")
    env = sub.add_parser("env")
    env.add_argument("action", choices=("list", "status", "prepare", "apply"))
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
        if args.token and not args.endpoint:
            raise ValueError("--token requires --endpoint")
        client = {"version": 1, "mode": "managed" if managed else "local"}
        if managed:
            client.update(endpoint=args.endpoint, token=args.token)
        elif args.endpoint:
            client["endpoint"] = args.endpoint
        _write(home / "client.json", client)
        if not (home / "task.json").exists():
            _write(home / "task.json", _default_task())
        return {"mode": client["mode"], "home": str(home), "task": str(home / "task.json")}
    if args.command == "config":
        client_path=home/'client.json'
        client=json.loads(client_path.read_text()) if client_path.exists() else {}
        if client.get('mode')=='managed':
            from autofusex.platform.client import API
            api=API(client['endpoint'],client['token'])
            if args.action=='show':return api.call('GET','/api/config/me')
            from .config import load
            return api.call('POST','/api/config/me',{'config':load(args.file) if args.action=='edit' else None})
        task = home / "task.json"
        if args.action == "reset":
            _write(task, _default_task())
        elif args.action == "edit":
            if not args.file:
                raise ValueError("config edit requires a JSON/YAML file path")
            from .config import load
            _write(task, load(args.file))
        elif not task.exists():
            _write(task, _default_task())
        return json.loads(task.read_text(encoding="utf-8"))
    if args.command == "start":
        if not (home / "client.json").exists():
            raise ValueError("run afx init first")
        result = _legacy(["daemon", "start", "--data-dir", str(home)])
        client_path = home / "client.json"
        client = json.loads(client_path.read_text())
        from ..runtime import request
        if client['mode'] == 'managed':
            from autofusex.platform.client import API
            from .artifacts import atomic
            effective = API(client['endpoint'], client['token']).call('GET','/api/config/me')
            atomic(home/'task.json', effective['config'])
        return request(home, {"action": "start_session"})
    if args.command == "status":
        return _legacy(["daemon", "status", "--data-dir", str(home)])
    if args.command == "stop":
        from ..runtime import request
        return request(home, {"action": "stop_session"})
    if args.command == "replay":
        _legacy(["daemon", "start", "--data-dir", str(home)])
        from ..runtime import request
        return request(home, {"action": "replay", "seeds": args.seeds})
    if args.command == "env":
        mapping = {"list": ["environment", "list"], "status": ["environment", "status", args.value or "local"],
                   "prepare": ["environment", "plan", args.value or "-"], "apply": ["environment", "apply", args.value or ""]}
        return _legacy([*mapping[args.action], "--data-dir", str(home)])
    raise ValueError("unsupported command")


def main(argv=None):
    try:
        result = run(argv)
        print(canonical({"schema_version": 1, "ok": True, "data": result}))
        return 0
    except Exception as exc:
        print(canonical({"schema_version": 1, "ok": False, "error": {"code": type(exc).__name__, "message": str(exc)[:400]}}))
        return 2
