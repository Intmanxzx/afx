"""afx platform: serve the platform API and manage offline-issued users and tokens."""
from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afx platform", description="AutofuseX platform server")
    parser.add_argument("--data", required=True, help="data directory holding core/content/stats databases")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP API (loopback; put a TLS proxy in front)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8100)

    user = sub.add_parser("user", help="manage users").add_subparsers(dest="action", required=True)
    add = user.add_parser("add")
    add.add_argument("user_id")
    add.add_argument("--role", choices=("admin", "developer", "viewer"), default="developer")
    add.add_argument("--name")
    for action in ("disable", "enable"):
        user.add_parser(action).add_argument("user_id")

    token = sub.add_parser("token", help="issue, list and revoke tokens").add_subparsers(dest="action", required=True)
    issue = token.add_parser("issue")
    issue.add_argument("user_id")
    issue.add_argument("--kind", choices=("client", "agent", "web"), required=True)
    issue.add_argument("--scopes", help="comma separated; default depends on kind")
    issue.add_argument("--expires-days", type=int)
    token.add_parser("list").add_argument("--user")
    token.add_parser("revoke").add_argument("prefix", help="token prefix shown by 'token list'")

    sub.add_parser("rebuild-stats", help="recompute stats.db from core.db")

    backup = sub.add_parser("backup", help="consistent online backup to one .tar archive")
    backup.add_argument("--out", required=True, help="directory for the archive")
    backup.add_argument("--core-only", action="store_true", help="skip content.db (logs, case specs)")
    backup.add_argument("--tmp", help="scratch directory for snapshots (default: system temp)")
    verify = sub.add_parser("verify-backup", help="check archive hashes and SQLite integrity")
    verify.add_argument("archive")
    verify.add_argument("--tmp")
    restore = sub.add_parser("restore", help="restore an archive into the empty --data directory")
    restore.add_argument("archive")
    restore.add_argument("--content-from", help="full archive supplying content.db for a core-only archive")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("backup", "verify-backup", "restore"):
        # Never open (and so never create or migrate) the data directory here.
        from . import backup as backups
        try:
            if args.command == "backup":
                path = backups.create(args.data, args.out, include_content=not args.core_only, tmp_dir=args.tmp)
                result = {"archive": str(path), "bytes": path.stat().st_size}
            elif args.command == "verify-backup":
                result = backups.verify(args.archive, tmp_dir=args.tmp)
            else:
                result = backups.restore(args.archive, args.data, args.content_from)
        except (ValueError, OSError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
            return 2
        print(json.dumps({"ok": True, **result}))
        return 0
    from . import auth
    from .store import Store
    store = Store(args.data)
    try:
        if args.command == "serve":
            from .server import create_server
            server = create_server(args.data, args.host, args.port)
            print(f"afx platform listening on http://{args.host}:{args.port}", file=sys.stderr)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
            return 0
        if args.command == "user":
            if args.action == "add":
                auth.add_user(store, args.user_id, args.role, args.name)
            else:
                auth.set_user_status(store, args.user_id, "disabled" if args.action == "disable" else "active")
            print(json.dumps({"ok": True}))
            return 0
        if args.command == "token":
            if args.action == "issue":
                scopes = args.scopes.split(",") if args.scopes else None
                token = auth.issue_token(store, args.user_id, args.kind, scopes, args.expires_days)
                print(json.dumps({"ok": True, "token": token,
                                  "note": "shown once; store it in the client or agent config"}))
            elif args.action == "list":
                print(json.dumps({"ok": True, "tokens": auth.list_tokens(store, args.user)}, indent=2))
            else:
                print(json.dumps({"ok": True, "revoked": auth.revoke_token(store, args.prefix)}))
            return 0
        if args.command == "rebuild-stats":
            from . import aggregates
            with store.write() as conn:
                aggregates.rebuild(conn, store.setting(conn, "perf.tie_threshold"))
            print(json.dumps({"ok": True}))
            return 0
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    return 2
