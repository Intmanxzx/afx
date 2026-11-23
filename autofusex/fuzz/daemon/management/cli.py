"""Stable, scriptable management CLI.

Every successful command writes one JSON envelope to stdout.  The command tree
is intentionally explicit so an Agent can discover it with ``--help`` without
parsing prose.  Model dependencies are imported only by task workers.
"""
from __future__ import annotations

import argparse
import copy
import math
import sqlite3
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from autofusex.contracts.configuration import canonical, parse
from autofusex.workloads.fuzz.configuration import validate
from ..runtime import Manager, request
from .store import TERMINAL

RUN_TERMINAL = {"completed", "failed", "stopped", "cancelled", "interrupted", "needs_attention"}


class AgentArgumentParser(argparse.ArgumentParser):
    """Keep syntax errors machine-readable; --help remains normal argparse text."""
    def error(self, message):
        print(canonical(dict(schema_version=1, ok=False,
                             error=dict(code="usage_error", message=message))), file=sys.stdout)
        raise SystemExit(EXIT_USAGE)


EXIT_OK = 0
EXIT_OPERATION_FAILED = 1
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3


def _common(parent: argparse.ArgumentParser) -> None:
    parent.add_argument("--data-dir", metavar="DIR", help="client data directory")
    parent.add_argument("--json", action="store_true", help="emit the stable JSON envelope (default)")
    parent.add_argument("--timeout", type=float, default=300.0, help="wait timeout in seconds (default: 300)")
    parent.add_argument("--request-key", help="idempotency key; reuse only with the same payload")


def _add_file(parent, name="FILE"):
    parent.add_argument(name.lower(), metavar=name, help="YAML or JSON specification file; - reads stdin")


def _build_parser() -> argparse.ArgumentParser:
    root = AgentArgumentParser(
        prog="afx",
        description="AutofuseX managed client CLI. All output is JSON for Agent automation.",
        epilog="Exit codes: 0 success/accepted, 1 terminal operation failure, 2 usage or validation error, 3 daemon unavailable.",
    )
    root.add_argument("--version", action="version", version="afx management protocol 1")
    groups = root.add_subparsers(dest="group", required=True, metavar="COMMAND", parser_class=AgentArgumentParser)

    def group(name, help_text):
        p = groups.add_parser(name, help=help_text, description=help_text)
        p.set_defaults(_group_parser=p)
        return p.add_subparsers(dest="action", required=True, metavar="ACTION", parser_class=AgentArgumentParser)

    def action(actions, name, help_text, *, options=()):
        p = actions.add_parser(name, help=help_text, description=help_text)
        _common(p)
        for option in options:
            option(p)
        return p

    daemon = group("daemon", "Start, inspect, or stop the local management daemon.")
    action(daemon, "start", "Start the supervised daemon and return when its local socket is ready.", options=[
        lambda p: p.add_argument("--foreground", action="store_true", help="run in this process")])
    action(daemon, "run", "Run the manager process in the foreground; used by the launcher.", options=[
        lambda p: p.add_argument("--foreground", action="store_true", help=argparse.SUPPRESS)])
    action(daemon, "supervise", "Run the stable launcher; it performs health checks and rollback.")
    action(daemon, "status", "Show local daemon, connection, operation, and task status.")
    action(daemon, "stop", "Stop the manager only when no operation or task is active.")

    device = group("device", "Enroll, pair, inspect, or revoke a managed device.")
    action(device, "enroll", "Enroll with an administrator token read from stdin; the token is never an argument.", options=[
        lambda p: p.add_argument("--token-stdin", action="store_true", required=True),
        lambda p: p.add_argument("--endpoint", default="https://autofuse.online"),
        lambda p: p.add_argument("--name", default=__import__("socket").gethostname()[:80])])
    action(device, "pair", "Start browser pairing for this local device.", options=[
        lambda p: p.add_argument("--endpoint", default="https://autofuse.online"),
        lambda p: p.add_argument("--name", default=__import__("socket").gethostname()[:80])])
    action(device, "status", "Show this device's local pairing and capability status.")
    action(device, "revoke", "Revoke this device on the service; requires a paired device.")

    environment = group("environment", "Inspect and prepare isolated CANN/Python execution environments.")
    action(environment, "inspect", "Detect Python, architecture, and locally installed managed environments.")
    action(environment, "list", "List locally managed environments.")
    action(environment, "plan", "Validate an environment YAML and create an immutable install plan.", options=[lambda p: _add_file(p)])
    action(environment, "apply", "Execute a previously reviewed install plan.", options=[lambda p: p.add_argument("plan_id", metavar="PLAN_ID", nargs="?"), lambda p: p.add_argument("--plan", dest="plan_option", metavar="PLAN_ID")])
    action(environment, "status", "Show a managed environment by ID or name.", options=[lambda p: p.add_argument("environment_id", metavar="ENVIRONMENT_ID")])

    task = group("task", "Validate, submit, inspect, and control managed fuzz tasks.")
    action(task, "validate", "Validate task YAML, resolve its environment, and return the effective snapshot.", options=[lambda p: _add_file(p), lambda p: p.add_argument("--wait", action="store_true")])
    action(task, "submit", "Submit a task; returns an operation ID and never waits by default.", options=[
        lambda p: _add_file(p), lambda p: p.add_argument("--workers", type=int),
        lambda p: p.add_argument("--wait", action="store_true", help="wait for terminal operation state")])
    action(task, "wait", "Wait for task completion, interruption, or attention; does not cancel on timeout.", options=[
        lambda p: p.add_argument("run_id", metavar="RUN_ID")])
    action(task, "results", "Read saved attempts for a task, without requiring the execution environment.", options=[
        lambda p: p.add_argument("run_id", metavar="RUN_ID"),
        lambda p: p.add_argument("--limit", type=int, default=50),
        lambda p: p.add_argument("--after", type=int, default=0),
        lambda p: p.add_argument("--status", choices=("ok", "fail", "error", "unsupported")),
        lambda p: p.add_argument("--seed"),
        lambda p: p.add_argument("--attempt", help="read one full attempt instead of a page")])
    action(task, "replay", "Replay seeds using a prior run's environment and execution/evaluation settings.", options=[
        lambda p: p.add_argument("seeds", nargs="+", metavar="SEED"),
        lambda p: p.add_argument("--from-run", required=True, metavar="RUN_ID"),
        lambda p: p.add_argument("--wait", action="store_true", help="wait for submission, not execution")])
    action(task, "list", "List locally submitted task runs.")
    action(task, "inspect", "Show one task run and its effective immutable snapshot.", options=[lambda p: p.add_argument("run_id", metavar="RUN_ID")])
    action(task, "status", "Alias for task inspect.", options=[lambda p: p.add_argument("run_id", metavar="RUN_ID")])
    for name, help_text in (("pause", "Pause generation and new dispatches at a safe boundary."),
                            ("resume", "Resume a paused task."),
                            ("stop", "Gracefully stop a task; use force-stop for owned process termination."),
                            ("force-stop", "Force-stop only processes owned by this managed task.")):
        action(task, name, help_text, options=[lambda p, n=name: p.add_argument("run_id", metavar="RUN_ID"),
                                               lambda p: p.add_argument("--wait", action="store_true"), lambda p: p.add_argument("--graceful", action="store_true", help="explicit alias; graceful is the default")])
    action(task, "reconfigure", "Apply a new YAML to a continuous task at the next safe epoch.", options=[
        lambda p: p.add_argument("run_id", metavar="RUN_ID"), lambda p: _add_file(p),
        lambda p: p.add_argument("--wait", action="store_true")])

    operation = group("operation", "Inspect, wait for, cancel, or read logs from a long operation.")
    action(operation, "inspect", "Return the current operation envelope.", options=[lambda p: p.add_argument("operation_id", metavar="OPERATION_ID")])
    action(operation, "wait", "Poll until terminal or --timeout expires.", options=[lambda p: p.add_argument("operation_id", metavar="OPERATION_ID")])
    action(operation, "logs", "Return bounded structured operation events.", options=[lambda p: p.add_argument("operation_id", metavar="OPERATION_ID"), lambda p: p.add_argument("--after", type=int, default=0)])
    action(operation, "cancel", "Cancel a queued operation or request cancellation of a running one.", options=[lambda p: p.add_argument("operation_id", metavar="OPERATION_ID")])

    build = group("build", "Build a locally authorized CANN RUN package.")
    action(build, "submit", "Submit a build specification; returns an operation ID.", options=[lambda p: _add_file(p), lambda p: p.add_argument("--wait", action="store_true")])
    action(build, "inspect", "Inspect a build operation by operation ID.", options=[lambda p: p.add_argument("operation_id", metavar="OPERATION_ID")])

    tool = group("tool", "Stage, inspect, and activate signed AutofuseX releases.")
    action(tool, "stage", "Stage a signed release manifest without activating it.", options=[lambda p: _add_file(p), lambda p: p.add_argument("--wait", action="store_true")])
    action(tool, "activate", "Activate a staged release through the stable launcher.", options=[lambda p: p.add_argument("release_id", metavar="RELEASE_ID"), lambda p: p.add_argument("--wait", action="store_true")])
    action(tool, "list", "List staged tool releases.")
    action(tool, "inspect", "Inspect one staged release.", options=[lambda p: p.add_argument("release_id", metavar="RELEASE_ID")])

    client = group("client", "Install and inspect the versioned client execution plane.")
    action(client, "update", "Install an exact autofusex version from PyPI through the daemon.", options=[
        lambda p: p.add_argument("version", metavar="VERSION"),
        lambda p: p.add_argument("--index-url"),
        lambda p: p.add_argument("--wait", action="store_true")])
    action(client, "status", "Show the active client release.")

    return root


# Public for tests and shell completion generators.
def parser():
    return _build_parser()


def _home(args):
    from ...client.config import data_home
    return data_home(getattr(args, "data_dir", None))


def _read_file(filename):
    if not filename or filename == "-":
        text = sys.stdin.read(65537)
    else:
        with Path(filename).open(encoding="utf-8") as stream:
            text = stream.read(65537)
    return parse(text)


def _request_key(args, default=None):
    return args.request_key or default or str(uuid.uuid4())


def _submit(home, args, kind, payload, *, wait=False):
    result = request(home, dict(action="submit", kind=kind, payload=payload,
                                request_key=_request_key(args)))
    if wait:
        return _wait(home, result["id"], args.timeout)
    return result


def _wait(home, operation_id, timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("--timeout must be positive")
    started = time.monotonic()
    while True:
        result = request(home, dict(action="operation", id=operation_id))
        if result["state"] in TERMINAL:
            return result
        if time.monotonic() - started >= timeout:
            raise TimeoutError("operation wait timed out; operation remains active")
        time.sleep(.2)


def _wait_run(home, run_id, timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("--timeout must be positive and finite")
    started = time.monotonic()
    while True:
        result = _local_query(home, "run", run_id)
        if result["state"] in RUN_TERMINAL:
            return result
        if time.monotonic() - started >= timeout:
            raise TimeoutError("task wait timed out; task remains unchanged; inspect " + run_id)
        time.sleep(.2)


def _local_query(home, query, value=None):
    return request(home, dict(action="query", query=query, value=value))


def _task_payload(args):
    payload = validate(_read_file(args.file))
    if getattr(args, "workers", None) is not None:
        if payload.get("kind") != "fuzz_task":
            raise ValueError("--workers is valid only for fuzz_task")
        payload.setdefault("execution", {}).setdefault("resources", {}).update(policy="manual", workers=args.workers)
    return validate(payload)


def _run(args):
    home = _home(args)
    group, action = args.group, args.action
    if group == "daemon":
        if action == "supervise":
            from ..supervisor import run
            return run(home)
        if action in {"run", "start"} and action == "run":
            Manager(home).run(); return EXIT_OK
        if action == "start":
            try:
                result = request(home, dict(action="status"))
            except (OSError, ValueError):
                log = (home / "manager-start.log").open("ab")
                try:
                    child = subprocess.Popen([sys.executable, "-m", "autofusex", "daemon", "supervise", "--data-dir", str(home)],
                                             start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                finally:
                    log.close()
                for _ in range(100):
                    if child.poll() is not None:
                        raise RuntimeError("daemon exited during startup")
                    try:
                        result = request(home, dict(action="status")); break
                    except OSError: time.sleep(.05)
                else: raise TimeoutError("daemon startup timed out")
            return result
        if action == "status": return request(home, dict(action="status"))
        if action == "stop": return request(home, dict(action="shutdown"))

    if group == "device":
        if action == "enroll":
            token = sys.stdin.readline(258).strip()
            if not 20 <= len(token) <= 256: raise ValueError("invalid registration token")
            return request(home, dict(action="enroll", endpoint=args.endpoint, name=args.name, token=token))
        if action == "pair": return request(home, dict(action="pair", endpoint=args.endpoint, name=args.name))
        if action == "status": return request(home, dict(action="status"))
        if action == "revoke": return request(home, dict(action="revoke"))

    if group == "client":
        if action == "status":
            result = request(home, dict(action="status"))
            active = home / 'active-client.json'
            if active.exists():
                result['active_client'] = json.loads(active.read_text())
            return result
        if action == "update":
            payload = dict(version=args.version, package='autofusex')
            if args.index_url: payload['index_url'] = args.index_url
            return _submit(home, args, 'client.update', payload, wait=args.wait)

    if group == "environment":
        if action in {"inspect", "list"}: return _local_query(home, "environments")
        if action == "status": return _local_query(home, "environment", args.environment_id)
        if action == "plan": return _submit(home, args, "environment.plan", _read_file(args.file))
        if action == "apply":
            plan_id = args.plan_id or args.plan_option
            if not plan_id: raise ValueError("provide PLAN_ID or --plan PLAN_ID")
            return _submit(home, args, "environment.apply", {"plan_id": plan_id})

    if group == "task":
        if action == "validate": return _submit(home, args, "task.validate", _task_payload(args), wait=args.wait)
        if action == "submit": return _submit(home, args, "task.submit", _task_payload(args), wait=args.wait)
        if action == "wait": return _wait_run(home, args.run_id, args.timeout)
        if action == "results":
            from .results import read
            run = _local_query(home, "run", args.run_id)
            return read(home, run, limit=args.limit, after=args.after, status=args.status,
                        seed=args.seed, attempt=args.attempt)
        if action == "replay":
            original = _local_query(home, "run", args.from_run)
            payload = copy.deepcopy(original["spec"]["spec"])
            payload["workload"] = {"mode": "replay", "seeds": args.seeds}
            payload.pop("termination", None)
            # Stored auto plans contain resolved workers/threads; replay those exact budgets.
            resources = payload.get("execution", {}).get("resources", {})
            if any(key in resources for key in ("workers", "compile_threads")):
                resources["policy"] = "manual"
            return _submit(home, args, "task.submit", validate(payload), wait=args.wait)
        if action == "list": return _local_query(home, "runs")
        if action in {"inspect", "status"}: return _local_query(home, "run", args.run_id)
        if action in {"pause", "resume", "stop", "force-stop"}:
            return _submit(home, args, "task.action", {"run_id": args.run_id, "action": action}, wait=args.wait)
        if action == "reconfigure":
            return _submit(home, args, "task.reconfigure", {"run_id": args.run_id, "spec": _read_file(args.file)}, wait=args.wait)

    if group == "operation":
        if action == "inspect": return request(home, dict(action="operation", id=args.operation_id))
        if action == "wait": return _wait(home, args.operation_id, args.timeout)
        if action == "logs": return request(home, dict(action="events", id=args.operation_id, after=args.after))
        if action == "cancel": return _submit(home, args, "operation.cancel", {"operation_id": args.operation_id})

    if group == "build":
        if action == "submit": return _submit(home, args, "build.submit", _read_file(args.file), wait=args.wait)
        if action == "inspect": return request(home, dict(action="operation", id=args.operation_id))

    if group == "tool":
        if action == "stage": return _submit(home, args, "tool.stage", _read_file(args.file), wait=args.wait)
        if action == "activate": return _submit(home, args, "tool.activate", {"release_id": args.release_id}, wait=args.wait)
        if action in {"list", "inspect"}: return _local_query(home, "tools", getattr(args, "release_id", None))
    raise ValueError(f"unsupported command: {group} {action}")


def _exit_for_result(result):
    if isinstance(result, dict) and result.get("state") in {"failed", "needs_attention", "cancelled", "interrupted", "stopped"}:
        return EXIT_OPERATION_FAILED
    return EXIT_OK


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = _run(args)
        print(canonical(dict(schema_version=1, ok=True, data=result)))
        return _exit_for_result(result)
    except (ValueError, TypeError, OSError, LookupError, PermissionError, TimeoutError, RuntimeError, sqlite3.Error) as exc:
        code = EXIT_UNAVAILABLE if isinstance(exc, OSError) and not isinstance(exc, TimeoutError) else EXIT_USAGE
        print(canonical(dict(schema_version=1, ok=False, error={"code": type(exc).__name__, "message": str(exc)[:400]})))
        return code
