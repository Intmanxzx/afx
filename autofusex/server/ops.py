"""Platform operations: client heartbeats, hourly rollups, one-click client update (design 4.7, 6.7).

No per-event detail is kept: heartbeats carry counter deltas that are folded
into one row per client per hour and expired after ROLLUP_DAYS.
"""
from __future__ import annotations

import json
import re
import uuid
import urllib.request
import threading

from autofusex.server.auth import Principal
from autofusex.server.issues import IssueError, idempotent
from autofusex.server.store import Store, now_ms

ROLLUP_DAYS = 90
ONLINE_MS = 5 * 60_000
COUNTERS = ("executions", "uploads", "upload_errors", "retests")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,31}$")
_VERSION = re.compile(r"^[A-Za-z0-9_.+-]{1,64}$")
_PYPI_CACHE = {}
_PYPI_LOCK = threading.Lock()
_UPDATE_TIMEOUT_MS = 300_000


def latest_pypi_version(store):
    now = now_ms()
    with _PYPI_LOCK:
        cached = dict(_PYPI_CACHE)
    if cached.get('version') and now - cached.get('at', 0) < 300_000:
        return cached['version']
    try:
        with urllib.request.urlopen('https://pypi.org/pypi/autofusex/json', timeout=5) as response:
            value = json.loads(response.read()).get('info', {}).get('version')
        if isinstance(value, str) and _VERSION.fullmatch(value):
            with _PYPI_LOCK:
                _PYPI_CACHE.update(version=value, at=now)
            return value
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return cached.get('version')


def _reconcile_update_timeouts(store):
    now = now_ms()
    with store.write() as conn:
        rows = conn.execute("SELECT client_id, target_afx_version, update_status FROM clients WHERE target_afx_version IS NOT NULL").fetchall()
        for row in rows:
            try:
                status = json.loads(row['update_status'] or '{}')
            except (TypeError, ValueError):
                status = {}
            phase = status.get('phase')
            started = status.get('started_at')
            if phase not in {'stopping', 'upgrading'} or not isinstance(started, int) or now - started <= _UPDATE_TIMEOUT_MS:
                continue
            failed_phase = 'stop_timeout' if phase == 'stopping' else 'upgrade_timeout'
            conn.execute("UPDATE clients SET update_status=?, target_afx_version=NULL, update_request_id=NULL, target_daemon_state=NULL, daemon_request_id=NULL WHERE client_id=?",
                         (json.dumps({'state': 'failed', 'version': row['target_afx_version'], 'phase': failed_phase, 'failed_at': now}), row['client_id']))


def is_online(last_heartbeat_at, now):
    return last_heartbeat_at is not None and now - last_heartbeat_at <= ONLINE_MS


def heartbeat(store: Store, principal: Principal, body: dict) -> dict:
    principal.require_write()
    _reconcile_update_timeouts(store)
    machine_id = body.get("machine_id")
    if not isinstance(machine_id, str) or not 1 <= len(machine_id) <= 128:
        raise IssueError("invalid", "machine_id is required")
    for key in ("hostname", "npu_arch", "afx_version"):
        value = body.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > 128):
            raise IssueError("invalid", f"{key} must be a short string")
    devices = body.get("devices")
    if devices is not None and (not isinstance(devices, list) or len(devices) > 64 or
            any(not isinstance(d, str) or not re.fullmatch(r"npu:[0-9]+", d) for d in devices) or
            len(set(devices)) != len(devices)):
        raise IssueError("invalid", "devices must be distinct logical npu:index identifiers")
    versions = body.get("backend_versions") or {}
    if not isinstance(versions, dict) or len(versions) > 16 or not all(
            isinstance(k, str) and _NAME.match(k) and (v is None or (isinstance(v, str) and len(v) <= 128))
            for k, v in versions.items()):
        raise IssueError("invalid", "backend_versions must map backend -> version string")
    counters = body.get("counters") or {}
    if not isinstance(counters, dict) or set(counters) - set(COUNTERS) or not all(
            isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 10**7 for v in counters.values()):
        raise IssueError("invalid", f"counters must map {', '.join(COUNTERS)} to non-negative integers")
    runtime = body.get("runtime")
    if runtime is not None and (not isinstance(runtime, list) or len(runtime) > 32 or not all(
            isinstance(r, dict) for r in runtime) or len(json.dumps(runtime)) > 128 * 1024):
        raise IssueError("invalid", "runtime must be a list of at most 32 run snapshots (128 KiB)")
    windows = body.get("windows") or []
    if not isinstance(windows, list) or len(windows) > 20 or not all(
            isinstance(w, dict) and isinstance(w.get("window_id"), str) and 1 <= len(w["window_id"]) <= 64
            and isinstance(w.get("rows"), list) for w in windows) or len(json.dumps(windows)) > 512 * 1024:
        raise IssueError("invalid", "windows must list at most 20 {window_id, rows} (512 KiB)")
    pending = body.get("pending_uploads")
    if pending is not None and (isinstance(pending, bool) or not isinstance(pending, int) or pending < 0):
        raise IssueError("invalid", "pending_uploads must be a non-negative integer")
    from autofusex.server.ingest import upsert_client
    now = now_ms()
    with store.write() as conn:
        client_id = upsert_client(conn, principal.user_id, {
            "machine_id": machine_id, "client_home": body.get("client_home"), "hostname": body.get("hostname"), "npu_arch": body.get("npu_arch"),
            "afx_version": body.get("afx_version")})
        conn.execute("UPDATE clients SET backend_versions=COALESCE(?, backend_versions), last_heartbeat_at=?, "
                     "busy=?, pending_uploads=?, deleted_at=NULL WHERE client_id=?",
                     (json.dumps(versions, sort_keys=True) if versions else None, now, int(bool(body.get("busy"))),
                      pending, client_id))
        if "devices" in body:
            conn.execute("UPDATE clients SET devices=? WHERE client_id=?",
                         (json.dumps(devices) if devices is not None else None, client_id))
        if runtime is not None:
            conn.execute("UPDATE clients SET runtime=?, runtime_at=? WHERE client_id=?",
                         (json.dumps(runtime), now, client_id))
        accepted = []
        for window in windows:
            conn.execute("INSERT OR REPLACE INTO efficiency_windows VALUES(?,?,?,?)",
                         (client_id, window["window_id"], now, json.dumps(window["rows"])))
            accepted.append(window["window_id"])
        if accepted:
            # keep the most recent 100 windows per client
            conn.execute("DELETE FROM efficiency_windows WHERE client_id=? AND window_id NOT IN (SELECT window_id "
                         "FROM efficiency_windows WHERE client_id=? ORDER BY received_at DESC LIMIT 100)",
                         (client_id, client_id))
        if any(counters.values()):
            hour = now // 3_600_000
            values = [counters.get(name, 0) for name in COUNTERS]
            conn.execute(
                f"INSERT INTO metric_rollups(client_id, hour, {', '.join(COUNTERS)}) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(client_id, hour) DO UPDATE SET "
                + ", ".join(f"{name}={name}+excluded.{name}" for name in COUNTERS), (client_id, hour, *values))
            if now % 100 == 0:  # occasional cheap expiry
                conn.execute("DELETE FROM metric_rollups WHERE hour<?", (hour - ROLLUP_DAYS * 24,))
        if body.get('update_status') is not None:
            progress = body['update_status']
            if not isinstance(progress, dict) or progress.get('state') not in {'idle','queued','running','activating','succeeded','failed','needs_attention','cancelled'}:
                raise IssueError('invalid', 'invalid update status')
            safe = dict(state=progress['state'], version=str(progress.get('version',''))[:64])
            # The server owns stop/upgrade phases and deadlines. Client progress can
            # update the visible state, but must not move an orchestration backwards.
            existing = conn.execute('SELECT target_afx_version, update_status FROM clients WHERE client_id=?', (client_id,)).fetchone()
            current = json.loads(existing['update_status'] or '{}') if existing and existing['update_status'] else {}
            if existing and existing['target_afx_version'] and current.get('phase') in {'stopping', 'upgrading'}:
                if progress['state'] in {'succeeded', 'failed', 'needs_attention'}:
                    safe.update(phase=current.get('phase'), started_at=current.get('started_at'))
                else:
                    safe.update(phase=current.get('phase'), started_at=current.get('started_at'))
            else:
                phase = str(progress.get('phase',''))[:32]
                started = progress.get('started_at')
                if phase or started is not None:
                    safe.update(phase=phase, started_at=started)
            conn.execute('UPDATE clients SET update_status=? WHERE client_id=?', (json.dumps(safe), client_id))
        row = conn.execute("SELECT target_afx_version, afx_version, update_request_id, target_daemon_state, daemon_request_id, daemon_state FROM clients WHERE client_id=?",
                           (client_id,)).fetchone()
        target = row["target_afx_version"]
        progress_row = conn.execute("SELECT update_status FROM clients WHERE client_id=?", (client_id,)).fetchone()
        progress = json.loads(progress_row[0]) if progress_row and progress_row[0] else {}
        if (target is not None and target == row["afx_version"]
                and body.get("update_ack") == row["update_request_id"]):
            conn.execute("UPDATE clients SET target_afx_version=NULL, update_request_id=NULL WHERE client_id=?", (client_id,))
            conn.execute("INSERT INTO audit_log(at, actor, action, detail) VALUES(?,?,?,?)",
                         (now, principal.user_id, "client.updated", f"client {client_id} -> {target}"))
            target = None
        daemon_target = row["target_daemon_state"]
        daemon_request = row["daemon_request_id"]
        reported_state = body.get("daemon_state")
        if reported_state in {"running", "stopped"}:
            conn.execute("UPDATE clients SET daemon_state=?, daemon_state_at=? WHERE client_id=?",
                         (reported_state, now, client_id))
        if (reported_state in {"running", "stopped"} and ('running' if daemon_target == 'restart' else daemon_target) == reported_state
                and body.get("daemon_request_id") == daemon_request):
            conn.execute("UPDATE clients SET target_daemon_state=NULL, daemon_request_id=NULL WHERE client_id=?", (client_id,))
            conn.execute("INSERT INTO audit_log(at, actor, action, detail) VALUES(?,?,?,?)",
                         (now, principal.user_id, "client.daemon_state", f"client {client_id} -> {reported_state}"))
            daemon_target = None
            daemon_request = None
    # Release an update only after the daemon has acknowledged a stopped state.
    # The daemon keeps heartbeating while workers drain, so control visibility
    # is retained throughout the transition.
        if target and reported_state == "stopped" and body.get('daemon_request_id') == row['daemon_request_id'] and progress.get('phase') == 'stopping':
            conn.execute("UPDATE clients SET update_status=? WHERE client_id=?",
                         (json.dumps({'state': 'running', 'version': target, 'phase': 'upgrading', 'started_at': now}), client_id))
            progress = {'state': 'running', 'version': target, 'phase': 'upgrading', 'started_at': now}
        update_ready = target is not None and progress.get('phase') == 'upgrading'
    return {"client_id": client_id, "update_to": target if update_ready else None,
            "update_request_id": row["update_request_id"] if target else None,
            "daemon_state": daemon_target,
            "daemon_request_id": daemon_request if daemon_target else None,
            "server_time": now, "windows_accepted": accepted}


def request_update(store: Store, principal: Principal, body: dict):
    principal.require_admin()
    ids, version = body.get("client_ids"), body.get("version")
    if not isinstance(ids, list) or not 1 <= len(ids) <= 200 or not all(
            isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise IssueError("invalid", "client_ids must list 1..200 client ids")
    if "version" in body and version is None:
        def cancel(conn):
            count = 0
            for client_id in ids:
                count += conn.execute("UPDATE clients SET target_afx_version=NULL, update_request_id=NULL, update_status=?, target_daemon_state=NULL, daemon_request_id=NULL WHERE client_id=?",
                                     (json.dumps({'state': 'cancelled'}), client_id)).rowcount
            return {'requested': count, 'cancelled': True}
        return idempotent(store, principal, "client-update", body.pop("request_key", None), body, cancel)
    if version is None:
        version = latest_pypi_version(store)
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise IssueError("unavailable", "最新 autofusex 版本暂不可用", 503)

    def action(conn):
        count = 0
        request_id = str(uuid.uuid4())
        for client_id in ids:
            daemon_request = str(uuid.uuid4())
            count += conn.execute(
                "UPDATE clients SET target_afx_version=?, update_request_id=?, update_status=?, "
                "target_daemon_state='stopped', daemon_request_id=? WHERE client_id=? AND deleted_at IS NULL",
                (version, request_id, json.dumps({'state': 'queued', 'version': version, 'phase': 'stopping', 'started_at': now_ms()}), daemon_request, client_id)).rowcount
        conn.execute("INSERT INTO audit_log(at, actor, action, detail) VALUES(?,?,?,?)",
                     (now_ms(), principal.user_id, "client.update_requested", f"{ids} -> {version}"))
        return {"requested": count, "version": version}
    return idempotent(store, principal, "client-update", body.pop("request_key", None), body, action)


def request_daemon_state(store: Store, principal: Principal, body: dict):
    """Queue an idempotent daemon lifecycle command for an owned client or any client as admin."""
    client_id, state = body.get("client_id"), body.get("state")
    if not isinstance(client_id, int) or isinstance(client_id, bool) or state not in {"running", "stopped", "restart"}:
        raise IssueError("invalid", "client_id and state are required")
    def action(conn):
        row = conn.execute("SELECT user_id FROM clients WHERE client_id=? AND deleted_at IS NULL", (client_id,)).fetchone()
        if row is None: raise IssueError("not_found", "client not found", 404)
        if principal.role != "admin" and row["user_id"] != principal.user_id:
            raise IssueError("forbidden", "client is owned by another user", 403)
        request_id = str(uuid.uuid4())
        conn.execute("UPDATE clients SET target_daemon_state=?, daemon_request_id=? WHERE client_id=?", (state, request_id, client_id))
        conn.execute("INSERT INTO audit_log(at, actor, action, detail) VALUES(?,?,?,?)", (now_ms(), principal.user_id, "client.daemon_requested", f"client {client_id} -> {state}"))
        return {"client_id": client_id, "state": state, "request_id": request_id}
    return idempotent(store, principal, "client-daemon", body.pop("request_key", None), body, action)


def delete_client(store: Store, principal: Principal, client_id: int):
    principal.require_admin()
    with store.write() as conn:
        row = conn.execute("SELECT client_id, display_name, last_heartbeat_at FROM clients WHERE client_id=? AND deleted_at IS NULL", (client_id,)).fetchone()
        if row is None:
            raise IssueError("not_found", "client not found", 404)
        if is_online(row[2], now_ms()):
            raise IssueError("online", "online clients cannot be deleted", 409)
        now = now_ms()
        conn.execute("UPDATE clients SET deleted_at=?, runtime=NULL, runtime_at=NULL WHERE client_id=?", (now, client_id))
        conn.execute("INSERT INTO audit_log(at, actor, action, detail) VALUES(?,?,?,?)", (now, principal.user_id, "client.deleted", f"client {client_id} ({row[1]})"))
    return {"client_id": client_id, "deleted": True}


def summary(store: Store, principal: Principal | None = None, scope: str = "mine") -> dict:
    """Fleet summary. Throughput is counted from the stored executions (by when they ran, CPU
    baselines excluded) so it matches the overview; heartbeat counters would also count re-uploads
    and backfills in the hour they arrive. Live runtime snapshots are returned only for the
    viewer's own clients (admins may ask for scope=all)."""
    _reconcile_update_timeouts(store)
    conn = store.reader()
    now = now_ms()
    hour = now // 3_600_000
    viewer = principal.user_id if principal is not None and not principal.anonymous else None
    # The operations dashboard is public read-only. Anonymous viewers may see
    # fleet aggregates; mutation routes still require authentication.
    see_all = scope == "all" and principal is not None and (principal.anonymous or principal.role == "admin")
    ran = {}
    for row in conn.execute(
            "SELECT client_id, COUNT(*) AS executions, COALESCE(SUM(run_kind<>'fuzz'),0) AS retests FROM executions "
            "WHERE created_at>? AND backend<>'cpu' GROUP BY client_id", (now - 86_400_000,)):
        ran[row["client_id"]] = (row["executions"], row["retests"])
    clients = []
    for row in conn.execute(
            "SELECT client_id, user_id, machine_id, client_home, display_name, hostname, npu_arch, afx_version, backend_versions, "
            "target_afx_version, update_status, target_daemon_state, daemon_request_id, daemon_state, daemon_state_at, last_heartbeat_at, last_seen_at, busy, pending_uploads, runtime, runtime_at, devices "
            "FROM clients WHERE deleted_at IS NULL "
            "ORDER BY COALESCE(last_heartbeat_at, last_seen_at) DESC"):
        item = dict(row)
        item["update_status"] = json.loads(item["update_status"]) if item["update_status"] else {"state":"idle"}
        item["backend_versions"] = json.loads(item["backend_versions"]) if item["backend_versions"] else {}
        mine = see_all or (viewer is not None and item["user_id"] == viewer)
        item["runtime"] = json.loads(item["runtime"]) if item["runtime"] and mine else []
        item["mine"] = viewer is not None and item["user_id"] == viewer
        item["online"] = is_online(item["last_heartbeat_at"], now)
        totals = conn.execute(
            f"SELECT {', '.join(f'COALESCE(SUM({n}),0) AS {n}' for n in COUNTERS)} FROM metric_rollups "
            "WHERE client_id=? AND hour>?", (item["client_id"], hour - 24)).fetchone()
        item["last_24h"] = dict(totals)
        item["last_24h"]["executions"], item["last_24h"]["retests"] = ran.get(item["client_id"], (0, 0))
        raw_devices = item.pop("devices")
        devices = json.loads(raw_devices) if raw_devices is not None else None
        item["devices"] = devices if mine else None
        item["card_count"] = len(devices) if mine and devices is not None else None
        clients.append(item)
    users = []
    user_ids = sorted({c["user_id"] for c in clients if see_all or c["user_id"] == viewer})
    for user_id in user_ids:
        owned = [c for c in clients if c["user_id"] == user_id]
        ids = [c["client_id"] for c in owned]
        marks = ",".join("?" for _ in ids)
        recent = conn.execute(f"SELECT COUNT(*) FROM executions WHERE client_id IN ({marks}) AND created_at>=? AND backend<>'cpu'",
                              [*ids, now - 600_000]).fetchone()[0] if ids else 0
        latest = conn.execute(f"SELECT MAX(created_at) FROM executions WHERE client_id IN ({marks}) AND backend<>'cpu'", ids).fetchone()[0] if ids else None
        users.append({"user_id": user_id, "clients": len(owned),
                      "online_clients": sum(c["online"] for c in owned),
                      "card_count": sum(c["card_count"] for c in owned) if all(c["card_count"] is not None for c in owned) else None,
                      "executions_24h": sum(c["last_24h"]["executions"] for c in owned),
                      "uploads_24h": sum(c["last_24h"]["uploads"] for c in owned),
                      "executions_10m": recent, "cases_per_minute": round(recent / 10, 2),
                      "last_execution_at": latest})
    buckets = {}
    for r in conn.execute(
            "SELECT created_at/3600000 AS hour, COUNT(*) AS executions, COALESCE(SUM(run_kind<>'fuzz'),0) AS retests "
            "FROM executions WHERE created_at>=? AND backend<>'cpu' GROUP BY 1", ((hour - 47) * 3_600_000,)):
        buckets[r["hour"]] = {"hour": r["hour"], "executions": r["executions"], "retests": r["retests"],
                              "uploads": 0, "upload_errors": 0}
    for r in conn.execute("SELECT hour, SUM(uploads) AS uploads, SUM(upload_errors) AS upload_errors FROM metric_rollups "
                          "WHERE hour>? GROUP BY hour", (hour - 48,)):
        entry = buckets.setdefault(r["hour"], {"hour": r["hour"], "executions": 0, "retests": 0, "uploads": 0, "upload_errors": 0})
        entry["uploads"], entry["upload_errors"] = r["uploads"], r["upload_errors"]
    series = [buckets[h] for h in sorted(buckets)]
    versions = [r[0] for r in conn.execute(
        "SELECT afx_version FROM clients WHERE afx_version IS NOT NULL GROUP BY afx_version")]
    latest = max(versions, key=_version_key) if versions else None
    sizes = {p.name: p.stat().st_size for p in store.dir.glob("*.db") if p.is_file()}
    queue = {r[0]: r[1] for r in conn.execute(
        "SELECT status, COUNT(*) FROM retest_tasks WHERE status IN ('pending','leased') GROUP BY status")}
    return {
        "online_clients": sum(c["online"] for c in clients),
        "executions_24h": sum(c["last_24h"]["executions"] for c in clients),
        "pending_uploads": sum(c["pending_uploads"] or 0 for c in clients if c["online"]),
        "retest_queue": queue, "db_bytes": sizes, "latest_afx_version": latest_pypi_version(store),
        "clients": clients, "users": users, "hourly": series,
    }


def efficiency(store: Store, client_id=None, limit: int = 100) -> dict:
    """Finished stage windows reported by clients (same rows the old tool-efficiency page used)."""
    conn = store.reader()
    sql = ("SELECT w.client_id, c.user_id, c.display_name, w.window_id, w.received_at, w.payload FROM efficiency_windows w "
           "JOIN clients c ON c.client_id=w.client_id")
    args = []
    if client_id is not None:
        sql += " WHERE w.client_id=?"
        args.append(client_id)
    windows = []
    for row in conn.execute(sql + " ORDER BY w.received_at DESC LIMIT ?", args + [min(limit, 200)]):
        windows.append({"client_id": row["client_id"], "user_id": row["user_id"], "client_name": row["display_name"],
                        "window_id": row["window_id"], "received_at": row["received_at"],
                        "rows": json.loads(row["payload"])})
    users = {}
    for window in windows:
        user = users.setdefault(window["user_id"], {"user_id": window["user_id"], "windows": 0, "clients": set(), "seconds": 0.0, "success": 0, "completed": 0, "failed": 0, "case_errors": 0})
        user["windows"] += 1; user["clients"].add(window["client_id"])
        for row in window["rows"]:
            user["seconds"] += float(row.get("window_seconds") or 0)
            for key in ("success", "completed", "failed", "case_errors"):
                user[key] += int(row.get(key) or 0)
    for user in users.values():
        user["clients"] = len(user["clients"]); user["success_per_minute"] = round(user["success"] * 60 / user["seconds"], 2) if user["seconds"] else None
    # A client may not have finished a stage window yet. Keep the efficiency
    # panel useful by deriving a recent per-user throughput fallback directly
    # from stored non-CPU executions.
    since = now_ms() - 86_400_000
    where = "e.created_at>=? AND e.backend<>'cpu'"
    args = [since]
    if client_id is not None:
        where += " AND e.client_id=?"; args.append(client_id)
    for row in conn.execute("SELECT e.user_id, COUNT(*) AS completed,"
                            "SUM(e.function_status='pass') AS success,"
                            "SUM(e.function_status<>'pass') AS failed,"
                            "COUNT(DISTINCT e.client_id) AS clients,"
                            "MIN(e.created_at) AS first_at, MAX(e.created_at) AS last_at "
                            f"FROM executions e WHERE {where} GROUP BY e.user_id", args):
        user = users.setdefault(row["user_id"], {"user_id": row["user_id"], "windows": 0, "clients": 0, "seconds": 0.0, "success": 0, "completed": 0, "failed": 0, "case_errors": 0})
        if not user["windows"]:
            user.update(completed=row["completed"], success=row["success"] or 0, failed=row["failed"] or 0, clients=row["clients"],
                        executions_24h=row["completed"], first_at=row["first_at"], last_at=row["last_at"], fallback=True)
            span = max(60.0, ((row["last_at"] or row["first_at"]) - (row["first_at"] or row["last_at"])) / 1000)
            user["success_per_minute"] = round((row["success"] or 0) * 60 / span, 2)
    return {"windows": windows, "users": list(users.values())}


def _version_key(version: str):
    return tuple(int(p) if p.isdigit() else -1 for p in re.split(r"[.+-]", version))
