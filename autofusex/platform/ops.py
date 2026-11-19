"""Platform operations: client heartbeats, hourly rollups, one-click client update (design 4.7, 6.7).

No per-event detail is kept: heartbeats carry counter deltas that are folded
into one row per client per hour and expired after ROLLUP_DAYS.
"""
from __future__ import annotations

import json
import re

from .auth import Principal
from .issues import IssueError, idempotent
from .store import Store, now_ms

ROLLUP_DAYS = 90
ONLINE_MS = 5 * 60_000
COUNTERS = ("executions", "uploads", "upload_errors", "retests")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,31}$")
_VERSION = re.compile(r"^[A-Za-z0-9_.+-]{1,64}$")


def heartbeat(store: Store, principal: Principal, body: dict) -> dict:
    principal.require_write()
    machine_id = body.get("machine_id")
    if not isinstance(machine_id, str) or not 1 <= len(machine_id) <= 128:
        raise IssueError("invalid", "machine_id is required")
    for key in ("hostname", "npu_arch", "afx_version"):
        value = body.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > 128):
            raise IssueError("invalid", f"{key} must be a short string")
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
    from .ingest import upsert_client
    now = now_ms()
    with store.write() as conn:
        client_id = upsert_client(conn, principal.user_id, {
            "machine_id": machine_id, "hostname": body.get("hostname"), "npu_arch": body.get("npu_arch"),
            "afx_version": body.get("afx_version")})
        conn.execute("UPDATE clients SET backend_versions=COALESCE(?, backend_versions), last_heartbeat_at=?, "
                     "busy=?, pending_uploads=? WHERE client_id=?",
                     (json.dumps(versions, sort_keys=True) if versions else None, now, int(bool(body.get("busy"))),
                      pending, client_id))
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
        row = conn.execute("SELECT target_afx_version, afx_version FROM clients WHERE client_id=?",
                           (client_id,)).fetchone()
        target = row["target_afx_version"]
        if target is not None and target == row["afx_version"]:
            conn.execute("UPDATE clients SET target_afx_version=NULL WHERE client_id=?", (client_id,))
            conn.execute("INSERT INTO audit_log(at, actor, action, detail) VALUES(?,?,?,?)",
                         (now, principal.user_id, "client.updated", f"client {client_id} -> {target}"))
            target = None
    return {"client_id": client_id, "update_to": target, "server_time": now, "windows_accepted": accepted}


def request_update(store: Store, principal: Principal, body: dict):
    principal.require_admin()
    ids, version = body.get("client_ids"), body.get("version")
    if not isinstance(ids, list) or not 1 <= len(ids) <= 200 or not all(
            isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise IssueError("invalid", "client_ids must list 1..200 client ids")
    if version is not None and (not isinstance(version, str) or not _VERSION.match(version)):
        raise IssueError("invalid", "version must be an exact package version or null to cancel")

    def action(conn):
        count = 0
        for client_id in ids:
            count += conn.execute("UPDATE clients SET target_afx_version=? WHERE client_id=?",
                                  (version, client_id)).rowcount
        conn.execute("INSERT INTO audit_log(at, actor, action, detail) VALUES(?,?,?,?)",
                     (now_ms(), principal.user_id, "client.update_requested", f"{ids} -> {version}"))
        return {"requested": count, "version": version}
    return idempotent(store, principal, "client-update", body.pop("request_key", None), body, action)


def summary(store: Store, principal: Principal | None = None, scope: str = "mine") -> dict:
    """Fleet summary. Throughput is counted from the stored executions (by when they ran, CPU
    baselines excluded) so it matches the overview; heartbeat counters would also count re-uploads
    and backfills in the hour they arrive. Live runtime snapshots are returned only for the
    viewer's own clients (admins may ask for scope=all)."""
    conn = store.reader()
    now = now_ms()
    hour = now // 3_600_000
    viewer = principal.user_id if principal is not None and not principal.anonymous else None
    see_all = scope == "all" and principal is not None and not principal.anonymous and principal.role == "admin"
    ran = {}
    for row in conn.execute(
            "SELECT client_id, COUNT(*) AS executions, COALESCE(SUM(run_kind<>'fuzz'),0) AS retests FROM executions "
            "WHERE created_at>? AND backend<>'cpu' GROUP BY client_id", (now - 86_400_000,)):
        ran[row["client_id"]] = (row["executions"], row["retests"])
    clients = []
    for row in conn.execute(
            "SELECT client_id, user_id, machine_id, display_name, hostname, npu_arch, afx_version, backend_versions, "
            "target_afx_version, last_heartbeat_at, last_seen_at, busy, pending_uploads, runtime, runtime_at "
            "FROM clients "
            "ORDER BY COALESCE(last_heartbeat_at, last_seen_at) DESC"):
        item = dict(row)
        item["backend_versions"] = json.loads(item["backend_versions"]) if item["backend_versions"] else {}
        mine = see_all or (viewer is not None and item["user_id"] == viewer)
        item["runtime"] = json.loads(item["runtime"]) if item["runtime"] and mine else []
        item["mine"] = viewer is not None and item["user_id"] == viewer
        seen = max(item["last_heartbeat_at"] or 0, item["last_seen_at"] or 0)
        item["online"] = now - seen <= ONLINE_MS
        totals = conn.execute(
            f"SELECT {', '.join(f'COALESCE(SUM({n}),0) AS {n}' for n in COUNTERS)} FROM metric_rollups "
            "WHERE client_id=? AND hour>?", (item["client_id"], hour - 24)).fetchone()
        item["last_24h"] = dict(totals)
        item["last_24h"]["executions"], item["last_24h"]["retests"] = ran.get(item["client_id"], (0, 0))
        clients.append(item)
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
        "retest_queue": queue, "db_bytes": sizes, "latest_afx_version": latest,
        "clients": clients, "hourly": series,
    }


def efficiency(store: Store, client_id=None, limit: int = 100) -> dict:
    """Finished stage windows reported by clients (same rows the old tool-efficiency page used)."""
    conn = store.reader()
    sql = ("SELECT w.client_id, c.display_name, w.window_id, w.received_at, w.payload FROM efficiency_windows w "
           "JOIN clients c ON c.client_id=w.client_id")
    args = []
    if client_id is not None:
        sql += " WHERE w.client_id=?"
        args.append(client_id)
    windows = []
    for row in conn.execute(sql + " ORDER BY w.received_at DESC LIMIT ?", args + [min(limit, 200)]):
        windows.append({"client_id": row["client_id"], "client_name": row["display_name"],
                        "window_id": row["window_id"], "received_at": row["received_at"],
                        "rows": json.loads(row["payload"])})
    return {"windows": windows}


def _version_key(version: str):
    return tuple(int(p) if p.isdigit() else -1 for p in re.split(r"[.+-]", version))


def not_implemented(feature: str):
    raise IssueError("not_implemented", f"{feature} is reserved for a later phase", 501)
