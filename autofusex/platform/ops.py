"""Platform operations: client heartbeats, hourly rollups, one-click client update (design 4.7, 6.7).

No per-event detail is kept: heartbeats carry counter deltas that are folded
into one row per client per hour and expired after ROLLUP_DAYS.
"""
from __future__ import annotations

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
    principal.require("upload")
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
    pending = body.get("pending_uploads")
    if pending is not None and (isinstance(pending, bool) or not isinstance(pending, int) or pending < 0):
        raise IssueError("invalid", "pending_uploads must be a non-negative integer")
    import json
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
    return {"client_id": client_id, "update_to": target, "server_time": now}


def request_update(store: Store, principal: Principal, body: dict):
    principal.require("admin")
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


def summary(store: Store) -> dict:
    conn = store.reader()
    now = now_ms()
    hour = now // 3_600_000
    clients = []
    for row in conn.execute(
            "SELECT client_id, user_id, machine_id, hostname, npu_arch, afx_version, backend_versions, "
            "target_afx_version, last_heartbeat_at, last_seen_at, busy, pending_uploads FROM clients "
            "ORDER BY COALESCE(last_heartbeat_at, last_seen_at) DESC"):
        item = dict(row)
        import json
        item["backend_versions"] = json.loads(item["backend_versions"]) if item["backend_versions"] else {}
        seen = max(item["last_heartbeat_at"] or 0, item["last_seen_at"] or 0)
        item["online"] = now - seen <= ONLINE_MS
        totals = conn.execute(
            f"SELECT {', '.join(f'COALESCE(SUM({n}),0) AS {n}' for n in COUNTERS)} FROM metric_rollups "
            "WHERE client_id=? AND hour>?", (item["client_id"], hour - 24)).fetchone()
        item["last_24h"] = dict(totals)
        clients.append(item)
    series = [dict(r) for r in conn.execute(
        f"SELECT hour, {', '.join(f'SUM({n}) AS {n}' for n in COUNTERS)} FROM metric_rollups WHERE hour>? "
        "GROUP BY hour ORDER BY hour", (hour - 48,))]
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


def _version_key(version: str):
    return tuple(int(p) if p.isdigit() else -1 for p in re.split(r"[.+-]", version))


def not_implemented(feature: str):
    raise IssueError("not_implemented", f"{feature} is reserved for a later phase", 501)
