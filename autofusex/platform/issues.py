"""Issue clustering and lifecycle (design 2, 4.5).

Clustering runs inside the ingest transaction whenever an execution becomes a
current head. Lifecycle writes (claim, renew, release, verify, knowledge) are
idempotent by request_key and guarded by the issue lease: only the lease holder
can move an issue forward; the web UI cannot bypass it.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3

from .auth import Principal
from .store import Store, now_ms

DIMENSION_PREFIX = {"function": "F", "accuracy": "A", "performance": "P"}
DIMENSION_LABEL = {"function": "功能", "accuracy": "精度", "performance": "性能"}
ACTIVE = ("claimed", "verifying", "regressing", "summarizing")
TERMINAL = ("resolved", "closed")
REPRO_LIMIT = 50


class IssueError(ValueError):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.status = code, status


# -- signatures -----------------------------------------------------------------
_NORMALIZE = [
    (re.compile(r"0x[0-9a-f]+"), "<hex>"),
    (re.compile(r"graph=\S+"), "graph=<graph>"),
    (re.compile(r"[0-9a-f]{12,}"), "<id>"),
    (re.compile(r"(?:[a-z]:)?(?:/[\w.+-]+){2,}"), "<path>"),
    (re.compile(r"'[^']*'|\"[^\"]*\""), "<str>"),
    (re.compile(r"\[[\d,\s]*\]|\([\d,\s]+\)"), "<shape>"),
    (re.compile(r"\d+"), "<n>"),
    (re.compile(r"\s+"), " "),
]


def normalize(text: str | None) -> str:
    """Strip the parts of an error line that vary between occurrences of one bug."""
    text = (text or "").strip().splitlines()[0].lower() if (text or "").strip() else ""
    for pattern, repl in _NORMALIZE:
        text = pattern.sub(repl, text)
    return text.strip()[:200]


def _digest(*parts) -> bytes:
    return hashlib.sha256("\x1f".join(parts).encode()).digest()[:16]


# Bugs usually live in the hardest op of a graph; one primary category keeps clusters coarse enough.
CATEGORY_PRIORITY = ("reduce", "broadcast", "view", "compare", "cast", "elementwise")


def primary_category(categories) -> str:
    if not categories:
        return "uncategorized"
    rank = {name: i for i, name in enumerate(CATEGORY_PRIORITY)}
    return min(categories, key=lambda c: (rank.get(c, len(rank)), c))


def case_traits(conn, case_id: int) -> dict:
    row = conn.execute("SELECT case_key, case_source, case_revision, op_types, dtypes, node_count FROM cases "
                       "WHERE case_id=?", (case_id,)).fetchone()
    categories = sorted(r[0] for r in conn.execute(
        "SELECT v.value FROM case_features f JOIN feature_registry r USING(feature_id) "
        "JOIN feature_values v ON v.value_id=f.value_id WHERE f.case_id=? AND r.name='op_category'", (case_id,)))
    return {"case_key": row["case_key"], "case_source": row["case_source"], "revision": row["case_revision"],
            "ops": row["op_types"] or "", "dtypes": row["dtypes"] or "", "node_count": row["node_count"] or 0,
            "categories": ",".join(categories) or "uncategorized", "primary": primary_category(categories)}


def function_signature(execution, traits):
    stage = execution["fail_stage"] or execution["function_status"]
    norm = normalize(execution["log_summary"])
    sig = _digest("function", execution["backend"], stage, norm, traits["primary"])
    title = f"[{execution['backend']}] {stage}: {norm[:80] or execution['function_status']} ({traits['primary']})"
    return sig, stage, title


def accuracy_bucket(execution) -> str:
    rel, absolute = execution["max_rel_err"], execution["max_abs_err"]
    if rel is None and absolute is None:
        return "unknown"
    if (rel or 0) >= 1 or (absolute or 0) >= 1:
        return "severe"
    return "mild"


def accuracy_signature(execution, traits):
    # Single-op graphs keep the exact op; larger graphs group by primary category.
    ops = traits["ops"] if traits["node_count"] == 1 else traits["primary"]
    bucket = accuracy_bucket(execution)
    ref = execution["accuracy_ref"] or "cpu"
    sig = _digest("accuracy", execution["backend"], traits["dtypes"], ops, bucket)
    label = {"severe": "严重", "mild": "轻微"}.get(bucket, "")
    title = f"[{execution['backend']}] 精度{label}失败 {traits['dtypes']} {ops}" + (
        " (EagerNPU 标杆)" if ref == "eager-npu" else "")
    return sig, bucket, title


def slowdown_bucket(ratio: float) -> str:
    if ratio >= 5:
        return ">5x"
    if ratio >= 2:
        return "2-5x"
    return "1.2-2x"


def performance_signature(backend, anchor, ratio, traits):
    bucket = slowdown_bucket(ratio)
    sig = _digest("performance", backend, anchor, bucket, traits["primary"])
    title = f"[{backend}] kernel 比 {anchor} 慢 {bucket} ({traits['primary']})"
    return sig, bucket, title


def failure_dimension(execution) -> str | None:
    if execution["function_status"] not in ("pass", "unsupported"):
        return "function"
    if execution["function_status"] == "pass" and execution["accuracy_status"] == "fail":
        return "accuracy"
    return None


def execution_signature(conn, execution, traits=None):
    """Signature stored on every failing execution of an issue backend (any run kind)."""
    dimension = failure_dimension(execution)
    if dimension is None:
        return None
    traits = traits or case_traits(conn, execution["case_id"])
    return (function_signature if dimension == "function" else accuracy_signature)(execution, traits)[0]


# -- clustering hook ------------------------------------------------------------
def _setting(conn, key):
    from .store import DEFAULT_SETTINGS
    row = conn.execute("SELECT value FROM platform_settings WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else DEFAULT_SETTINGS[key]


def _execution(conn, exec_id):
    return conn.execute(
        "SELECT exec_id, case_id, case_revision, backend, npu_arch, function_status, fail_stage, accuracy_status, "
        "accuracy_ref, max_abs_err, max_rel_err, kernel_ns, log_summary, created_at FROM executions WHERE exec_id=?",
        (exec_id,)).fetchone()


def config(conn) -> dict:
    """Clustering settings, read once per upload transaction."""
    return {"backends": _setting(conn, "issue.backends"), "anchor": _setting(conn, "issue.perf_anchor"),
            "slowdown": _setting(conn, "issue.perf_slowdown")}


def on_head(conn: sqlite3.Connection, case_id: int, exec_id: int, cfg: dict | None = None) -> None:
    """Called after an execution became the current head of (case, backend, arch)."""
    cfg = cfg or config(conn)
    execution = _execution(conn, exec_id)
    backends, anchor = cfg["backends"], cfg["anchor"]
    if execution["backend"] not in backends and execution["backend"] != anchor:
        return
    traits = case_traits(conn, case_id)
    if execution["backend"] in backends:
        dimension = failure_dimension(execution)
        if dimension is not None:
            make = function_signature if dimension == "function" else accuracy_signature
            sig, category, title = make(execution, traits)
            conn.execute("UPDATE executions SET issue_signature=? WHERE exec_id=?", (sig, exec_id))
            _attach(conn, dimension, execution["backend"], sig, category, title, execution["log_summary"],
                    execution, traits)
    _check_performance(conn, case_id, execution["npu_arch"], backends, anchor, cfg["slowdown"], traits)


def _check_performance(conn, case_id, arch, backends, anchor, slowdown, traits):
    heads = {row["backend"]: row for row in conn.execute(
        "SELECT e.* FROM execution_heads h JOIN executions e ON e.exec_id=h.exec_id "
        "WHERE h.case_id=? AND h.npu_arch=?", (case_id, arch))}
    base = heads.get(anchor)
    if base is None or not _timed(base):
        return
    for backend in backends:
        row = heads.get(backend)
        if row is None or not _timed(row):
            continue
        ratio = row["kernel_ns"] / base["kernel_ns"]
        if ratio >= slowdown:
            sig, category, title = performance_signature(backend, anchor, ratio, traits)
            _attach(conn, "performance", backend, sig, category, title,
                    f"kernel {row['kernel_ns']}ns vs {anchor} {base['kernel_ns']}ns ({ratio:.2f}x)", row, traits)


def _timed(row) -> bool:
    return (row["function_status"] == "pass" and row["accuracy_status"] != "fail"
            and row["kernel_ns"] is not None and row["kernel_ns"] > 0)


def _attach(conn, dimension, backend, sig, category, title, summary, execution, traits):
    now = now_ms()
    alias = conn.execute("SELECT issue_id FROM issue_signature_alias WHERE dimension=? AND backend=? AND signature=?",
                         (dimension, backend, sig)).fetchone()
    if alias is not None:
        issue = conn.execute("SELECT * FROM issues WHERE issue_id=?", (alias[0],)).fetchone()
    else:
        issue = conn.execute("SELECT * FROM issues WHERE dimension=? AND backend=? AND signature=?",
                             (dimension, backend, sig)).fetchone()
    if issue is None:
        cur = conn.execute(
            "INSERT INTO issues(issue_id, dimension, backend, signature, category, title, log_summary, status, "
            "first_seen_at, last_seen_at, updated_at) VALUES('', ?,?,?,?,?,?, 'open', ?,?,?)",
            (dimension, backend, sig, category, title[:300], (summary or "")[:512], now, now, now))
        issue_id = f"{DIMENSION_PREFIX[dimension]}{cur.lastrowid}"  # e.g. F254: one token, easy to copy
        conn.execute("UPDATE issues SET issue_id=? WHERE issue_no=?", (issue_id, cur.lastrowid))
        event(conn, issue_id, "system", "created", None, "open", title[:300])
        issue = conn.execute("SELECT * FROM issues WHERE issue_id=?", (issue_id,)).fetchone()
    issue_id = issue["issue_id"]
    inserted = conn.execute(
        "INSERT OR IGNORE INTO issue_cases(issue_id, case_id, role, npu_arch, case_revision, first_exec_id, "
        "last_exec_id, added_at) VALUES(?,?, 'affected', ?,?,?,?,?)",
        (issue_id, execution["case_id"], execution["npu_arch"], traits["revision"], execution["exec_id"],
         execution["exec_id"], now)).rowcount
    if not inserted:
        conn.execute("UPDATE issue_cases SET last_exec_id=?, case_revision=?, stale=0 WHERE issue_id=? "
                     "AND role='affected' AND case_id=? AND npu_arch=?",
                     (execution["exec_id"], traits["revision"], issue_id, execution["case_id"], execution["npu_arch"]))
    conn.execute("UPDATE issues SET case_count=case_count+?, priority=case_count+?, last_seen_at=?, updated_at=? "
                 "WHERE issue_id=?", (inserted, inserted, now, now, issue_id))
    if issue["status"] == "resolved" and _reopens(conn, issue, execution):
        conn.execute("UPDATE issues SET status='open', owner_user=NULL, agent_id=NULL, lease_id=NULL, "
                     "lease_expires_at=NULL, updated_at=? WHERE issue_id=?", (now, issue_id))
        event(conn, issue_id, "system", "reopened", "resolved", "open",
              f"new failure exec {execution['exec_id']} after resolution")


def _reopens(conn, issue, execution) -> bool:
    """A resolved issue reopens only for a failure on a backend version that was not already known broken.

    Until the fix is merged, fuzz keeps running the old versions and keeps failing; those
    failures are expected and must not reopen the issue.
    """
    if execution["created_at"] <= (issue["resolved_at"] or 0):
        return False
    version = conn.execute("SELECT v.backend_version FROM executions e JOIN environments v USING(env_id) "
                           "WHERE e.exec_id=?", (execution["exec_id"],)).fetchone()[0]
    if version is not None and version == issue["fix_version"]:
        return True
    known_broken = {r[0] for r in conn.execute(
        "SELECT DISTINCT v.backend_version FROM issue_cases ic JOIN executions e ON e.exec_id=ic.first_exec_id "
        "JOIN environments v ON v.env_id=e.env_id WHERE ic.issue_id=? AND ic.role='affected' AND e.created_at<=?",
        (issue["issue_id"], issue["resolved_at"]))}
    return version not in known_broken


def mark_stale(conn, case_id: int, revision: int) -> None:
    """Imported case graph replaced: its old issue memberships no longer describe the case."""
    rows = conn.execute("SELECT DISTINCT issue_id FROM issue_cases WHERE case_id=? AND case_revision<? AND stale=0",
                        (case_id, revision)).fetchall()
    conn.execute("UPDATE issue_cases SET stale=1 WHERE case_id=? AND case_revision<?", (case_id, revision))
    for row in rows:
        event(conn, row[0], "system", "case_stale", None, None, f"case {case_id} graph replaced (rev {revision})")


def event(conn, issue_id, actor, kind, from_status, to_status, detail=None):
    conn.execute("INSERT INTO issue_events(issue_id, at, actor, kind, from_status, to_status, detail) "
                 "VALUES(?,?,?,?,?,?,?)", (issue_id, now_ms(), actor, kind, from_status, to_status, detail))


# -- idempotent writes ------------------------------------------------------------
def idempotent(store: Store, principal: Principal, route: str, request_key, body: dict, action):
    """Run action(conn) once per (user, request_key); identical retries return the first response."""
    if not isinstance(request_key, str) or not 1 <= len(request_key) <= 128:
        raise IssueError("invalid", "request_key must be a string of 1..128 chars")
    body_hash = hashlib.sha256(json.dumps([route, body], sort_keys=True).encode()).digest()
    with store.write() as conn:
        row = conn.execute("SELECT body_hash, response FROM agent_requests WHERE user_id=? AND request_key=?",
                           (principal.user_id, request_key)).fetchone()
        if row is not None:
            if row["body_hash"] != body_hash:
                raise IssueError("conflict", "request_key reused with a different request", 409)
            return json.loads(row["response"])
        expire_leases(conn)
        result = action(conn)
        conn.execute("INSERT INTO agent_requests VALUES(?,?,?,?,?,?,?)",
                     (principal.user_id, request_key, route, body_hash, 200, json.dumps(result), now_ms()))
        return result


def expire_leases(conn) -> None:
    now = now_ms()
    for row in conn.execute("SELECT issue_id, status, owner_user, agent_id FROM issues WHERE lease_id IS NOT NULL "
                            "AND lease_expires_at<=?", (now,)).fetchall():
        to_status = "open" if row["status"] == "claimed" else row["status"]
        conn.execute("UPDATE issues SET status=?, owner_user=NULL, agent_id=NULL, lease_id=NULL, "
                     "lease_expires_at=NULL, updated_at=? WHERE issue_id=?", (to_status, now, row["issue_id"]))
        event(conn, row["issue_id"], "system", "lease_expired", row["status"], to_status,
              f"{row['owner_user']}/{row['agent_id']}")


def _holder(conn, principal: Principal, issue_id: str, lease_id) -> sqlite3.Row:
    issue = conn.execute("SELECT * FROM issues WHERE issue_id=?", (issue_id,)).fetchone()
    if issue is None:
        raise IssueError("not_found", "issue not found", 404)
    if not isinstance(lease_id, str) or issue["lease_id"] != lease_id or issue["owner_user"] != principal.user_id:
        raise IssueError("lease_lost", "lease is not held by this caller (expired, released or taken over)", 409)
    return issue


def _set_status(conn, issue, to_status, actor, kind, detail=None, **fields):
    fields["status"] = to_status
    fields["updated_at"] = now_ms()
    assignments = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE issues SET {assignments} WHERE issue_id=?", (*fields.values(), issue["issue_id"]))
    event(conn, issue["issue_id"], actor, kind, issue["status"], to_status, detail)


# -- lifecycle ----------------------------------------------------------------------
_AGENT = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def claim(store: Store, principal: Principal, body: dict):
    principal.require("issue")
    agent_id = body.get("agent_id")
    if not isinstance(agent_id, str) or not _AGENT.match(agent_id):
        raise IssueError("invalid", "agent_id must be 1..64 chars of letters, digits, _ . : -")
    wanted = body.get("issue_id")
    dimension, backend = body.get("dimension"), body.get("backend")
    if dimension is not None and dimension not in DIMENSION_PREFIX:
        raise IssueError("invalid", "dimension must be function, accuracy or performance")

    def action(conn):
        held = conn.execute("SELECT issue_id FROM issues WHERE owner_user=? AND agent_id=? AND lease_id IS NOT NULL",
                            (principal.user_id, agent_id)).fetchone()
        if held is not None:
            raise IssueError("agent_busy", f"agent already holds {held[0]}; release it first", 409)
        if wanted is not None:
            issue = conn.execute("SELECT * FROM issues WHERE issue_id=?", (wanted,)).fetchone()
            if issue is None:
                raise IssueError("not_found", "issue not found", 404)
            if issue["status"] in TERMINAL or issue["lease_id"] is not None:
                raise IssueError("unavailable", f"issue is {issue['status']} or held by someone else", 409)
        else:
            where, args = ["status='open'", "lease_id IS NULL"], []
            if dimension:
                where.append("dimension=?")
                args.append(dimension)
            if backend:
                where.append("backend=?")
                args.append(backend)
            issue = conn.execute(f"SELECT * FROM issues WHERE {' AND '.join(where)} "
                                 "ORDER BY priority DESC, issue_no LIMIT 1", args).fetchone()
            if issue is None:
                return None
        lease_id = secrets.token_hex(16)
        now = now_ms()
        to_status = "claimed" if issue["status"] == "open" else issue["status"]
        _set_status(conn, issue, to_status, principal.user_id, "claimed", agent_id, owner_user=principal.user_id,
                    agent_id=agent_id, lease_id=lease_id, claimed_at=now,
                    lease_expires_at=now + _setting(conn, "issue.lease_minutes") * 60_000)
        return dict(detail(conn, issue["issue_id"], repro=True), lease_id=lease_id)
    return idempotent(store, principal, "claim", body.pop("request_key", None), body, action)


def renew(store: Store, principal: Principal, issue_id: str, body: dict):
    principal.require("issue")

    def action(conn):
        issue = _holder(conn, principal, issue_id, body.get("lease_id"))
        expires = now_ms() + _setting(conn, "issue.lease_minutes") * 60_000
        conn.execute("UPDATE issues SET lease_expires_at=?, updated_at=? WHERE issue_id=?",
                     (expires, now_ms(), issue["issue_id"]))
        return {"issue_id": issue_id, "lease_expires_at": expires}
    return idempotent(store, principal, f"renew:{issue_id}", body.pop("request_key", None), body, action)


def release(store: Store, principal: Principal, issue_id: str, body: dict):
    principal.require("issue")

    def action(conn):
        issue = _holder(conn, principal, issue_id, body.get("lease_id"))
        to_status = "open" if issue["status"] == "claimed" else issue["status"]
        _set_status(conn, issue, to_status, principal.user_id, "released", issue["agent_id"], owner_user=None,
                    agent_id=None, lease_id=None, lease_expires_at=None)
        return {"issue_id": issue_id, "status": to_status}
    return idempotent(store, principal, f"release:{issue_id}", body.pop("request_key", None), body, action)


_VERSION = re.compile(r"^[A-Za-z0-9_.+-]{1,64}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


def verify(store: Store, principal: Principal, issue_id: str, body: dict):
    """Freeze the affected cases and queue a verification round on their own arch (design 2)."""
    principal.require("issue")
    version, sha = body.get("fix_version"), body.get("fix_sha256")
    if not isinstance(version, str) or not _VERSION.match(version):
        raise IssueError("invalid", "fix_version must be an exact package version")
    if not isinstance(sha, str) or not _SHA.match(sha):
        raise IssueError("invalid", "fix_sha256 must be 64 lowercase hex chars")
    fix_ref = body.get("fix_ref")
    if fix_ref is not None and (not isinstance(fix_ref, str) or len(fix_ref) > 512):
        raise IssueError("invalid", "fix_ref must be a string of at most 512 chars")

    def action(conn):
        from . import retests
        issue = _holder(conn, principal, issue_id, body.get("lease_id"))
        if issue["status"] != "claimed":
            raise IssueError("wrong_state", f"verify needs status claimed, issue is {issue['status']}", 409)
        cases = conn.execute(
            "SELECT case_id, npu_arch FROM issue_cases WHERE issue_id=? AND role='affected' AND stale=0 "
            "ORDER BY added_at DESC LIMIT ?", (issue_id, _setting(conn, "verify.max_cases"))).fetchall()
        if not cases:
            raise IssueError("no_cases", "issue has no current affected cases to verify", 409)
        round_no = issue["round_no"] + 1
        conn.execute("INSERT INTO issue_rounds(issue_id, round_no, kind, fix_version, fix_sha256, started_at) "
                     "VALUES(?,?, 'verify', ?,?,?)", (issue_id, round_no, version, sha, now_ms()))
        tasks = retests.create_tasks(conn, issue, round_no, [(c["case_id"], c["npu_arch"]) for c in cases],
                                     version, sha)
        _set_status(conn, issue, "verifying", principal.user_id, "verify_submitted",
                    f"round {round_no} version {version} tasks {tasks}", round_no=round_no, fix_version=version,
                    fix_sha256=sha, fix_ref=fix_ref or issue["fix_ref"])
        return {"issue_id": issue_id, "round_no": round_no, "tasks": tasks, "status": "verifying"}
    return idempotent(store, principal, f"verify:{issue_id}", body.pop("request_key", None), body, action)


def pin_cases(store: Store, principal: Principal, issue_id: str, body: dict):
    """Lease holder (or admin) names must-run regression cases before regression starts (design 2.1)."""
    principal.require("issue")
    keys, reason = body.get("case_keys"), body.get("reason")
    if not isinstance(keys, list) or not 1 <= len(keys) <= 200 or not all(isinstance(k, str) for k in keys):
        raise IssueError("invalid", "case_keys must be a list of 1..200 strings")
    if reason is not None and (not isinstance(reason, str) or len(reason) > 500):
        raise IssueError("invalid", "reason must be at most 500 chars")

    def action(conn):
        issue = conn.execute("SELECT * FROM issues WHERE issue_id=?", (issue_id,)).fetchone()
        if issue is None:
            raise IssueError("not_found", "issue not found", 404)
        if principal.role != "admin":
            _holder(conn, principal, issue_id, body.get("lease_id"))
        if issue["status"] not in ("open", "claimed", "verifying"):
            raise IssueError("wrong_state", "regression set is frozen once regression starts", 409)
        ids = _case_ids(conn, keys)
        for case_id in ids:
            conn.execute("INSERT INTO regression_pins(scope, scope_value, case_id, added_by, reason, created_at) "
                         "VALUES('issue', ?,?,?,?,?)", (issue_id, case_id, principal.user_id, reason, now_ms()))
        event(conn, issue_id, principal.user_id, "regression_pinned", None, None, f"{len(ids)} cases")
        return {"issue_id": issue_id, "pinned": len(ids)}
    return idempotent(store, principal, f"pin:{issue_id}", body.pop("request_key", None), body, action)


def _case_ids(conn, keys):
    ids = []
    for key in dict.fromkeys(keys):
        row = conn.execute("SELECT case_id FROM cases WHERE case_key=?", (key,)).fetchone()
        if row is None:
            raise IssueError("not_found", f"unknown case {key[:64]}", 404)
        ids.append(row[0])
    return ids


def submit_knowledge(store: Store, principal: Principal, issue_id: str, body: dict):
    principal.require("knowledge")
    from . import knowledge
    fields = knowledge.validate(body.get("knowledge"))

    def action(conn):
        issue = _holder(conn, principal, issue_id, body.get("lease_id"))
        if issue["status"] != "summarizing":
            raise IssueError("wrong_state", f"knowledge needs status summarizing, issue is {issue['status']}", 409)
        knowledge_id = knowledge.create(conn, principal, fields, confidence="verified", status="published",
                                        agent_id=issue["agent_id"], issue=issue)
        _set_status(conn, issue, "resolved", principal.user_id, "resolved", knowledge_id, knowledge_id=knowledge_id,
                    resolved_at=now_ms(), owner_user=None, agent_id=None, lease_id=None, lease_expires_at=None)
        return {"issue_id": issue_id, "knowledge_id": knowledge_id, "status": "resolved"}
    return idempotent(store, principal, f"knowledge:{issue_id}", body.pop("request_key", None), body, action)


# -- admin --------------------------------------------------------------------------
def close(store: Store, principal: Principal, issue_id: str, body: dict):
    principal.require("admin")
    reason = body.get("reason")
    if not isinstance(reason, str) or not 1 <= len(reason) <= 500:
        raise IssueError("invalid", "reason is required (1..500 chars)")

    def action(conn):
        issue = conn.execute("SELECT * FROM issues WHERE issue_id=?", (issue_id,)).fetchone()
        if issue is None:
            raise IssueError("not_found", "issue not found", 404)
        if issue["status"] in TERMINAL:
            raise IssueError("wrong_state", f"issue is already {issue['status']}", 409)
        _cancel_tasks(conn, issue_id)
        _set_status(conn, issue, "closed", principal.user_id, "closed", reason, close_reason=reason,
                    owner_user=None, agent_id=None, lease_id=None, lease_expires_at=None)
        return {"issue_id": issue_id, "status": "closed"}
    return idempotent(store, principal, f"close:{issue_id}", body.pop("request_key", None), body, action)


def merge(store: Store, principal: Principal, issue_id: str, body: dict):
    """Fold a too-fine cluster into another; its signature keeps routing to the target."""
    principal.require("admin")
    target_id = body.get("into")

    def action(conn):
        source = conn.execute("SELECT * FROM issues WHERE issue_id=?", (issue_id,)).fetchone()
        target = conn.execute("SELECT * FROM issues WHERE issue_id=?", (target_id,)).fetchone()
        if source is None or target is None or source["issue_id"] == target["issue_id"]:
            raise IssueError("invalid", "source and target must be two existing issues", 400)
        if (source["dimension"], source["backend"]) != (target["dimension"], target["backend"]):
            raise IssueError("invalid", "only issues of the same dimension and backend can merge", 400)
        if source["status"] in ACTIVE or target["status"] in TERMINAL:
            raise IssueError("wrong_state", "source must not be in progress and target must be open", 409)
        moved = conn.execute(
            "INSERT OR IGNORE INTO issue_cases(issue_id, case_id, role, npu_arch, case_revision, stale, first_exec_id, "
            "last_exec_id, added_at) SELECT ?, case_id, role, npu_arch, case_revision, stale, first_exec_id, "
            "last_exec_id, added_at FROM issue_cases WHERE issue_id=? AND role='affected'",
            (target_id, issue_id)).rowcount
        conn.execute("INSERT OR REPLACE INTO issue_signature_alias VALUES(?,?,?,?)",
                     (source["dimension"], source["backend"], source["signature"], target_id))
        conn.execute("UPDATE issue_signature_alias SET issue_id=? WHERE issue_id=?", (target_id, issue_id))
        conn.execute("UPDATE issues SET case_count=case_count+?, priority=case_count+?, updated_at=? WHERE issue_id=?",
                     (moved, moved, now_ms(), target_id))
        _cancel_tasks(conn, issue_id)
        _set_status(conn, source, "closed", principal.user_id, "merged", target_id, duplicate_of=target_id,
                    close_reason=f"merged into {target_id}")
        event(conn, target_id, principal.user_id, "merged_from", None, None, f"{issue_id} (+{moved} cases)")
        return {"issue_id": issue_id, "merged_into": target_id, "moved_cases": moved}
    return idempotent(store, principal, f"merge:{issue_id}", body.pop("request_key", None), body, action)


def _cancel_tasks(conn, issue_id):
    conn.execute("UPDATE retest_tasks SET status='cancelled', finished_at=? WHERE issue_id=? "
                 "AND status IN ('pending','leased')", (now_ms(), issue_id))


# -- reads --------------------------------------------------------------------------
_ISSUE_COLUMNS = ("issue_id, dimension, backend, category, title, log_summary, status, priority, case_count, "
                  "owner_user, agent_id, lease_expires_at, fix_version, fix_sha256, fix_ref, knowledge_id, caused_by, "
                  "duplicate_of, close_reason, round_no, first_seen_at, last_seen_at, claimed_at, resolved_at, "
                  "updated_at")


def list_issues(store: Store, *, dimension=None, backend=None, status=None, after=None, limit=50) -> dict:
    if not 1 <= limit <= 200:
        raise IssueError("invalid", "limit must be 1..200")
    where, args = ["issue_no<?"], [after if after is not None else 1 << 62]
    for column, value in (("dimension", dimension), ("backend", backend), ("status", status)):
        if value is not None:
            where.append(f"{column}=?")
            args.append(value)
    rows = store.reader().execute(f"SELECT issue_no, {_ISSUE_COLUMNS} FROM issues WHERE {' AND '.join(where)} "
                                  "ORDER BY issue_no DESC LIMIT ?", args + [limit]).fetchall()
    items = [dict(row) for row in rows]
    cursor = items[-1]["issue_no"] if len(items) == limit else None
    for item in items:
        item.pop("issue_no")
    return {"issues": items, "next_cursor": cursor}


def detail(conn, issue_id: str, *, repro: bool = False) -> dict:
    row = conn.execute(f"SELECT {_ISSUE_COLUMNS} FROM issues WHERE issue_id=?", (issue_id,)).fetchone()
    if row is None:
        raise IssueError("not_found", "issue not found", 404)
    out = dict(row)
    out["rounds"] = [dict(r) for r in conn.execute(
        "SELECT round_no, kind, fix_version, fix_sha256, started_at, finished_at, outcome, pass_count, fail_count, "
        "invalid_count FROM issue_rounds WHERE issue_id=? ORDER BY round_no", (issue_id,))]
    out["events"] = [dict(r) for r in conn.execute(
        "SELECT at, actor, kind, from_status, to_status, detail FROM issue_events WHERE issue_id=? "
        "ORDER BY event_id DESC LIMIT 50", (issue_id,))]
    out["arches"] = [r[0] for r in conn.execute(
        "SELECT DISTINCT npu_arch FROM issue_cases WHERE issue_id=? AND role='affected' AND stale=0", (issue_id,))]
    if repro:
        out["repro"] = _repro(conn, issue_id, out["backend"])
        from . import knowledge
        out["related_knowledge"] = knowledge.related(conn, out)
    return out


def _repro(conn, issue_id, backend):
    rows = conn.execute(
        "SELECT c.case_key, c.case_source, ic.npu_arch, e.exec_uuid, e.log_summary FROM issue_cases ic "
        "JOIN cases c ON c.case_id=ic.case_id LEFT JOIN executions e ON e.exec_id=ic.last_exec_id "
        "WHERE ic.issue_id=? AND ic.role='affected' AND ic.stale=0 ORDER BY ic.added_at DESC LIMIT ?",
        (issue_id, REPRO_LIMIT)).fetchall()
    cases = []
    for row in rows:
        command = (f"afx fuzz replay {row['case_key']} --backend {backend}" if row["case_source"] == "nnsmith"
                   else None)
        cases.append({"case_key": row["case_key"], "case_source": row["case_source"], "npu_arch": row["npu_arch"],
                      "exec_uuid": row["exec_uuid"], "log_summary": row["log_summary"], "command": command})
    return {"cases": cases, "truncated": len(rows) == REPRO_LIMIT}


def issue_detail(store: Store, issue_id: str) -> dict:
    return detail(store.reader(), issue_id)


def issue_cases(store: Store, issue_id: str, role: str = "affected", after=None, limit=100) -> dict:
    if role not in ("affected", "regression"):
        raise IssueError("invalid", "role must be affected or regression")
    rows = store.reader().execute(
        "SELECT ic.case_id, c.case_key, ic.npu_arch, ic.pick_reason, ic.stale, ic.round_no, ic.result, ic.added_at "
        "FROM issue_cases ic JOIN cases c ON c.case_id=ic.case_id WHERE ic.issue_id=? AND ic.role=? AND ic.case_id>? "
        "ORDER BY ic.case_id LIMIT ?", (issue_id, role, after or 0, limit)).fetchall()
    items = [dict(r) for r in rows]
    return {"cases": items, "next_cursor": items[-1]["case_id"] if len(items) == limit else None}
