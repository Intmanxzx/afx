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

from autofusex.server.auth import Principal
from autofusex.server.store import Store, now_ms, get_content

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
    (re.compile(r"0x[0-9a-fA-F]+"), "<address>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[- T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)*"), "<time>"),
    (re.compile(r"graph=\S+"), "graph=<graph>"),
    (re.compile(r"\s+"), " "),
]


def normalize(text: str | None) -> str:
    """Normalize metadata only; preserve op names, dtypes, codes and values."""
    text = next((line.strip() for line in (text or "").splitlines() if line.strip()), "")
    text = re.sub(r"(\[ERROR\]\s+\w+)\([^)]*\):", r"\1:", text)
    text = re.sub(r":\d+\]\d*", "]", text)
    for pattern, replacement in _NORMALIZE:
        text = pattern.sub(replacement, text)
    return text.strip()


def _digest(*parts) -> bytes:
    return hashlib.sha256("\x1f".join(parts).encode()).digest()[:16]


# Bugs usually live in the hardest op of a graph; one primary category keeps clusters coarse enough.
CATEGORY_PRIORITY = ("Reduce", "Transpose", "Concat", "Split", "Gather", "Scatter",
                     "View", "Reshape", "Broadcast", "Elementwise")


def primary_category(categories) -> str:
    canonical = {name.lower(): name for name in CATEGORY_PRIORITY}
    found = {canonical.get(str(value).lower(), "Unknown") for value in categories or []}
    return next((name for name in CATEGORY_PRIORITY if name in found), "Unknown")


def case_traits(conn, case_id: int) -> dict:
    row = conn.execute("SELECT case_key, case_source, case_revision, op_types, dtypes, node_count FROM cases "
                       "WHERE case_id=?", (case_id,)).fetchone()
    categories = sorted(r[0] for r in conn.execute(
        "SELECT v.value FROM case_features f JOIN feature_registry r USING(feature_id) "
        "JOIN feature_values v ON v.value_id=f.value_id WHERE f.case_id=? AND r.name='op_category'", (case_id,)))
    return {"case_key": row["case_key"], "case_source": row["case_source"], "revision": row["case_revision"],
            "ops": row["op_types"] or "", "dtypes": row["dtypes"] or "", "node_count": row["node_count"] or 0,
            "categories": ",".join(categories) or "uncategorized", "primary": primary_category(categories)}


_PLOG_PREFIX = re.compile(
    r"^\[ERROR\]\s+\w+\([^)]*\):\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)*\s+\[[^\]\r\n]+:\d+\]\d*\s*", re.I)


def function_title(backend, stage, summary, primary):
    line = next((line.strip() for line in (summary or "").splitlines() if line.strip()), "")
    body = _PLOG_PREFIX.sub("", line).strip() or stage
    prefix, suffix = f"[{backend}] {stage}: ", f" ({primary})"
    return prefix + body[:max(0, 300 - len(prefix) - len(suffix))] + suffix


def _display_issue(row):
    out = dict(row)
    # Older rows store the normalized signature as their title. Rebuild from
    # the retained original summary so existing issues benefit without a migration.
    if out.get("cluster_generation", 1) == 1 and out["dimension"] == "function" and out["log_summary"]:
        category = re.search(r" \(([^()]*)\)$", out["title"])
        if category:
            out["title"] = function_title(out["backend"], out["category"],
                                          out["log_summary"], category[1])
    return out


def function_signature(execution, traits):
    status, stage = execution["function_status"], execution["fail_stage"] or "unknown"
    norm = normalize(execution["log_summary"]) or "No diagnostic evidence"
    attribution = status + ":" + stage
    # Graph composition is a filter trait, not part of a functional failure.
    sig = _digest("function", attribution, norm)
    return sig, attribution, f"{attribution}: {norm} ({traits['primary']})"[:300]


def accuracy_bucket(execution) -> str:
    rel, absolute = execution["max_rel_err"], execution["max_abs_err"]
    if rel is None and absolute is None:
        return "unknown"
    if (rel or 0) >= 1 or (absolute or 0) >= 1:
        return "severe"
    return "mild"


def accuracy_signature(execution, traits):
    primary = traits["primary"]
    return _digest("accuracy", primary), primary, f"精度失败 · {primary}"


def slowdown_bucket(ratio: float) -> str:
    if ratio >= 5:
        return ">5x"
    if ratio >= 2:
        return "2-5x"
    return "1.2-2x"


def performance_signature(backend, anchor, ratio, traits):
    primary = traits["primary"]
    return _digest("performance", primary), primary, f"性能问题 · {primary}"


def failure_dimension(execution) -> str | None:
    if execution["function_status"] not in ("pass", "unsupported", "invalid"):
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


def same_function_issue(conn, left, right):
    """Concurrent pre-upgrade repairs can retain IDs for the same new signature."""
    rows = conn.execute("SELECT category,log_summary FROM issues WHERE issue_id IN (?,?) "
                        "AND dimension='function' AND cluster_generation=2", (left, right)).fetchall()
    return len(rows) == 2 and rows[0]['category'] == rows[1]['category'] and (
        normalize(rows[0]['log_summary']) == normalize(rows[1]['log_summary']))


# -- clustering hook ------------------------------------------------------------
def _setting(conn, key):
    from autofusex.server.store import DEFAULT_SETTINGS
    row = conn.execute("SELECT value FROM platform_settings WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else DEFAULT_SETTINGS[key]


def _execution(conn, exec_id):
    return conn.execute(
        "SELECT exec_id, case_id, case_revision, backend, npu_arch, function_status, fail_stage, accuracy_status, "
        "accuracy_ref, max_abs_err, max_rel_err, kernel_ns, log_summary, created_at, detail_content_id, log_content_id FROM executions WHERE exec_id=?",
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
    heads = {row["backend"]: row for row in conn.execute(
        "SELECT e.* FROM execution_heads h JOIN executions e ON e.exec_id=h.exec_id "
        "WHERE h.case_id=? AND h.npu_arch=?", (case_id, execution["npu_arch"]))}
    for backend in (backends if execution["backend"] == anchor else [execution["backend"]]):
        target = heads.get(backend)
        if target is not None:
            reconcile(conn, target, heads.get(anchor), cfg=cfg)


def reconcile(conn, execution, anchor=None, *, cfg=None, source_id=None):
    """Classify once; record pass, fail or skip against the original issue."""
    cfg = cfg or config(conn)
    traits = case_traits(conn, execution["case_id"])
    if execution["case_revision"] != traits["revision"]:
        return None
    dimension = failure_dimension(execution)
    signature = None
    execution = dict(execution)
    if dimension == "function":
        from autofusex.server.error_logs import issue_summary
        detail = get_content(conn, execution['detail_content_id']) if execution.get('detail_content_id') else None
        log = get_content(conn, execution['log_content_id']) if execution.get('log_content_id') else None
        execution['log_summary'] = issue_summary(detail, log, execution['log_summary'], execution['function_status'])
    summary = execution["log_summary"]
    if dimension:
        signature = {"function": function_signature, "accuracy": accuracy_signature}[dimension](execution, traits)
    measured = anchor is not None and _timed(anchor) and _timed(execution)
    if dimension is None and measured:
        ratio = execution["kernel_ns"] / anchor["kernel_ns"]
        if ratio >= cfg["slowdown"]:
            dimension = "performance"
            signature = performance_signature(execution["backend"], cfg["anchor"], ratio, traits)
            summary = f"kernel {ratio:.2f}x slower than {cfg['anchor']}"
    target_id = None
    if signature:
        source = conn.execute("SELECT * FROM issues WHERE issue_id=?", (source_id,)).fetchone() if source_id else None
        # Frozen pre-upgrade verification keeps its original ID and membership.
        if (dimension == 'function' and source is not None and source['dimension'] == 'function'
                and source['status'] in ('verifying', 'regressing', 'summarizing')
                and source['category'] == signature[1]
                and normalize(source['log_summary']) == normalize(summary)):
            target_id = source_id
        else:
            target_id = _attach(conn, dimension, execution["backend"], *signature, summary, execution, traits)
    conn.execute("UPDATE executions SET issue_signature=? WHERE exec_id=?",
                 (signature[0] if signature else None, execution['exec_id']))
    if source_id is None:
        previous = conn.execute("SELECT issue_id FROM issue_current WHERE case_id=? AND backend=? AND npu_arch=?",
                                (execution['case_id'],execution['backend'],execution['npu_arch'])).fetchone()
        if target_id:
            conn.execute("INSERT INTO issue_current VALUES(?,?,?,?,?) ON CONFLICT(case_id,backend,npu_arch) "
                         "DO UPDATE SET issue_id=excluded.issue_id,exec_id=excluded.exec_id",
                         (execution['case_id'],execution['backend'],execution['npu_arch'],target_id,execution['exec_id']))
        elif execution['function_status'] != 'pass' or measured or not previous or conn.execute(
                "SELECT dimension FROM issues WHERE issue_id=?", (previous[0],)).fetchone()[0] != 'performance':
            conn.execute("DELETE FROM issue_current WHERE case_id=? AND backend=? AND npu_arch=?",
                         (execution['case_id'],execution['backend'],execution['npu_arch']))
        if previous and previous[0] != target_id:
            event(conn,previous[0],'system','current_membership_changed',None,None,
                  f"case {execution['case_id']} {execution['backend']} → {target_id or 'none'}")
    rows = conn.execute(
        "SELECT ic.issue_id, i.dimension, i.status FROM issue_cases ic JOIN issues i USING(issue_id) "
        "WHERE ic.case_id=? AND ic.npu_arch=? AND ic.backend=? AND ic.role='affected' "
        "AND ic.stale=0 AND COALESCE(ic.result,'')<>'skip'",
        (execution["case_id"], execution["npu_arch"], execution["backend"])).fetchall()
    for row in rows:
        if source_id is not None and row["issue_id"] != source_id:
            continue
        if source_id is None and row["status"] in ACTIVE:
            continue
        if target_id is None and (execution["function_status"] != "pass" or
                                  row["dimension"] == "performance" and not measured):
            continue
        same = row['issue_id'] == target_id or (source_id is not None and target_id is not None
                                              and same_function_issue(conn, row['issue_id'], target_id))
        result = "pass" if target_id is None else "fail" if same else "skip"
        reason = f"→ {target_id}" if result == "skip" else None
        conn.execute("UPDATE issue_cases SET result=?, pick_reason=?, last_exec_id=? WHERE issue_id=? "
                     "AND role='affected' AND case_id=? AND npu_arch=? AND backend=?",
                     (result, reason, execution["exec_id"], row["issue_id"], execution["case_id"], execution["npu_arch"], execution["backend"]))
        if result == "skip":
            event(conn, row["issue_id"], "system", "case_reclassified", None, None,
                  f"{traits['case_key']} ({execution['npu_arch']}) → {target_id}")
    return target_id


def _timed(row) -> bool:
    return (row["function_status"] == "pass" and row["accuracy_status"] != "fail"
            and row["kernel_ns"] is not None and row["kernel_ns"] > 0)


def _attach(conn, dimension, backend, sig, category, title, summary, execution, traits):
    now = now_ms()
    alias = conn.execute("SELECT issue_id FROM issue_signature_alias WHERE dimension=? AND backend=? AND signature=?",
                         (dimension, "", sig)).fetchone()
    if alias is not None:
        issue = conn.execute("SELECT * FROM issues WHERE issue_id=?", (alias[0],)).fetchone()
    else:
        issue = conn.execute("SELECT * FROM issues WHERE dimension=? AND cluster_generation=2 AND signature=?",
                             (dimension, sig)).fetchone()
    if issue is None:
        cur = conn.execute(
            "INSERT INTO issues(issue_id, dimension, backend, signature, category, title, log_summary, status, "
            "first_seen_at, last_seen_at, updated_at) VALUES('', ?,?,?,?,?,?, 'open', ?,?,?)",
            (dimension, backend, sig, category, title[:300], (summary or ""), now, now, now))
        issue_id = f"{DIMENSION_PREFIX[dimension]}{cur.lastrowid}"  # e.g. F254: one token, easy to copy
        conn.execute("UPDATE issues SET issue_id=?, primary_category=?, cluster_generation=2 WHERE issue_no=?", (issue_id, traits["primary"], cur.lastrowid))
        event(conn, issue_id, "system", "created", None, "open", title[:300])
        issue = conn.execute("SELECT * FROM issues WHERE issue_id=?", (issue_id,)).fetchone()
    issue_id = issue["issue_id"]
    new_case = conn.execute("SELECT 1 FROM issue_cases WHERE issue_id=? AND case_id=? AND role='affected' LIMIT 1",
                            (issue_id,execution['case_id'])).fetchone() is None
    inserted = conn.execute(
        "INSERT OR IGNORE INTO issue_cases(issue_id, case_id, role, npu_arch, case_revision, first_exec_id, "
        "last_exec_id, added_at, backend, result) VALUES(?,?, 'affected', ?,?,?,?,?,?, 'fail')",
        (issue_id, execution["case_id"], execution["npu_arch"], traits["revision"], execution["exec_id"],
         execution["exec_id"], now, backend)).rowcount
    if not inserted and issue["status"] not in ACTIVE:
        conn.execute("UPDATE issue_cases SET last_exec_id=?, case_revision=?, stale=0, result='fail', pick_reason=NULL WHERE issue_id=? "
                     "AND role='affected' AND case_id=? AND npu_arch=? AND backend=?",
                     (execution["exec_id"], traits["revision"], issue_id, execution["case_id"], execution["npu_arch"], backend))
    conn.execute("UPDATE issues SET case_count=case_count+?, last_seen_at=?, updated_at=? WHERE issue_id=?",
                 (int(new_case), now, now, issue_id))
    if issue["status"] == "resolved" and _reopens(conn, issue, execution):
        conn.execute("UPDATE issues SET status='open', owner_user=NULL, agent_id=NULL, lease_id=NULL, "
                     "lease_expires_at=NULL, updated_at=? WHERE issue_id=?", (now, issue_id))
        event(conn, issue_id, "system", "reopened", "resolved", "open",
              f"new failure exec {execution['exec_id']} after resolution")

    return issue_id


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
    conn.execute("DELETE FROM issue_current WHERE case_id=?", (case_id,))
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
    principal.require_write()
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
            if issue["treatment"] == "ignore" or issue["status"] in TERMINAL or issue["lease_id"] is not None:
                raise IssueError("unavailable", f"issue is {issue['status']} or held by someone else", 409)
        else:
            where, args = ["status='open'", "lease_id IS NULL", "treatment<>'ignore'", "EXISTS (SELECT 1 FROM issue_current cur WHERE cur.issue_id=issues.issue_id)"], []
            if dimension:
                where.append("dimension=?")
                args.append(dimension)
            if backend:
                where.append("EXISTS (SELECT 1 FROM issue_cases ic WHERE ic.issue_id=issues.issue_id AND ic.backend=?)")
                args.append(backend)
            issue = conn.execute(f"SELECT * FROM issues WHERE {' AND '.join(where)} "
                                 "ORDER BY CASE treatment WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END, case_count DESC, issue_no LIMIT 1", args).fetchone()
            if issue is None:
                return None
        lease_id = secrets.token_hex(16)
        now = now_ms()
        to_status = "claimed" if issue["status"] == "open" else issue["status"]
        _set_status(conn, issue, to_status, principal.user_id, "claimed", agent_id, owner_user=principal.user_id,
                    agent_id=agent_id, lease_id=lease_id, claimed_at=now,
                    lease_expires_at=now + _setting(conn, "issue.lease_days") * 86_400_000)
        return dict(detail(conn, issue["issue_id"], repro=True), lease_id=lease_id)
    return idempotent(store, principal, "claim", body.pop("request_key", None), body, action)


def renew(store: Store, principal: Principal, issue_id: str, body: dict):
    principal.require_write()

    def action(conn):
        issue = _holder(conn, principal, issue_id, body.get("lease_id"))
        expires = now_ms() + _setting(conn, "issue.lease_days") * 86_400_000
        conn.execute("UPDATE issues SET lease_expires_at=?, updated_at=? WHERE issue_id=?",
                     (expires, now_ms(), issue["issue_id"]))
        return {"issue_id": issue_id, "lease_expires_at": expires}
    return idempotent(store, principal, f"renew:{issue_id}", body.pop("request_key", None), body, action)


def release(store: Store, principal: Principal, issue_id: str, body: dict):
    principal.require_write()

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
    principal.require_write()
    version, sha = body.get("fix_version"), body.get("fix_sha256")
    if not isinstance(version, str) or not _VERSION.match(version):
        raise IssueError("invalid", "fix_version must be an exact package version")
    if not isinstance(sha, str) or not _SHA.match(sha):
        raise IssueError("invalid", "fix_sha256 must be 64 lowercase hex chars")
    fix_ref = body.get("fix_ref")
    if fix_ref is not None and (not isinstance(fix_ref, str) or len(fix_ref) > 512):
        raise IssueError("invalid", "fix_ref must be a string of at most 512 chars")

    def action(conn):
        from autofusex.server import retests
        issue = _holder(conn, principal, issue_id, body.get("lease_id"))
        if issue["status"] != "claimed":
            raise IssueError("wrong_state", f"verify needs status claimed, issue is {issue['status']}", 409)
        cases = conn.execute(
            "SELECT case_id,npu_arch,backend FROM issue_current WHERE issue_id=? "
            "ORDER BY exec_id DESC LIMIT ?", (issue_id, _setting(conn, "verify.max_cases"))).fetchall()
        if not cases:
            raise IssueError("no_cases", "issue has no current affected cases to verify", 409)
        round_no = issue["round_no"] + 1
        conn.execute("INSERT INTO issue_rounds(issue_id, round_no, kind, fix_version, fix_sha256, started_at) "
                     "VALUES(?,?, 'verify', ?,?,?)", (issue_id, round_no, version, sha, now_ms()))
        tasks = retests.create_tasks(conn, issue, round_no, [(c["case_id"], c["npu_arch"], c["backend"]) for c in cases],
                                     version, sha)
        _set_status(conn, issue, "verifying", principal.user_id, "verify_submitted",
                    f"round {round_no} version {version} tasks {tasks}", round_no=round_no, fix_version=version,
                    fix_sha256=sha, fix_ref=fix_ref or issue["fix_ref"])
        return {"issue_id": issue_id, "round_no": round_no, "tasks": tasks, "status": "verifying"}
    return idempotent(store, principal, f"verify:{issue_id}", body.pop("request_key", None), body, action)


def pin_cases(store: Store, principal: Principal, issue_id: str, body: dict):
    """Lease holder (or admin) names must-run regression cases before regression starts (design 2.1)."""
    principal.require_write()
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
    principal.require_write()
    from autofusex.server import knowledge
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
    principal.require_admin()
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
    principal.require_admin()
    target_id = body.get("into")

    def action(conn):
        source = conn.execute("SELECT * FROM issues WHERE issue_id=?", (issue_id,)).fetchone()
        target = conn.execute("SELECT * FROM issues WHERE issue_id=?", (target_id,)).fetchone()
        if source is None or target is None or source["issue_id"] == target["issue_id"]:
            raise IssueError("invalid", "source and target must be two existing issues", 400)
        if source["dimension"] != target["dimension"]:
            raise IssueError("invalid", "only issues of the same dimension can merge", 400)
        if source["status"] in ACTIVE or target["status"] in TERMINAL:
            raise IssueError("wrong_state", "source must not be in progress and target must be open", 409)
        moved = conn.execute(
            "INSERT OR IGNORE INTO issue_cases(issue_id, case_id, role, npu_arch, case_revision, stale, first_exec_id, "
            "last_exec_id, added_at, backend) SELECT ?, case_id, role, npu_arch, case_revision, stale, first_exec_id, "
            "last_exec_id, added_at, backend FROM issue_cases WHERE issue_id=? AND role='affected'",
            (target_id, issue_id)).rowcount
        conn.execute("INSERT OR REPLACE INTO issue_signature_alias VALUES(?,?,?,?)",
                     (source["dimension"], "", source["signature"], target_id))
        conn.execute("UPDATE issue_signature_alias SET issue_id=? WHERE issue_id=?", (target_id, issue_id))
        conn.execute("UPDATE issue_current SET issue_id=? WHERE issue_id=?", (target_id,issue_id))
        conn.execute("UPDATE issues SET case_count=(SELECT COUNT(DISTINCT case_id) FROM issue_cases WHERE issue_id=? "
                     "AND role='affected'), updated_at=? WHERE issue_id=?", (target_id,now_ms(),target_id))
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
                  "updated_at, treatment, primary_category, cluster_generation")


_ISSUE_RANK = "CASE WHEN status IN ('claimed','verifying','regressing','summarizing') THEN 0 WHEN status='open' THEN 1 ELSE 2 END"
_ISSUE_ORDER = f"{_ISSUE_RANK} ASC, case_count DESC, issue_no DESC"


def affected_backends(conn, issue_id):
    return [r[0] for r in conn.execute("SELECT DISTINCT backend FROM issue_cases WHERE issue_id=? "
            "AND role='affected' ORDER BY backend", (issue_id,))]


def issue_classifications(conn, issue_id):
    # Current membership carries the same attribution shown in discovery. Older
    # issues may retain a stage-only category, so never parse it as an execution.
    rows = conn.execute(
        "SELECT DISTINCT e.function_status,e.fail_stage,e.accuracy_status,e.accuracy_ref,e.invalid_reason "
        "FROM issue_current ic JOIN executions e ON e.exec_id=ic.exec_id WHERE ic.issue_id=? "
        "ORDER BY e.function_status,e.fail_stage", (issue_id,)).fetchall()
    if not rows:
        rows = conn.execute(
            "SELECT DISTINCT e.function_status,e.fail_stage,e.accuracy_status,e.accuracy_ref,e.invalid_reason "
            "FROM issue_cases ic JOIN executions e ON e.exec_id=ic.last_exec_id "
            "WHERE ic.issue_id=? AND ic.role='affected' ORDER BY e.function_status,e.fail_stage",
            (issue_id,)).fetchall()
    return [dict(row) for row in rows]


def set_treatment(store, principal, issue_id, body):
    principal.require_admin()
    treatment, reason = body.get("treatment"), body.get("reason", "")
    if treatment not in {"ignore", "low", "normal", "high"}:
        raise IssueError("invalid", "invalid treatment")
    if not isinstance(reason, str) or len(reason) > 500:
        raise IssueError("invalid", "reason must be at most 500 characters")
    def action(conn):
        row = conn.execute("SELECT * FROM issues WHERE issue_id=?", (issue_id,)).fetchone()
        if row is None:
            raise IssueError("not_found", "issue not found", 404)
        # Ignoring explicitly withdraws work; stale workers cannot complete it.
        if treatment == "ignore" and row["status"] in ACTIVE:
            _cancel_tasks(conn, issue_id)
            conn.execute("UPDATE issue_rounds SET outcome='invalid',finished_at=? WHERE issue_id=? "
                         "AND finished_at IS NULL", (now_ms(), issue_id))
            _set_status(conn, row, "open", principal.user_id, "work_withdrawn", reason,
                        owner_user=None, agent_id=None, lease_id=None, lease_expires_at=None)
        conn.execute("UPDATE issues SET treatment=?,updated_at=? WHERE issue_id=?",
                     (treatment, now_ms(), issue_id))
        event(conn, issue_id, principal.user_id, "treatment_changed", None, None,
              json.dumps(dict(before=row["treatment"], after=treatment, reason=reason), ensure_ascii=False))
        return dict(issue_id=issue_id, treatment=treatment)
    return idempotent(store, principal, f"treatment:{issue_id}", body.pop("request_key", None), body, action)


def list_issues(store, *, dimension=None, backend=None, status=None, q=None, after=None,
                limit=50, page=None, results=None, selected_backends=None, chips=None,
                categories=None, treatment=None, scope="all", case_q=None):
    from autofusex.server import search
    if not 1 <= limit <= 200 or (page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1)):
        raise IssueError("invalid", "invalid page or limit")
    if page is not None and after is not None:
        raise IssueError("invalid", "page and after cannot be combined")
    if scope not in {"all", "current"}:
        raise IssueError("invalid", "scope must be all or current")
    query = search.visual_query(case_q, results, selected_backends or backend, chips)
    hs, ha = query.head_sql() if query else ("1", [])
    cs, ca = query.case_sql() if query else ("1", [])
    membership = ("FROM issue_cases ic JOIN cases c ON c.case_id=ic.case_id "
                  "JOIN executions e ON e.exec_id=ic.last_exec_id ")
    match = "ic.issue_id=issues.issue_id AND ic.role='affected'"
    if scope == "current":
        membership = ("FROM issue_current ic JOIN cases c ON c.case_id=ic.case_id "
                      "JOIN executions e ON e.exec_id=ic.exec_id ")
        match = "ic.issue_id=issues.issue_id"
    match += f" AND ({hs}) AND ({cs})"
    match_args = [*ha, *ca]
    where, args = [], []
    for column, value in (("dimension", dimension), ("status", status), ("treatment", treatment),
                          ("primary_category", categories)):
        if value:
            values = value.split(",")
            if len(values) > 20:
                raise IssueError("invalid", "too many filter values")
            where.append(f"issues.{column} IN ({','.join('?' for _ in values)})")
            args.extend(values)
    if q:
        if len(q) > 128:
            raise IssueError("invalid", "q must be at most 128 characters")
        where.append("(issue_id LIKE ? OR title LIKE ? OR log_summary LIKE ?)")
        args.extend(["%" + q + "%"] * 3)
    if query or scope == "current":
        where.append(f"EXISTS (SELECT 1 {membership} WHERE {match})")
        args.extend(match_args)
    offset = 0
    if after is not None:
        try:
            offset = int(after)
            if offset < 0: raise ValueError()
        except (ValueError, TypeError):
            raise IssueError("invalid", "after must be a nonnegative offset")
    clause = "WHERE " + " AND ".join(where) if where else ""
    conn = store.reader()
    conn.execute("BEGIN")
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM issues {clause}", args).fetchone()[0]
        pages = (total + limit - 1) // limit
        if page is not None:
            page = min(page, max(1, pages)); offset = (page - 1) * limit
        rows = conn.execute(f"SELECT {_ISSUE_COLUMNS}, "
            f"(SELECT COUNT(DISTINCT ic.case_id) {membership} WHERE {match}) AS matched_case_count "
            f"FROM issues {clause} ORDER BY {_ISSUE_RANK}, "
            "CASE treatment WHEN 'high' THEN 0 WHEN 'normal' THEN 1 WHEN 'low' THEN 2 ELSE 3 END, "
            "case_count DESC, issue_no DESC LIMIT ? OFFSET ?", [*match_args, *args, limit, offset]).fetchall()
        items = [_display_issue(row) for row in rows]
        for item in items:
            item["backends"] = affected_backends(conn, item["issue_id"])
            item["classifications"] = issue_classifications(conn, item["issue_id"])
            item["current_case_count"] = conn.execute("SELECT COUNT(DISTINCT case_id) FROM issue_current WHERE issue_id=?", (item["issue_id"],)).fetchone()[0]
    finally:
        conn.execute("ROLLBACK")
    return dict(issues=items, page=page or offset // limit + 1, page_size=limit,
                total=total, total_pages=pages, next_cursor=str(offset + limit) if offset + limit < total else None)


def detail(conn, issue_id: str, *, repro: bool = False) -> dict:
    row = conn.execute(f"SELECT {_ISSUE_COLUMNS} FROM issues WHERE issue_id=?", (issue_id,)).fetchone()
    if row is None:
        raise IssueError("not_found", "issue not found", 404)
    out = _display_issue(row)
    out["backends"] = affected_backends(conn, issue_id)
    out["classifications"] = issue_classifications(conn, issue_id)
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
        from autofusex.server import knowledge
        out["related_knowledge"] = knowledge.related(conn, out)
    return out


def _repro(conn, issue_id, backend):
    rows = conn.execute(
        "SELECT c.case_key, c.case_source, ic.npu_arch, ic.backend, e.exec_uuid FROM issue_cases ic "
        "JOIN cases c ON c.case_id=ic.case_id LEFT JOIN executions e ON e.exec_id=ic.last_exec_id "
        "WHERE ic.issue_id=? AND ic.role='affected' AND ic.stale=0 AND COALESCE(ic.result,'')<>'skip' ORDER BY ic.added_at DESC LIMIT ?",
        (issue_id, REPRO_LIMIT)).fetchall()
    cases = []
    for row in rows:
        command = (f"afx fuzz replay {row['case_key']} --backend {row['backend']}" if row["case_source"] == "nnsmith"
                   else None)
        cases.append({"case_key": row["case_key"], "case_source": row["case_source"], "npu_arch": row["npu_arch"],
                      "backend": row["backend"], "exec_uuid": row["exec_uuid"], "command": command})
    return {"cases": cases, "truncated": len(rows) == REPRO_LIMIT}


def issue_detail(store: Store, issue_id: str) -> dict:
    return detail(store.reader(), issue_id)


def issue_cases(store, issue_id, role="affected", after=None, limit=100):
    if role not in {"affected", "regression"}:
        raise IssueError("invalid", "invalid role")
    offset = int(after or 0)
    rows = store.reader().execute(
        "SELECT ic.case_id,c.case_key,ic.npu_arch,ic.backend,ic.pick_reason,ic.stale,ic.round_no,ic.result,ic.added_at, "
        "e.function_status,e.fail_stage,e.accuracy_status,e.accuracy_ref,e.invalid_reason "
        "FROM issue_cases ic JOIN cases c USING(case_id) LEFT JOIN executions e ON e.exec_id=ic.last_exec_id "
        "WHERE ic.issue_id=? AND ic.role=? "
        "ORDER BY ic.case_id,ic.npu_arch,ic.backend LIMIT ? OFFSET ?", (issue_id, role, limit, offset)).fetchall()
    return dict(cases=[dict(r) for r in rows], next_cursor=offset+limit if len(rows)==limit else None)


def refresh_classified_executions(conn, exec_ids):
    """Reconcile only changed current heads, in the caller's transaction."""
    cfg = config(conn)
    for exec_id in set(exec_ids):
        head = conn.execute("SELECT case_id FROM execution_heads WHERE exec_id=?", (exec_id,)).fetchone()
        if head:
            on_head(conn, head[0], exec_id, cfg)
