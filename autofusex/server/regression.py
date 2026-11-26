"""Regression case selection (design 2.1).

regression set = pinned cases (not counted in the budget) + four layers filled
in order with leftover quota rolling forward:
neighbor (shares features with the affected cases), sentinel (one per op
category x dtype), history (cases of resolved issues), stratified random.
Only cases whose current head on the issue backend and arch passes are taken
(pinned cases excepted). The random seed is derived from issue and round so a
preview and the frozen selection agree.
"""
from __future__ import annotations

import hashlib
import random

from autofusex.server.issues import _setting

RECENT_PER_VALUE = 3000
NEIGHBOR_WEIGHTS = {"op": 2, "op_category": 1, "dtype": 1}
LAYERS = ("neighbor", "sentinel", "history", "random")


def selection_seed(issue_id: str, round_no: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{issue_id}:{round_no}".encode()).digest()[:8], "big") >> 2


def select(conn, issue, round_no: int, budget: int | None = None):
    """Return [(case_id, npu_arch, reason)] for the regression round."""
    budget = budget if budget is not None else _setting(conn, "regression.budget")
    budget = min(budget, _setting(conn, "regression.max_budget"))
    shares = _setting(conn, "regression.shares")
    rng = random.Random(selection_seed(issue["issue_id"], round_no))
    arches = [r[0] for r in conn.execute(
        "SELECT DISTINCT npu_arch FROM issue_cases WHERE issue_id=? AND role='affected' AND stale=0",
        (issue["issue_id"],))]
    if not arches:
        return []
    affected = {r[0] for r in conn.execute(
        "SELECT case_id FROM issue_cases WHERE issue_id=? AND role='affected'", (issue["issue_id"],))}
    picked: dict = {}

    def passing_arch(case_id):
        for arch in arches:
            row = conn.execute(
                "SELECT e.function_status, e.accuracy_status FROM execution_heads h JOIN executions e "
                "ON e.exec_id=h.exec_id WHERE h.case_id=? AND h.npu_arch=? AND h.backend IN "
                "(SELECT backend FROM issue_cases WHERE issue_id=? AND role='affected' AND npu_arch=?) "
                "GROUP BY h.case_id HAVING MIN(e.function_status='pass' AND COALESCE(e.accuracy_status,'')<>'fail')=1 "
                "AND COUNT(DISTINCT h.backend)=(SELECT COUNT(DISTINCT backend) FROM issue_cases WHERE issue_id=? "
                "AND role='affected' AND npu_arch=?)",
                (case_id, arch, issue["issue_id"], arch, issue["issue_id"], arch)).fetchone()
            if row is not None and row[0] == "pass" and row[1] != "fail":
                return arch
        return None

    def take(case_ids, reason, quota, require_pass=True):
        taken = 0
        for case_id in case_ids:
            if taken >= quota:
                break
            if case_id in picked or case_id in affected:
                continue
            arch = passing_arch(case_id)
            if arch is None:
                if require_pass:
                    continue
                arch = arches[0]
            picked[case_id] = (arch, reason)
            taken += 1
        return taken

    for case_id in _pins(conn, issue["issue_id"], affected):
        take([case_id], "pinned", 1, require_pass=False)

    leftover = 0
    for layer in LAYERS:
        quota = budget * shares[layer] // 100 + leftover
        if layer == LAYERS[-1]:
            quota = budget - sum(1 for _, reason in picked.values() if reason != "pinned")
        candidates = {"neighbor": _neighbors, "sentinel": _sentinels, "history": _history,
                      "random": _random}[layer](conn, issue, affected, rng)
        leftover = quota - take(candidates, layer, max(0, quota))
    return [(case_id, arch, reason) for case_id, (arch, reason) in picked.items()]


def _pins(conn, issue_id, affected):
    feature_values = {f"{r[0]}={r[1]}" for r in conn.execute(
        "SELECT DISTINCT fr.name, v.value FROM issue_cases ic JOIN case_features f ON f.case_id=ic.case_id "
        "JOIN feature_registry fr ON fr.feature_id=f.feature_id JOIN feature_values v ON v.value_id=f.value_id "
        "WHERE ic.issue_id=? AND ic.role='affected'", (issue_id,))}
    rows = conn.execute("SELECT scope, scope_value, case_id FROM regression_pins WHERE revoked_at IS NULL AND "
                        "(scope='global' OR (scope='issue' AND scope_value=?) OR scope='feature') ORDER BY pin_id",
                        (issue_id,)).fetchall()
    return [r["case_id"] for r in rows if r["scope"] != "feature" or r["scope_value"] in feature_values]


def _recent(conn, value_id, limit=RECENT_PER_VALUE):
    return [r[0] for r in conn.execute(
        "SELECT case_id FROM case_features WHERE value_id=? ORDER BY case_id DESC LIMIT ?", (value_id, limit))]


def _neighbors(conn, issue, affected, rng):
    weights = {}
    for row in conn.execute(
            "SELECT f.value_id, fr.name, COUNT(*) FROM case_features f JOIN feature_registry fr USING(feature_id) "
            f"WHERE f.case_id IN ({','.join('?' * len(affected))}) AND fr.name IN ('op','op_category','dtype') "
            "GROUP BY 1, 2", tuple(affected)):
        weights[row[0]] = NEIGHBOR_WEIGHTS[row[1]]
    scores: dict = {}
    for value_id, weight in weights.items():
        for case_id in _recent(conn, value_id):
            scores[case_id] = scores.get(case_id, 0) + weight
    return sorted(scores, key=lambda c: (-scores[c], -c))


def _sentinels(conn, issue, affected, rng):
    """One recent case per op_category x dtype combination."""
    categories = _values(conn, "op_category")
    dtypes = {value_id: set(_recent(conn, value_id)) for value_id in _values(conn, "dtype")}
    out = []
    for category in categories:
        recent = _recent(conn, category)
        for members in dtypes.values():
            hit = next((c for c in recent if c in members), None)
            if hit is not None:
                out.append(hit)
    return out


def _values(conn, name):
    return [r[0] for r in conn.execute(
        "SELECT v.value_id FROM feature_values v JOIN feature_registry fr USING(feature_id) WHERE fr.name=? "
        "ORDER BY v.value_id", (name,))]


def _history(conn, issue, affected, rng):
    """Up to three representative cases per resolved issue, most recently resolved first."""
    per_issue_limit, seen, filtered = 3, {}, []
    for row in conn.execute(
            "SELECT ic.issue_id, ic.case_id FROM issues i JOIN issue_cases ic ON ic.issue_id=i.issue_id "
            "AND ic.role='affected' WHERE i.status='resolved' AND ic.backend IN (SELECT backend FROM issue_cases WHERE issue_id=?) AND i.issue_id<>? "
            "ORDER BY i.resolved_at DESC, ic.added_at LIMIT 3000", (issue["issue_id"], issue["issue_id"])):
        if seen.get(row[0], 0) < per_issue_limit:
            seen[row[0]] = seen.get(row[0], 0) + 1
            filtered.append(row[1])
    return filtered


def _random(conn, issue, affected, rng):
    """Round-robin over op categories, random order inside each category."""
    pools = []
    for category in _values(conn, "op_category"):
        pool = _recent(conn, category)
        rng.shuffle(pool)
        pools.append(pool)
    out = []
    while any(pools):
        for pool in pools:
            if pool:
                out.append(pool.pop())
    return out


def preview(store, issue_id: str) -> dict:
    conn = store.reader()
    issue = conn.execute("SELECT * FROM issues WHERE issue_id=?", (issue_id,)).fetchone()
    if issue is None:
        from autofusex.server.issues import IssueError
        raise IssueError("not_found", "issue not found", 404)
    round_no = issue["round_no"] + (2 if issue["status"] in ("open", "claimed") else 1)
    picks = select(conn, issue, round_no)
    keys = {r[0]: r[1] for r in conn.execute(
        f"SELECT case_id, case_key FROM cases WHERE case_id IN ({','.join('?' * len(picks))})",
        [p[0] for p in picks])} if picks else {}
    counts: dict = {}
    for _, _, reason in picks:
        counts[reason] = counts.get(reason, 0) + 1
    return {"issue_id": issue_id, "round_no": round_no, "budget": _setting(conn, "regression.budget"),
            "counts": counts,
            "cases": [{"case_key": keys[c], "npu_arch": a, "reason": r} for c, a, r in picks]}


# -- pins (admin) --------------------------------------------------------------
def list_pins(store, scope=None) -> list:
    where, args = "revoked_at IS NULL", ()
    if scope is not None:
        where += " AND p.scope=?"
        args = (scope,)
    return [dict(r) for r in store.reader().execute(
        "SELECT p.pin_id, p.scope, p.scope_value, c.case_key, p.added_by, p.reason, p.created_at "
        f"FROM regression_pins p JOIN cases c ON c.case_id=p.case_id WHERE {where} ORDER BY p.pin_id", args)]


def add_pins(store, principal, body: dict):
    from autofusex.server.issues import IssueError, _case_ids, idempotent
    from autofusex.server.store import now_ms
    principal.require_admin()
    scope, value, keys = body.get("scope"), body.get("scope_value", ""), body.get("case_keys")
    if scope not in ("global", "feature"):
        raise IssueError("invalid", "scope must be global or feature (issue pins go through the issue API)")
    if scope == "feature" and (not isinstance(value, str) or value.count("=") != 1 or len(value) > 128):
        raise IssueError("invalid", "feature scope_value must look like op_category=reduce")
    if not isinstance(keys, list) or not 1 <= len(keys) <= 200000 or not all(isinstance(k, str) for k in keys):
        raise IssueError("invalid", "case_keys must be a list of 1..200000 strings")
    reason = body.get("reason") if isinstance(body.get("reason"), str) else None

    def action(conn):
        ids = _case_ids(conn, keys)
        for case_id in ids:
            conn.execute("INSERT INTO regression_pins(scope, scope_value, case_id, added_by, reason, created_at) "
                         "VALUES(?,?,?,?,?,?)", (scope, value if scope == "feature" else "", case_id,
                                                 principal.user_id, (reason or "")[:500] or None, now_ms()))
        return {"pinned": len(ids)}
    return idempotent(store, principal, "pins-add", body.pop("request_key", None), body, action)


def revoke_pin(store, principal, pin_id: int, body: dict):
    from autofusex.server.issues import IssueError, idempotent
    from autofusex.server.store import now_ms
    principal.require_admin()

    def action(conn):
        if conn.execute("UPDATE regression_pins SET revoked_at=? WHERE pin_id=? AND revoked_at IS NULL",
                        (now_ms(), pin_id)).rowcount != 1:
            raise IssueError("not_found", "pin not found", 404)
        return {"pin_id": pin_id, "revoked": True}
    return idempotent(store, principal, f"pins-revoke:{pin_id}", body.pop("request_key", None), body, action)
