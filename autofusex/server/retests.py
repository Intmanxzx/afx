"""Verification and regression retest tasks pulled by clients (design 2).

A task is one case on one arch for one round. Clients declare the candidate
package they have installed; only matching tasks are handed out. Results are
uploaded as normal executions (run kind verify/regress, never current heads)
and bound to the task on completion. When every task of a round is terminal
the round is evaluated and the issue moves on.
"""
from __future__ import annotations

import json
import re
import secrets

from autofusex.server.auth import Principal
from autofusex.server.issues import IssueError, _setting, event, idempotent
from autofusex.server.store import Store, get_content, now_ms

MAX_CLAIM = 20
# Failures caused by the test machine rather than the candidate package never count against a fix.
_INFRA = re.compile(r"device busy|owns this physical NPU|device lease|out of memory|ACL_ERROR_RT_MEMORY|"
                    r"No space left on device|Connection (?:reset|refused)|Too many open files", re.I)
MAX_ATTEMPTS = 3


def create_tasks(conn, issue, round_no: int, targets, version: str, sha: str, role: str = "affected") -> int:
    now, count = now_ms(), 0
    for target in targets:
        case_id, arch = target[:2]
        candidates = [target[2]] if len(target) > 2 else [r[0] for r in conn.execute(
            "SELECT DISTINCT backend FROM issue_cases WHERE issue_id=? AND role='affected' AND npu_arch=?",
            (issue["issue_id"], arch))]
        for backend in candidates:
            backends = list(dict.fromkeys(["cpu", backend] + (
                [_setting(conn, "issue.perf_anchor")] if issue["dimension"] == "performance" else [])))
            conn.execute(
                "INSERT INTO retest_tasks(issue_id,round_no,case_id,npu_arch,backends,required_backend,"
                "required_version,required_sha256,status,created_at) VALUES(?,?,?,?,?,?,?,?,'pending',?)",
                (issue["issue_id"],round_no,case_id,arch,",".join(backends),backend,version,sha,now))
            conn.execute("UPDATE retest_tasks SET case_revision=(SELECT case_revision FROM cases WHERE case_id=?) "
                         "WHERE task_id=last_insert_rowid()", (case_id,))
            if role == "regression":
                conn.execute("INSERT OR IGNORE INTO issue_cases(issue_id,case_id,role,npu_arch,backend,case_revision,added_at) "
                             "SELECT ?,case_id,'regression',?,?,case_revision,? FROM cases WHERE case_id=?",
                             (issue["issue_id"],arch,backend,now,case_id))
            conn.execute("UPDATE issue_cases SET round_no=?,result=NULL WHERE issue_id=? AND role=? "
                         "AND case_id=? AND npu_arch=? AND backend=?",
                         (round_no,issue["issue_id"],role,case_id,arch,backend))
            count += 1
    return count


def _expire(conn) -> None:
    now = now_ms()
    conn.execute("UPDATE retest_tasks SET status=CASE WHEN attempts>=? THEN 'invalid' ELSE 'pending' END, "
                 "result=CASE WHEN attempts>=? THEN 'invalid' ELSE result END, "
                 "detail=CASE WHEN attempts>=? THEN 'lease expired too often' ELSE detail END, "
                 "client_id=NULL, lease_id=NULL, lease_expires_at=NULL "
                 "WHERE status='leased' AND lease_expires_at<=?", (MAX_ATTEMPTS,) * 3 + (now,))
    # Rounds whose last task just became invalid need evaluating too.
    for row in conn.execute("SELECT DISTINCT t.issue_id, t.round_no FROM retest_tasks t JOIN issue_rounds r "
                            "ON r.issue_id=t.issue_id AND r.round_no=t.round_no WHERE r.finished_at IS NULL "
                            "AND t.status='invalid'").fetchall():
        _maybe_finish(conn, row["issue_id"], row["round_no"])


_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,31}$")
_VERSION = re.compile(r"^[A-Za-z0-9_.+-]{1,64}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


def _installed(body) -> dict:
    installed = body.get("installed")
    if not isinstance(installed, dict) or not 1 <= len(installed) <= 16:
        raise IssueError("invalid", "installed must map backend -> {version, sha256}")
    out = {}
    for backend, info in installed.items():
        if not isinstance(backend, str) or not _NAME.match(backend) or not isinstance(info, dict):
            raise IssueError("invalid", "installed entries must be backend -> {version, sha256}")
        version, sha = info.get("version"), info.get("sha256")
        if not isinstance(version, str) or not _VERSION.match(version) or not isinstance(sha, str) \
                or not _SHA.match(sha):
            raise IssueError("invalid", f"installed.{backend} needs an exact version and sha256")
        out[backend] = (version, sha)
    return out


def claim(store: Store, principal: Principal, body: dict):
    principal.require_write()
    machine_id, arch = body.get("machine_id"), body.get("npu_arch")
    if not isinstance(machine_id, str) or not 1 <= len(machine_id) <= 128:
        raise IssueError("invalid", "machine_id is required")
    if not isinstance(arch, str) or not 1 <= len(arch) <= 32:
        raise IssueError("invalid", "npu_arch is required")
    installed = _installed(body)
    limit = body.get("limit", 5)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_CLAIM:
        raise IssueError("invalid", f"limit must be 1..{MAX_CLAIM}")

    def action(conn):
        from autofusex.server.ingest import upsert_client
        _expire(conn)
        client_id = upsert_client(conn, principal.user_id, {"machine_id": machine_id, "npu_arch": arch})
        lease_ms = _setting(conn, "retest.lease_minutes") * 60_000
        tasks = []
        for backend, (version, sha) in installed.items():
            # Manual retests belong to a user, independently of client identity.
            # Issue rounds retain their exact candidate-package matching.
            rows = conn.execute(
                "SELECT t.*, COALESCE(r.kind, t.task_kind) AS kind FROM retest_tasks t LEFT JOIN issue_rounds r "
                "ON r.issue_id=t.issue_id AND r.round_no=t.round_no WHERE t.status='pending' AND t.npu_arch=? "
                "AND t.required_backend=? AND ((t.required_version=? AND t.required_sha256=?) "
                "OR t.required_version='*') AND (t.issue_id<>'' OR t.assigned_user=?) "
                "ORDER BY t.task_id LIMIT ?",
                (arch, backend, version, sha, principal.user_id, limit - len(tasks))).fetchall()
            for row in rows:
                lease_id = secrets.token_hex(16)
                conn.execute("UPDATE retest_tasks SET status='leased', client_id=?, lease_id=?, lease_expires_at=?, "
                             "attempts=attempts+1 WHERE task_id=?",
                             (client_id, lease_id, now_ms() + lease_ms, row["task_id"]))
                tasks.append(_task_payload(conn, row, lease_id))
            if len(tasks) >= limit:
                break
        return {"tasks": tasks}
    return idempotent(store, principal, "retest-claim", body.pop("request_key", None), body, action)


def _task_payload(conn, row, lease_id):
    case = conn.execute("SELECT case_key, case_source, case_revision, spec_content_id FROM cases WHERE case_id=?",
                        (row["case_id"],)).fetchone()
    spec = None
    if case["case_source"] != "nnsmith" and case["spec_content_id"] is not None:
        data = get_content(conn, case["spec_content_id"])
        spec = json.loads(data) if data is not None else None
    return {"task_id": row["task_id"], "lease_id": lease_id, "issue_id": row["issue_id"], "round_no": row["round_no"],
            "run_kind": row["kind"],
            "run_uuid": (f"{row['issue_id']}-r{row['round_no']}-t{row['task_id']}" if row["issue_id"]
                         else f"retest-t{row['task_id']}"),
            "case_key": case["case_key"], "case_source": case["case_source"], "case_revision": case["case_revision"],
            "spec": spec, "npu_arch": row["npu_arch"], "backends": row["backends"].split(","),
            "required": {"backend": row["required_backend"], "version": row["required_version"],
                         "sha256": row["required_sha256"]}}


def complete(store: Store, principal: Principal, task_id: int, body: dict):
    principal.require_write()
    terminal_reason = body.get("terminal_reason")
    if terminal_reason is not None and (not isinstance(terminal_reason, str) or not 1 <= len(terminal_reason) <= 500):
        raise IssueError("invalid", "terminal_reason must be 1..500 characters")
    exec_uuids = body.get("exec_uuids")
    if not isinstance(exec_uuids, list) or not (0 if terminal_reason else 1) <= len(exec_uuids) <= 8 \
            or not all(isinstance(u, str) and len(u) <= 64 for u in exec_uuids):
        raise IssueError("invalid", "exec_uuids must list 1..8 uploaded execution ids")
    installed = _installed(body)

    def action(conn):
        task = conn.execute("SELECT t.*, COALESCE(r.kind, t.task_kind) AS kind FROM retest_tasks t "
                            "LEFT JOIN issue_rounds r ON r.issue_id=t.issue_id AND r.round_no=t.round_no "
                            "WHERE t.task_id=?", (task_id,)).fetchone()
        if task is None:
            raise IssueError("not_found", "task not found", 404)
        client = conn.execute("SELECT user_id FROM clients WHERE client_id=?", (task["client_id"],)).fetchone()
        if task["status"] != "leased" or task["lease_id"] != body.get("lease_id") or client is None \
                or client["user_id"] != principal.user_id:
            raise IssueError("lease_lost", "task lease is not held by this caller", 409)
        if terminal_reason:
            if exec_uuids:
                raise IssueError("invalid", "terminal_reason cannot accompany executions")
            conn.execute("UPDATE retest_tasks SET status='invalid',result='invalid',detail=?,finished_at=?, "
                         "lease_id=NULL,lease_expires_at=NULL WHERE task_id=?",
                         (terminal_reason, now_ms(), task_id))
            if terminal_reason == 'no_autofuse_coverage' and task['required_backend'] in {'af-mlir', 'af-legacy'} and task['case_revision'] is not None:
                conn.execute("INSERT OR REPLACE INTO case_coverage VALUES(?,?,?,?,?,?,?,?)",
                             (task['case_id'], task['case_revision'], task['required_backend'], task['npu_arch'],
                              terminal_reason, task_id, json.dumps(installed, sort_keys=True), now_ms()))
            if task["issue_id"]:
                _maybe_finish(conn, task["issue_id"], task["round_no"])
            return {"task_id": task_id, "result": "invalid", "detail": terminal_reason}
        rows = []
        for exec_uuid in dict.fromkeys(exec_uuids):
            row = conn.execute("SELECT * FROM executions WHERE exec_uuid=?", (exec_uuid,)).fetchone()
            if row is None or row["user_id"] != principal.user_id:
                raise IssueError("not_found", f"execution {exec_uuid[:64]} not uploaded by this user", 404)
            if (row["case_id"], row["run_kind"]) != (task["case_id"], task["kind"]) or \
                    row["backend"] not in task["backends"].split(",") or \
                    (row["backend"] != "cpu" and row["npu_arch"] != task["npu_arch"]):
                raise IssueError("mismatch", f"execution {exec_uuid[:64]} does not belong to this task", 409)
            expected_run = (f"{task['issue_id']}-r{task['round_no']}-t{task_id}" if task["issue_id"]
                            else f"retest-t{task_id}")
            run = conn.execute("SELECT run_uuid FROM runs WHERE run_id=?", (row["run_id"],)).fetchone()[0]
            if run != expected_run:
                raise IssueError("mismatch", "execution belongs to another retest run", 409)
            rows.append(row)
        current_revision = conn.execute("SELECT case_revision FROM cases WHERE case_id=?", (task["case_id"],)).fetchone()[0]
        if any(r["case_revision"] != current_revision for r in rows):
            raise IssueError("mismatch", "execution uses an obsolete case revision", 409)
        result, detail = _judge(conn, task, rows, installed)
        if result != 'invalid':
            conn.execute("DELETE FROM case_coverage WHERE case_id=? AND case_revision=? AND backend=? AND npu_arch=?",
                         (task['case_id'], current_revision, task['required_backend'], task['npu_arch']))
        target = next((r for r in rows if r["backend"] == task["required_backend"]), None)
        if result != "invalid" and target is not None:
            from autofusex.server.issues import reconcile
            anchor = next((r for r in rows if r["backend"] == _setting(conn, "issue.perf_anchor")), None)
            destination = reconcile(conn, target, anchor, source_id=task["issue_id"])
            if task["issue_id"] and destination and destination != task["issue_id"]:
                result, detail = "skip", f"→ {destination}"

        conn.execute("UPDATE retest_tasks SET status=?, result=?, detail=?, finished_at=?, lease_id=NULL, "
                     "lease_expires_at=NULL WHERE task_id=?",
                     ("invalid" if result == "invalid" else "done", result, detail, now_ms(), task_id))
        conn.executemany("INSERT OR IGNORE INTO retest_task_execs VALUES(?,?)", [(task_id, r["exec_id"]) for r in rows])
        if not task["issue_id"]:
            return {"task_id": task_id, "result": result, "detail": detail}
        role = "affected" if task["kind"] == "verify" else "regression"
        target = next((r for r in rows if r["backend"] == task["required_backend"]), None)
        conn.execute("UPDATE issue_cases SET result=?, last_exec_id=COALESCE(?, last_exec_id) WHERE issue_id=? "
                     "AND role=? AND case_id=? AND npu_arch=? AND backend=?",
                     (result, target["exec_id"] if target else None, task["issue_id"], role, task["case_id"],
                      task["npu_arch"], task["required_backend"]))
        _maybe_finish(conn, task["issue_id"], task["round_no"])
        return {"task_id": task_id, "result": result, "detail": detail}
    return idempotent(store, principal, f"retest-complete:{task_id}", body.pop("request_key", None), body, action)


def _judge(conn, task, rows, installed):
    """pass / fail / invalid for one task. Invalid never counts as a failure of the fix."""
    required = (task["required_version"], task["required_sha256"])
    if task["required_version"] != "*" and installed.get(task["required_backend"]) != required:
        return "invalid", "installed package does not match the version under test"
    by_backend = {r["backend"]: r for r in rows}
    target = by_backend.get(task["required_backend"])
    if target is None:
        return "invalid", f"no {task['required_backend']} result"
    if target["function_status"] != "pass" and _INFRA.search(target["log_summary"] or ""):
        return "invalid", f"environment problem, not the fix: {(target['log_summary'] or '')[:160]}"
    ok = target["function_status"] == "pass" and target["accuracy_status"] != "fail"
    issue = conn.execute("SELECT dimension FROM issues WHERE issue_id=?", (task["issue_id"],)).fetchone()
    if not ok:
        return "fail", f"{target['function_status']}/{target['accuracy_status']}"
    if issue is not None and issue["dimension"] == "performance" and task["kind"] == "verify":
        anchor = by_backend.get(_setting(conn, "issue.perf_anchor"))
        if anchor is None or not anchor["kernel_ns"] or not target["kernel_ns"]:
            return "invalid", "performance verification needs kernel time of both backends"
        ratio = target["kernel_ns"] / anchor["kernel_ns"]
        if ratio >= _setting(conn, "issue.perf_slowdown"):
            return "fail", f"still {ratio:.2f}x slower"
        return "pass", f"{ratio:.2f}x"
    return "pass", None


def _maybe_finish(conn, issue_id, round_no) -> None:
    counts = {row[0]: row[1] for row in conn.execute(
        "SELECT COALESCE(result, status), COUNT(*) FROM retest_tasks WHERE issue_id=? AND round_no=? "
        "AND status<>'cancelled' GROUP BY 1", (issue_id, round_no))}
    if counts.get("pending") or counts.get("leased"):
        return
    rnd = conn.execute("SELECT * FROM issue_rounds WHERE issue_id=? AND round_no=?", (issue_id, round_no)).fetchone()
    if rnd is None or rnd["finished_at"] is not None:
        return
    passed, failed, invalid = counts.get("pass", 0), counts.get("fail", 0), counts.get("invalid", 0)
    skipped = counts.get("skip", 0)
    outcome = "failed" if failed else ("invalid" if invalid else "passed")
    conn.execute("UPDATE issue_rounds SET finished_at=?, outcome=?, pass_count=?, fail_count=?, invalid_count=? "
                 "WHERE issue_id=? AND round_no=?", (now_ms(), outcome, passed, failed, invalid, issue_id, round_no))
    issue = conn.execute("SELECT * FROM issues WHERE issue_id=?", (issue_id,)).fetchone()
    detail = f"round {round_no} {rnd['kind']}: {passed} pass, {failed} fail, {invalid} invalid, {skipped} skip"
    if outcome != "passed":
        # Back to the developer; without a live lease the issue returns to the pool.
        to_status = "claimed" if issue["lease_id"] else "open"
        _move(conn, issue, to_status, f"{rnd['kind']}_{outcome}", detail)
        return
    if rnd["kind"] == "regress":
        _move(conn, issue, "summarizing", "regress_passed", detail)
        return
    from autofusex.server import regression
    picks = regression.select(conn, issue, round_no + 1)
    if not picks:
        _move(conn, issue, "summarizing", "verify_passed", detail + "; no regression candidates")
        return
    conn.execute("INSERT INTO issue_rounds(issue_id, round_no, kind, fix_version, fix_sha256, selection_seed, "
                 "started_at) VALUES(?,?, 'regress', ?,?,?,?)",
                 (issue_id, round_no + 1, rnd["fix_version"], rnd["fix_sha256"],
                  regression.selection_seed(issue_id, round_no + 1), now_ms()))
    create_tasks(conn, issue, round_no + 1, [(c, a) for c, a, _ in picks], rnd["fix_version"], rnd["fix_sha256"],
                 role="regression")
    for case_id, arch, reason in picks:
        conn.execute("UPDATE issue_cases SET pick_reason=? WHERE issue_id=? AND role='regression' AND case_id=? AND npu_arch=?",
                     (reason,issue_id,case_id,arch))
    conn.execute("UPDATE issues SET round_no=? WHERE issue_id=?", (round_no + 1, issue_id))
    _move(conn, issue, "regressing", "verify_passed", f"{detail}; regression round {round_no + 1} "
                                                      f"with {len(picks)} cases")


def _move(conn, issue, to_status, kind, detail):
    conn.execute("UPDATE issues SET status=?, updated_at=? WHERE issue_id=?", (to_status, now_ms(), issue["issue_id"]))
    event(conn, issue["issue_id"], "system", kind, issue["status"], to_status, detail)


def round_tasks(store: Store, issue_id: str, round_no: int) -> list:
    return [dict(row) for row in store.reader().execute(
        "SELECT t.task_id, c.case_key, t.npu_arch, t.required_backend AS backend, t.status, t.result, t.detail, t.attempts, t.finished_at "
        "FROM retest_tasks t JOIN cases c ON c.case_id=t.case_id WHERE t.issue_id=? AND t.round_no=? "
        "ORDER BY t.task_id", (issue_id, round_no))]


# -- manual retests from the discovery page (design 6.3) -------------------------
def create_manual(store: Store, principal: Principal, body: dict):
    """Re-run cases on assigned users' matching clients; results become current heads."""
    principal.require_write()
    keys, backends = body.get("case_keys"), body.get("backends")
    if not isinstance(keys, list) or not 1 <= len(keys) <= 200000 or not all(isinstance(k, str) for k in keys):
        raise IssueError("invalid", "case_keys must be a list of 1..200000 strings")
    if not isinstance(backends, list) or not 1 <= len(backends) <= 4 \
            or not all(isinstance(b, str) and _NAME.match(b) and b != "cpu" for b in backends):
        raise IssueError("invalid", "backends must list 1..4 NPU backends")
    arch = body.get("npu_arch")
    if arch is not None and (not isinstance(arch, str) or not 1 <= len(arch) <= 32):
        raise IssueError("invalid", "npu_arch must be a short string")
    repetitions = body.get("repetitions", 1)
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or not 1 <= repetitions <= 5:
        raise IssueError("invalid", "repetitions must be 1..5")
    if type(body.get("force", False)) is not bool:
        raise IssueError("invalid", "force must be boolean")
    users = body.get("user_ids")
    if users is not None:
        principal.require_admin()
        if not isinstance(users, list) or not users or not all(isinstance(u, str) for u in users):
            raise IssueError("invalid", "user_ids must be a nonempty user list")
        users = list(dict.fromkeys(users))
    else:
        users = [principal.user_id]
    request_key = body.get("request_key")

    def action(conn):
        from autofusex.server.issues import _case_ids
        for user in users:
            if not conn.execute("SELECT 1 FROM users WHERE user_id=? AND status='active' AND role IN ('admin','developer')", (user,)).fetchone():
                raise IssueError("invalid", f"inactive or unknown user: {user}")
        now, created, skipped, excluded = now_ms(), 0, [], 0
        for case_id in _case_ids(conn, keys):
            for backend in dict.fromkeys(backends):
                arches = [arch] if arch else [r[0] for r in conn.execute(
                    "SELECT DISTINCT npu_arch FROM execution_heads WHERE case_id=? AND backend=?", (case_id, backend))]
                if not arches and arch is None:
                    # A newly selected backend inherits this case's known NPU chips.
                    arches = [r[0] for r in conn.execute(
                        "SELECT DISTINCT npu_arch FROM execution_heads WHERE case_id=? "
                        "AND backend<>'cpu' AND npu_arch<>'' ORDER BY npu_arch", (case_id,))]
                if not arches:
                    skipped.append(case_id)
                    continue
                for target_arch in arches:
                    revision = conn.execute("SELECT case_revision FROM cases WHERE case_id=?", (case_id,)).fetchone()[0]
                    if not body.get('force', False) and conn.execute(
                            "SELECT 1 FROM case_coverage WHERE case_id=? AND case_revision=? AND backend=? AND npu_arch=?",
                            (case_id, revision, backend, target_arch)).fetchone():
                        excluded += 1
                        continue
                    for _ in range(repetitions):
                        conn.execute(
                            "INSERT INTO retest_tasks(issue_id, round_no, case_id, npu_arch, backends, required_backend, "
                            "required_version, required_sha256, status, created_at, task_kind, requested_by, batch, assigned_user) "
                            "VALUES('', 0, ?,?,?,?, '*', '*', 'pending', ?, 'retest', ?, ?, ?)",
                            (case_id, target_arch, f"cpu,{backend}", backend, now, principal.user_id, request_key, users[created % len(users)]))
                        conn.execute("UPDATE retest_tasks SET case_revision=? WHERE task_id=last_insert_rowid()", (revision,))
                        created += 1
        return {"batch": request_key, "tasks": created, "skipped_without_arch": len(skipped), "skipped_no_coverage": excluded}
    return idempotent(store, principal, "retest-manual", body.pop("request_key", None), body, action)


def list_manual(store: Store, principal: Principal, limit: int = 50) -> list:
    if principal.anonymous:
        return []
    rows = store.reader().execute(
        "SELECT batch, MIN(created_at) AS created_at, COUNT(*) AS tasks, "
        "SUM(status IN ('pending','leased')) AS running, SUM(result='pass') AS passed, "
        "SUM(result='fail') AS failed, SUM(result='invalid' OR status='invalid') AS invalid "
        "FROM retest_tasks WHERE requested_by=? AND batch IS NOT NULL GROUP BY batch "
        "ORDER BY MIN(task_id) DESC LIMIT ?", (principal.user_id, limit)).fetchall()
    batches = [dict(r) for r in rows]
    conn = store.reader()
    for batch in batches:
        tasks = conn.execute(
            "SELECT t.task_id, c.case_key, t.backends, t.npu_arch, t.status, t.result, t.detail, t.assigned_user "
            "FROM retest_tasks t JOIN cases c ON c.case_id=t.case_id "
            "WHERE t.requested_by=? AND t.batch=? ORDER BY t.task_id LIMIT 100",
            (principal.user_id, batch['batch'])).fetchall()
        batch['items_truncated'] = batch['tasks'] > 100
        batch['assignments'] = [dict(r) for r in conn.execute(
            "SELECT assigned_user AS user_id, COUNT(*) AS tasks FROM retest_tasks "
            "WHERE requested_by=? AND batch=? GROUP BY assigned_user",
            (principal.user_id, batch['batch']))]
        batch['items'] = []
        for row in tasks:
            task = dict(row)
            task['executions'] = [dict(e) for e in conn.execute(
                "SELECT e.exec_uuid, e.backend FROM retest_task_execs x "
                "JOIN executions e ON e.exec_id=x.exec_id WHERE x.task_id=? ORDER BY e.exec_id",
                (task['task_id'],))]
            batch['items'].append(task)
    return batches
