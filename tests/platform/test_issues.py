import re
import json
import threading
import http.client as httpclient

def _short(n):
    digits, alphabet = "", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    while n:
        n, d = divmod(n, 36)
        digits = alphabet[d] + digits
    return "E" + (digits or "0").zfill(26)


import pytest

from autofusex.server import auth
from autofusex.server import issues
from autofusex.server import knowledge
from autofusex.server import regression
from autofusex.server import retests
from autofusex.server.ingest import Ingestor
from autofusex.server.issues import IssueError
from autofusex.server.server import PlatformServer
from autofusex.server.store import Store

ENV = {"cann": "8.2", "backend_version": "0.3.0"}
FIX = {"fix_version": "0.3.1+fix1", "fix_sha256": "a" * 64}
INSTALLED = {"af-mlir": {"version": FIX["fix_version"], "sha256": FIX["fix_sha256"]}}


def seed(n):
    return _short(n)


def case(n, category="elementwise", ops=("add", "mul"), dtypes=("float16",), nodes=6):
    return {"case_key": seed(n), "case_source": "nnsmith", "is_dynamic": False, "shape_count": 1,
            "node_count": nodes, "op_types": list(ops), "dtypes": list(dtypes), "max_numel": 4096,
            "features": {"op_category": [category]}}


def item(uuid, n, backend="af-mlir", status="pass", accuracy="pass", summary=None, created=1000, **extra):
    out = {"exec_uuid": uuid, "case": extra.pop("c", None) or case(n), "backend": backend,
           "npu_arch": "" if backend == "cpu" else "910b", "env": ENV, "function_status": status,
           "accuracy_status": accuracy if status == "pass" else None, "e2e_ns": 10, "created_at": created,
           "log_summary": summary}
    out.update(extra)
    return out


class World:
    def __init__(self, tmp_path):
        self.store = Store(tmp_path / "data")
        auth.add_user(self.store, "dev", "developer")
        auth.add_user(self.store, "root", "admin")
        self.client = self.agent = self._principal("dev")  # one token per user, used for everything
        self.admin = self._principal("root")
        self.ingestor = Ingestor(self.store)
        self.keys = 0

    def _principal(self, user):
        return auth.authenticate(self.store, "Bearer " + auth.issue_token(self.store, user))

    def key(self):
        self.keys += 1
        return f"k{self.keys:05d}"

    def upload(self, items, kind="fuzz", run=None):
        body = {"request_key": self.key(), "client": {"machine_id": "m1", "npu_arch": "910b"},
                "run": {"run_uuid": run or self.key(), "kind": kind}, "items": items}
        result = self.ingestor.upload(self.client, body, json.dumps(body).encode())
        assert all(r["status"] == "accepted" for r in result["results"]), result
        return result

    def claim_tasks(self, installed=INSTALLED, limit=20):
        return retests.claim(self.store, self.client, {"request_key": self.key(), "machine_id": "m1",
                                                       "npu_arch": "910b", "installed": installed,
                                                       "limit": limit})["tasks"]

    def run_tasks(self, tasks, outcome, installed=INSTALLED):
        """Upload a result for each task and complete it. outcome(case_key) -> function_status."""
        results = []
        for task in tasks:
            n = int(task["case_key"][1:], 36)
            status = outcome(task["case_key"])
            uuid = f"t{task['task_id']:06d}x{self.keys}"
            self.upload([item(uuid, n, status=status, summary="Schedule failed at 0x1f tile 32" if status != "pass" else None,
                              created=5000, fail_stage="schedule" if status != "pass" else None)], kind=task["run_kind"], run=task["run_uuid"])
            results.append(retests.complete(self.store, self.client, task["task_id"], {
                "request_key": self.key(), "lease_id": task["lease_id"], "exec_uuids": [uuid],
                "installed": installed}))
        return results

    def issue(self, issue_id):
        return issues.issue_detail(self.store, issue_id)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def seed_failures(world):
    """Two cases share one schedule bug, one case has another bug, eight cases pass."""
    world.upload([
        item("exec-0001", 1, status="compile_fail", fail_stage="schedule", summary="Schedule failed at 0x1f tile 32"),
        item("exec-0002", 2, status="compile_fail", fail_stage="schedule", summary="Schedule failed at 0x2a tile 32"),
        item("exec-0003", 3, status="runtime_fail", summary="aclrtSynchronize failed", c=case(3, "reduce")),
        *[item(f"exec-1{n:03d}", n) for n in range(10, 18)],
        *[item(f"exec-2{n:03d}", n, backend="cpu") for n in range(1, 4)],
    ])
    listed = issues.list_issues(world.store)["issues"]
    return {i["category"].split(":")[-1]: i for i in listed}


def test_clustering_groups_same_normalized_error(world):
    by_category = seed_failures(world)
    assert set(by_category) == {"schedule", "unknown_retest"}
    schedule = by_category["schedule"]
    assert schedule["case_count"] == 2 and schedule["status"] == "open" and re.fullmatch(r"F\d+", schedule["issue_id"])
    assert schedule["dimension"] == "function" and schedule["backend"] == "af-mlir"
    assert issues.normalize("Error at 0x1F in /usr/lib/x.so: shape [3, 4] 'foo'") == \
        "Error at <address> in /usr/lib/x.so: shape [3, 4] 'foo'"


def test_function_title_new_and_legacy(world):
    summary = ('[ERROR] GE(12345,python):2026-11-20-18:00:00.123.456 '
               '[./autofuse/lowering/view.cpp:812]12345 Operator Reshape failed: shape [3, 4], axis 2')
    world.upload([item('exec-title', 301, status='compile_fail', fail_stage='lowering',
                       summary=summary, c=case(301, 'view'))])
    row = issues.list_issues(world.store)['issues'][0]
    assert 'Operator Reshape failed: shape [3, 4], axis 2' in row['title']
    assert row['log_summary'] == summary
    assert row['category'] == 'compile_fail:unknown_retest'
    assert row['primary_category'] == 'View'


def test_function_title_plain_missing_and_long_summary():
    assert issues.function_title('af-mlir', 'lowering', 'Mismatch: expected 32, got 64', 'view') == (
        '[af-mlir] lowering: Mismatch: expected 32, got 64 (view)')
    assert issues.function_title('af-mlir', 'lowering', None, 'view') == (
        '[af-mlir] lowering: lowering (view)')
    assert len(issues.function_title('af-mlir', 'lowering', 'x' * 500, 'view')) == 300


@pytest.mark.parametrize('legacy', ['af_fronted', 'graph_check', 'lowering'])
def test_issue_classification_comes_from_execution_not_legacy_category(world, legacy):
    world.upload([item('legacy-label', 302, status='compile_fail', fail_stage='schedule',
                       summary='Schedule failed')])
    issue_id = issues.list_issues(world.store)['issues'][0]['issue_id']
    with world.store.write() as conn:
        conn.execute('UPDATE issues SET category=? WHERE issue_id=?', (legacy, issue_id))
    expected = dict(function_status='compile_fail', fail_stage='schedule',
                    accuracy_status=None, accuracy_ref=None, invalid_reason=None)
    assert issues.list_issues(world.store)['issues'][0]['classifications'] == [expected]
    assert world.issue(issue_id)['classifications'] == [expected]
    row = issues.issue_cases(world.store, issue_id)['cases'][0]
    assert {key: row[key] for key in expected} == expected
    with world.store.write() as conn:
        conn.execute('DELETE FROM issue_current WHERE issue_id=?', (issue_id,))
    assert world.issue(issue_id)['classifications'] == [expected]


def test_issue_cases_keep_classification_per_backend_and_missing_regression(world):
    world.upload([item('case-mlir', 303, status='compile_fail', fail_stage='schedule', summary='Schedule failed'),
                  item('case-legacy', 303, backend='af-legacy', status='compile_fail', fail_stage='schedule',
                       summary='Schedule failed'), item('case-regression', 304)])
    issue_id = issues.list_issues(world.store)['issues'][0]['issue_id']
    with world.store.write() as conn:
        legacy_id = conn.execute("SELECT exec_id FROM executions WHERE exec_uuid='case-legacy'").fetchone()[0]
        conn.execute("UPDATE executions SET function_status='runtime_fail',fail_stage='aclnn' WHERE exec_id=?",
                     (legacy_id,))
        regression_id = conn.execute('SELECT case_id FROM cases WHERE case_key=?', (seed(304),)).fetchone()[0]
        retests.create_tasks(conn, dict(issue_id=issue_id, dimension='function'), 1,
                            [(regression_id, '910b', 'af-mlir')], FIX['fix_version'], FIX['fix_sha256'], 'regression')
    rows = {row['backend']: row for row in issues.issue_cases(world.store, issue_id)['cases']}
    assert rows['af-mlir']['fail_stage'] == 'schedule'
    assert rows['af-legacy']['function_status'] == 'runtime_fail'
    assert rows['af-legacy']['fail_stage'] == 'aclnn'
    assert len(world.issue(issue_id)['classifications']) == 2
    regression_row = issues.issue_cases(world.store, issue_id, role='regression')['cases'][0]
    assert regression_row['case_key'] == seed(304) and regression_row['function_status'] is None


def test_reclassifies_case_when_failure_dimension_changes(world):
    world.upload([item("exec-reclass-f", 101, status="compile_fail", fail_stage="schedule",
                        summary="schedule failure")])
    function_issue = issues.list_issues(world.store)["issues"][0]
    world.upload([item("exec-reclass-a", 101, status="pass", accuracy="fail",
                        max_abs_err=2.0, max_rel_err=2.0, summary="accuracy failure")])
    listed = {row["issue_id"]: row for row in issues.list_issues(world.store)["issues"]}
    assert listed[function_issue["issue_id"]]["case_count"] == 1
    old = issues.issue_cases(world.store, function_issue["issue_id"])["cases"][0]
    assert old["stale"] == 0 and old["result"] == "skip"
    assert any(row["dimension"] == "accuracy" and row["case_count"] == 1 for row in listed.values())


def test_claimed_issues_are_listed_before_open_issues(world):
    world.upload([item("exec-order-1", 201, status="compile_fail", fail_stage="schedule", summary="one"),
                  item("exec-order-2", 202, status="compile_fail", fail_stage="tiling", summary="two")])
    rows = issues.list_issues(world.store)["issues"]
    claimed = issues.claim(world.store, world.agent, {"request_key": "order-claim", "agent_id": "order-agent",
                                                       "issue_id": rows[-1]["issue_id"]})
    ordered = issues.list_issues(world.store)["issues"]
    assert ordered[0]["issue_id"] == claimed["issue_id"]


def test_accuracy_and_performance_issues(world):
    world.upload([
        item("exec-0001", 1, accuracy="fail", accuracy_ref="eager-npu", max_rel_err=3.0, nodes=2,
             c=case(1, ops=("div",), nodes=2)),
        item("exec-0002", 2, kernel_ns=300),
        item("exec-0003", 2, backend="triton-npu", kernel_ns=100),
        item("exec-0004", 3, kernel_ns=110),
        item("exec-0005", 3, backend="triton-npu", kernel_ns=100),
    ])
    by_dim = {i["dimension"]: i for i in issues.list_issues(world.store)["issues"]}
    assert set(by_dim) == {"accuracy", "performance"}
    assert by_dim["accuracy"]["category"] == "Elementwise"
    assert by_dim["performance"]["category"] == "Elementwise" and by_dim["performance"]["case_count"] == 1


def test_full_lifecycle(world):
    schedule = seed_failures(world)["schedule"]
    issue_id = schedule["issue_id"]
    world.upload([item("exec-3001", 30, c=case(30, "reduce"))])
    regression.add_pins(world.store, world.admin, {"request_key": world.key(), "scope": "global",
                                                   "case_keys": [seed(30)], "reason": "key reduce graph"})

    claim_body = {"request_key": "claim-1", "agent_id": "agent-1"}
    claimed = issues.claim(world.store, world.agent, dict(claim_body))
    assert claimed["issue_id"] == issue_id and claimed["status"] == "claimed"
    assert claimed["repro"]["cases"][0]["command"].startswith("afx fuzz replay E")
    assert issues.claim(world.store, world.agent, dict(claim_body)) == claimed  # retry is idempotent
    with pytest.raises(IssueError) as busy:
        issues.claim(world.store, world.agent, {"request_key": "claim-2", "agent_id": "agent-1"})
    assert busy.value.code == "agent_busy"
    lease = claimed["lease_id"]

    # round 1: one case still fails -> back to claimed
    out = issues.verify(world.store, world.agent, issue_id, dict(FIX, request_key=world.key(), lease_id=lease))
    assert out == {"issue_id": issue_id, "round_no": 1, "tasks": 2, "status": "verifying"}
    wrong = {"af-mlir": {"version": "0.3.0", "sha256": "b" * 64}}
    assert world.claim_tasks(installed=wrong) == []
    tasks = world.claim_tasks()
    assert len(tasks) == 2 and tasks[0]["run_kind"] == "verify" and tasks[0]["backends"] == ["cpu", "af-mlir"]
    world.run_tasks(tasks, lambda key: "compile_fail" if key == seed(2) else "pass")
    detail = world.issue(issue_id)
    assert detail["status"] == "claimed" and detail["rounds"][0]["outcome"] == "failed"
    assert detail["rounds"][0]["pass_count"] == 1 and detail["rounds"][0]["fail_count"] == 1

    # verify-run results never become current heads
    assert world.store.reader().execute(
        "SELECT COUNT(*) FROM execution_heads h JOIN executions e USING(exec_id) WHERE e.run_kind='verify'"
    ).fetchone()[0] == 0

    # round 2 passes -> regression round with the pinned case first
    issues.verify(world.store, world.agent, issue_id, dict(FIX, request_key=world.key(), lease_id=lease))
    world.run_tasks(world.claim_tasks(), lambda key: "pass")
    detail = world.issue(issue_id)
    assert detail["status"] == "regressing" and detail["round_no"] == 3
    picks = issues.issue_cases(world.store, issue_id, "regression")["cases"]
    reasons = {c["case_key"]: c["pick_reason"] for c in picks}
    assert reasons[seed(30)] == "pinned" and seed(1) not in reasons
    assert set(reasons.values()) <= {"pinned", "neighbor", "sentinel", "history", "random"}
    assert len(picks) == 9  # 8 passing elementwise cases + pinned reduce case; the failing reduce case excluded

    # regression passes -> summarizing -> knowledge -> resolved
    regress_tasks = world.claim_tasks()
    assert {t["run_kind"] for t in regress_tasks} == {"regress"}
    world.run_tasks(regress_tasks, lambda key: "pass")
    assert world.issue(issue_id)["status"] == "summarizing"
    with pytest.raises(IssueError):
        issues.submit_knowledge(world.store, world.agent, issue_id, {
            "request_key": world.key(), "lease_id": "wrong", "knowledge": {}})
    kb = issues.submit_knowledge(world.store, world.agent, issue_id, {
        "request_key": world.key(), "lease_id": lease, "knowledge": {
            "kind": "bugfix_function", "title": "schedule tiling 轴不匹配 导致融合失败",
            "keywords": ["schedule", "tiling", "融合"], "ops": ["add", "mul"], "dtypes": ["float16"],
            "symptom": "compile fails at schedule", "root_cause": "tile axis not aligned",
            "solution": "align tile axis with broadcast dim", "fix_ref": "https://example.invalid/mr/1"}})
    detail = world.issue(issue_id)
    assert detail["status"] == "resolved" and detail["knowledge_id"] == kb["knowledge_id"]
    assert [e["kind"] for e in reversed(detail["events"])][:3] == ["created", "claimed", "verify_submitted"]

    conn = world.store.reader()
    entry = knowledge.get(conn, kb["knowledge_id"])
    assert entry["confidence"] == "verified" and entry["evidence"]["issue_id"] == issue_id
    assert entry["version_range"] == ">=0.3.1+fix1" and entry["backends"] == ["af-mlir"]
    assert [i["knowledge_id"] for i in knowledge.search(conn, q="tiling")["items"]] == [kb["knowledge_id"]]
    assert knowledge.search(conn, q="融合")["items"]  # two-char term via LIKE
    assert knowledge.search(conn, q="schedule", op="add", backend="af-mlir")["items"]
    assert not knowledge.search(conn, q="schedule", op="sub")["items"]
    assert not knowledge.search(conn, q="%")["items"]
    lines = list(knowledge.export_jsonl(conn))
    assert len(lines) == 1 and json.loads(lines[0])["knowledge_id"] == kb["knowledge_id"]

    # the old, still-unmerged version keeps failing: expected, the issue stays resolved
    later = 10**13
    world.upload([item("exec-9001", 40, status="compile_fail", fail_stage="schedule", summary="Schedule failed at 0x99 tile 32",
                       created=later)])
    assert world.issue(issue_id)["status"] == "resolved"
    # the same bug on a newer version reopens it
    world.upload([item("exec-9002", 41, status="compile_fail", fail_stage="schedule", summary="Schedule failed at 0x98 tile 32",
                       created=later, env={"cann": "8.2", "backend_version": "0.3.2"})])
    detail = world.issue(issue_id)
    assert detail["status"] == "open" and detail["case_count"] == 4
    related = issues.claim(world.store, world.agent, {"request_key": world.key(), "agent_id": "agent-1",
                                                     "issue_id": issue_id})["related_knowledge"]
    assert related[0]["knowledge_id"] == kb["knowledge_id"]


def test_invalid_results_do_not_count_as_failures(world):
    issue_id = seed_failures(world)["schedule"]["issue_id"]
    lease = issues.claim(world.store, world.agent, {"request_key": "c1", "agent_id": "a"})["lease_id"]
    issues.verify(world.store, world.agent, issue_id, dict(FIX, request_key="v1", lease_id=lease))
    tasks = world.claim_tasks()
    # client claims with the right package but reports a different one at completion
    other = {"af-mlir": {"version": FIX["fix_version"], "sha256": "c" * 64}}
    results = world.run_tasks(tasks, lambda key: "pass", installed=other)
    assert {r["result"] for r in results} == {"invalid"}
    detail = world.issue(issue_id)
    assert detail["status"] == "claimed" and detail["rounds"][0]["outcome"] == "invalid"


def test_lease_expiry_and_takeover(world):
    issue_id = seed_failures(world)["schedule"]["issue_id"]
    first = issues.claim(world.store, world.agent, {"request_key": "c1", "agent_id": "a"})
    with world.store.write() as conn:
        conn.execute("UPDATE issues SET lease_expires_at=1 WHERE issue_id=?", (issue_id,))
    second = issues.claim(world.store, world.agent, {"request_key": "c2", "agent_id": "b", "issue_id": issue_id})
    assert second["lease_id"] != first["lease_id"]
    with pytest.raises(IssueError) as lost:
        issues.renew(world.store, world.agent, issue_id, {"request_key": "r1", "lease_id": first["lease_id"]})
    assert lost.value.code == "lease_lost"
    kinds = [e["kind"] for e in world.issue(issue_id)["events"]]
    assert "lease_expired" in kinds
    released = issues.release(world.store, world.agent, issue_id, {"request_key": "r2",
                                                                  "lease_id": second["lease_id"]})
    assert released["status"] == "open"


def test_request_key_reuse_with_other_body_conflicts(world):
    seed_failures(world)
    issues.claim(world.store, world.agent, {"request_key": "same", "agent_id": "a"})
    with pytest.raises(IssueError) as conflict:
        issues.claim(world.store, world.agent, {"request_key": "same", "agent_id": "b"})
    assert conflict.value.status == 409


def test_admin_merge_and_close(world):
    world.upload([
        item("exec-0001", 1, status="compile_fail", fail_stage="schedule", summary="axis mismatch"),
        item("exec-0002", 2, status="compile_fail", fail_stage="schedule", summary="axis not aligned"),
    ])
    a, b = [i["issue_id"] for i in issues.list_issues(world.store)["issues"]]
    with pytest.raises(auth.AuthError):
        issues.merge(world.store, world.agent, a, {"request_key": "m0", "into": b})
    merged = issues.merge(world.store, world.admin, a, {"request_key": "m1", "into": b})
    assert merged["moved_cases"] == 1
    assert world.issue(a)["status"] == "closed" and world.issue(b)["case_count"] == 2
    # the merged signature now routes new cases to the target
    world.upload([item("exec-0003", 3, status="compile_fail", fail_stage="schedule", summary="axis mismatch")])
    assert world.issue(b)["case_count"] == 3
    issues.close(world.store, world.admin, b, {"request_key": "x1", "reason": "won't fix: unsupported pattern"})
    assert world.issue(b)["status"] == "closed"


def test_pinning_frozen_after_regression_starts(world):
    issue_id = seed_failures(world)["schedule"]["issue_id"]
    lease = issues.claim(world.store, world.agent, {"request_key": "c1", "agent_id": "a"})["lease_id"]
    pinned = issues.pin_cases(world.store, world.agent, issue_id, {"request_key": "p1", "lease_id": lease,
                                                                  "case_keys": [seed(10)]})
    assert pinned["pinned"] == 1
    preview = regression.preview(world.store, issue_id)
    assert preview["cases"][0] == {"case_key": seed(10), "npu_arch": "910b", "reason": "pinned"}
    issues.verify(world.store, world.agent, issue_id, dict(FIX, request_key="v1", lease_id=lease))
    world.run_tasks(world.claim_tasks(), lambda key: "pass")
    with pytest.raises(IssueError) as frozen:
        issues.pin_cases(world.store, world.agent, issue_id, {"request_key": "p2", "lease_id": lease,
                                                             "case_keys": [seed(11)]})
    assert frozen.value.code == "wrong_state"


def test_regression_budget_respected(world):
    world.upload([item(f"exec-{n:04d}", n, c=case(n, ["elementwise", "reduce", "view"][n % 3])) for n in range(10, 60)])
    world.upload([item("exec-0001", 1, status="compile_fail", fail_stage="schedule", summary="x")])
    issue = world.store.reader().execute("SELECT * FROM issues").fetchone()
    picks = regression.select(world.store.reader(), issue, 2, budget=10)
    assert len(picks) == 10 and len({p[0] for p in picks}) == 10
    assert picks == regression.select(world.store.reader(), issue, 2, budget=10)  # deterministic


# -- HTTP ----------------------------------------------------------------------
def test_numbered_pages(world):
    seed_failures(world)
    rows = issues.list_issues(world.store)['issues']
    issues.claim(world.store, world.agent, {'request_key': 'page-claim', 'agent_id': 'p',
                                          'issue_id': rows[-1]['issue_id']})
    expected = issues.list_issues(world.store)['issues']
    first = issues.list_issues(world.store, page=1, limit=1)
    second = issues.list_issues(world.store, page=2, limit=1)
    assert first == dict(issues=expected[:1], page=1, page_size=1, total=2, total_pages=2, next_cursor="1")
    assert second['issues'] == expected[1:]
    assert issues.list_issues(world.store, page=10**30, limit=1) == second
    cursor = issues.list_issues(world.store, limit=1)['next_cursor']
    assert issues.list_issues(world.store, after=cursor, limit=1)['issues'] == second['issues']
    filtered = issues.list_issues(world.store, page=2, limit=1, dimension='function',
                                  backend='af-mlir', status='claimed', q=expected[0]['issue_id'])
    assert filtered['total'] == filtered['total_pages'] == filtered['page'] == 1
    assert filtered['issues'] == expected[:1]
    assert issues.list_issues(world.store, page=5, q='no matching issue') == dict(
        issues=[], page=1, page_size=50, total=0, total_pages=0, next_cursor=None)
    with world.store.write() as conn:
        conn.execute("UPDATE issues SET status='closed' WHERE issue_id=?", (expected[1]['issue_id'],))
    shrunk = issues.list_issues(world.store, page=2, limit=1, status='claimed')
    assert shrunk['page'] == shrunk['total_pages'] == shrunk['total'] == 1


@pytest.mark.parametrize('kwargs', [dict(page=0), dict(page=-1), dict(page=1.5),
                                    dict(page=True), dict(page=1, after='1:1:1'),
                                    dict(page=1, limit=201)])
def test_invalid_numbered_pages(world, kwargs):
    with pytest.raises(IssueError):
        issues.list_issues(world.store, **kwargs)


def test_page_id_query_uses_ordered_indexes(world):
    conn = world.store.reader()
    for clause, args, index in [('', [], 'issues_page'),
                                ('WHERE dimension=?', ['function'], 'issues_page_dimension'),
                                ('WHERE dimension=? AND status=?', ['function', 'open'], 'issues_page_status')]:
        plan = ' '.join(row[3] for row in conn.execute(
            f'EXPLAIN QUERY PLAN SELECT issue_no FROM issues {clause} '
            f'ORDER BY {issues._ISSUE_ORDER} LIMIT ? OFFSET ?', args + [50, 10000]))
        assert index in plan
        assert 'TEMP B-TREE' not in plan


def test_http_issue_flow(world):
    server = PlatformServer(world.store, ("127.0.0.1", 0))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    agent_token = client_token = auth.issue_token(world.store, "dev")

    def call(method, path, payload=None, token=agent_token):
        conn = httpclient.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        data = json.dumps(payload).encode() if payload is not None else None
        conn.request(method, path, body=data, headers={"Authorization": "Bearer " + token,
                                                       "Content-Type": "application/json"})
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, (json.loads(raw) if response.getheader("Content-Type").startswith(
            "application/json") else raw)

    try:
        seed_failures(world)
        status, listed = call("GET", "/api/issues?dimension=function")
        assert status == 200 and len(listed["data"]["issues"]) == 2
        status, paged = call("GET", "/api/issues?dimension=function&page=2&limit=1")
        assert status == 200 and paged['data']['page'] == 2 and paged['data']['total_pages'] == 2
        assert call("GET", "/api/issues?page=0")[0] == 400
        assert call("GET", "/api/issues?page=abc")[0] == 400
        status, claimed = call("POST", "/api/issues/claim", {"request_key": "h1", "agent_id": "a"})
        issue_id, lease = claimed["data"]["issue_id"], claimed["data"]["lease_id"]
        status, verify = call("POST", f"/api/issues/{issue_id}/verify",
                              dict(FIX, request_key="h2", lease_id=lease))
        assert status == 200 and verify["data"]["status"] == "verifying"
        # a developer token may upload (an empty body is rejected as invalid, not forbidden)
        status, _ = call("POST", "/api/client/executions", {"request_key": "x"}, token=agent_token)
        assert status != 403
        status, tasks = call("POST", "/api/client/retests/claim", {"request_key": "h3", "machine_id": "m1",
                                                                   "npu_arch": "910b", "installed": INSTALLED},
                             token=client_token)
        assert status == 200 and len(tasks["data"]["tasks"]) == 2
        status, rounds = call("GET", f"/api/issues/{issue_id}/rounds/1/tasks")
        assert {t["status"] for t in rounds["data"]} == {"leased"}
        assert call("GET", "/api/issues/F999999")[0] == 404
        assert call("POST", f"/api/admin/issues/{issue_id}/close", {"request_key": "h4", "reason": "x"})[0] == 403
        status, search = call("GET", "/api/knowledge/search?q=%E8%9E%8D%E5%90%88")
        assert status == 200 and search["data"]["items"] == []
        status, raw = call("GET", "/api/knowledge/export")
        assert status == 200 and raw == b""
    finally:
        server.shutdown()
        server.server_close()


def test_http_release_returns_claim_to_queue(world):
    from fastapi.testclient import TestClient
    from autofusex.server.server import create_app

    issue_id = seed_failures(world)["schedule"]["issue_id"]
    headers = {"Authorization": "Bearer " + auth.issue_token(world.store, "dev")}
    auth.add_user(world.store, "other", "developer")
    other = {"Authorization": "Bearer " + auth.issue_token(world.store, "other")}
    with TestClient(create_app(world.store)) as client:
        claimed = client.post("/api/issues/claim", headers=headers, json={
            "request_key": "release-claim", "agent_id": "a", "issue_id": issue_id})
        assert claimed.status_code == 200, claimed.text
        lease = claimed.json()["data"]["lease_id"]
        path = f"/api/issues/{issue_id}/release"
        payload = {"request_key": "release-1", "lease_id": lease}
        assert client.post(path, json=payload).status_code == 401
        assert client.post(path, headers=headers, json={"request_key": "missing-lease"}).status_code == 400
        assert client.post("/api/issues/F999999/release", headers=headers, json=payload).status_code == 404
        denied = client.post(path, headers=other, json=payload)
        assert denied.status_code == 409 and denied.json()["error"]["code"] == "lease_lost"
        denied = client.post(path, headers=headers, json={**payload, "lease_id": "wrong"})
        assert denied.status_code == 409 and denied.json()["error"]["code"] == "lease_lost"
        assert world.issue(issue_id)["owner_user"] == "dev"

        released = client.post(path, headers=headers, json=payload)
        assert released.status_code == 200, released.text
        assert released.json()["data"] == {"issue_id": issue_id, "status": "open"}
        row = world.store.reader().execute("SELECT owner_user, agent_id, lease_id, lease_expires_at FROM issues "
                                           "WHERE issue_id=?", (issue_id,)).fetchone()
        assert all(value is None for value in row)
        queued = client.get("/api/issues", params={"status": "open", "scope": "current"}).json()["data"]["issues"]
        assert issue_id in {row["issue_id"] for row in queued}
        # The former agent is free to work on another issue.
        next_issue = client.post("/api/issues/claim", headers=headers, json={
            "request_key": "release-next", "agent_id": "a"})
        assert next_issue.status_code == 200 and next_issue.json()["data"] is not None
        returned = client.post(f"/api/issues/{next_issue.json()['data']['issue_id']}/release", headers=headers,
                               json={"request_key": "release-next-return", "lease_id": next_issue.json()["data"]["lease_id"]})
        assert returned.status_code == 200, returned.text
        taken = client.post("/api/issues/claim", headers=other, json={
            "request_key": "release-takeover", "agent_id": "b"})
        assert taken.status_code == 200 and taken.json()["data"]["issue_id"] == issue_id
        assert taken.json()["data"]["lease_id"] != lease
        # Retrying the old successful request must not release the new holder.
        assert client.post(path, headers=headers, json=payload).json() == released.json()
        assert world.issue(issue_id)["owner_user"] == "other"
        stale = client.post(path, headers=headers, json={**payload, "request_key": "release-stale"})
        assert stale.status_code == 409 and stale.json()["error"]["code"] == "lease_lost"
        conflict = client.post(path, headers=headers, json={**payload, "lease_id": "different"})
        assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "conflict"
        assert sum(event["kind"] == "released" for event in world.issue(issue_id)["events"]) == 2


def test_device_busy_is_invalid_not_failure(world):
    issue_id = seed_failures(world)["schedule"]["issue_id"]
    lease = issues.claim(world.store, world.agent, {"request_key": "c1", "agent_id": "a"})["lease_id"]
    issues.verify(world.store, world.agent, issue_id, dict(FIX, request_key="v1", lease_id=lease))
    tasks = world.claim_tasks()
    for task in tasks:
        n = int(task["case_key"][1:], 36)
        uuid = f"busy{task['task_id']:06d}"
        world.upload([item(uuid, n, status="runtime_fail", fail_stage="capability",
                           summary="ValueError: device busy: another managed test owns this physical NPU",
                           created=5000)], kind="verify", run=task["run_uuid"])
        result = retests.complete(world.store, world.client, task["task_id"], {
            "request_key": world.key(), "lease_id": task["lease_id"], "exec_uuids": [uuid], "installed": INSTALLED})
        assert result["result"] == "invalid"
    detail = world.issue(issue_id)
    assert detail["rounds"][0]["outcome"] == "invalid" and detail["status"] == "claimed"


def test_short_id_migration_renames_everywhere(tmp_path):
    import sqlite3
    from autofusex.server import schema
    store = Store(tmp_path / "data")
    with store.write() as conn:
        conn.execute("INSERT INTO issues(issue_id, dimension, backend, signature, category, title, status, first_seen_at, "
                     "last_seen_at, updated_at, knowledge_id) VALUES('ISS-F-000254','function','af-mlir',x'01','c','t',"
                     "'resolved',0,0,0,'KB-000012')")
        conn.execute("INSERT INTO issue_cases(issue_id, case_id, role, npu_arch, case_revision, added_at) "
                     "VALUES('ISS-F-000254', 1, 'affected', 'a', 1, 0)")
        conn.execute("INSERT INTO knowledge_links VALUES('KB-000012','issue','ISS-F-000254')")
        conn.execute("DELETE FROM schema_migrations WHERE version=5")
    with store.connect(readonly=False) as conn:
        schema.apply(conn, "main", schema.CORE)
    r = store.reader()
    assert r.execute("SELECT issue_id, knowledge_id FROM issues").fetchone()[:] == ("F254", "K12")
    assert r.execute("SELECT issue_id FROM issue_cases").fetchone()[0] == "F254"
    assert r.execute("SELECT knowledge_id, target_id FROM knowledge_links").fetchone()[:] == ("K12", "F254")


@pytest.mark.parametrize('destination', ['function', 'accuracy', 'performance', 'pass'])
def test_verified_result_reclassifies_original_issue(world, destination):
    issue_id = seed_failures(world)['schedule']['issue_id']
    lease = issues.claim(world.store, world.agent, dict(request_key=world.key(), agent_id='a', issue_id=issue_id))['lease_id']
    issues.verify(world.store, world.agent, issue_id, dict(FIX, request_key=world.key(), lease_id=lease))
    tasks = world.claim_tasks()
    for task in tasks:
        n = int(task['case_key'][1:], 36)
        extra = {
            'function': dict(status='runtime_fail', summary='another failure'),
            'accuracy': dict(accuracy='fail'),
            'performance': dict(kernel_ns=1000),
            'pass': {},
        }[destination]
        uuid = f'verified-{task["task_id"]}'
        world.upload([item(uuid, n, **extra)], kind=task['run_kind'], run=task['run_uuid'])
        # Include the benchmark in this same verified run, never a historical head.
        if destination == 'performance':
            world.upload([item(uuid+'-anchor', n, backend='triton-npu', kernel_ns=100)],
                         kind=task['run_kind'], run=task['run_uuid'])
            with world.store.write() as conn:
                conn.execute("UPDATE retest_tasks SET backends=backends || ',triton-npu' WHERE task_id=?", (task['task_id'],))
        ids = [uuid] + ([uuid+'-anchor'] if destination == 'performance' else [])
        result = retests.complete(world.store, world.client, task['task_id'], dict(
            request_key=world.key(), lease_id=task['lease_id'], exec_uuids=ids, installed=INSTALLED))
        assert result['result'] == ('pass' if destination == 'pass' else 'skip')
    members = issues.issue_cases(world.store, issue_id)['cases']
    assert {r['result'] for r in members} == ({'pass'} if destination == 'pass' else {'skip'})
    assert not any(r['stale'] for r in members)
    assert world.issue(issue_id)['rounds'][0]['outcome'] == 'passed'
    if destination != 'pass':
        targets = [r for r in issues.list_issues(world.store)['issues'] if r['issue_id'] != issue_id and r['dimension'] == destination]
        assert any({c['result'] for c in issues.issue_cases(world.store, r['issue_id'])['cases']} == {'fail'} for r in targets)


def test_shared_backend_cluster_and_precise_filters(world):
    world.upload([item('cross-aa', 501, status='compile_fail', fail_stage='schedule', summary='Schedule failed tile 32'),
                  item('cross-bb', 501, backend='af-legacy', status='compile_fail', fail_stage='schedule', summary='Schedule failed tile 32'),
                  item('cross-cc', 502, status='runtime_fail', summary='another failure')])
    rows = issues.list_issues(world.store, results='schedule', selected_backends='af-legacy', scope='current')['issues']
    assert len(rows) == 1
    assert rows[0]['backends'] == ['af-legacy', 'af-mlir']
    assert rows[0]['case_count'] == rows[0]['matched_case_count'] == 1
    assert len(issues.issue_cases(world.store, rows[0]['issue_id'])['cases']) == 2
    assert not issues.list_issues(world.store, results='runtime_unknown', selected_backends='af-legacy', scope='current')['issues']
    lease = issues.claim(world.store, world.agent, dict(request_key=world.key(),agent_id='cross',issue_id=rows[0]['issue_id']))
    result = issues.verify(world.store,world.agent,rows[0]['issue_id'],dict(FIX,request_key=world.key(),lease_id=lease['lease_id']))
    assert result['tasks'] == 2
    tasks = world.store.reader().execute('SELECT required_backend FROM retest_tasks').fetchall()
    assert {r[0] for r in tasks} == {'af-legacy','af-mlir'}


def test_classification_change_moves_current_membership_with_active_lease(world):
    world.upload([item('change-a', 601, status='compile_fail', fail_stage='schedule', summary='ERROR lowering')])
    old = issues.list_issues(world.store)['issues'][0]
    lease = issues.claim(world.store,world.agent,dict(request_key=world.key(),agent_id='change'))
    with world.store.write() as conn:
        exec_id = conn.execute("SELECT exec_id FROM executions WHERE exec_uuid='change-a'").fetchone()[0]
        conn.execute("UPDATE executions SET fail_stage='mlir' WHERE exec_id=?", (exec_id,))
        issues.refresh_classified_executions(conn,[exec_id])
    rows = issues.list_issues(world.store,scope='current')['issues']
    assert len(rows) == 1 and rows[0]['issue_id'] != old['issue_id']
    assert rows[0]['category'] == 'compile_fail:mlir'
    assert world.issue(old['issue_id'])['status'] == 'claimed'
    assert not issues.list_issues(world.store,scope='current',results='schedule')['issues']
    # Frozen evidence remains available, while current filtering follows classification.
    assert issues.issue_cases(world.store,old['issue_id'])['cases']


def test_treatment_permissions_idempotency_and_reactivation(world):
    seed_failures(world)
    row=issues.list_issues(world.store)['issues'][0]
    body=dict(request_key=world.key(),treatment='ignore',reason='out of scope')
    with pytest.raises(auth.AuthError):
        issues.set_treatment(world.store,world.agent,row['issue_id'],dict(body))
    first=issues.set_treatment(world.store,world.admin,row['issue_id'],dict(body))
    assert issues.set_treatment(world.store,world.admin,row['issue_id'],dict(body)) == first
    with pytest.raises(IssueError):
        issues.claim(world.store,world.agent,dict(request_key=world.key(),agent_id='ignore',issue_id=row['issue_id']))
    issues.set_treatment(world.store,world.admin,row['issue_id'],dict(request_key=world.key(),treatment='high'))
    assert issues.claim(world.store,world.agent,dict(request_key=world.key(),agent_id='high'))['issue_id'] == row['issue_id']


def test_rebuild_preview_and_apply_keep_executions(world):
    from autofusex.server.issue_rebuild import rebuild
    seed_failures(world)
    before=[dict(r) for r in world.store.reader().execute('SELECT * FROM executions')]
    old={r['issue_id'] for r in issues.list_issues(world.store)['issues']}
    assert rebuild(world.store)['applied'] is False
    assert {r['issue_id'] for r in issues.list_issues(world.store)['issues']} == old
    assert rebuild(world.store,apply=True)['new_issues']['function'] == 2
    assert not old & {r['issue_id'] for r in issues.list_issues(world.store)['issues']}
    assert before == [dict(r) for r in world.store.reader().execute('SELECT * FROM executions')]


def test_shared_primary_category_only_accuracy_and_performance(world):
    world.upload([item('acc-one1',701,accuracy='fail',max_rel_err=.01,c=case(701,category='Reduce',ops=('sum',),dtypes=('float32',))),
                  item('acc-two2',702,backend='af-legacy',accuracy='fail',max_rel_err=10,c=case(702,category='Reduce',ops=('mean',))),
                  item('perf-one',703,kernel_ns=200,c=case(703,category='Reduce')),
                  item('perf-ref',703,backend='triton-npu',kernel_ns=100,c=case(703,category='Reduce')),
                  item('perf-two',704,backend='af-legacy',kernel_ns=900,c=case(704,category='Reduce')),
                  item('perf-ref2',704,backend='triton-npu',kernel_ns=100,c=case(704,category='Reduce'))])
    rows=issues.list_issues(world.store)['issues']
    assert len(rows)==2
    assert {r['dimension'] for r in rows}=={'accuracy','performance'}
    assert all(r['case_count']==2 for r in rows)
    assert issues.primary_category(['Elementwise','Broadcast','Reshape','View','Scatter','Gather','Split','Concat','Transpose','Reduce'])=='Reduce'


def test_treatment_http_and_openapi(world):
    from fastapi.testclient import TestClient
    from autofusex.server.server import create_app
    seed_failures(world)
    issue_id=issues.list_issues(world.store)['issues'][0]['issue_id']
    with TestClient(create_app(world.store)) as client:
        path=f'/api/admin/issues/{issue_id}/treatment'
        body=dict(request_key='http-label',treatment='ignore')
        assert client.post(path,json=body,headers={'Authorization':'Bearer '+auth.issue_token(world.store,'dev')}).status_code==403
        token=auth.issue_token(world.store,'root')
        assert client.post(path,json=body,headers={'Authorization':'Bearer '+token}).status_code==200
        assert client.get('/api/issues?treatment=ignore').json()['data']['issues'][0]['issue_id']==issue_id
        assert '/api/admin/issues/{issue_id}/treatment' in client.get('/openapi.json').json()['paths']
        assert client.get('/api/issues?treatment=low,normal,high&categories=Reduce,Elementwise&selected_backends=af-mlir,af-legacy').status_code == 200


def test_issue_log_preserves_operator_dtype_and_error_code():
    from autofusex.server.error_logs import issue_summary
    line='Cast: Check dtype failed input_dtypes: [DT_INT64], output_dtypes: [DT_FLOAT]'
    assert issue_summary(None,None,line)==line
    assert issues.normalize(line)==line
    assert issues.normalize('ERROR code 161002') != issues.normalize('ERROR code 161003')
    assert issue_summary(None,None,'deprecated old API','timeout')=='Execution timed out'


def test_classification_rollback_refreshes_issue_without_new_upload(world):
    from autofusex.server.reclassification import rollback, FIELDS
    world.upload([item('rollback-issue',801,status='compile_fail',fail_stage='schedule',summary='Schedule failed')])
    with world.store.write() as conn:
        row=conn.execute("SELECT * FROM executions WHERE exec_uuid='rollback-issue'").fetchone()
        before=[row[k] for k in FIELDS]
        after=[row['function_status'],'mlir','mlir',999]
        conn.execute('INSERT INTO classification_changes VALUES(?,?,?,?)',('issue-test',row['exec_id'],json.dumps(before),json.dumps(after)))
        conn.execute('UPDATE executions SET function_status=?,fail_stage=?,failure_owner=?,classifier_version=? WHERE exec_id=?',(*after,row['exec_id']))
        issues.refresh_classified_executions(conn,[row['exec_id']])
    assert issues.list_issues(world.store,scope='current')['issues'][0]['category']=='compile_fail:mlir'
    rollback(world.store,'issue-test')
    rows=issues.list_issues(world.store,scope='current')['issues']
    assert len(rows)==1 and rows[0]['category']=='compile_fail:schedule'
