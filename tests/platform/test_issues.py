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

from autofusex.platform import auth, issues, knowledge, regression, retests
from autofusex.platform.ingest import Ingestor
from autofusex.platform.issues import IssueError
from autofusex.platform.server import PlatformServer
from autofusex.platform.store import Store

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
            self.upload([item(uuid, n, status=status, summary="error at 0x1f tile 32" if status != "pass" else None,
                              created=5000)], kind=task["run_kind"], run=task["run_uuid"])
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
        item("exec-0001", 1, status="compile_fail", fail_stage="schedule", summary="error at 0x1f tile 32"),
        item("exec-0002", 2, status="compile_fail", fail_stage="schedule", summary="error at 0x2a tile 64"),
        item("exec-0003", 3, status="runtime_fail", summary="aclrtSynchronize failed", c=case(3, "reduce")),
        *[item(f"exec-1{n:03d}", n) for n in range(10, 18)],
        *[item(f"exec-2{n:03d}", n, backend="cpu") for n in range(1, 4)],
    ])
    listed = issues.list_issues(world.store)["issues"]
    return {i["category"]: i for i in listed}


def test_clustering_groups_same_normalized_error(world):
    by_category = seed_failures(world)
    assert set(by_category) == {"schedule", "runtime_fail"}
    schedule = by_category["schedule"]
    assert schedule["case_count"] == 2 and schedule["status"] == "open" and re.fullmatch(r"F\d+", schedule["issue_id"])
    assert schedule["dimension"] == "function" and schedule["backend"] == "af-mlir"
    assert issues.normalize("Error at 0x1F in /usr/lib/x.so: shape [3, 4] 'foo'") == \
        "error at <hex> in <path>: shape <shape> <str>"


def test_reclassifies_case_when_failure_dimension_changes(world):
    world.upload([item("exec-reclass-f", 101, status="compile_fail", fail_stage="schedule",
                        summary="schedule failure")])
    function_issue = issues.list_issues(world.store)["issues"][0]
    world.upload([item("exec-reclass-a", 101, status="pass", accuracy="fail",
                        max_abs_err=2.0, max_rel_err=2.0, summary="accuracy failure")])
    listed = {row["issue_id"]: row for row in issues.list_issues(world.store)["issues"]}
    assert listed[function_issue["issue_id"]]["case_count"] == 0
    old = issues.issue_cases(world.store, function_issue["issue_id"])["cases"][0]
    assert old["stale"] == 1 and old["result"] == "skip"
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
    assert by_dim["accuracy"]["category"] == "severe" and "EagerNPU" in by_dim["accuracy"]["title"]
    assert by_dim["performance"]["category"] == "2-5x" and by_dim["performance"]["case_count"] == 1


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
    world.upload([item("exec-9001", 40, status="compile_fail", fail_stage="schedule", summary="error at 0x99 tile 8",
                       created=later)])
    assert world.issue(issue_id)["status"] == "resolved"
    # the same bug on a newer version reopens it
    world.upload([item("exec-9002", 41, status="compile_fail", fail_stage="schedule", summary="error at 0x98 tile 8",
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
    from autofusex.platform import schema
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
