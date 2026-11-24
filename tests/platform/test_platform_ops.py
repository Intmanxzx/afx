import http.client as httpclient
import json
import threading

def _short(n):
    digits, alphabet = "", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    while n:
        n, d = divmod(n, 36)
        digits = alphabet[d] + digits
    return "E" + (digits or "0").zfill(26)


import pytest

from autofusex.server import auth
from autofusex.server import ops
from autofusex.server import profile
from autofusex.server import queries
from autofusex.server import retests
from autofusex.server import search
from autofusex.client.platform_transport import af_error_line, function_status
from autofusex.server.ingest import Ingestor
from autofusex.server.failure_attribution import classify as classify_failure
from autofusex.server.issues import IssueError
from autofusex.server.server import PlatformServer
from autofusex.server.store import Store, now_ms

ENV = {"cann": "8.2", "backend_version": "0.3.0"}


def seed(n):
    return _short(n)


def item(uuid, n, backend="af-mlir", status="pass", accuracy="pass", kernel=None, category="reduce", op="aten.sum.dim_IntList"):
    return {"exec_uuid": uuid, "backend": backend, "npu_arch": "" if backend == "cpu" else "910b", "env": ENV,
            "case": {"case_key": seed(n), "case_source": "nnsmith", "is_dynamic": False, "shape_count": 1,
                     "node_count": 3, "op_types": [op], "dtypes": ["float16"], "max_numel": 64 * n, "max_dim": n,
                     "features": {"op_category": [category]}},
            "function_status": status, "accuracy_status": accuracy if status == "pass" else None,
            "kernel_ns": kernel, "e2e_ns": 5, "created_at": 1000 + n,
            "log_summary": "error: tiling failed" if status != "pass" else None}


@pytest.fixture
def world(tmp_path):
    store = Store(tmp_path / "data")
    auth.add_user(store, "dev", "developer")
    auth.add_user(store, "root", "admin")
    client = auth.authenticate(store, "Bearer " + auth.issue_token(store, "dev"))
    admin = auth.authenticate(store, "Bearer " + auth.issue_token(store, "root"))
    ingestor = Ingestor(store)
    items = []
    for n in range(1, 41):
        category, op = ("reduce", "aten.sum.dim_IntList") if n % 2 else ("elementwise", "aten.add.Tensor")
        items.append(item(f"exec-a{n:04d}", n, kernel=50 if n % 2 else 200, category=category, op=op,
                          status="compile_fail" if n % 5 == 0 else "pass"))
        items.append(item(f"exec-b{n:04d}", n, backend="triton-npu", kernel=100, category=category, op=op))
    body = {"request_key": "r1", "client": {"machine_id": "m1", "npu_arch": "910b"},
            "run": {"run_uuid": "run-00001", "kind": "fuzz"}, "items": items}
    ingestor.upload(client, body, json.dumps(body).encode())
    return store, client, admin


def test_search_parser_rejects_unsafe_input():
    assert search.parse(None) is None
    for bad in ["backend:x;DROP", "status:nope", "unknown:x", "a" * 300, " ".join(["backend:x"] * 9),
                "error:ab", "backend:" + "x" * 80, "op:a'b"]:
        with pytest.raises(search.SearchError):
            search.parse(bad)
    query = search.parse("backend:af-mlir,af-legacy -stage:schedule category:reduce status:fail error:tiling")
    assert len(query.head) == 4 and len(query.case) == 1
    # user text only ever travels as bound parameters
    assert all("tiling" not in sql and "af-mlir" not in sql for sql, _ in query.head + query.case)


def test_search_filters_cases_and_overview(world):
    store, _, _ = world
    page = queries.case_page(store, q="backend:af-mlir status:fail category:reduce")
    keys = {g["case"]["case_key"] for g in page["groups"]}
    assert keys == {seed(n) for n in (5, 15, 25, 35)}
    pipe = queries.pipeline(store, q="category:elementwise")
    assert pipe["total_cases"] == 20 and pipe["q"] == "category:elementwise"
    lanes = {lane["backend"]: lane for lane in pipe["lanes"]}
    assert lanes["af-mlir"]["executed"] == 20
    by_dtype = queries.backend_stats(store, "af-mlir", "op_category", q="error:tiling")
    assert {g["value"] for g in by_dtype["groups"]} == {"reduce", "elementwise"}
    assert queries.case_page(store, q=seed(7)[:45])["groups"][0]["case"]["case_key"].startswith(seed(7)[:45])
    with pytest.raises(queries.QueryError):
        queries.case_page(store, q="category:reduce", backend="af-mlir", status="fail")


def test_search_timeout_maps_to_408(world):
    store, _, _ = world
    conn = store.reader()
    with pytest.raises(search.SearchError) as exc:
        with search.time_limit(conn, seconds=0):
            conn.execute("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT COUNT(*) FROM n").fetchone()
    assert exc.value.status == 408


def test_heartbeat_rollups_and_update(world):
    store, client, admin = world
    first = ops.heartbeat(store, client, {"machine_id": "m1", "npu_arch": "910b", "afx_version": "0.2.0",
                                         "backend_versions": {"af-mlir": "9.2.0"}, "counters": {"executions": 7},
                                         "pending_uploads": 3})
    assert first["update_to"] is None
    ops.request_update(store, admin, {"request_key": "u1", "client_ids": [first["client_id"]], "version": "0.2.1"})
    waiting = ops.heartbeat(store, client, {"machine_id": "m1", "afx_version": "0.2.0",
                                            "counters": {"executions": 3}})
    assert waiting["update_to"] is None and waiting["daemon_state"] == "stopped"
    assert ops.heartbeat(store, client, {"machine_id": "m1", "afx_version": "0.2.0",
                                         "daemon_state": "stopped", "daemon_request_id": waiting["daemon_request_id"],
                                         "counters": {"executions": 3}})["update_to"] == "0.2.1"
    ack = ops.heartbeat(store, client, {"machine_id": "m1"})["update_request_id"]
    assert ops.heartbeat(store, client, {"machine_id": "m1", "afx_version": "0.2.1", "update_ack": ack})["update_to"] is None
    # Throughput comes from stored executions by when they ran, not from heartbeat counters.
    with store.write() as conn:
        conn.execute("UPDATE executions SET created_at=? WHERE exec_id IN "
                     "(SELECT exec_id FROM executions WHERE backend='af-mlir' ORDER BY exec_id LIMIT 10)", (now_ms(),))
    summary = ops.summary(store)
    me = next(c for c in summary["clients"] if c["client_id"] == first["client_id"])
    assert me["last_24h"]["executions"] == 10 and me["online"] and me["backend_versions"] == {"af-mlir": "9.2.0"}
    assert summary["executions_24h"] == 10 and sum(h["executions"] for h in summary["hourly"]) == 10
    with pytest.raises(auth.AuthError):
        ops.request_update(store, client, {"request_key": "u2", "client_ids": [1], "version": "0.3"})
    with pytest.raises(IssueError):
        ops.heartbeat(store, client, {"machine_id": "m1", "counters": {"executions": -1}})


def test_daemon_control_is_owner_scoped_and_idempotent(world):
    store, client, admin = world
    created = ops.heartbeat(store, client, {"machine_id": "daemon-control"})
    request = {"request_key": "daemon-stop", "client_id": created["client_id"], "state": "stopped"}
    queued = ops.request_daemon_state(store, client, request.copy())
    assert queued["state"] == "stopped"
    assert ops.request_daemon_state(store, client, request.copy()) == queued
    beat = ops.heartbeat(store, client, {"machine_id": "daemon-control", "daemon_state": "stopped", "daemon_request_id": queued["request_id"]})
    assert beat["daemon_state"] is None
    assert ops.request_daemon_state(store, admin, {"request_key": "daemon-start", "client_id": created["client_id"], "state": "running"})["state"] == "running"


def test_efficiency_counts_recent_shapes_independently_of_windows(world, monkeypatch):
    store, client, admin = world
    now = now_ms()
    monkeypatch.setattr(ops, 'now_ms', lambda: now)
    # A deleted installation and an old window must not shadow the current run.
    old = ops.heartbeat(store, client, {"machine_id": "old", "windows": [
        {"window_id": "old-window", "rows": [{"success": 9999, "completed": 9999}]}]})['client_id']
    current = ops.heartbeat(store, client, {"machine_id": "m1", "windows": [
        {"window_id": "stale-window", "rows": [{"success": 8888}]},
        {"window_id": "recent-window", "rows": [{"success": 7777}]}]})['client_id']
    ingestor = Ingestor(store)
    records = []
    for n, backend, status, shapes in [
            (101, 'af-mlir', 'pass', [('pass', 'pass'), ('pass', 'fail'), ('runtime_fail', None)]),
            (102, 'af-mlir', 'compile_fail', []),
            (103, 'cpu', 'pass', [('pass', 'pass')]),
            (104, 'af-mlir', 'pass', [('pass', 'pass')]),
            (105, 'af-mlir', 'pass', [('pass', None)]),
            (106, 'af-mlir', 'pass', [])]:
        record = item(f'eff-exec-{n}', n, backend=backend, status=status)
        record['created_at'] = now - 120_000
        record['case'].update(is_dynamic=len(shapes)>1, shape_count=max(1,len(shapes)))
        record['shapes'] = [dict(function_status=f, accuracy_status=a) for f,a in shapes]
        records.append(record)
    body = {"request_key": "eff-upload", "client": {"machine_id": "m1"},
            "run": {"run_uuid": "eff-retest", "kind": "retest"}, "items": records}
    assert ingestor.upload(client, body, json.dumps(body).encode())['accepted'] == len(records)
    with store.write() as conn:
        conn.execute("UPDATE clients SET deleted_at=? WHERE client_id=?", (now, old))
        conn.execute("UPDATE efficiency_windows SET received_at=? WHERE window_id='stale-window'", (now-86_400_001,))
        conn.execute("UPDATE executions SET created_at=? WHERE exec_uuid='eff-exec-104'", (now-86_400_001,))
        # Recent execution from a deleted client must also be excluded.
        conn.execute("UPDATE executions SET client_id=?,created_at=? WHERE exec_uuid='exec-a0001'", (old, now))
    result = ops.efficiency(store, limit=1)
    user = next(u for u in result['users'] if u['user_id']=='dev')
    assert {k:user[k] for k in ('executions_24h','clients','success','completed','failed','case_errors')} == dict(
        executions_24h=4, clients=1, success=2, completed=4, failed=2, case_errors=1)
    assert user['success_per_minute'] == 1.0
    assert [w['window_id'] for w in ops.efficiency(store)['windows']] == ['recent-window']
    assert ops.efficiency(store, client_id=old)['users'] == []
    assert ops.efficiency(store, client_id=current, limit=0)['users'] == result['users']
    summary = ops.summary(store, admin, 'all')
    assert next(u for u in summary['users'] if u['user_id']=='dev')['executions_24h'] == user['executions_24h']


def test_summary_uses_unique_cases_and_reports_active_npu_usage(world):
    store, client, admin = world
    now = now_ms()
    with store.write() as conn:
        conn.execute("UPDATE executions SET created_at=?", (now,))
    ops.heartbeat(store, client, {
        "machine_id": "m1",
        "devices": ["npu:0", "npu:1"],
        "runtime": [{"run_id": "r1", "state": "running", "devices": [
            {"uuid": "card-0", "logical": "npu:0", "busy": True}]}],
    })
    user = next(u for u in ops.summary(store, admin, "all")["users"] if u["user_id"] == "dev")
    assert user["executions_10m"] == 40
    assert user["npu_used"] == 1 and user["npu_total"] == 2


def test_admin_can_delete_offline_client_and_heartbeat_restores_it(world):
    store, client, admin = world
    created = ops.heartbeat(store, client, {"machine_id": "delete-me"})
    with store.write() as conn:
        conn.execute("UPDATE clients SET last_heartbeat_at=? WHERE client_id=?", (now_ms() - ops.ONLINE_MS - 1, created["client_id"]))
    assert ops.delete_client(store, admin, created["client_id"])["deleted"]
    assert not any(c["client_id"] == created["client_id"] for c in ops.summary(store, admin, "all")["clients"])
    ops.heartbeat(store, client, {"machine_id": "delete-me"})
    assert any(c["client_id"] == created["client_id"] for c in ops.summary(store, admin, "all")["clients"])


def test_manual_retest_runs_on_any_build_and_updates_head(world):
    store, client, _ = world
    created = retests.create_manual(store, client, {"request_key": "b1", "case_keys": [seed(5)],
                                                    "backends": ["af-mlir"]})
    assert created["tasks"] == 1
    installed = {"af-mlir": {"version": "whatever", "sha256": "e" * 64}}
    tasks = retests.claim(store, client, {"request_key": "c1", "machine_id": "m1", "npu_arch": "910b",
                                          "installed": installed})["tasks"]
    assert len(tasks) == 1 and tasks[0]["run_kind"] == "retest" and tasks[0]["run_uuid"].startswith("retest-t")
    fixed = item("exec-retest01", 5)
    fixed["created_at"] = 10**12
    body = {"request_key": "r2", "client": {"machine_id": "m1"}, "run": {"run_uuid": tasks[0]["run_uuid"],
                                                                            "kind": "retest"}, "items": [fixed]}
    Ingestor(store).upload(client, body, json.dumps(body).encode())
    done = retests.complete(store, client, tasks[0]["task_id"], {"request_key": "d1", "lease_id": tasks[0]["lease_id"],
                                                                 "exec_uuids": ["exec-retest01"], "installed": installed})
    assert done["result"] == "pass"
    head = queries.case_page(store, case_key=seed(5))["groups"][0]["executions"]
    assert next(e for e in head if e["exec_uuid"] == "exec-retest01")["is_head"]
    batch = retests.list_manual(store, client)[0]
    assert batch["passed"] == 1
    task = batch["items"][0]
    assert task["case_key"] == seed(5)
    assert task["result"] == "pass"
    assert task["executions"] == [{"exec_uuid": "exec-retest01", "backend": "af-mlir"}]


@pytest.mark.parametrize('requester_is_admin', [False, True])
def test_manual_retest_only_requester_can_claim_even_after_expiry(world, requester_is_admin):
    store, uploader, admin = world
    auth.add_user(store, 'requester', 'developer')
    requester = admin if requester_is_admin else auth.authenticate(
        store, 'Bearer ' + auth.issue_token(store, 'requester'))
    retests.create_manual(store, requester, {'request_key': 'own-batch',
        'case_keys': [seed(5)], 'backends': ['af-mlir']})

    def claim(user, key, arch='910b', backend='af-mlir'):
        return retests.claim(store, user, {'request_key': key, 'machine_id': 'm1',
            'npu_arch': arch, 'installed': {backend: {'version': 'any', 'sha256': 'e' * 64}},
            'limit': 1})['tasks']

    assert claim(uploader, 'uploader-claim') == []
    if not requester_is_admin:
        assert claim(admin, 'admin-claim') == []
    assert claim(requester, 'wrong-arch', arch='910c') == []
    assert claim(requester, 'wrong-backend', backend='af-legacy') == []
    tasks = claim(requester, 'requester-claim')
    assert len(tasks) == 1
    with store.write() as conn:
        conn.execute('UPDATE retest_tasks SET lease_expires_at=0 WHERE task_id=?',
                     (tasks[0]['task_id'],))
    assert claim(uploader, 'uploader-reclaim') == []
    reclaimed = claim(requester, 'requester-reclaim')
    assert [t['task_id'] for t in reclaimed] == [tasks[0]['task_id']]
    assert reclaimed[0]['lease_id'] != tasks[0]['lease_id']


def test_profile_points_patterns_and_drafts(world):
    store, _, admin = world
    profile.MIN_PATTERN_CASES = 5
    profile._cache.clear()
    pts = profile.points(store, "af-mlir", "triton-npu", "node_count", "max_numel", "910b")
    assert len(pts["points"]) == 32 and pts["columns"][2] == "log2_ratio"
    pat = profile.patterns(store, "af-mlir", "triton-npu", "910b")
    assert pat["best_ops"][0]["op"] == "aten.sum.dim_IntList" and pat["best_ops"][0]["geomean_ratio"] == 0.5
    assert pat["importance"][0]["dim"] == "op_category"
    profile.DRAFT_MIN_CASES = 10
    drafts = profile.create_drafts(store, admin, {"request_key": "p1", "x": "af-mlir", "y": "triton-npu",
                                                  "npu_arch": "910b"})
    assert len(drafts["created"]) == 2
    from autofusex.server import knowledge
    entry = knowledge.get(store.reader(), drafts["created"][0], include_unpublished=True)
    assert entry["confidence"] == "measured" and entry["status"] == "draft"
    assert not knowledge.search(store.reader())["items"]  # drafts are hidden until published


def test_client_failure_attribution_uses_real_error_lines():
    generated = ("InductorError: RuntimeError: Failed to execute graph code:from autofuse.pyautofuse import ascir\n"
                 "scheduled_x = fuser.schedule(x)\n Check dtype failed for graph_hint/cast Cast; input_dtypes: [DT_INT64]")
    assert af_error_line(generated).startswith("Check dtype failed")
    status, stage, line = function_status("af-mlir", "error", "first_call_combined", generated, None, -1)
    assert (status, stage) == ("compile_fail", "graph_check") and line.startswith("Check dtype")
    assert function_status("af-mlir", "ok", "", "", None, -1) == ("pass", None, None)
    assert function_status("af-mlir", "unsupported", "", "", None, -1)[0] == "unsupported"


def test_npu_operator_error_at_first_call_is_runtime_failure_despite_lowering_warning():
    detail = {
        "execution": {"error_stage": "first_call_combined"},
        "shape_results": [{
            "error_stage": "first_call_combined",
            "error_summary": "RuntimeError: all: NPU function error: call aclnnAll failed",
        }],
    }
    log = ("torch/_inductor/lowering.py:2156: UserWarning: Torchinductor does not support code generation for complex operators\n"
           "AclNN_Parameter_Error: Tensor self not implemented for DT_COMPLEX64")
    assert classify_failure(backend="af-legacy", declared_status="compile_fail",
                            declared_stage="lowering", summary=detail["shape_results"][0]["error_summary"],
                            log=log, detail=detail)[:3] == ("runtime_fail", "first_call_combined", "npu_runtime")


def test_http_static_reserved_and_security_headers(world):
    store, _, _ = world
    server = PlatformServer(store, ("127.0.0.1", 0))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    token = auth.issue_token(store, "dev")

    def call(method, path, payload=None, auth_header=True):
        conn = httpclient.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        headers = {"Content-Type": "application/json"}
        if auth_header:
            headers["Authorization"] = "Bearer " + token
        conn.request(method, path, body=json.dumps(payload).encode() if payload is not None else None, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response, data

    try:
        response, data = call("GET", "/", auth_header=False)
        assert response.status == 200 and b"AutofuseX" in data
        assert "script-src 'self'" in response.getheader("Content-Security-Policy")
        assert response.getheader("X-Frame-Options") == "DENY"
        assert call("GET", "/app.js", auth_header=False)[0].status == 200
        assert call("GET", "/../secret.key", auth_header=False)[0].status == 404
        assert call("GET", "/static/../../core.db", auth_header=False)[0].status == 404
        assert call("GET", "/healthz", auth_header=False)[0].status == 200
        assert call("POST", "/api/import/baize", {"request_key": "x"})[0].status == 404
        assert call("GET", "/api/models/jobs/abc")[0].status == 404
        response, data = call("GET", "/api/cases?q=" + "backend:x%3BDROP")
        assert response.status == 400 and json.loads(data)["error"]["code"] == "invalid_query"
        response, data = call("GET", "/api/cases?q=category:reduce%20status:fail%20backend:af-mlir")
        assert response.status == 200 and len(json.loads(data)["data"]["groups"]) == 4
        assert call("POST", "/api/client/heartbeat", {"machine_id": "m9"})[0].status == 200
    finally:
        server.shutdown()
        server.server_close()


def test_keep_alive_survives_rejected_bodies(world):
    """A 404/401 for a request with a body must not leave the body on a reused connection."""
    store, _, _ = world
    server = PlatformServer(store, ("127.0.0.1", 0))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    token = auth.issue_token(store, "dev")
    try:
        conn = httpclient.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        junk = json.dumps({"backends": ["cpu", "af-mlir"], "pad": "x" * 5000}).encode()
        for path, headers in (("/api/legacy/upload", {"Authorization": "Bearer " + token}), ("/api/me", {})):
            conn.request("POST", path, body=junk, headers={"Content-Type": "application/json", **headers})
            response = conn.getresponse()
            response.read()
            assert response.status in (401, 404)
        conn.request("POST", "/api/client/heartbeat", body=json.dumps({"machine_id": "m1"}).encode(),
                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
        response = conn.getresponse()
        assert response.status == 200 and json.loads(response.read())["ok"]
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


def test_mapper_context_cache_survives_eviction(tmp_path):
    import sqlite3
    from autofusex.client.platform_transport import Mapper
    db = tmp_path / "results.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE contexts(id TEXT PRIMARY KEY, kind TEXT, payload TEXT)")
    conn.executemany("INSERT INTO contexts VALUES(?,?,?)", [(f"c{i}", "x", json.dumps({"i": i})) for i in range(2105)])
    conn.commit()
    mapper = Mapper(str(db))
    assert [mapper.context(f"c{i}")["i"] for i in range(2105)] == list(range(2105))
    assert mapper.context("missing") == {}


def test_summary_line_skips_wrappers():
    from autofusex.client.platform_transport import summary_line
    text = ("BackendCompilerFailed: backend='inductor' raised:\nTraceback (most recent call last):\n"
            "  File \"x.py\", line 1\nTypeError: int_noop() got an unexpected keyword argument 'decimals'\n"
            "Set TORCH_LOGS=\"+dynamo\" and TORCHDYNAMO_VERBOSE=1 for more information")
    assert summary_line(text) == "TypeError: int_noop() got an unexpected keyword argument 'decimals'"
    assert summary_line("RuntimeError: aclnnAdd failed") == "RuntimeError: aclnnAdd failed"
    only_wrapper = "InductorError: RuntimeError: Failed to execute graph code:from autofuse.pyautofuse import ascir"
    assert summary_line(only_wrapper) == only_wrapper


def test_reads_are_public_writes_need_login(world):
    store, _, _ = world
    server = PlatformServer(store, ("127.0.0.1", 0))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def call(method, path, payload=None, token=None):
        conn = httpclient.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        conn.request(method, path, body=json.dumps(payload).encode() if payload is not None else None, headers=headers)
        response = conn.getresponse()
        body = json.loads(response.read() or b"null")
        conn.close()
        return response.status, body

    try:
        for path in ("/api/overview/pipeline", "/api/overview/matrix?backends=af-mlir,triton-npu", "/api/cases?limit=5", "/api/issues", "/api/knowledge/search",
                     "/api/ops/summary", "/api/settings", "/api/retests"):
            status, body = call("GET", path)
            assert status == 200 and body["ok"], path
        assert call("GET", "/api/retests")[1]["data"] == []
        assert call("GET", "/api/me")[0] == 401
        assert call("GET", "/api/cases", token="afx_bogus")[0] == 401  # a wrong token is still an error
        for method, path, payload in (("POST", "/api/retests", {"request_key": "r", "case_keys": [seed(1)], "backends": ["af-mlir"]}),
                                      ("POST", "/api/issues/claim", {"request_key": "r", "agent_id": "a"}),
                                      ("POST", "/api/client/heartbeat", {"machine_id": "m"}),
                                      ("PUT", "/api/admin/settings", {"changes": {}, "base_revisions": {}}),
                                      ("POST", "/api/client/executions", {"request_key": "r"})):
            status, body = call(method, path, payload)
            assert status == 401 and body["error"]["code"] == "unauthorized", path
    finally:
        server.shutdown()
        server.server_close()


def test_display_names_history_spec_and_windows(tmp_path):
    store = Store(tmp_path / "data")
    auth.add_user(store, "dev", "developer", "张三")
    client = auth.authenticate(store, "Bearer " + auth.issue_token(store, "dev"))
    auth.add_user(store, "root", "admin")
    admin = auth.authenticate(store, "Bearer " + auth.issue_token(store, "root"))
    spec = {"graph": {"nodes": [{"op": "aten.add.Tensor", "display_name": "add", "inputs": ["x"], "outputs": ["y"]}],
                      "outputs": ["y"]}, "input_shapes": {"x": [2, 3]}, "input_dtypes": {"x": "float16"}, "weights": []}
    ingestor = Ingestor(store)
    for i, machine in enumerate(("m-a", "m-b", "m-a")):
        it = item(f"exec-h{i:04d}", 7)
        it["case"]["spec"] = spec
        it["case"]["features"]["op_display"] = ["add"]
        it["created_at"] = 1000 + i
        it["detail"] = {"diagnostic_view": {"shape_log": "ascir text"}, "stage_timeline": []}
        body = {"request_key": f"r{i}", "client": {"machine_id": machine}, "run": {"run_uuid": f"run-000{i}", "kind": "fuzz"},
                "items": [it]}
        ingestor.upload(client, body, json.dumps(body).encode())
    names = sorted(c["display_name"] for c in queries.clients(store))
    assert names == ["张三-npu0", "张三-npu1"]
    history = queries.case_history(store, seed(7), limit=2)
    assert history["total"] == 3 and len(history["executions"]) == 2 and history["next_cursor"]
    assert history["executions"][0]["user_name"] == "张三" and history["executions"][0]["client_name"] == "张三-npu0"
    rest = queries.case_history(store, seed(7), limit=2, after=history["next_cursor"])
    assert len(rest["executions"]) == 1 and rest["next_cursor"] is None
    assert queries.case_spec(store, seed(7))["graph"]["nodes"][0]["display_name"] == "add"
    assert queries.execution_detail(store, "exec-h0002")["detail"]["diagnostic_view"]["shape_log"] == "ascir text"
    assert queries.case_page(store, q="op:add")["groups"][0]["case"]["op_display"] == "add"
    assert queries.case_page(store, q=seed(7)[:10])["groups"][0]["case"]["case_key"] == seed(7)
    reply = ops.heartbeat(store, client, {"machine_id": "m-a", "runtime": [{"run_id": "r1", "state": "running", "elapsed": 60, "completed": 6}],
                                          "windows": [{"window_id": "w1", "rows": [{"backend": "af-mlir", "window_seconds": 60}]}]})
    assert reply["windows_accepted"] == ["w1"]
    a = next(c for c in ops.summary(store, client)["clients"] if c["display_name"] == "张三-npu0")
    assert a["runtime"][0]["run_id"] == "r1" and a["mine"]
    # runtime snapshots are private to their owner unless an admin asks for everything
    a = next(c for c in ops.summary(store)["clients"] if c["display_name"] == "张三-npu0")
    assert a["runtime"] == [] and not a["mine"]
    assert next(c for c in ops.summary(store, admin)["clients"] if c["display_name"] == "张三-npu0")["runtime"] == []
    assert next(c for c in ops.summary(store, admin, "all")["clients"] if c["display_name"] == "张三-npu0")["runtime"]
    assert ops.efficiency(store)["windows"][0]["client_name"] == "张三-npu0"


def test_client_retries_transient_errors(monkeypatch):
    from autofusex.client import platform_transport as platform_client
    calls = []

    def flaky(self, method, path, body=None):
        calls.append(path)
        if len(calls) < 3:
            raise platform_client.APIError(502, "http_error", "Bad Gateway")
        return {"ok": True}

    monkeypatch.setattr(platform_client.API, "_call", flaky)
    monkeypatch.setattr(platform_client.time, "sleep", lambda s: None)
    api = platform_client.API("http://127.0.0.1:1", "afx_x")
    assert api.call("POST", "/api/client/heartbeat", {}) == {"ok": True} and len(calls) == 3
    calls.clear()
    monkeypatch.setattr(platform_client.API, "_call", lambda self, *a: (_ for _ in ()).throw(platform_client.APIError(400, "invalid", "bad")))
    with pytest.raises(platform_client.APIError):
        api.call("POST", "/x", {})


def test_heartbeat_accepts_large_window_payload(world):
    store, _, _ = world
    server = PlatformServer(store, ("127.0.0.1", 0))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    token = auth.issue_token(store, "dev")
    try:
        rows = [{"backend": "af-mlir", "window_seconds": 60, "pad": "x" * 5000} for _ in range(20)]
        body = {"machine_id": "m1", "windows": [{"window_id": "w-big", "rows": rows}]}
        conn = httpclient.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        conn.request("POST", "/api/client/heartbeat", body=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
        response = conn.getresponse()
        assert response.status == 200 and json.loads(response.read())["data"]["windows_accepted"] == ["w-big"]
    finally:
        server.shutdown()
        server.server_close()


def test_negated_stage_keeps_rows_without_stage(world):
    store, _, _ = world
    # failing af-mlir rows in the fixture have no fail_stage; excluding a stage must not drop them
    keys = {g["case"]["case_key"] for g in queries.case_page(store, q="backend:af-mlir status:function_fail -stage:schedule,graph_check")["groups"]}
    assert keys == {seed(n) for n in (5, 10, 15, 20, 25, 30, 35, 40)}


def test_schedule_stage_from_optimize_module_logs():
    from autofusex.client.platform_transport import function_status
    ge = ("InductorError: RuntimeError: Failed to execute graph code:from autofuse.pyautofuse import ascir\n"
          "[ERROR] GE(12345,python3):2026-11-17-18:00:00.123.456 [./autofuse/optimize/autoschedule/autoschedule.cpp:812]12345 GenTilingGroup: "
          "[AUTOFUSE]tiling group gen failed for node reduce_0.\n Codegen generate kernel failed or abort")
    status, stage, line = function_status("af-mlir", "error", "first_call_combined", ge, None, -1)
    assert (status, stage) == ("compile_fail", "schedule") and "autoschedule.cpp" in line
    raised = "Traceback...\nRuntimeError: Optimize fail"
    assert function_status("af-legacy", "error", "", raised, None, -1)[1] == "schedule"
    # GE errors from files outside optimize, or on non-AF backends, are not schedule failures
    other = ("[ERROR] GE(1,p):t [./autofuse/compiler/py_module/pyautofuse.cpp:269]1 CodegenImpl: Codegen generate kernel "
             "failed\nRuntimeError: Codegen generate kernel failed")
    assert function_status("af-mlir", "error", "", "[ERROR] GE(1,p):t [schedule_utils.cpp:3]1 x: y", None, -1)[1] == "schedule"
    assert function_status("af-mlir", "error", "", other, None, -1)[1] != "schedule"
    assert function_status("triton-npu", "error", "", raised, None, -1)[1] != "schedule"
    # ASCIR graph construction happens before schedule
    check = ("InductorError: RuntimeError: Failed to execute graph code:from autofuse.pyautofuse import ascir\n"
             "Check dtype failed for graph_hint/cast Cast; input_dtypes: [DT_INT64]\nRuntimeError: Optimize fail")
    assert function_status("af-mlir", "error", "", check, None, -1)[1] == "graph_check"


def test_plog_errors_by_worker_pid_and_window(tmp_path):
    import time as _time
    from autofusex.client.platform_transport import attempt_window, plog_errors, function_status
    root = tmp_path / "plog"
    root.mkdir()
    base = _time.mktime((2026, 11, 18, 8, 56, 45, 0, 0, -1))
    stamp = lambda t: _time.strftime("%Y-%m-%d-%H:%M:%S", _time.localtime(t)) + ".971.759"
    (root / "plog-4025872_20261118085646150.log").write_text("\n".join([
        f"[ERROR] GE(4025872,python):{stamp(base - 60)} [./autofuse/optimize/x.cpp:1]4025872 Old: earlier case",
        f"[ERROR] GE(4025872,python):{stamp(base + 1)} [./autofuse/optimize/autoschedule/autoschedule.cpp:431]4025872 "
        "PrepareTilingCases: ErrorNo: 4294967295(failed)No valid tiling cases for graph",
        f"[INFO] GE(4025872,python):{stamp(base + 2)} [a.cpp:1]4025872 noise",
    ]))
    (root / "plog-999_20261118085646150.log").write_text(
        f"[ERROR] GE(999,python):{stamp(base + 1)} [./autofuse/optimize/y.cpp:2]999 other worker\n")
    import os as _os
    for f in root.iterdir():
        _os.utime(f, (base + 10, base + 10))  # files are modified after their own lines
    detail = {"stage_timeline": [{"kind": "first_call_combined", "worker_pid": 4025872, "lease": {"device_uuid": "u"}},
                                 {"kind": "reference", "worker_pid": 999, "lease": {}}],
              "lifecycle": [{"event": "dispatched", "wall_time": datetime_iso(base)},
                            {"event": "finished", "wall_time": datetime_iso(base + 5)}]}
    pids, start, end = attempt_window(detail)
    assert pids == {4025872}
    lines = plog_errors(pids, start, end, root=root)
    assert len(lines) == 1 and "PrepareTilingCases" in lines[0]
    status, stage, line = function_status("af-mlir", "error", "first_call_combined",
                                          "RuntimeError: Codegen generate kernel failed", None, -1, "\n".join(lines))
    assert stage == "schedule" and "autoschedule.cpp" in line


def datetime_iso(epoch):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def test_environment_errors_are_not_uploaded(tmp_path):
    import sqlite3
    from pathlib import Path as _Path
    from autofusex.client.platform_transport import Mapper, State, Counters, upload_new
    db = tmp_path / "results.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE attempts(id TEXT PRIMARY KEY, seed TEXT, backend TEXT, status TEXT, error_summary TEXT, "
                 "execution_id TEXT, evaluation_id TEXT, environment_id TEXT, prepare_ns INT, error_stage TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE shapes(attempt_id TEXT, shape_index INT, correctness TEXT, error_summary TEXT)")
    conn.execute("CREATE TABLE contexts(id TEXT PRIMARY KEY, kind TEXT, payload TEXT)")
    conn.execute("INSERT INTO attempts(id, backend, status, error_summary) VALUES('a1','af-mlir','error',"
                 "'AttributeError: module ''triton'' has no attribute ''language''')")
    conn.commit()

    class NoAPI:
        def call(self, *a, **k):
            raise AssertionError("environment failures must not be uploaded")

    state = State(_Path(tmp_path / "state.sqlite"))
    statuses = upload_new(NoAPI(), Mapper(str(db)), state, Counters(), key_prefix="fuzz")
    assert statuses == {"a1": "environment"}
    assert state.conn.execute("SELECT reason FROM rejected").fetchone()[0].startswith("environment:")


def test_task_configuration_inheritance_and_user_isolation(world):
    from autofusex.server import task_config
    store, dev, admin = world
    default=task_config.default()
    default['termination']={'mode':'continuous'}
    task_config.put(store,admin,{'config':default},administrator=True)
    assert task_config.get(store,dev)['config']['termination']['mode']=='continuous'
    personal=task_config.default()
    personal['workload']['generation']={'shape_mode':'dynamic','dtype_mode':'float32','max_nodes':3}
    task_config.put(store,dev,{'config':personal})
    assert task_config.get(store,dev)['source']=='personal'
    assert task_config.get(store,admin)['config']['termination']['mode']=='continuous'
    with pytest.raises(auth.AuthError):
        task_config.put(store,dev,{'config':personal},administrator=True)
    task_config.put(store,dev,{'config':None})
    assert task_config.get(store,dev)['config']['termination']['mode']=='continuous'


def test_task_configuration_http(world):
    from autofusex.client.platform_transport import API
    store, dev, admin=world
    token=auth.issue_token(store,'dev')
    server=PlatformServer(store,('127.0.0.1',0))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        api=API('http://127.0.0.1:'+str(server.server_port),token)
        result=api.call('GET','/api/config/me')
        spec=result['config'];spec['workload']['generation']={'shape_mode':'dynamic','dtype_mode':'float32'}
        saved=api.call('POST','/api/config/me',{'config':spec})
        assert saved['source']=='personal'
        assert api.call('GET','/api/config/me')['config']['workload']['generation']['shape_mode']=='dynamic'
        assert len(api.call('GET','/api/config/operators')['items'])>100
    finally:
        server.shutdown();server.server_close();thread.join()


def test_device_inventory_independent_of_runs(world):
    store, client, admin = world
    def count():
        return next(c for c in ops.summary(store, principal=admin, scope="all")["clients"]
                    if c["machine_id"] == "m1")["card_count"]
    assert count() is None
    for devices, expected in [(["npu:0", "npu:1"], 2), ([], 0), (None, None)]:
        ops.heartbeat(store, client, {"machine_id": "m1", "devices": devices, "runtime": []})
        assert count() == expected
    for devices in [["npu:0", "npu:0"], ["cpu"], [True], "npu:0"]:
        with pytest.raises(IssueError):
            ops.heartbeat(store, client, {"machine_id": "m1", "devices": devices})


def test_upgrade_pre_attribution_database(tmp_path):
    import sqlite3
    from autofusex.server import schema
    conn = sqlite3.connect(tmp_path / 'legacy.db', isolation_level=None)
    old = [(v, sql.replace('ALTER TABLE executions ADD COLUMN classifier_version INTEGER;', '')
            .replace('ALTER TABLE executions ADD COLUMN failure_owner TEXT;', ''))
           for v, sql in schema.CORE if v <= 10]
    schema.apply(conn, 'main', old)
    schema.apply(conn, 'main', schema.CORE)
    schema.apply(conn, 'main', schema.CORE)
    columns = {r[1] for r in conn.execute('PRAGMA table_info(executions)')}
    assert {'classifier_version', 'failure_owner'} <= columns
    assert conn.execute('SELECT count(*) FROM schema_migrations WHERE version=11').fetchone()[0] == 1
    conn.close()


def test_update_request_identity_survives_retries_and_changes_on_new_request(world):
    store, client, admin = world
    body = {'machine_id':'upgrade-retry', 'afx_version':'0.2.0'}
    cid = ops.heartbeat(store, client, body)['client_id']
    request = {'client_ids':[cid], 'version':'0.2.1', 'request_key':'upgrade-first'}
    ops.request_update(store, admin, dict(request))
    first = ops.heartbeat(store, client, body)['update_request_id']
    assert first
    ops.request_update(store, admin, dict(request))
    assert ops.heartbeat(store, client, body)['update_request_id'] == first
    ops.request_update(store, admin, dict(request, request_key='upgrade-second'))
    assert ops.heartbeat(store, client, body)['update_request_id'] != first
    ack = ops.heartbeat(store, client, body)['update_request_id']
    complete = ops.heartbeat(store, client, dict(body, afx_version='0.2.1', update_ack=ack))
    assert complete['update_to'] is None and complete['update_request_id'] is None
    ops.request_update(store, admin, dict(request, request_key='upgrade-third'))
    ops.request_update(store, admin, dict(request, request_key='upgrade-cancel', version=None))
    cancelled = ops.heartbeat(store, client, body)
    assert cancelled['update_to'] is None and cancelled['update_request_id'] is None


def test_execution_upload_does_not_overwrite_control_version(world):
    store, client, _ = world
    cid = ops.heartbeat(store, client, {'machine_id':'separate-runtime','afx_version':'0.2.2.9001'})['client_id']
    with store.write() as conn:
        Ingestor(store)._client(conn, client.user_id, {'machine_id':'separate-runtime','afx_version':'0.2.2.5'})
        assert conn.execute('SELECT afx_version FROM clients WHERE client_id=?',(cid,)).fetchone()[0] == '0.2.2.9001'


def test_update_requests_stop_before_releasing_package(world):
    store, client, admin = world
    cid = ops.heartbeat(store, client, {'machine_id': 'stop-before-update', 'afx_version': '0.2.0',
                                        'daemon_state': 'running'})['client_id']
    assert ops.request_update(store, admin, {'request_key': 'stop-first', 'client_ids': [cid],
                                             'version': '0.2.3.2'})['requested'] == 1
    waiting = ops.heartbeat(store, client, {'machine_id': 'stop-before-update', 'afx_version': '0.2.0'})
    assert waiting['update_to'] is None and waiting['daemon_state'] == 'stopped'
    ready = ops.heartbeat(store, client, {'machine_id': 'stop-before-update', 'afx_version': '0.2.0',
                                          'daemon_state': 'stopped', 'daemon_request_id': waiting['daemon_request_id']})
    assert ready['update_to'] == '0.2.3.2'


def test_restart_requires_matching_running_ack_and_reports_upgrade_progress(world):
    store, client, _ = world
    cid=ops.heartbeat(store,client,{'machine_id':'restart-check'})['client_id']
    cmd=ops.request_daemon_state(store,client,{'client_id':cid,'state':'restart','request_key':'restart-check'})
    for state, identity in [('stopped',cmd['request_id']),('running','stale')]:
        assert ops.heartbeat(store,client,{'machine_id':'restart-check','daemon_state':state,'daemon_request_id':identity})['daemon_state']=='restart'
    assert ops.heartbeat(store,client,{'machine_id':'restart-check','daemon_state':'running','daemon_request_id':cmd['request_id'],'update_status':{'state':'activating','version':'9.0'}})['daemon_state'] is None
    row=next(c for c in ops.summary(store,client)['clients'] if c['client_id']==cid)
    assert row['update_status']=={'state':'activating','version':'9.0'}


def test_reported_daemon_state_survives_command_ack(world):
    store, client, admin = world
    body = {"machine_id": "visible-state", "daemon_state": "stopped"}
    cid = ops.heartbeat(store, client, body)["client_id"]
    command = ops.request_daemon_state(store, client, {"client_id": cid, "state": "running", "request_key": "visible-start"})
    item = next(c for c in ops.summary(store, client)["clients"] if c["client_id"] == cid)
    assert item["daemon_state"] == "stopped"
    assert item["target_daemon_state"] == "running"
    ops.heartbeat(store, client, dict(body, daemon_state="running", daemon_request_id=command["request_id"]))
    ops.heartbeat(store, client, {"machine_id": "visible-state"})
    item = next(c for c in ops.summary(store, client)["clients"] if c["client_id"] == cid)
    assert item["daemon_state"] == "running"
    assert item["daemon_state_at"] is not None
    assert item["target_daemon_state"] is None
