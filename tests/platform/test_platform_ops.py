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

from autofusex.platform import auth, ops, profile, queries, retests, search
from autofusex.platform.client import af_error_line, function_status
from autofusex.platform.ingest import Ingestor
from autofusex.platform.issues import IssueError
from autofusex.platform.server import PlatformServer
from autofusex.platform.store import Store

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
    client = auth.authenticate(store, "Bearer " + auth.issue_token(store, "dev", "client"))
    admin = auth.authenticate(store, "Bearer " + auth.issue_token(store, "root", "web", ["admin"]))
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
    assert ops.heartbeat(store, client, {"machine_id": "m1", "afx_version": "0.2.0",
                                         "counters": {"executions": 3}})["update_to"] == "0.2.1"
    assert ops.heartbeat(store, client, {"machine_id": "m1", "afx_version": "0.2.1"})["update_to"] is None
    summary = ops.summary(store)
    me = next(c for c in summary["clients"] if c["client_id"] == first["client_id"])
    assert me["last_24h"]["executions"] == 10 and me["online"] and me["backend_versions"] == {"af-mlir": "9.2.0"}
    assert summary["executions_24h"] == 10
    with pytest.raises(auth.AuthError):
        ops.request_update(store, client, {"request_key": "u2", "client_ids": [1], "version": "0.3"})
    with pytest.raises(IssueError):
        ops.heartbeat(store, client, {"machine_id": "m1", "counters": {"executions": -1}})


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
    assert retests.list_manual(store, client)[0]["passed"] == 1


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
    from autofusex.platform import knowledge
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


def test_http_static_reserved_and_security_headers(world):
    store, _, _ = world
    server = PlatformServer(store, ("127.0.0.1", 0))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    token = auth.issue_token(store, "dev", "client")

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
        assert call("POST", "/api/import/baize", {"request_key": "x"})[0].status == 501
        assert call("GET", "/api/models/jobs/abc")[0].status == 501
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
    token = auth.issue_token(store, "dev", "client")
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
    from autofusex.platform.client import Mapper
    db = tmp_path / "results.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE contexts(id TEXT PRIMARY KEY, kind TEXT, payload TEXT)")
    conn.executemany("INSERT INTO contexts VALUES(?,?,?)", [(f"c{i}", "x", json.dumps({"i": i})) for i in range(2105)])
    conn.commit()
    mapper = Mapper(str(db))
    assert [mapper.context(f"c{i}")["i"] for i in range(2105)] == list(range(2105))
    assert mapper.context("missing") == {}


def test_summary_line_skips_wrappers():
    from autofusex.platform.client import summary_line
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
        for path in ("/api/overview/pipeline", "/api/cases?limit=5", "/api/issues", "/api/knowledge/search",
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
    client = auth.authenticate(store, "Bearer " + auth.issue_token(store, "dev", "client"))
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
    summary = ops.summary(store)
    a = next(c for c in summary["clients"] if c["display_name"] == "张三-npu0")
    assert a["runtime"][0]["run_id"] == "r1"
    assert ops.efficiency(store)["windows"][0]["client_name"] == "张三-npu0"


def test_client_retries_transient_errors(monkeypatch):
    from autofusex.platform import client as platform_client
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
    token = auth.issue_token(store, "dev", "client")
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
