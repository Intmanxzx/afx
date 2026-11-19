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

from autofusex.platform import aggregates, auth, queries
from autofusex.platform.ingest import Ingestor, UploadError, check_stats, update_settings
from autofusex.platform.server import PlatformServer
from autofusex.platform.store import Store

SEED_A = _short(1)
SEED_B = _short(2)
FP1, FP2 = "1" * 64, "2" * 64
ENV = {"cann": "8.2", "torch": "2.6", "backend_version": "0.3.0", "os": "ubuntu22", "cpu_arch": "arm"}


def case(key=SEED_A, source="nnsmith", fingerprint=None, ops=("add",), category=("elementwise",)):
    return {"case_key": key, "case_source": source, "graph_fingerprint": fingerprint, "is_dynamic": False,
            "shape_count": 1, "node_count": 3, "op_types": list(ops), "dtypes": ["float16"],
            "max_dim": 128, "max_numel": 4096, "features": {"op_category": list(category)},
            "spec": {"graph": key}}


def item(uuid, backend="af-mlir", status="pass", accuracy="pass", kernel=None, created=1000, c=None, **extra):
    out = {"exec_uuid": uuid, "case": c or case(), "backend": backend,
           "npu_arch": "" if backend == "cpu" else "910b", "env": ENV, "function_status": status,
           "accuracy_status": accuracy if status == "pass" else None, "e2e_ns": 10, "kernel_ns": kernel,
           "created_at": created}
    out.update(extra)
    return out


def body(items, key="req-1", run="run-0001", kind="fuzz"):
    return {"request_key": key, "client": {"machine_id": "m1", "npu_arch": "910b"},
            "run": {"run_uuid": run, "kind": kind}, "items": items}


@pytest.fixture
def env(tmp_path):
    store = Store(tmp_path / "data")
    auth.add_user(store, "alice", "developer")
    auth.add_user(store, "root", "admin")
    token = auth.issue_token(store, "alice")
    principal = auth.authenticate(store, "Bearer " + token)
    return store, Ingestor(store), principal


def upload(ingestor, principal, payload):
    raw = json.dumps(payload).encode()
    return ingestor.upload(principal, payload, raw)


def stats_snapshot(store):
    conn = store.reader()
    return (sorted(map(tuple, conn.execute("SELECT * FROM stats.stat_buckets"))),
            sorted((r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], round(r[8], 9))
                   for r in conn.execute("SELECT * FROM stats.perf_pair_buckets")))


# -- auth ----------------------------------------------------------------------
def test_token_lifecycle(env):
    store, _, principal = env
    assert principal.user_id == "alice"
    principal.require_write()
    with pytest.raises(auth.AuthError):
        principal.require_admin()
    token = auth.issue_token(store, "alice")
    assert auth.authenticate(store, "Bearer " + token).user_id == "alice"
    auth.revoke_token(store, token[:12])
    with pytest.raises(auth.AuthError):
        auth.authenticate(store, "Bearer " + token)
    with pytest.raises(auth.AuthError):
        auth.authenticate(store, "Bearer afx_not-a-token")
    with pytest.raises(ValueError):
        auth.issue_token(store, "nobody")


def test_reissue_revokes_previous_token(env):
    store, _, _ = env
    first = auth.issue_token(store, "alice")
    second = auth.issue_token(store, "alice")
    with pytest.raises(auth.AuthError):
        auth.authenticate(store, "Bearer " + first)
    assert auth.authenticate(store, "Bearer " + second).user_id == "alice"


def test_disabled_user_rejected(env):
    store, _, _ = env
    token = auth.issue_token(store, "alice")
    auth.set_user_status(store, "alice", "disabled")
    with pytest.raises(auth.AuthError):
        auth.authenticate(store, "Bearer " + token)


def test_tokens_stored_as_hmac_only(env):
    store, _, _ = env
    token = auth.issue_token(store, "alice")
    raw = (store.dir / "core.db").read_bytes()
    assert token.encode() not in raw


# -- upload --------------------------------------------------------------------
def test_upload_idempotent_and_conflicts(env):
    store, ingestor, principal = env
    payload = body([item("exec-0001"), {"exec_uuid": "bad"}])
    first = upload(ingestor, principal, payload)
    assert [r["status"] for r in first["results"]] == ["accepted", "rejected"]
    assert upload(ingestor, principal, payload) == first
    changed = body([item("exec-0001", status="compile_fail")])
    with pytest.raises(UploadError) as exc:
        upload(ingestor, principal, changed)
    assert exc.value.status == 409
    second = upload(ingestor, principal, dict(changed, request_key="req-2"))
    assert second["results"][0]["status"] == "conflict"
    third = upload(ingestor, principal, body([item("exec-0001")], key="req-3"))
    assert third["results"][0]["status"] == "duplicate"


def test_viewer_cannot_upload(env):
    store, ingestor, _ = env
    auth.add_user(store, "eve", "viewer")
    viewer = auth.authenticate(store, "Bearer " + auth.issue_token(store, "eve"))
    with pytest.raises(auth.AuthError):
        upload(ingestor, viewer, body([item("exec-0001")]))


def test_validation_rejects_bad_case_keys(env):
    _, ingestor, principal = env
    bad_seed = item("exec-0001", c=case(key=str(10 * 2**160 + 1)))
    bad_baize = item("exec-0002", c=case(key="torch/x", source="baize_torch", fingerprint=FP1))
    accuracy_on_failure = item("exec-0003", status="compile_fail")
    accuracy_on_failure["accuracy_status"] = "fail"
    result = upload(ingestor, principal, body([bad_seed, bad_baize, accuracy_on_failure]))
    assert [r["status"] for r in result["results"]] == ["rejected"] * 3


def test_environment_and_content_deduplicated(env):
    store, ingestor, principal = env
    log = "error: schedule failed\n" * 100
    items = [item(f"exec-{i:04d}", c=case(key=_short(100 + i)), status="compile_fail",
                  fail_stage="schedule", log=log) for i in range(5)]
    upload(ingestor, principal, body(items))
    conn = store.reader()
    assert conn.execute("SELECT COUNT(*) FROM environments").fetchone()[0] == 1
    # five distinct specs + one shared log
    assert conn.execute("SELECT COUNT(*) FROM content.contents").fetchone()[0] == 6
    detail = queries.execution_detail(store, "exec-0000")
    assert detail["log"] == log and detail["env"]["cann"] == "8.2"


# -- heads and aggregates -------------------------------------------------------
def test_pipeline_and_head_replacement(env):
    store, ingestor, principal = env
    upload(ingestor, principal, body([
        item("exec-0001", backend="cpu"),
        item("exec-0002", backend="af-mlir", status="compile_fail", fail_stage="schedule", created=1000),
        item("exec-0003", backend="af-legacy", accuracy="fail"),
        item("exec-0004", backend="af-mlir", c=case(key=SEED_B)),
    ]))
    lanes = {lane["backend"]: lane for lane in queries.pipeline(store)["lanes"]}
    assert lanes["af-mlir"]["stage_fail"] == {"schedule": 1} and lanes["af-mlir"]["pass"] == 1
    assert lanes["af-legacy"]["accuracy_fail"] == 1 and lanes["cpu"]["baseline"]
    # newer retest replaces the head; an older late upload does not
    upload(ingestor, principal, body([item("exec-0005", created=2000)], key="req-2", run="run-0002", kind="retest"))
    upload(ingestor, principal, body([item("exec-0006", status="timeout", created=500)], key="req-3", run="run-0003"))
    lanes = {lane["backend"]: lane for lane in queries.pipeline(store)["lanes"]}
    assert lanes["af-mlir"]["pass"] == 2 and lanes["af-mlir"]["function_fail"] == 0
    assert queries.pipeline(store)["total_cases"] == 2
    incremental = stats_snapshot(store)
    with store.write() as conn:
        aggregates.rebuild(conn, 0.05)
    assert stats_snapshot(store) == incremental


def test_verify_runs_never_become_current(env):
    store, ingestor, principal = env
    upload(ingestor, principal, body([item("exec-0001", status="compile_fail")]))
    upload(ingestor, principal, body([item("exec-0002", created=9999)], key="req-2", run="run-0002", kind="verify"))
    groups = queries.case_page(store)["groups"]
    heads = [e["exec_uuid"] for e in groups[0]["executions"] if e["is_head"]]
    assert heads == ["exec-0001"] and len(groups[0]["executions"]) == 2


def test_perf_pair_and_flip(env):
    store, ingestor, principal = env
    upload(ingestor, principal, body([
        item("exec-0001", backend="af-mlir", kernel=50),
        item("exec-0002", backend="triton-npu", kernel=100),
        item("exec-0003", backend="af-mlir", kernel=102, c=case(key=SEED_B)),
        item("exec-0004", backend="triton-npu", kernel=100, c=case(key=SEED_B)),
    ]))
    forward = queries.perf_pair(store, "af-mlir", "triton-npu")["groups"][0]
    assert (forward["faster"], forward["tie"], forward["slower"]) == (1, 1, 0)
    backward = queries.perf_pair(store, "triton-npu", "af-mlir")["groups"][0]
    assert (backward["faster"], backward["tie"], backward["slower"]) == (0, 1, 1)
    assert forward["geomean_ratio"] == pytest.approx((0.5 * 1.02) ** 0.5)
    by_category = queries.perf_pair(store, "af-mlir", "triton-npu", dim="op_category")["groups"]
    assert by_category[0]["value"] == "elementwise" and by_category[0]["total"] == 2
    # a failing head leaves the comparison
    upload(ingestor, principal, body([item("exec-0005", backend="triton-npu", status="runtime_fail",
                                           created=2000)], key="req-2", run="run-0002"))
    assert queries.perf_pair(store, "af-mlir", "triton-npu")["groups"][0]["total"] == 1


def test_tie_threshold_change_rebuilds(env):
    store, ingestor, principal = env
    upload(ingestor, principal, body([item("exec-0001", kernel=103), item("exec-0002", backend="triton-npu",
                                                                           kernel=100)]))
    assert queries.perf_pair(store, "af-mlir", "triton-npu")["groups"][0]["tie"] == 1
    revision = store.settings()["perf.tie_threshold"]["revision"]
    update_settings(store, "root", {"perf.tie_threshold": 0.01}, {"perf.tie_threshold": revision})
    assert queries.perf_pair(store, "af-mlir", "triton-npu")["groups"][0]["slower"] == 1
    with pytest.raises(UploadError):
        update_settings(store, "root", {"perf.tie_threshold": 0.02}, {"perf.tie_threshold": revision})
    with pytest.raises(UploadError):
        update_settings(store, "root", {"regression.shares": {"neighbor": 90}}, {"regression.shares": 1})


def test_baize_graph_update_retires_old_results(env):
    store, ingestor, principal = env
    key = "B:torch/ops/test_add_001"
    v1 = case(key=key, source="baize_torch", fingerprint=FP1)
    v2 = case(key=key, source="baize_torch", fingerprint=FP2, category=("reduce",))
    upload(ingestor, principal, body([item("exec-0001", c=v1)]))
    assert queries.pipeline(store)["lanes"][0]["pass"] == 1
    upload(ingestor, principal, body([item("exec-0002", c=v2, status="compile_fail", created=2000)],
                                     key="req-2", run="run-0002", kind="import"))
    group = queries.case_page(store)["groups"][0]
    assert group["case"]["case_revision"] == 2
    old = next(e for e in group["executions"] if e["exec_uuid"] == "exec-0001")
    assert old["stale_graph"] and not old["is_head"]
    lane = queries.pipeline(store)["lanes"][0]
    assert lane["pass"] == 0 and lane["function_fail"] == 1
    assert queries.backend_stats(store, "af-mlir", "op_category")["groups"][0]["value"] == "reduce"
    # a late upload of the old graph is kept as history of revision 1 only
    upload(ingestor, principal, body([item("exec-0003", c=v1, created=3000)], key="req-3", run="run-0003"))
    group = queries.case_page(store)["groups"][0]
    assert group["case"]["case_revision"] == 2
    assert next(e for e in group["executions"] if e["exec_uuid"] == "exec-0003")["case_revision"] == 1
    assert queries.pipeline(store)["lanes"][0]["function_fail"] == 1
    snapshot = stats_snapshot(store)
    with store.write() as conn:
        aggregates.rebuild(conn, 0.05)
    assert stats_snapshot(store) == snapshot


def test_stats_rebuilt_when_out_of_step(env):
    store, ingestor, principal = env
    upload(ingestor, principal, body([item("exec-0001")]))
    expected = stats_snapshot(store)
    with store.write() as conn:
        conn.execute("DELETE FROM stats.stat_buckets")
        conn.execute("UPDATE stats.stats_meta SET value='-1'")
    assert check_stats(store) is True
    assert stats_snapshot(store) == expected
    assert check_stats(store) is False


def test_case_page_filters_and_cursor(env):
    store, ingestor, principal = env
    items = [item(f"exec-{i:04d}", c=case(key=_short(100 + i)),
                  status="compile_fail" if i % 2 else "pass") for i in range(6)]
    upload(ingestor, principal, body(items))
    page = queries.case_page(store, limit=4)
    assert len(page["groups"]) == 4 and page["next_cursor"] is not None
    rest = queries.case_page(store, limit=4, after=page["next_cursor"])
    assert len(rest["groups"]) == 2 and rest["next_cursor"] is None
    failing = queries.case_page(store, backend="af-mlir", status="function_fail")
    assert len(failing["groups"]) == 3
    with pytest.raises(queries.QueryError):
        queries.case_page(store, status="fail")


# -- HTTP ----------------------------------------------------------------------
@pytest.fixture
def http(env):
    store, _, _ = env
    server = PlatformServer(store, ("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = auth.issue_token(store, "alice")

    def call(method, path, payload=None, token=token, raw=None, content_type="application/json"):
        conn = httpclient.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        headers = {"Content-Type": content_type}
        if token:
            headers["Authorization"] = "Bearer " + token
        data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
        conn.request(method, path, body=data, headers=headers)
        response = conn.getresponse()
        result = response.status, json.loads(response.read())
        conn.close()
        return result

    yield call
    server.shutdown()
    server.server_close()


def test_http_roundtrip(http):
    assert http("GET", "/api/me", token=None)[0] == 401
    assert http("GET", "/api/me", token="afx_wrong")[0] == 401
    status, me = http("GET", "/api/me")
    assert status == 200 and me["data"]["user_id"] == "alice"
    status, result = http("POST", "/api/client/executions", body([item("exec-0001", kernel=10),
                                                                   item("exec-0002", backend="triton-npu",
                                                                        kernel=20)]))
    assert status == 200 and result["data"]["accepted"] == 2
    status, pipe = http("GET", "/api/overview/pipeline?arch=910b")
    assert status == 200 and {lane["backend"] for lane in pipe["data"]["lanes"]} == {"af-mlir", "triton-npu"}
    status, perf = http("GET", "/api/overview/perf?x=af-mlir&y=triton-npu")
    assert perf["data"]["groups"][0]["faster"] == 1
    status, cases = http("GET", f"/api/cases?case_key={SEED_A}")
    assert len(cases["data"]["groups"][0]["executions"]) == 2
    assert http("GET", "/api/executions/exec-0001")[1]["data"]["backend"] == "af-mlir"
    assert http("GET", "/api/executions/missing-id")[0] == 404
    assert http("GET", "/api/nope")[0] == 404


def test_http_rejects_bad_input(http):
    assert http("GET", "/api/cases?backend=x%27%20OR%201=1")[0] == 400
    assert http("GET", "/api/cases?limit=100000")[0] == 400
    assert http("POST", "/api/client/executions", raw=b"{not json")[0] == 400
    assert http("POST", "/api/client/executions", raw=b"{}", content_type="text/plain")[0] == 415
    status, payload = http("PUT", "/api/admin/settings", {"changes": {}, "base_revisions": {}})
    assert status == 403 and payload["error"]["code"] == "forbidden"
