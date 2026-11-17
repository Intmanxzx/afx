import json
import tarfile

def _short(n):
    digits, alphabet = "", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    while n:
        n, d = divmod(n, 36)
        digits = alphabet[d] + digits
    return "E" + (digits or "0").zfill(26)


import pytest

from autofusex.platform import auth, backup, queries
from autofusex.platform.cli import main as cli_main
from autofusex.platform.ingest import Ingestor, check_stats
from autofusex.platform.store import Store

ENV = {"cann": "8.2", "torch": "2.6"}


def _item(i, log=None):
    seed = _short(i)
    out = {"exec_uuid": f"exec-{i:06d}", "backend": "af-mlir", "npu_arch": "910b", "env": ENV,
           "function_status": "pass", "accuracy_status": "pass", "e2e_ns": 1, "kernel_ns": 5, "created_at": 1 + i,
           "case": {"case_key": seed, "case_source": "nnsmith", "is_dynamic": False, "shape_count": 1,
                    "dtypes": ["float16"], "op_types": ["add"]}}
    if log:
        out["log"] = log
    return out


def _upload(store, items, key):
    principal = auth.authenticate(store, "Bearer " + auth.issue_token(store, "alice", "client"))
    payload = {"request_key": key, "client": {"machine_id": "m1"}, "run": {"run_uuid": key, "kind": "fuzz"},
               "items": items}
    return Ingestor(store).upload(principal, payload, json.dumps(payload).encode())


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "data")
    auth.add_user(store, "alice", "developer")
    _upload(store, [_item(i, log=f"log {i}") for i in range(20)], "run-00000001")
    return store


def test_full_backup_restore_roundtrip(store, tmp_path):
    token = auth.issue_token(store, "alice", "agent")
    archive = backup.create(store.dir, tmp_path / "out")
    assert backup.verify(archive)["files"] == ["content.db.gz", "core.db.gz", "secret.key"]
    target = tmp_path / "restored"
    backup.restore(archive, target)
    restored = Store(target)
    assert check_stats(restored) is True  # stats.db is never in the archive
    assert queries.pipeline(restored) == queries.pipeline(store)
    assert queries.execution_detail(restored, "exec-000003")["log"] == "log 3"
    assert auth.authenticate(restored, "Bearer " + token).user_id == "alice"  # same secret.key


def test_core_only_restore_reports_missing_logs(store, tmp_path):
    full = backup.create(store.dir, tmp_path / "full")
    _upload(store, [_item(100, log="newer log")], "run-00000002")
    core = backup.create(store.dir, tmp_path / "core", include_content=False)
    target = tmp_path / "restored"
    backup.restore(core, target, content_archive=full)
    restored = Store(target)
    assert queries.execution_detail(restored, "exec-000003")["log"] == "log 3"
    assert queries.execution_detail(restored, "exec-000100")["log_missing"] is True


def test_tampered_archive_rejected(store, tmp_path):
    archive = backup.create(store.dir, tmp_path / "out")
    bad = tmp_path / "bad.tar"
    with tarfile.open(archive) as src, tarfile.open(bad, "w") as dst:
        for info in src.getmembers():
            data = src.extractfile(info).read()
            if info.name == "secret.key":
                data = b"x" * len(data)
            info.size = len(data)
            dst.addfile(info, __import__("io").BytesIO(data))
    with pytest.raises(ValueError, match="SHA-256"):
        backup.verify(bad)


def test_restore_refuses_non_empty_directory(store, tmp_path):
    archive = backup.create(store.dir, tmp_path / "out")
    with pytest.raises(ValueError, match="not empty"):
        backup.restore(archive, store.dir)


def test_cli_backup_does_not_create_data_dir(tmp_path, capsys):
    assert cli_main(["--data", str(tmp_path / "missing"), "backup", "--out", str(tmp_path / "o")]) == 2
    assert not (tmp_path / "missing").exists()


def test_backup_consistent_while_writing(store, tmp_path):
    import threading
    stop = threading.Event()

    def writer():
        n = 1000
        while not stop.is_set():
            _upload(store, [_item(n)], f"run-w{n:08d}")
            n += 1

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        archives = [backup.create(store.dir, tmp_path / f"o{i}") for i in range(3)]
    finally:
        stop.set()
        thread.join()
    for i, archive in enumerate(archives):
        target = tmp_path / f"r{i}"
        backup.restore(archive, target)
        restored = Store(target)
        check_stats(restored)
        conn = restored.reader()
        executions = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
        lane = queries.pipeline(restored)["lanes"][0]
        assert lane["executed"] == executions  # heads/stats agree with facts
        missing = conn.execute("SELECT COUNT(*) FROM executions e LEFT JOIN content.contents c "
                               "ON c.content_id=e.log_content_id WHERE e.log_content_id IS NOT NULL "
                               "AND c.content_id IS NULL").fetchone()[0]
        assert missing == 0
