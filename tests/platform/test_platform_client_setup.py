"""'afx platform client setup': bootstraps a new host (permissions, directory, one
heartbeat) without ever writing or inventing a token."""
import json
import stat
import threading
from argparse import Namespace

import pytest

from autofusex.server import auth
from autofusex.client.platform_transport import setup
from autofusex.server.server import PlatformServer
from autofusex.server.store import Store


@pytest.fixture
def server(tmp_path):
    store = Store(tmp_path / "data")
    auth.add_user(store, "alice", "developer")
    token = auth.issue_token(store, "alice")
    instance = PlatformServer(store, ("127.0.0.1", 0))
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{instance.server_address[1]}", token
    instance.shutdown()
    instance.server_close()


def args(**kw):
    base = dict(token_file=None, results=None, endpoint=None, state=None, device=None,
                retest=False, interval=30.0)
    base.update(kw)
    return Namespace(**base)


def test_missing_token_file_is_refused_without_guessing(tmp_path):
    rc = setup(args(token_file=str(tmp_path / "missing.token"), results=str(tmp_path / "r/results.sqlite"),
                    endpoint="http://127.0.0.1:1"))
    assert rc == 2


def test_empty_token_file_is_refused(tmp_path):
    token_file = tmp_path / "client.token"
    token_file.write_text("")
    rc = setup(args(token_file=str(token_file), results=str(tmp_path / "r/results.sqlite"),
                    endpoint="http://127.0.0.1:1"))
    assert rc == 2


def test_unreachable_endpoint_is_reported_not_raised(tmp_path):
    token_file = tmp_path / "client.token"
    token_file.write_text("afx_something")
    rc = setup(args(token_file=str(token_file), results=str(tmp_path / "r/results.sqlite"),
                    endpoint="http://127.0.0.1:1"))
    assert rc == 2


def test_wrong_token_is_reported_not_raised(server, tmp_path, capsys):
    endpoint, _ = server
    token_file = tmp_path / "client.token"
    token_file.write_text("afx_wrong")
    rc = setup(args(token_file=str(token_file), results=str(tmp_path / "r/results.sqlite"), endpoint=endpoint))
    assert rc == 2
    captured = capsys.readouterr()
    assert "rejected the token" in captured.out + captured.err


def test_fixes_permissions_creates_results_dir_and_confirms_with_one_heartbeat(server, tmp_path, capsys):
    endpoint, token = server
    token_file = tmp_path / "client.token"
    token_file.write_text(token)
    token_file.chmod(0o644)  # wrong on purpose: setup must fix this, not just complain
    results = tmp_path / "data" / "results" / "results.sqlite"
    rc = setup(args(token_file=str(token_file), results=str(results), endpoint=endpoint, device=["npu:0"]))
    assert rc == 0
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    assert results.parent.is_dir()
    out = capsys.readouterr().out
    assert '"ok": true' in out.lower()
    assert "afx fuzz run --save" in out
    assert f"--data-dir {results.parent.parent}" in out  # 'fuzz run --save' only ever writes <data-dir>/results/results.sqlite
    assert "--token-file" in out and str(token_file) in out
    assert "--device npu:0" in out
    assert "warnings" not in out


def test_warns_when_results_path_does_not_match_the_data_dir_layout(server, tmp_path, capsys):
    """'fuzz run --save' always writes <data-dir>/results/results.sqlite and has no flag to
    redirect it, so a --results path that doesn't follow that layout can never be populated;
    setup must flag this instead of silently printing a next step that can't work."""
    endpoint, token = server
    token_file = tmp_path / "client.token"
    token_file.write_text(token)
    results = tmp_path / "custom.sqlite"  # not .../results/results.sqlite
    rc = setup(args(token_file=str(token_file), results=str(results), endpoint=endpoint))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["next_steps"][0] == "afx fuzz run --save"  # no --data-dir: nothing sane to derive
    assert "warnings" in payload and "results/results.sqlite" in payload["warnings"][0]
