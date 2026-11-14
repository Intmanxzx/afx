"""Deployment entrypoint checks; real installation is exercised on Linux/NPU."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_deployment_help_does_not_require_runtime():
    for path in ('deploy/fuzz/scripts/install_client.sh', 'deploy/fuzz/scripts/install_server.sh',
                 'deploy/fuzz/scripts/ensure_sqlite.sh',
                 'autofusedebugger/nnsmith_integration/install_af_fuzz.sh'):
        result=subprocess.run(['bash', str(ROOT/path), '--help'], capture_output=True, text=True)
        assert result.returncode==0, result.stderr
        assert 'Usage:' in result.stdout


def test_nnsmith_never_resets_an_unmanaged_checkout(tmp_path):
    source=tmp_path/'shared'; source.mkdir()
    marker=source/'user-work.py'; marker.write_text('KEEP = True\n')
    env=dict(os.environ, PYTHON_BIN=sys.executable, NNSMITH_DIR=str(source), INSTALL_AF_FUZZ_SKIP_PIP='1')
    result=subprocess.run(['bash', str(ROOT/'autofusedebugger/nnsmith_integration/install_af_fuzz.sh')],
                          env=env, capture_output=True, text=True)
    assert result.returncode!=0
    assert 'unmanaged directory' in result.stderr
    assert marker.read_text()=='KEEP = True\n'


def test_deploy_missing_arguments_fail_before_changes():
    for name in ('install_client.sh', 'install_server.sh'):
        script=ROOT/'deploy/fuzz/scripts'/name
        for args in ([], ['--wheel']):
            result=subprocess.run(['bash', str(script), *args], capture_output=True, text=True)
            assert result.returncode==2
            assert 'Usage:' in result.stderr
