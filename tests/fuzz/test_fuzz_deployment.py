"""Deployment entrypoint checks; real installation is exercised on Linux/NPU."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def test_deployment_help_does_not_require_runtime():
    for path in ('deploy/fuzz/scripts/install_client.sh', 'deploy/fuzz/scripts/install_server.sh',
                 'deploy/fuzz/scripts/ensure_sqlite.sh'):
        result=subprocess.run(['bash', str(ROOT/path), '--help'], capture_output=True, text=True)
        assert result.returncode==0, result.stderr
        assert 'Usage:' in result.stdout


def test_deploy_missing_arguments_fail_before_changes():
    for name in ('install_client.sh', 'install_server.sh'):
        script=ROOT/'deploy/fuzz/scripts'/name
        for args in ([], ['--wheel']):
            result=subprocess.run(['bash', str(script), *args], capture_output=True, text=True)
            assert result.returncode==2
            assert 'Usage:' in result.stderr
