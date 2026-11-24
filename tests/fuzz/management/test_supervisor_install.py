"""Stable launcher bundle survives ordinary client source updates."""
import json
from pathlib import Path
import subprocess
from autofusex.client.supervisor.install import ensure


def test_bundle_is_independent_and_not_replaced(tmp_path):
    command = ensure(tmp_path)
    bundle = tmp_path / 'supervisor/v2'
    source = bundle / 'autofusex/client/supervisor/__init__.py'
    original = source.read_bytes()
    source.write_bytes(original + b'\n# installed revision\n')
    assert ensure(tmp_path) == command
    assert source.read_bytes().endswith(b'# installed revision\n')
    assert not (bundle / 'autofusex/client/daemon.py').exists()
    probe = subprocess.run([command[0], '-I', '-c',
        'import sys; sys.path.insert(0,sys.argv[1]); import autofusex.client.supervisor; '
        'assert "autofusex.client.daemon" not in sys.modules; '
        'assert "yaml" not in sys.modules; print("ready")', str(bundle)], capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == 'ready'
    assert json.loads((bundle / 'ready.json').read_text())['protocol'] == 3
