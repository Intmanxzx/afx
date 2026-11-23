import json
import subprocess
import sys
import uuid

import pytest
from autofusex.execution.environment import collect


def test_managed_worker_uses_installation_home_in_child_process(tmp_path, monkeypatch):
    home = tmp_path / 'installation'
    queue_home = home / 'runs'
    queue_home.mkdir(parents=True)
    identity = str(uuid.uuid4())
    (home / 'config.json').write_text(json.dumps(dict(version=1, source_id=identity, cpu_workers=1)))
    monkeypatch.setenv('AFX_TASK_HOME', str(home))
    script = 'from autofusex.execution.environment import collect; import sys; print(collect(sys.argv[1])["installation_id"])'
    result = subprocess.run([sys.executable, '-c', script, str(queue_home)], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == identity


def test_managed_missing_configuration_fails(tmp_path, monkeypatch):
    monkeypatch.setenv('AFX_TASK_HOME', str(tmp_path))
    with pytest.raises(ValueError, match='installation configuration is missing'):
        collect(tmp_path / 'runs')
