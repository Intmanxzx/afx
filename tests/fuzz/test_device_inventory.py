import subprocess
from types import SimpleNamespace

import pytest
from autofusex.fuzz.client.scheduling import devices


@pytest.mark.parametrize('output,code,expected', [
    ('''warning
{"count":2}
noise''', 0, ['npu:0', 'npu:1']),
    ('{"count":0}', 0, []), ('{"count":true}', 0, None),
    ('{"count":2}', 1, None), ('broken', 0, None),
])
def test_inventory(monkeypatch, output, code, expected):
    monkeypatch.setattr(devices, '_inventory_at', None)
    monkeypatch.setattr(devices.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=output, returncode=code))
    assert devices.device_inventory() == expected


def test_timeout_and_cache(monkeypatch):
    monkeypatch.setattr(devices, '_inventory_at', None)
    calls = []
    def timeout(*a, **k):
        calls.append(1)
        raise subprocess.TimeoutExpired('probe', 30)
    monkeypatch.setattr(devices.subprocess, 'run', timeout)
    assert devices.device_inventory() is None
    assert devices.device_inventory() is None
    assert len(calls) == 1
