"""No hardware dependencies: synthetic cgroup mount and budget regressions."""
import json
import os
from pathlib import Path

import pytest

from autofusex.client.paths import load_config, setup
from autofusex.execution.resources import cgroup_limits, resource_inventory, budget_warnings


def test_nested_v2_limits_include_parent_constraints():
    membership = "0::/tenant/job"
    mounts = "21 20 0:28 /tenant /cg rw - cgroup2 cgroup rw"
    files = {"/cg/job/cpu.max": "800000 100000", "/cg/cpu.max": "150000 100000",
             "/cg/job/memory.max": "max", "/cg/memory.max": "1073741824"}
    result = cgroup_limits(membership, mounts, lambda p: files.get(str(p)))
    assert result == {"cpu_quota_cores": 1.5, "memory_limit_bytes": 1073741824}


def test_v1_cpu_memory_mounts_and_unlimited_sentinel():
    membership = "2:cpu,cpuacct:/job\n3:memory:/job"
    mounts = "21 20 0:28 / /cg/cpu rw - cgroup cgroup rw,cpu,cpuacct\n22 20 0:29 / /cg/mem rw - cgroup cgroup rw,memory"
    files = {"/cg/cpu/job/cpu.cfs_quota_us": "-1", "/cg/cpu/job/cpu.cfs_period_us": "100000",
             "/cg/cpu/cpu.cfs_quota_us": "200000", "/cg/cpu/cpu.cfs_period_us": "100000",
             "/cg/mem/job/memory.limit_in_bytes": str(2**63-4096)}
    result = cgroup_limits(membership, mounts, lambda p: files.get(str(p)))
    assert result == {"cpu_quota_cores": 2, "memory_limit_bytes": None}


def test_invalid_or_unavailable_cgroups_are_unknown():
    mounts = "21 20 0:28 / /cg rw - cgroup2 cgroup rw"
    assert cgroup_limits("0::/../escape", mounts, lambda p: pytest.fail("unsafe read")) == {
        "cpu_quota_cores": None, "memory_limit_bytes": None}
    assert cgroup_limits("", "") == {"cpu_quota_cores": None, "memory_limit_bytes": None}


def test_doctor_inventory_does_not_create_home(tmp_path):
    home = tmp_path / "absent" / "directory"
    inventory = resource_inventory(home)
    assert not home.exists()
    assert inventory["disk_probe_path"] == str(tmp_path)
    warnings = budget_warnings({"cpu_workers": 3, "compile_threads": 4, "memory_budget_bytes": 100},
                               {"effective_cpu_cores": 2, "effective_memory_bytes": 50, "disk_free_bytes": None})
    assert len(warnings) == 2


@pytest.mark.parametrize("field,value", [("identity", []), ("sync", "invalid"),
    ("source_id", "not-uuid"), ("cpu_workers", True), ("compile_threads", 0),
    ("memory_budget_bytes", -1), ("cache_budget_bytes", 1.5)])
def test_damaged_config_is_not_repaired_silently(tmp_path, field, value):
    setup(tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    config[field] = value
    original = json.dumps(config)
    path.write_text(original)
    with pytest.raises(ValueError):
        setup(tmp_path)
    assert path.read_text() == original
    assert not (tmp_path / ".setup.lock").exists()


def test_budget_update_preserves_source_identity_and_key_redaction(tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("never-echo-this-key")
    if os.name == "posix":
        secret.chmod(0o600)
    first = setup(tmp_path, user_id="user", display_name="Unique", key_file=secret)
    second = setup(tmp_path, compile_threads=2, memory_budget_bytes=1024, cache_budget_bytes=2048)
    assert first["source_id"] == second["source_id"]
    assert second["identity"]["configured"] and not second["identity"]["warnings"]
    assert "never-echo-this-key" not in json.dumps(second)
    config = load_config(tmp_path)
    assert config["memory_budget_bytes"] == 1024
    assert config["cache_budget_bytes"] == 2048


def test_invalid_identity_and_empty_key_preserve_config(tmp_path):
    setup(tmp_path)
    original = (tmp_path / "config.json").read_bytes()
    with pytest.raises(ValueError):
        setup(tmp_path, display_name="\n")
    key = tmp_path / "empty"
    key.touch()
    with pytest.raises(ValueError):
        setup(tmp_path, key_file=key)
    assert (tmp_path / "config.json").read_bytes() == original


def test_doctor_redacts_invalid_json(tmp_path, capsys):
    from autofusex.cli import main
    (tmp_path / "config.json").write_text("my-secret-is-not-json")
    assert main(["doctor", "--data-dir", str(tmp_path)]) == 2
    assert "my-secret" not in capsys.readouterr().err


@pytest.mark.parametrize("platform,env,expected", [
    ("win32", {"LOCALAPPDATA": "/windows/local"}, "/windows/local/autofusex"),
    ("darwin", {}, "/mock/home/Library/Application Support/autofusex"),
    ("linux", {"XDG_DATA_HOME": "/xdg/data"}, "/xdg/data/autofusex"),
])
def test_platform_path_selection_without_host_execution(monkeypatch, platform, env, expected):
    from autofusex.client import paths as config
    monkeypatch.delenv("AUTOFUSEX_HOME", raising=False)
    monkeypatch.setattr(config.sys, "platform", platform)
    monkeypatch.setattr(Path, "home", lambda: Path("/mock/home"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert str(config.data_home()) == expected


@pytest.mark.parametrize('cores,devices,memory_gib,expected', [
    (40, 2, 200, (4, 8)),
    (36, 2, 105, (4, 8)),
    (16, 2, 64, (4, 4)),
    (3, 2, 64, (3, 1)),
    (1, 1, 2, (1, 1)),
    (36, 2, 4, (2, 1)),
    (36, 2, 16, (4, 1)),
    (64, 4, 200, (6, 8)),
    (None, 2, None, (1, 1)),
    (24, 1, 64, (3, 8)),
    (64, 8, 200, (12, 5)),
    (8, 0, 32, (4, 1)),
])
def test_npu_auto_budget_distributes_effective_cores(monkeypatch, cores, devices, memory_gib, expected):
    from types import SimpleNamespace
    from autofusex.execution import resources
    memory = memory_gib * 1024**3 if memory_gib else None
    monkeypatch.setattr(resources, 'resource_inventory', lambda _: {
        'host_cpu_count': 128, 'cpu_worker_ceiling': cores,
        'effective_memory_bytes': memory*2 if memory else None})
    actual = resources.npu_first_defaults('.', devices, SimpleNamespace(memory_budget_bytes=memory))
    assert (actual['cpu_workers'], actual['compile_threads']) == expected
    assert actual['cpu_workers'] * actual['compile_threads'] <= (cores or 1)


def test_npu_auto_budget_respects_smaller_configured_memory(monkeypatch):
    from types import SimpleNamespace
    from autofusex.execution import resources
    monkeypatch.setattr(resources, 'resource_inventory', lambda _: {
        'cpu_worker_ceiling': 36, 'effective_memory_bytes': 200*1024**3})
    assert resources.npu_first_defaults('.', 2, SimpleNamespace(memory_budget_bytes=8*1024**3)) == {
        'cpu_workers': 4, 'compile_threads': 1}


@pytest.mark.parametrize('config,requested,workers,threads,expected', [
    (None, None, None, None, True),
    ({'resource_policy': 'auto'}, None, None, None, True),
    ({'cpu_workers': 6}, None, None, None, False),
    ({'resource_policy': 'manual'}, None, None, None, False),
    ({'resource_policy': 'auto'}, None, 2, None, False),
    ({'resource_policy': 'auto'}, None, None, 2, False),
    ({'resource_policy': 'auto'}, False, None, None, False),
    ({'cpu_workers': 6}, True, None, None, True),
])
def test_auto_policy_preserves_explicit_and_legacy_settings(config, requested, workers, threads, expected):
    from autofusex.execution.resources import use_auto_resources
    assert use_auto_resources(config, requested, workers, threads) is expected


def test_auto_policy_rejects_conflicting_overrides():
    from autofusex.execution.resources import use_auto_resources
    with pytest.raises(ValueError, match='cannot be combined'):
        use_auto_resources(None, True, 4, None)


def test_auto_policy_persisted_without_identity_changes(tmp_path):
    setup(tmp_path, cpu_workers=6, compile_threads=6)
    before = load_config(tmp_path)
    setup(tmp_path, resource_policy='auto')
    assert load_config(tmp_path) == dict(before, resource_policy='auto')
    original = (tmp_path/'config.json').read_bytes()
    with pytest.raises(ValueError, match='resource_policy'):
        setup(tmp_path, resource_policy='unknown')
    assert (tmp_path/'config.json').read_bytes() == original


def test_auto_plan_explains_unused_cpu_capacity(monkeypatch):
    from types import SimpleNamespace
    from autofusex.execution import resources
    monkeypatch.setattr(resources, 'resource_inventory', lambda _: {
        'host_cpu_count': 40, 'cpu_worker_ceiling': 36, 'effective_memory_bytes': 200*1024**3})
    plan = resources.npu_resource_plan('.', 2, SimpleNamespace(memory_budget_bytes=105*1024**3))
    assert plan['selected'] == {'cpu_workers': 4, 'compile_threads': 8}
    assert plan['version'] == 3 and plan['reasons'] and plan['inventory']['host_cpu_count'] == 40


def test_auto_plan_clamps_stale_machine_budgets(monkeypatch):
    from types import SimpleNamespace
    from autofusex.execution import resources
    monkeypatch.setattr(resources.os, 'name', 'posix')
    monkeypatch.setattr(resources.Path, 'exists', lambda _: True)
    monkeypatch.setattr(resources, 'resource_inventory', lambda _: {
        'cpu_worker_ceiling': 8, 'effective_memory_bytes': 16*1024**3,
        'disk_free_bytes': 10*1024**3})
    plan = resources.npu_resource_plan('.', 2, SimpleNamespace(
        memory_budget_bytes=100*1024**3, cache_budget_bytes=20*1024**3))
    assert plan['budget_limits'] == {
        'memory_budget_bytes': 8*1024**3, 'cache_budget_bytes': 2*1024**3}
    assert plan['selected'] == {'cpu_workers':4,'compile_threads':1}


@pytest.mark.parametrize('configured', [None, 100*1024**3])
def test_unknown_ram_never_uses_saved_capacity(monkeypatch, configured):
    from types import SimpleNamespace
    from autofusex.execution import resources
    monkeypatch.setattr(resources, 'resource_inventory', lambda _: {
        'cpu_worker_ceiling': 64, 'effective_memory_bytes': None})
    plan = resources.npu_resource_plan('.', 8, SimpleNamespace(memory_budget_bytes=configured))
    assert plan['selected'] == {'cpu_workers': 1, 'compile_threads': 1}
    assert plan['memory_budget_bytes'] is None
    assert 'memory_budget_bytes' not in plan['budget_limits']
    assert any('unknown' in reason for reason in plan['reasons'])


def test_subcore_quota_is_reported_not_hidden(monkeypatch):
    from types import SimpleNamespace
    from autofusex.execution import resources
    monkeypatch.setattr(resources, 'resource_inventory', lambda _: {
        'effective_cpu_cores': .5, 'cpu_worker_ceiling': 1,
        'effective_memory_bytes': 32*1024**3})
    plan = resources.npu_resource_plan('.', 2, SimpleNamespace(memory_budget_bytes=None))
    assert plan['selected'] == {'cpu_workers': 1, 'compile_threads': 1}
    assert any('throttling' in reason for reason in plan['reasons'])


def test_plan_reports_budget_corrections(monkeypatch):
    from types import SimpleNamespace
    from autofusex.execution import resources
    monkeypatch.setattr(resources, 'resource_inventory', lambda _: {
        'cpu_worker_ceiling': 8, 'effective_memory_bytes': 16*1024**3,
        'disk_free_bytes': 10*1024**3})
    plan = resources.npu_resource_plan('.', 2, SimpleNamespace(
        memory_budget_bytes=100*1024**3, cache_budget_bytes=20*1024**3))
    assert plan['budget_corrections']['cache_budget_bytes'] == {
        'configured': 20*1024**3, 'selected': 2*1024**3}


@pytest.mark.parametrize('cores,memory_gib,expected', [
    (36, 105, (4, 8)), (32, 72, (4, 8)),
    (31, 72, (4, 7)), (36, 64, (4, 7)), (36, 16, (4, 1)),
])
def test_eight_thread_policy_keeps_cpu_and_memory_bounds(monkeypatch, cores, memory_gib, expected):
    from types import SimpleNamespace
    from autofusex.execution import resources
    memory = memory_gib * 1024**3
    monkeypatch.setattr(resources, 'resource_inventory', lambda _: {
        'cpu_worker_ceiling': cores, 'effective_memory_bytes': memory*2})
    plan = resources.npu_resource_plan('.', 2, SimpleNamespace(memory_budget_bytes=memory))
    assert plan['version'] == 3
    assert (plan['selected']['cpu_workers'], plan['selected']['compile_threads']) == expected
    assert expected[0]*expected[1] <= cores
    assert expected[0]*(expected[1]+1)*2*1024**3 <= memory
