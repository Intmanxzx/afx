import json
import sys
import time
from pathlib import Path
import pytest
from autofusex.client import tasks
from autofusex.client.artifacts import atomic
from autofusex.contracts.configuration import canonical
from autofusex.client.configuration import validate
from autofusex.client.store import Store


def prepared(tmp_path):
    store=Store(tmp_path/'home')
    manifest=dict(id='fixture',name='fixture',python=sys.executable,env={})
    with store.connect() as db:db.execute('INSERT INTO environments VALUES (?,?,?,?)',('fixture','fixture',canonical(manifest),'ready'))
    spec=dict(schema_version=1,kind='fuzz_task',environment='fixture',workload=dict(mode='replay',seeds=['E0F8A5E5JE7A4QBAOWXZCIT10YY']))
    spec['execution'] = dict(devices=['npu:0'], resources=dict(policy='manual', workers=1, compile_threads=1))
    spec['results'] = dict(save=True, upload=False)
    return store,spec


def test_task_budget_uses_configuration_without_legacy_authority(tmp_path):
    store,spec=prepared(tmp_path)
    atomic(store.home/'trust.json',dict(resource_limits=dict(allowed_devices=['npu:0'],max_workers=1,max_compile_threads=1,memory_budget='1GiB')))
    store.put('effective_policy',validate(dict(schema_version=1,kind='client_policy')))
    spec['execution']=dict(devices=['npu:1'], resources=dict(policy='manual',workers=2,compile_threads=3,memory_budget='2GiB'))
    result=tasks.prepare(store,spec)
    assert result['spec']['environment']=='fixture'
    assert result['spec']['execution']['devices']==['npu:1']
    assert result['policy']['resources']['max_workers']==2
    assert result['policy']['resources']['max_compile_threads']==3
    assert result['policy']['resources']['memory_budget']=='2GiB'
    assert spec['execution']['resources']==dict(policy='manual',workers=2,compile_threads=3,memory_budget='2GiB')


def test_sync_policy_apply_does_not_require_task_authority(tmp_path):
    from autofusex.client.operations import execute
    store=Store(tmp_path)
    policy=validate(dict(schema_version=1,kind='client_policy',sync=dict(interval='20s')))
    result=execute(store,dict(id='fixture',kind='config.apply',payload=policy),{})
    assert result['scope']=='sync'
    assert store.get('effective_policy')['sync']['interval']=='20s'
    assert not (tmp_path/'trust.json').exists()


def test_all_devices_expand_without_trust_file(tmp_path, monkeypatch):
    from autofusex.client import execution_runtime
    store,spec=prepared(tmp_path)
    spec['results']={'save':True,'upload':False}
    monkeypatch.setattr(execution_runtime, 'facts', lambda store, environment: {'inventory': {'devices': ['npu:0','npu:1']}})
    spec['execution']=dict(devices=['all'],backends=['cpu','af-mlir'],resources=dict(policy='manual',workers=1,compile_threads=1))
    result=tasks.prepare(store,spec)
    assert result['spec']['execution']['devices']==['npu:0','npu:1']
    assert not (store.home/'trust.json').exists()


def test_all_devices_discovery_failure_never_reuses_stale_authority(tmp_path, monkeypatch):
    from autofusex.client import execution_runtime
    store,spec=prepared(tmp_path)
    spec['execution']['devices']=['all']
    monkeypatch.setattr(execution_runtime, 'facts', lambda *args: {'inventory': {'devices': None}})
    with pytest.raises(ValueError, match='cannot discover'):
        tasks.prepare(store,spec)


def test_auto_resources_ignore_legacy_authority(tmp_path, monkeypatch):
    from autofusex.execution import resources
    store,spec=prepared(tmp_path)
    atomic(store.home/'trust.json',dict(resource_limits=dict(max_workers=1,max_compile_threads=1)))
    store.put('effective_policy',validate(dict(schema_version=1,kind='client_policy')))
    spec['execution']['resources']={'policy':'auto','memory_budget':'4GiB'}
    def plan(home, device_count, budget):
        assert device_count == 1
        assert budget.memory_budget_bytes == 4 * 1024**3
        return dict(selected=dict(cpu_workers=4,compile_threads=2),budget_limits={})
    monkeypatch.setattr(resources, 'npu_resource_plan', plan)
    resolved=tasks.prepare(store,spec)
    assert resolved['spec']['execution']['resources']['workers']==4
    assert resolved['policy']['resources']['max_compile_threads']==2


def test_owned_task_actions_and_no_pid_kill(tmp_path):
    store,spec=prepared(tmp_path)
    directory=store.home/'runs'/'run';directory.mkdir(parents=True)
    with store.connect() as db:db.execute('INSERT INTO runs VALUES (?,?,?,?,?)',('run','{}','running',12345,None))
    with pytest.raises(ValueError):tasks.action(store,'run','force-stop',{})
    assert tasks.action(store,'run','pause',{'run':object()})['state']=='pausing'
    assert json.loads((directory/'control.json').read_text())['action']=='pause'
    assert tasks.action(store,'run','stop',{'run':object()})['state']=='draining'


def test_task_lock_survives_orphan_and_blocks_new_run(tmp_path):
    import fcntl
    store,spec=prepared(tmp_path)
    with (store.home/'managed-task.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='still owns'):tasks.start(store,spec,'op',{})


def test_continuous_reconfigure_keeps_old_snapshot_until_boundary(tmp_path):
    store,spec=prepared(tmp_path)
    spec['workload']={'mode':'generate'};spec['termination']={'mode':'continuous'}
    current=tasks.prepare(store,spec);current.update(run_id='run',directory=str(store.home/'runs/run'))
    directory=store.home/'runs/run';directory.mkdir(parents=True);atomic(directory/'spec.json',current)
    with store.connect() as db:db.execute('INSERT INTO runs VALUES (?,?,?,?,?)',('run',canonical(current),'running',123,None))
    spec['execution']['timeout']='180s'
    result=tasks.reconfigure(store,'run',spec,{'run':object()})
    assert result['state']=='waiting_safe_point'
    assert json.loads((directory/'spec.json').read_text())['spec']['execution']['timeout']=='120s'
    assert json.loads((directory/'next-spec.json').read_text())['spec']['execution']['timeout']=='180s'
    assert json.loads((directory/'control.json').read_text())['action']=='reconfigure'
    with pytest.raises(ValueError):tasks.reconfigure(store,'run',spec,{'run':object()})


def test_restart_reconciliation_does_not_kill_pid(tmp_path):
    store,spec=prepared(tmp_path)
    directory=store.home/'runs'/'orphan';directory.mkdir(parents=True);atomic(directory/'spec.json',{})
    with store.connect() as db:db.execute('INSERT INTO runs VALUES (?,?,?,?,?)',('orphan','{}','needs_attention',1,None))
    tasks.action(store,'orphan','stop',{})
    tasks.refresh(store,{})
    with store.connect() as db:assert db.execute('SELECT state FROM runs WHERE id=?',('orphan',)).fetchone()[0]=='interrupted'


def test_resolved_memory_budget_can_be_reused_after_upgrade(tmp_path):
    store, spec = prepared(tmp_path)
    spec['execution']['resources'] = {'policy': 'manual', 'workers': 1,
                         'compile_threads': 1, 'memory_budget': '112742891520B'}
    resolved = tasks.prepare(store, spec)
    assert tasks.prepare(store, resolved['spec'])['spec'] == resolved['spec']


def test_upgrade_resume_accepts_legacy_resolved_auto_resources(tmp_path, monkeypatch):
    store, spec = prepared(tmp_path)
    spec['execution']['resources'] = {'policy': 'manual', 'workers': 1,
                         'compile_threads': 1, 'memory_budget': '112742891520B'}
    resolved = tasks.prepare(store, spec)
    legacy = json.loads(json.dumps(resolved['spec']))
    legacy['execution']['resources']['policy'] = 'auto'
    directory = store.home / 'runs' / 'legacy'
    directory.mkdir(parents=True)
    atomic(directory / 'spec.json', legacy)
    monkeypatch.setattr(tasks, 'prepare', lambda _store, value: {'spec': value, 'environment': {
        'id': 'fixture', 'python': sys.executable, 'env': {}}, 'policy': {}, 'hash': 'x'})
    class Child:
        pid = 123
    monkeypatch.setattr(tasks.subprocess, 'Popen', lambda *a, **kw: Child())
    result = tasks.start(store, legacy, 'op', {}, resume_run='legacy')
    assert result['run_id'] == 'legacy'
    assert json.loads((directory / 'spec.json').read_text())['spec']['execution']['resources']['policy'] == 'manual'
