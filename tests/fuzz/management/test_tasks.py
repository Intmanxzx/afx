import json
import sys
import time
from pathlib import Path
import pytest
from autofusex.fuzz.daemon.management import tasks
from autofusex.fuzz.daemon.management.artifacts import atomic
from autofusex.fuzz.daemon.management.config import canonical,validate
from autofusex.fuzz.daemon.management.store import Store


def prepared(tmp_path):
    store=Store(tmp_path/'home')
    manifest=dict(id='fixture',name='fixture',python=sys.executable,env={})
    with store.connect() as db:db.execute('INSERT INTO environments VALUES (?,?,?,?)',('fixture','fixture',canonical(manifest),'ready'))
    atomic(store.home/'trust.json',dict(resource_limits=dict(allowed_devices=['npu:0'],max_workers=1,max_compile_threads=1)))
    store.put('effective_policy',validate(dict(schema_version=1,kind='client_policy')))
    spec=dict(schema_version=1,kind='fuzz_task',environment='fixture',workload=dict(mode='replay',seeds=['E0F8A5E5JE7A4QBAOWXZCIT10YY']))
    spec['results'] = dict(save=True, upload=False)
    return store,spec


def test_task_immutable_resolution_and_resource_limits(tmp_path):
    store,spec=prepared(tmp_path)
    result=tasks.prepare(store,spec)
    assert result['spec']['environment']=='fixture'
    assert 'execution' not in spec
    spec['execution']=dict(resources=dict(policy='manual',workers=10))
    with pytest.raises(ValueError):tasks.prepare(store,spec)
    spec['execution']=dict(devices=['npu:9'])
    with pytest.raises(PermissionError):tasks.prepare(store,spec)


def test_all_devices_expand_for_local_and_policy_wildcards(tmp_path, monkeypatch):
    store,spec=prepared(tmp_path)
    spec['results']={'save':True,'upload':False}
    atomic(store.home/'trust.json',dict(resource_limits=dict(allowed_devices=['all'],max_workers=1,max_compile_threads=1)))
    store.put('effective_policy',validate(dict(schema_version=1,kind='client_policy',
        resources=dict(allowed_devices=['all'],max_workers=1,max_compile_threads=1))))
    monkeypatch.setattr(tasks, '_expand_devices', lambda values: ['npu:0','npu:1'] if 'all' in values else values)
    spec['execution']=dict(devices=['all'],backends=['cpu','af-mlir'],resources=dict(policy='manual',workers=1,compile_threads=1))
    result=tasks.prepare(store,spec)
    assert result['spec']['execution']['devices']==['npu:0','npu:1']


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
    spec['execution']={'timeout':'180s'}
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
    spec['execution'] = {'resources': {'policy': 'manual', 'workers': 1,
                         'compile_threads': 1, 'memory_budget': '112742891520B'}}
    resolved = tasks.prepare(store, spec)
    assert tasks.prepare(store, resolved['spec'])['spec'] == resolved['spec']


def test_upgrade_resume_accepts_legacy_resolved_auto_resources(tmp_path, monkeypatch):
    store, spec = prepared(tmp_path)
    spec['execution'] = {'resources': {'policy': 'manual', 'workers': 1,
                         'compile_threads': 1, 'memory_budget': '112742891520B'}}
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
