import json
import sys
import time
from pathlib import Path
import pytest
from autofusex.fuzz.management import tasks
from autofusex.fuzz.management.artifacts import atomic
from autofusex.fuzz.management.config import canonical,validate
from autofusex.fuzz.management.store import Store


def prepared(tmp_path):
    store=Store(tmp_path/'home')
    manifest=dict(id='fixture',name='fixture',python=sys.executable,env={})
    with store.connect() as db:db.execute('INSERT INTO environments VALUES (?,?,?,?)',('fixture','fixture',canonical(manifest),'ready'))
    atomic(store.home/'trust.json',dict(resource_limits=dict(allowed_devices=['npu:0'],max_workers=1,max_compile_threads=1)))
    store.put('effective_policy',validate(dict(schema_version=1,kind='client_policy')))
    spec=dict(schema_version=1,kind='fuzz_task',environment='fixture',workload=dict(mode='replay',seeds=['C000000016']))
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
