import json
import sys
from pathlib import Path
import pytest
from autofusex.client.artifacts import atomic
from autofusex.client.store import Store
from autofusex.client import tasks
from autofusex.workloads.imported.registry import register


def fixture(tmp_path):
    store=Store(tmp_path/'home')
    atomic(store.home/'config.json',dict(version=1,source_id='12345678-1234-1234-1234-123456789abc',cpu_workers=1))
    script=tmp_path/'case.py';script.write_text('raise RuntimeError("fixture execution")')
    case=register(store.home,script)
    with store.connect() as db:
        db.execute('INSERT INTO environments VALUES (?,?,?,?)',('local','local',json.dumps(dict(id='local',name='local',python=sys.executable,env={})), 'ready'))
    spec=dict(schema_version=1,kind='imported_task',environment='local',
        workload=dict(case_key=case['case_key'],revision=case['revision']),
        execution=dict(devices=['npu:0']),results=dict(save=True,upload=False))
    return store,spec


def test_source_admission_freezes_revision_and_rejects_fuzz_controls(tmp_path):
    store,spec=fixture(tmp_path)
    value=tasks.prepare(store,spec)
    assert value['spec']['workload']==spec['workload']
    with store.connect() as db:
        db.execute('INSERT INTO runs VALUES (?,?,?,?,?)',('run',json.dumps(value),'running',123,None))
    (store.home/'runs/run').mkdir(parents=True)
    with pytest.raises(ValueError, match='pause/resume'):
        tasks.action(store,'run','pause',{'run': object()})
    assert tasks.action(store,'run','force-stop',{'run':object()})['state']=='cancelling'


def test_model_task_records_execution_exception(tmp_path,monkeypatch):
    from autofusex.workloads.model.task import run
    from autofusex.workloads.model import script_runner
    store,spec=fixture(tmp_path)
    value=tasks.prepare(store,spec)
    directory=store.home/'runs/run'; directory.mkdir(parents=True)
    value.update(directory=str(directory),run_id='run')
    monkeypatch.setenv('AFX_TASK_HOME',str(store.home))
    def fail(**kwargs): raise RuntimeError('fixture failed')
    monkeypatch.setattr(script_runner,'run_framework_script',fail)
    monkeypatch.setattr('autofusex.workloads.model.task.acquire_device',lambda device:None)
    assert run(value)==1
    result=json.loads((directory/'result.json').read_text())
    assert result['status']=='execution_failed'
    assert result['message']=='fixture failed'
    assert result['case']['revision']==spec['workload']['revision']
    assert not result['artifacts']


def test_source_task_stop_owns_only_its_child(tmp_path,monkeypatch):
    import subprocess
    from autofusex.execution import source_task
    directory=tmp_path/'run';directory.mkdir()
    atomic(directory/'control.json', {'action':'stop'})
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
    monkeypatch.setattr(source_task.subprocess,'Popen',lambda *a,**kw:child)
    spec=dict(directory=str(directory),spec=dict(execution=dict(resources={},timeout='1h')))
    assert source_task.run(directory/'spec.json',spec)==1
    assert child.poll() is not None
    assert json.loads((directory/'state.json').read_text())['reason']=='stop'


def test_source_receipt_journal_survives_duplicate_claim(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from autofusex.client import source_jobs
    from autofusex.workloads.imported.registry import export
    store,spec=fixture(tmp_path)
    atomic(store.home/'client.json',{'mode':'managed'})
    case_key=spec['workload']['case_key']
    bundle=export(store.home,case_key)
    calls=[]
    class API:
        def call(self,method,path,body=None):
            calls.append(path)
            if path.endswith('/claim'):return {'job':dict(job_id='job',lease_id='lease',spec=spec,bundle=bundle)}
            return {'state':'completed'}
    manager=SimpleNamespace(store=store,processes={},active=None)
    api=API();source_jobs.synchronize(manager,api,'machine')
    first=store.get('source_job')
    store.put('source_job',None)
    source_jobs.synchronize(manager,api,'machine')
    assert store.get('source_job')['operation_id']==first['operation_id']
    store.state(first['operation_id'],'failed',{'message':'environment unavailable'})
    source_jobs.synchronize(manager,api,'machine')
    assert store.get('source_job') is None
    assert calls[-1]=='/api/source-jobs/job/complete'


def test_manager_executes_claimed_source_operation_and_returns_receipt(tmp_path, monkeypatch):
    import time
    from autofusex.client.daemon import Manager
    from autofusex.client import source_jobs
    from autofusex.workloads.imported.registry import export
    store, spec = fixture(tmp_path)
    atomic(store.home/'client.json', {'mode': 'managed'})
    bundle = export(store.home, spec['workload']['case_key'])
    completed = []
    class API:
        def call(self, method, path, body=None):
            if path.endswith('/claim'):
                return {'job': dict(job_id='job', lease_id='lease', spec=spec, bundle=bundle)}
            completed.append(body)
            return {'state': 'completed'}
    def start(store, payload, operation_id, processes):
        prepared = tasks.prepare(store, payload)
        assert prepared['spec']['kind'] == 'imported_task'
        with store.connect() as db:
            db.execute('INSERT INTO runs VALUES (?,?,?,?,?)', ('run', json.dumps(prepared), 'stopped', None, None))
        return {'run_id': 'run'}
    monkeypatch.setattr(tasks, 'start', start)
    manager = Manager(store.home)
    try:
        api = API()
        source_jobs.synchronize(manager, api, 'machine')
        operation_id = store.get('source_job')['operation_id']
        deadline = time.monotonic() + 5
        while store.operation(operation_id)['state'] != 'succeeded' and time.monotonic() < deadline:
            manager.tick()
            time.sleep(.01)
        assert store.operation(operation_id)['state'] == 'succeeded'
        source_jobs.synchronize(manager, api, 'machine')
        assert completed[-1]['result']['status'] == 'stopped'
        assert completed[-1]['result']['case']['case_key'] == spec['workload']['case_key']
        assert store.get('source_job') is None
    finally:
        manager.pool.shutdown(wait=True)
