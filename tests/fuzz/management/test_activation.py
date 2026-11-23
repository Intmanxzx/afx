import json
import sys
import time
from types import SimpleNamespace
import pytest
from autofusex.client.store import Store
from autofusex.client.activation import select, finish, ready
from autofusex.client import client_updates


def test_staging_failure_preserves_pointer_and_running_tasks(tmp_path, monkeypatch):
    s = Store(tmp_path)
    s.put('supervisor_protocol', 2)
    pointer = tmp_path/'active-client.json'
    pointer.write_text(json.dumps({'version':'old','python':sys.executable}))
    before = pointer.read_bytes()
    def fail(*a, **k): raise RuntimeError('download failed')
    monkeypatch.setattr(client_updates, 'command', fail)
    processes = {'run': object()}
    from autofusex.client import upgrade_drain
    monkeypatch.setattr(upgrade_drain, 'begin', lambda *a: ['run'])
    monkeypatch.setattr(upgrade_drain, 'wait', lambda *a: processes.clear())
    with pytest.raises(RuntimeError):
        client_updates.install(s, {'version':'0.2.2.6'}, processes)
    assert pointer.read_bytes() == before
    assert not processes
    assert s.get('client_switch')['state'] == 'stage_failed'


def test_activation_requires_process_identity_heartbeat_and_restoration(tmp_path):
    s = Store(tmp_path)
    (tmp_path/'client.json').write_text('{"mode":"managed"}')
    switch = dict(target=dict(python='/new/python',version='new'),previous=dict(python='/old/python',version='old'),resume=True)
    select(s, switch)
    status = dict(agent_python='/old/python',agent_version='new',runs=[dict(id='new-run',state='running')])
    s.put('connection', dict(state='ok',time=time.time()))
    assert not ready(s,switch,status)
    status['agent_python']='/new/python'
    assert ready(s,switch,status)
    status['runs']=[]
    assert not ready(s,switch,status)
    status['runs']=[dict(id='new-run',state='running')]
    s.put('connection', dict(state='ok',time=0))
    assert not ready(s,switch,status)
    finish(s,switch,False)
    assert json.loads((tmp_path/'active-client.json').read_text()) == switch['previous']
    assert s.get('client_switch')['state']=='rolled_back'


def test_old_launcher_requires_explicit_migration(tmp_path):
    with pytest.raises(ValueError,match='migration'):
        client_updates.install(Store(tmp_path),{'version':'0.2.2.6'}, {})


def test_supervisor_switches_real_process_and_rolls_back_failed_boot(tmp_path):
    import subprocess
    from autofusex.client.daemon import request
    s=Store(tmp_path)
    child=subprocess.Popen([sys.executable,'-m','autofusex.client','daemon','supervise','--data-dir',str(tmp_path)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    def wait_for(fn):
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            try:
                value=fn()
                if value:return value
            except (OSError,ValueError):pass
            time.sleep(.05)
        raise AssertionError('supervisor transition timed out')
    try:
        before=wait_for(lambda:request(tmp_path,{'action':'status'}))
        assert before['pid']!=child.pid
        import importlib.metadata
        target=dict(python=sys.executable,version=importlib.metadata.version('autofusex'))
        op=s.submit('client.update',{'version':target['version']},'switch')
        s.state(op['id'],'activating')
        s.put('client_switch',dict(state='requested',target=target,previous=target,resume=False,operation_id=op['id']))
        wait_for(lambda:s.get('client_switch',{}).get('state')=='succeeded')
        after=request(tmp_path,{'action':'status'})
        assert after['pid']!=before['pid']
        assert s.operation(op['id'])['state']=='succeeded'
        s.put('client_switch',dict(state='requested',target=dict(python='/usr/bin/false',version='bad'),previous=target,resume=False))
        wait_for(lambda:s.get('client_switch',{}).get('state')=='rolled_back')
        restored=wait_for(lambda:request(tmp_path,{'action':'status'}))
        assert restored['agent_python']==sys.executable
        request(tmp_path,{'action':'shutdown'})
        assert child.wait(timeout=10)==0
    finally:
        if child.poll() is None:
            try:request(tmp_path,{'action':'shutdown'})
            except (OSError,ValueError):pass
            child.wait(timeout=10)


def test_local_stop_keeps_control_process_available(tmp_path):
    from autofusex.client.daemon import Manager
    manager=Manager(tmp_path)
    try:
        assert manager.dispatch({'action':'stop_session'}) == {'stopping':True}
        manager.tick()
        assert not manager.stop.is_set()
        assert manager.dispatch({'action':'status'})['pid'] > 0
    finally:manager.pool.shutdown()


def test_task_uses_registered_execution_python_not_active_control(tmp_path, monkeypatch):
    from autofusex.client import tasks
    from autofusex.client.artifacts import atomic
    s=Store(tmp_path)
    prepared={'environment':{'id':'execution','python':'/execution/python','env':{}},'hash':'test'}
    monkeypatch.setattr(tasks,'prepare',lambda *a:dict(prepared))
    atomic(tmp_path/'active-client.json',dict(python=sys.executable,version='control'))
    commands=[]
    monkeypatch.setattr(tasks.subprocess,'Popen',lambda argv,**kw:commands.append(argv) or SimpleNamespace(pid=123))
    tasks.start(s,{},'op',{})
    assert commands[0][0]=='/execution/python'
