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
    s.put('supervisor_protocol', 3)
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


def test_activation_requires_process_identity_heartbeat_and_restoration(tmp_path, monkeypatch):
    installed = []
    monkeypatch.setattr('autofusex.client.activation.install_release', installed.append)
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
    assert installed == [switch['target'], switch['previous']]
    assert not (tmp_path/'active-client.json').exists()
    assert s.get('client_switch')['state']=='rolled_back'


def test_old_launcher_requires_explicit_migration(tmp_path):
    with pytest.raises(ValueError,match='migration'):
        client_updates.install(Store(tmp_path),{'version':'0.2.2.6'}, {})


def test_supervisor_ignores_old_installation_pointers(tmp_path):
    import subprocess
    from autofusex.client.daemon import request
    s = Store(tmp_path)
    (tmp_path/'active-client.json').write_text(json.dumps({'python':'/missing/old/python'}))
    s.put('active_tool', 'obsolete')
    child = subprocess.Popen([sys.executable, '-m', 'autofusex.client', 'daemon', 'supervise',
                              '--data-dir', str(tmp_path)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for _ in range(100):
            try:
                status = request(tmp_path, {'action':'status'})
                break
            except OSError: time.sleep(.1)
        else: raise AssertionError('supervisor did not start')
        assert status['agent_python'] == sys.executable
        assert s.get('supervisor_protocol') == 3
        request(tmp_path, {'action':'shutdown'})
        assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None: child.terminate(); child.wait(timeout=10)


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
