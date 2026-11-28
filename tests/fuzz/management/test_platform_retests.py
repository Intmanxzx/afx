import json
import pytest
from types import SimpleNamespace
from unittest.mock import Mock
from autofusex.client import platform_retests as retests
from autofusex.client.store import Store


@pytest.mark.parametrize('selection, expected', [(['npu:1'], ['npu:1']), (['all'], ['npu:0','npu:1'])])
def test_retests_use_task_devices_without_trust(tmp_path, monkeypatch, selection, expected):
    from autofusex.client import platform_transport, retest_queue
    from autofusex.execution import environment
    from autofusex.execution.scheduling import devices
    (tmp_path/'client.json').write_text(json.dumps(dict(endpoint='https://fixture.invalid', token='fixture')))
    (tmp_path/'task.json').write_text(json.dumps(dict(execution=dict(devices=selection))))
    client=Mock()
    create=Mock(return_value=client)
    monkeypatch.setattr(platform_transport, 'Client', create)
    monkeypatch.setattr(devices, 'device_inventory_info', lambda: dict(devices=['npu:0','npu:1'], model='fixture'))
    monkeypatch.setattr(environment, 'collect', lambda *args: {})
    monkeypatch.setattr(retest_queue, 'synchronize', lambda *args: 0)
    retests.run(tmp_path)
    assert create.call_args.kwargs['devices']==expected
    assert not (tmp_path/'trust.json').exists()
    client.state.conn.close.assert_called_once()


def test_runner_uses_execution_environment_and_does_not_duplicate(tmp_path, monkeypatch):
    store = Store(tmp_path)
    (tmp_path/'client.json').write_text(json.dumps({'mode':'managed','platform':{'retest':True}}))
    manifest = {'python':'/execution/bin/python','env':{'PYTHONPATH':'/cann/python'}}
    with store.connect() as db:
        db.execute("INSERT INTO environments VALUES ('local','local',?,'ready')", (json.dumps(manifest),))
    manager = SimpleNamespace(store=store, processes={'run':object()})
    child = Mock(pid=123);child.poll.return_value=None
    spawn = Mock(return_value=child)
    monkeypatch.setattr(retests.subprocess,'Popen',spawn)
    monkeypatch.setenv('PYTHONPATH','/control/python')
    retests.tick(manager);retests.tick(manager)
    assert spawn.call_count == 1
    args,kwargs=spawn.call_args
    assert args[0][0]=='/execution/bin/python'
    assert kwargs['env']['PYTHONPATH'].endswith(':/cann/python')
    assert (tmp_path / 'retest').is_dir()
    assert kwargs['stdout'].name.endswith('/retest/runner.log')
    assert kwargs['start_new_session']
    kill = Mock();monkeypatch.setattr(retests.os,'killpg',kill)
    (tmp_path/'daemon-control.json').write_text(json.dumps({'state':'stopped'}))
    retests.tick(manager)
    kill.assert_called_once_with(123,retests.signal.SIGTERM)
    assert manager._retest_process is None


def test_explicitly_stopped_client_does_not_claim(tmp_path, monkeypatch):
    store=Store(tmp_path)
    (tmp_path/'client.json').write_text(json.dumps({'mode':'managed','platform':{'retest':True}}))
    (tmp_path/'daemon-control.json').write_text(json.dumps({'state':'stopped'}))
    spawn=Mock();monkeypatch.setattr(retests.subprocess,'Popen',spawn)
    retests.tick(SimpleNamespace(store=store,processes={}))
    spawn.assert_not_called()


def test_retest_runs_without_fuzz_and_switches_independently(tmp_path, monkeypatch):
    from autofusex.client.workload_control import receive, states
    store = Store(tmp_path)
    (tmp_path/'client.json').write_text(json.dumps({'mode':'managed','platform':{'retest':False}}))
    with store.connect() as db:
        db.execute("INSERT INTO environments VALUES ('local','local',?,'ready')",
                   (json.dumps(dict(python='/execution/bin/python', env={})),))
    manager = SimpleNamespace(store=store, processes={})
    child = Mock(pid=123); child.poll.return_value = None
    spawn = Mock(return_value=child)
    monkeypatch.setattr(retests.subprocess, 'Popen', spawn)
    kill = Mock(); monkeypatch.setattr(retests.os, 'killpg', kill)
    receive(store, dict(fuzz=dict(state='stopped', request_id='f1'),
                        retest=dict(state='running', request_id='r1')))
    retests.tick(manager)
    spawn.assert_called_once()
    assert states(store)['retest'] == dict(state='running', request_id='r1')
    receive(store, dict(retest=dict(state='stopped', request_id='r2')))
    retests.tick(manager)
    kill.assert_not_called()
    assert states(store)['retest']['state'] == 'stopping'
    child.poll.return_value = 0
    retests.tick(manager)
    assert states(store)['retest']['state'] == 'stopped'
    assert json.loads((tmp_path/'daemon-control.json').read_text())['request_id'] == 'f1'


@pytest.mark.parametrize('already_running', [False, True])
def test_local_start_after_platform_stop_restores_retest_runner(tmp_path, monkeypatch, already_running):
    from autofusex.client.daemon import Manager
    from autofusex.client import local_session
    from autofusex.client.session_control import reconcile
    from autofusex.client.artifacts import atomic
    store = Store(tmp_path)
    atomic(tmp_path/'client.json', dict(mode='managed', platform=dict(retest=True)))
    atomic(tmp_path/'daemon-control.json', dict(request_id='old-stop', state='stopped'))
    store.put('session_control', dict(request_id='old-stop', state='stopped', status='succeeded'))
    with store.connect() as db:
        db.execute("INSERT INTO environments VALUES ('local','local',?,'ready')",
                   (json.dumps(dict(python='/execution/bin/python', env={})),))
        db.execute("INSERT INTO runs VALUES ('run','{}','running',1,NULL)")
    manager = SimpleNamespace(store=store, processes={'run':object()} if already_running else {}, active=None)
    def start(m, dump=None):
        m.processes['run'] = object()
        return dict(run_ids=['run'], state='running')
    monkeypatch.setattr(local_session, 'start', start)
    child = Mock(pid=123)
    child.poll.return_value = None
    spawn = Mock(return_value=child)
    monkeypatch.setattr(retests.subprocess, 'Popen', spawn)
    retests.tick(manager)
    spawn.assert_not_called()
    assert Manager.dispatch(manager, dict(action='start_session'))['run_ids'] == ['run']
    reconcile(manager)
    retests.tick(manager)
    retests.tick(manager)
    assert store.get('session_control')['state'] == 'running'
    assert store.get('session_control')['status'] == 'succeeded'
    assert json.loads((tmp_path/'daemon-control.json').read_text())['state'] == 'running'
    spawn.assert_called_once()


def test_failed_local_start_preserves_stop_command(tmp_path, monkeypatch):
    from autofusex.client.daemon import Manager
    from autofusex.client import local_session
    from autofusex.client.artifacts import atomic
    store = Store(tmp_path)
    command = dict(request_id='old-stop', state='stopped')
    atomic(tmp_path/'daemon-control.json', command)
    manager = SimpleNamespace(store=store, processes={})
    def fail(*args, **kwargs):
        raise ValueError('invalid configuration')
    monkeypatch.setattr(local_session, 'start', fail)
    with pytest.raises(ValueError, match='invalid configuration'):
        Manager.dispatch(manager, dict(action='start_session'))
    assert json.loads((tmp_path/'daemon-control.json').read_text()) == command


def test_claim_journal_resumes_without_claiming_again(tmp_path):
    underlying = Mock()
    underlying.call.return_value = {'tasks':[{'task_id':7},{'task_id':8}]}
    path=tmp_path/'claims.json'
    api=retests.JournalAPI(underlying,path)
    assert len(api.call('POST','/api/client/retests/claim',{'request_key':'one'})['tasks'])==2
    restarted=retests.JournalAPI(underlying,path)
    assert len(restarted.call('POST','/api/client/retests/claim',{'request_key':'two'})['tasks'])==2
    assert underlying.call.call_count==1
    restarted.call('POST','/api/client/retests/7/complete',{})
    assert json.loads(path.read_text())['tasks']==[{'task_id':8}]


def test_standalone_replay_reaches_scheduler(tmp_path, monkeypatch):
    from autofusex.workloads.fuzz.scheduling import staged
    from autofusex.client import paths as config
    from autofusex.execution.scheduling.budgets import Budget
    from autofusex.workloads.fuzz.contracts import Seed
    monkeypatch.setattr(config,'data_home',lambda *args:tmp_path)
    monkeypatch.setattr(Budget,'configured',lambda *args:Budget(cpu_workers=1,compile_threads=1))
    drain=Mock(return_value={'interrupted':0,'failed':0})
    monkeypatch.setattr(staged,'drain',drain)
    args=SimpleNamespace(seed='F0SQR0WLFHGE5JG59LC7NA60RQ1',backend='cpu',device='cpu',
        database=str(tmp_path/'results.sqlite'),save=False,compile_threads=1,performance=False,
        warmup_steps=1,measured_steps=1,static_only=True)
    assert staged.replay(args)==0
    drain.assert_called_once()
