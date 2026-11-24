import json
from types import SimpleNamespace
from unittest.mock import Mock
from autofusex.client import platform_retests as retests
from autofusex.client.store import Store


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


def test_stopped_client_does_not_claim(tmp_path, monkeypatch):
    store=Store(tmp_path)
    (tmp_path/'client.json').write_text(json.dumps({'mode':'managed','platform':{'retest':True}}))
    spawn=Mock();monkeypatch.setattr(retests.subprocess,'Popen',spawn)
    retests.tick(SimpleNamespace(store=store,processes={}))
    spawn.assert_not_called()


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
