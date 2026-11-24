import json
import pytest
from autofusex.client.modern_cli import run
from autofusex.contracts.execution import ExecutionConfig, EvaluationConfig
from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
from autofusex.workloads.fuzz.scheduling.ledger import TaskLedger


def test_init_is_isolated_and_local_even_with_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME',str(tmp_path/'unrelated'))
    result=run(['--data-dir',str(tmp_path/'client'),'init','--endpoint','https://example.com'])
    assert result['mode']=='local'
    assert not (tmp_path/'unrelated').exists()
    assert json.loads((tmp_path/'client/task.json').read_text())['results']['upload'] is False


def test_init_device_sets_initial_local_task(tmp_path):
    run(['--data-dir', str(tmp_path/'client'), 'init', '--local', '--device', 'npu:1', '--device', 'npu:0'])
    client = json.loads((tmp_path/'client/client.json').read_text())
    task = json.loads((tmp_path/'client/task.json').read_text())
    assert client['devices'] == ['npu:1', 'npu:0']
    assert task['execution']['devices'] == ['npu:1', 'npu:0']


@pytest.mark.parametrize('dump', [False, True])
def test_managed_start_uses_fetched_configuration(tmp_path, monkeypatch, dump):
    from autofusex.client import modern_cli, daemon
    from autofusex.client.platform_transport import API
    prefix = ['--data-dir', str(tmp_path/'client')]
    run([*prefix, 'init', '--token', 'fixture', '--device', 'npu:0'])
    config = modern_cli._default_task()
    config['execution']['devices'] = ['npu:1']
    config['execution']['resources'] = dict(policy='manual', workers=2, compile_threads=3)
    monkeypatch.setattr(API, 'call', lambda *args: {'config': config})
    monkeypatch.setattr(modern_cli, '_legacy', lambda *args: {})
    def request(home, body):
        assert body == {'action': 'start_session', 'dump': True if dump else None}
        assert json.loads((home/'task.json').read_text()) == config
        return {'state': 'preparing'}
    monkeypatch.setattr(daemon, 'request', request)
    assert run([*prefix, 'start', *(['--dump'] if dump else [])]) == {'state': 'preparing'}


def test_replay_priority_survives_restart_and_is_fifo(tmp_path):
    ledger=TaskLedger(tmp_path/'queue.sqlite')
    seed=str(SamplingConfig().seed(42))
    ex=ExecutionConfig('cpu','cpu',1);ev=EvaluationConfig()
    normal=ledger.enqueue(seed,ex,ev)
    first=ledger.enqueue_many([(seed,ex,ev,None)],priority=100)[0]
    second=ledger.enqueue_many([(seed,ex,ev,None)],priority=100)[0]
    restored=TaskLedger(ledger.path)
    assert restored.claim()['id']==first
    assert restored.claim()['id']==second
    assert restored.claim()['id']==normal


def test_invalid_priority_does_not_enqueue(tmp_path):
    ledger=TaskLedger(tmp_path/'queue.sqlite')
    with pytest.raises(ValueError):
        ledger.enqueue_many([],priority=True)


def test_shape_mode_is_encoded_in_generated_seeds(tmp_path):
    from dataclasses import asdict
    from autofusex.workloads.fuzz.scheduling.campaign import initialize, refill
    for mode, dynamic in [('static',False),('dynamic',True)]:
        ledger=TaskLedger(tmp_path/(mode+'.sqlite'))
        initialize(ledger,[ExecutionConfig('cpu','cpu',1)],EvaluationConfig(),1,
                   sampling=asdict(SamplingConfig()),shape_mode=mode)
        assert refill(ledger)
        from autofusex.workloads.fuzz.contracts import Seed
        assert Seed.parse(ledger.claim()['seed']).dynamic is dynamic


def test_replay_uses_history_and_enqueues_before_generation(tmp_path):
    import sqlite3
    from autofusex.client.store import Store
    from autofusex.client.replay import submit
    from autofusex.client.modern_cli import _default_task
    from autofusex.execution.scheduling.budgets import Budget, write_policy
    store=Store(tmp_path)
    spec=_default_task();spec['execution']['backends']=['cpu']
    prepared={'spec':spec,'environment':{'id':'local'}}
    with store.connect() as db:
        db.execute('INSERT INTO runs VALUES (?,?,?,?,?)',('run',json.dumps(prepared),'running',0,None))
    directory=tmp_path/'runs/run';directory.mkdir(parents=True)
    seed=str(SamplingConfig().seed(7))
    with sqlite3.connect(directory/'results.sqlite') as db:
        db.execute('CREATE TABLE attempts(seed TEXT)');db.execute('INSERT INTO attempts VALUES (?)',(seed,))
    ledger=TaskLedger(directory/'tasks.sqlite');write_policy(ledger,Budget())
    normal=ledger.enqueue(str(SamplingConfig().seed(8)),ExecutionConfig('cpu','cpu',1),EvaluationConfig())
    first=submit(store,[seed],{'run':object()})
    second=submit(store,[seed],{'run':object()})
    assert set(first['task_ids']).isdisjoint(second['task_ids'])
    assert ledger.claim()['id']==first['task_ids'][0]
    assert ledger.claim()['id']==second['task_ids'][0]
    assert ledger.claim()['id']==normal


def test_config_roundtrip_and_reset(tmp_path):
    prefix = ['--data-dir', str(tmp_path / 'client')]
    run([*prefix, 'init', '--local'])
    spec = run([*prefix, 'config', 'export'])
    spec['execution']['devices'] = ['npu:1']
    path = tmp_path / 'task.json'
    path.write_text(json.dumps(spec))
    assert run([*prefix, 'config', 'validate', str(path)]) == spec
    assert run([*prefix, 'config'])['config']['execution']['devices'] != ['npu:1']
    assert run([*prefix, 'config', 'edit', str(path)])['config'] == spec
    assert run([*prefix, 'config', 'reset'])['config']['results']['upload'] is False
    path.write_text(json.dumps({'schema_version': 1, 'kind': 'client_policy'}))
    with pytest.raises(ValueError, match='task configuration'):
        run([*prefix, 'config', 'validate', str(path)])


def test_managed_config_export_and_validation(tmp_path, monkeypatch):
    from autofusex.client.platform_transport import API
    from autofusex.client.modern_cli import _default_task
    prefix = ['--data-dir', str(tmp_path / 'client')]
    run([*prefix, 'init', '--token', 'test-token'])
    spec = _default_task()
    calls = []
    def call(self, method, path, body=None):
        calls.append((method, path, body))
        return {'config': spec, 'source': 'personal'}
    monkeypatch.setattr(API, 'call', call)
    assert run([*prefix, 'config', 'export']) == spec
    path = tmp_path / 'task.json'
    path.write_text(json.dumps(spec))
    assert run([*prefix, 'config', 'validate', str(path)]) == spec
    assert len(calls) == 1
    run([*prefix, 'config', 'edit', str(path)])
    assert calls[-1] == ('POST', '/api/config/me', {'config': spec})
