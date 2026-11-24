import json

import pytest

from autofusex.client.daemon import Manager
from autofusex.client.modern_cli import _default_task
from autofusex.contracts.execution import ExecutionConfig, EvaluationConfig
from autofusex.execution.scheduling.budgets import Budget, write_policy
from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
from autofusex.workloads.fuzz.scheduling.ledger import TaskLedger


@pytest.fixture
def active(tmp_path):
    manager = Manager(tmp_path)
    spec = _default_task()
    spec['execution']['devices'] = ['npu:1']
    with manager.store.connect() as db:
        db.execute('INSERT INTO runs VALUES (?,?,?,?,?)',
                   ('active', json.dumps({'spec': spec}), 'running', 0, None))
    directory = tmp_path / 'runs' / 'active'
    directory.mkdir(parents=True)
    (directory / 'control.json').write_text(json.dumps({'action': 'run'}))
    ledger = TaskLedger(directory / 'tasks.sqlite')
    with ledger._connection():
        pass
    write_policy(ledger, Budget(compile_threads=2))
    manager.processes['active'] = object()
    yield manager, ledger
    manager.pool.shutdown()


@pytest.mark.parametrize('backend,expected', [(None, ['cpu', 'af-mlir']), ('cpu', ['cpu']), ('af-legacy', ['af-legacy'])])
def test_arbitrary_seed_joins_active_run_with_priority(active, backend, expected):
    manager, ledger = active
    seed = str(SamplingConfig().seed(123))
    normal = ledger.enqueue(seed, ExecutionConfig('cpu', 'cpu', 1), EvaluationConfig())
    result = manager.dispatch({'action': 'fuzz_replay', 'options': {
        'seed': seed, 'backend': backend, 'performance': True, 'static_only': True,
        'warmup_steps': 1, 'measured_steps': 2}})
    assert result['queue'] == str(ledger.path)
    assert not (manager.store.home / 'tasks').exists()
    assert not (ledger.path.parent / 'results.sqlite').exists()  # no history needed
    rows = [ledger.claim() for _ in expected]
    assert [r['id'] for r in rows] == result['task_ids']
    configs = [json.loads(r['execution']) for r in rows]
    assert [c['backend'] for c in configs] == expected
    assert all(c['device'] == ('cpu' if c['backend'] == 'cpu' else 'npu:1') for c in configs)
    assert all(c['compile_threads'] == 2 and c['static_only'] for c in configs)
    assert all(json.loads(r['evaluation'])['performance'] for r in rows)
    assert ledger.claim()['id'] == normal


@pytest.mark.parametrize('options', [{'device': 'npu:0'}, {'compile_threads': 3}, {'backend': 'invalid'}])
def test_invalid_options_do_not_enqueue(active, options):
    manager, ledger = active
    with pytest.raises(ValueError):
        manager.dispatch({'action': 'fuzz_replay', 'options': {'seed': str(SamplingConfig().seed(1)), **options}})
    assert ledger.claim() is None


def test_stopping_run_rejects_instead_of_starting_standalone(active):
    manager, ledger = active
    (ledger.path.parent / 'control.json').write_text(json.dumps({'action': 'stop'}))
    with pytest.raises(ValueError, match='stopping'):
        manager.dispatch({'action': 'fuzz_replay', 'options': {}})
    assert ledger.claim() is None


def test_cli_routes_to_daemon_before_device_probe(active, monkeypatch):
    from autofusex.contracts import ipc
    from autofusex.workloads.fuzz import cli
    from autofusex.workloads.fuzz.scheduling import staged
    manager, ledger = active
    monkeypatch.setattr(staged, '_wait_replay', lambda queue, ids: 0 if queue.path == ledger.path and len(ids) == 1 else 1)
    monkeypatch.setattr(ipc, 'request', lambda home, body: manager.dispatch(body))
    monkeypatch.setattr('autofusex.client.paths.data_home', lambda *args: manager.store.home)
    assert cli.replay([str(SamplingConfig().seed(4)), '--backend', 'cpu']) == 0
    assert json.loads(ledger.claim()['execution'])['backend'] == 'cpu'


def test_idle_daemon_allows_standalone(tmp_path):
    manager = Manager(tmp_path)
    try:
        assert manager.dispatch({'action': 'fuzz_replay', 'options': {}}) is None
    finally:
        manager.pool.shutdown()


def test_daemon_rejection_does_not_fall_back(active, monkeypatch):
    from autofusex.contracts import ipc
    from autofusex.workloads.fuzz import cli
    manager, ledger = active
    def reject(home, body):
        raise ValueError('run is preparing')
    monkeypatch.setattr(ipc, 'request', reject)
    monkeypatch.setattr('autofusex.client.paths.data_home', lambda *args: manager.store.home)
    with pytest.raises(ValueError, match='preparing'):
        cli.replay([str(SamplingConfig().seed(4)), '--backend', 'cpu'])
    assert ledger.claim() is None


def test_absent_daemon_preserves_standalone(active, monkeypatch):
    from autofusex.contracts import ipc
    from autofusex.workloads.fuzz import cli
    from autofusex.workloads.fuzz.scheduling import staged
    from autofusex.execution.scheduling.budgets import Budget
    manager, ledger = active
    def absent(home, body):
        raise FileNotFoundError()
    monkeypatch.setattr(ipc, 'request', absent)
    monkeypatch.setattr('autofusex.client.paths.data_home', lambda *args: manager.store.home)
    monkeypatch.setattr(Budget, 'configured', lambda *args: Budget())
    paths = []
    def drain(queue, database, **kwargs):
        paths.append(queue.path)
        assert json.loads(queue.claim()['execution'])['backend'] == 'cpu'
        return {'interrupted': 0}
    monkeypatch.setattr(staged, 'drain', drain)
    assert cli.replay([str(SamplingConfig().seed(4)), '--backend', 'cpu']) == 0
    assert len(paths) == 1 and '.replay-stages-' in str(paths[0])
