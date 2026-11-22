import json
from autofusex.fuzz.daemon.management.store import Store
from autofusex.fuzz.daemon.management.retest_queue import submit
from autofusex.fuzz.client.scheduling.ledger import TaskLedger
from autofusex.fuzz.client.scheduling.budgets import Budget, write_policy
from autofusex.fuzz.client.scheduling.placement import candidates


def test_retest_inherits_active_run_and_is_idempotent(tmp_path):
    store = Store(tmp_path)
    directory = tmp_path / 'runs' / 'one'
    directory.mkdir(parents=True)
    spec = {'execution': {'devices': ['npu:0', 'npu:1'], 'static_only': False},
            'evaluation': {'performance': {'enabled': True, 'warmup_steps': 7, 'measured_steps': 13}}}
    (directory / 'spec.json').write_text(json.dumps({'spec': spec}))
    (directory / 'control.json').write_text(json.dumps({'action': 'run'}))
    database = directory / 'results.sqlite'
    (directory / 'tasks.sqlite.results.json').write_text(json.dumps({'database': str(database)}))
    with store.connect() as db:
        db.execute("INSERT INTO runs(id,spec,state) VALUES ('one',?,'running')", (json.dumps({'spec': spec}),))
    ledger = TaskLedger(directory / 'tasks.sqlite')
    write_policy(ledger, Budget(cpu_workers=2, compile_threads=3))
    task = {'task_id': 7, 'lease_id': 'lease-one', 'case_key': 'F0SQR0WLFHGE5JG59LC7NA60RQ1', 'backends': ['af-mlir']}
    saved = submit(tmp_path, task)
    assert saved['database'] == str(database)
    assert submit(tmp_path, task)['task_ids'] == saved['task_ids']
    row = ledger.get(saved['task_ids'][0])
    assert json.loads(row['evaluation']) == {'correctness': True, 'performance': True, 'warmup_steps': 7, 'measured_steps': 13}
    assert json.loads(row['execution'])['compile_threads'] == 3
    assert candidates(ledger, row) == ['npu:0', 'npu:1']
    assert ledger.priorities()[row['id']] == 100
    # Lost response after claim must not enqueue the same remote task again.
    ledger.claim(row['id'])
    assert submit(tmp_path, task)['task_ids'] == saved['task_ids']
    assert len(ledger.page()['items']) == 1


def test_saved_database_path_is_used_without_cli_option(tmp_path):
    from autofusex.fuzz.client.config import results_database
    configured = tmp_path / 'archive' / 'custom.sqlite'
    (tmp_path / 'client.json').write_text(json.dumps({'platform': {'results': str(configured)}}))
    assert results_database(tmp_path) == configured
    assert results_database(tmp_path, tmp_path / 'override.sqlite') == tmp_path / 'override.sqlite'
