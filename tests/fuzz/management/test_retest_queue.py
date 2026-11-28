import json
from autofusex.client.store import Store
from autofusex.client.retest_queue import submit
from autofusex.workloads.fuzz.scheduling.ledger import TaskLedger
from autofusex.execution.scheduling.budgets import Budget, write_policy
from autofusex.execution.scheduling.placement import candidates


def test_retest_without_active_fuzz_creates_only_explicit_tasks(tmp_path):
    Store(tmp_path)
    spec = {'execution': {'devices': ['npu:0'], 'static_only': False},
            'evaluation': {'performance': {'enabled': False, 'warmup_steps': 1, 'measured_steps': 1}}}
    (tmp_path/'task.json').write_text(json.dumps(spec))
    task = dict(task_id=77, lease_id='isolated', run_uuid='retest-run',
                case_key='F0SQR0WLFHGE5JG59LC7NA60RQ1', backends=['af-mlir'])
    saved = submit(tmp_path, task)
    assert saved['standalone']
    ledger = TaskLedger(saved['queue'])
    assert len(ledger.page()['items']) == 1
    assert submit(tmp_path, task)['task_ids'] == saved['task_ids']
    with ledger._connection() as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='campaign'").fetchone()


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
    from autofusex.client.paths import results_database
    configured = tmp_path / 'archive' / 'custom.sqlite'
    (tmp_path / 'client.json').write_text(json.dumps({'platform': {'results': str(configured)}}))
    assert results_database(tmp_path) == configured
    assert results_database(tmp_path, tmp_path / 'override.sqlite') == tmp_path / 'override.sqlite'


def test_cancelled_retest_reports_terminal_receipt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from autofusex.client import retest_queue, platform_transport, daemon
    task = dict(task_id=202, lease_id='lease', case_source='nnsmith')
    saved = dict(queue=str(tmp_path/'tasks.sqlite'), task_ids=['local'])
    api = Mock(); api.call.return_value = {'tasks': [task]}
    client = SimpleNamespace(api=api, devices=['npu:0'], environment=lambda: {}, identity=lambda: {})
    monkeypatch.setattr(platform_transport, 'fresh_installed', lambda: {'af-mlir': {'version':'1','sha256':'a'*64}})
    monkeypatch.setattr(daemon, 'request', lambda *a: saved)
    ledger = Mock(); ledger.get.return_value = dict(state='cancelled', reason='no_autofuse_coverage')
    monkeypatch.setattr(retest_queue, 'TaskLedger', lambda _: ledger)
    assert retest_queue._synchronize(client, tmp_path) == 1
    body = api.call.call_args.args[2]
    assert body['terminal_reason'] == 'no_autofuse_coverage'
    assert body['exec_uuids'] == []
    assert saved['delivered']


def test_retest_uploads_baseline_but_completes_original_task(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from autofusex.client import retest_queue, platform_transport, daemon
    task = dict(task_id=203, lease_id='lease', case_source='nnsmith',run_uuid='run',run_kind='retest')
    saved = dict(queue=str(tmp_path/'tasks.sqlite'), database=str(tmp_path/'results.sqlite'),task_ids=['local'])
    api = Mock(); api.call.return_value = {'tasks':[task]}
    client = SimpleNamespace(api=api,devices=['npu:0'],environment=lambda: {},identity=lambda: {},state=Mock(),counters={})
    monkeypatch.setattr(platform_transport,'fresh_installed',lambda: {'af-mlir': {'version': '1'}})
    monkeypatch.setattr(daemon,'request',lambda *a:saved)
    ledger = Mock(); ledger.get.return_value = dict(state='completed')
    monkeypatch.setattr(retest_queue,'TaskLedger',lambda _:ledger)
    monkeypatch.setattr(retest_queue,'completed',lambda _:['target'])
    mapper = Mock()
    mapper.conn.execute.return_value.fetchone.return_value={'execution_id':'context'}
    mapper.context.return_value={'diagnostics':json.dumps({'baseline_exec_uuid':'baseline'})}
    monkeypatch.setattr(platform_transport,'Mapper',lambda _:mapper)
    upload = Mock(return_value={'target':'accepted','baseline':'accepted'})
    monkeypatch.setattr(platform_transport,'upload_new',upload)
    assert retest_queue._synchronize(client,tmp_path)==1
    assert upload.call_args.kwargs['attempt_ids']==['baseline','target']
    assert api.call.call_args.args[2]['exec_uuids']==['target']
    mapper.conn.close.assert_called_once()


def test_journal_task_rejected_before_enqueue_after_package_change(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from autofusex.client import platform_transport, retest_queue, daemon
    old = dict(version='8.2', sha256='a'*64)
    new = dict(version='8.2', sha256='b'*64)
    task = dict(task_id=7, lease_id='old-lease', case_source='nnsmith', required=dict(backend='af-mlir', **old))
    api = Mock(); api.call.return_value = {'tasks': [task]}
    client = SimpleNamespace(api=api, devices=['npu:0'], identity=lambda: {})
    monkeypatch.setattr(platform_transport, 'fresh_installed', lambda: {'af-mlir': new})
    enqueue = Mock(); monkeypatch.setattr(daemon, 'request', enqueue)
    assert retest_queue._synchronize(client, tmp_path) == 1
    enqueue.assert_not_called()
    assert api.call.call_args.args[2]['terminal_reason'] == 'environment_changed'


def test_fresh_package_collection_bypasses_client_cache(monkeypatch):
    from autofusex.client import platform_transport
    from autofusex.execution import environment
    values = iter([dict(cann='8.2', autofuse_binary='a'*64), dict(cann='8.2', autofuse_binary='b'*64)])
    monkeypatch.setattr(environment, 'collect', lambda: next(values))
    assert platform_transport.fresh_installed() != platform_transport.fresh_installed()


def test_changed_during_execution_is_terminal_without_result_upload(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from autofusex.client import platform_transport
    old = {'af-mlir': dict(version='8.2', sha256='a'*64)}
    new = {'af-mlir': dict(version='8.2', sha256='b'*64)}
    task = dict(task_id=8, lease_id='lease', required=dict(backend='af-mlir', **old['af-mlir']))
    client = SimpleNamespace(api=Mock())
    monkeypatch.setattr(platform_transport, 'fresh_installed', lambda: new)
    assert platform_transport.Client._reject_changed_task(client, task, old)
    body = client.api.call.call_args.args[2]
    assert body['exec_uuids'] == [] and body['installed'] == new
    assert body['terminal_reason'] == 'environment_changed'


def test_noaf_receipt_waits_for_audit_commit(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from autofusex.client import retest_queue, platform_transport, daemon
    task = dict(task_id=205, lease_id='lease', case_source='nnsmith')
    saved = dict(queue=str(tmp_path/'tasks.sqlite'), task_ids=['local'], evidence_version=1)
    api = Mock(); api.call.return_value = {'tasks': [task]}
    client = SimpleNamespace(api=api, devices=['npu:0'], environment=lambda: {}, identity=lambda: {})
    monkeypatch.setattr(platform_transport, 'fresh_installed', lambda: {'af-mlir': {'version':'1','sha256':'a'*64}})
    monkeypatch.setattr(daemon, 'request', lambda *a: saved)
    ledger = Mock(); ledger.get.return_value = dict(state='cancelled', reason='no_autofuse_coverage')
    monkeypatch.setattr(retest_queue, 'TaskLedger', lambda _: ledger)
    assert retest_queue._synchronize(client, tmp_path) == 0
    assert api.call.call_count == 1
    assert not saved.get('delivered')
    observed = {'torch_npu': 'worker-build'}
    monkeypatch.setattr(retest_queue, 'skipped_evidence', lambda _: [{'environment': observed}])
    assert retest_queue._synchronize(client, tmp_path) == 1
    assert api.call.call_args.args[2]['coverage_environment'] == observed
