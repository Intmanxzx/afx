from autofusex.fuzz.client.generation.sampling import SamplingConfig
CURRENT_SEED = str(SamplingConfig.profile('small', max_nodes=1, dtype_mode='float32').seed(3))
"""First migration slice: no fictitious split, no inferred historic metrics."""
import json
import sqlite3
import sys
import time
import pytest

from autofusex.fuzz.client.contracts import ExecutionConfig, EvaluationConfig
from autofusex.fuzz.client.scheduling.ledger import TaskLedger
from autofusex.fuzz.client.scheduling import supervisor, campaign
from autofusex.fuzz.client.scheduling.budgets import Budget, write_policy
from autofusex.fuzz.client.storage.lifecycle import append
from autofusex.fuzz.client.storage import ResultRepository


@pytest.mark.skipif(sys.platform != 'linux', reason='managed flock/SQLite isolation requires Linux')
def test_idle_device_takes_campaign_task_not_preassigned_card(tmp_path, monkeypatch):
    ledger = TaskLedger(tmp_path/'queue')
    campaign.initialize(ledger, [ExecutionConfig('af-mlir', 'npu:0', 1, save_results=False)], EvaluationConfig(), 2, ['npu:0', 'npu:1'])
    for _ in range(2): assert campaign.refill(ledger, target_pending=3)
    # Both initial hints target card 0; eligible card 1 must still take work.
    with ledger._connection() as conn:
        conn.execute("UPDATE tasks SET execution=json_set(execution,'$.device','npu:0')")
    write_policy(ledger, Budget(cpu_workers=2))
    monkeypatch.setattr(Budget, 'admit', lambda *a: None)
    seen = []
    def command(l, d, t):
        seen.append(json.loads(t['execution'])['device'])
        return [sys.executable, '-c', 'import time; time.sleep(.3)']
    monkeypatch.setattr(supervisor, '_worker_command', command)
    assert supervisor.drain(ledger, tmp_path/'db')['completed'] == 2
    assert seen == ['npu:0', 'npu:1']


@pytest.mark.skipif(sys.platform != 'linux', reason='managed flock/SQLite isolation requires Linux')
def test_slow_finalizer_does_not_delay_other_timeout(tmp_path, monkeypatch):
    ledger = TaskLedger(tmp_path/'queue')
    for _ in range(2): ledger.enqueue(CURRENT_SEED, ExecutionConfig('cpu', 'cpu', 1), EvaluationConfig())
    write_policy(ledger, Budget(cpu_workers=2))
    monkeypatch.setattr(Budget, 'admit', lambda *a: None)
    monkeypatch.setattr(supervisor, '_worker_command', lambda *a: [sys.executable, '-c', 'import time;time.sleep(5)'])
    starts = []
    def save(*a):
        starts.append(time.monotonic())
        time.sleep(.5)
    monkeypatch.setattr(supervisor, '_save_worker_failure', save)
    assert supervisor.drain(ledger, tmp_path/'db', timeout=.2)['interrupted'] == 2
    assert abs(starts[0]-starts[1]) < .3


def test_worker_failure_never_generates_graph(tmp_path, monkeypatch):
    ledger=TaskLedger(tmp_path/'queue')
    ledger.enqueue(CURRENT_SEED,ExecutionConfig('cpu','cpu',1),EvaluationConfig())
    task=ledger.claim()
    monkeypatch.setattr('autofusex.fuzz.client.generation.materialize', lambda *a: (_ for _ in ()).throw(AssertionError('no generation')))
    supervisor._save_worker_failure(ledger,tmp_path/'db',task,'timeout','')
    assert supervisor._has_result(tmp_path/'db',task)
    assert ResultRepository(tmp_path/'db').detail(task['attempt_id'])['shape_results']==[]


def test_lifecycle_idempotent_separate_from_result_hash(tmp_path):
    from fuzz_fixtures import bundle
    from test_fuzz_precise_metrics import Adapter
    from autofusex.fuzz.client.execution import execute
    from autofusex.fuzz.client.sync.outbox import pending, acknowledge
    result=execute(bundle('C16'),Adapter(),EvaluationConfig())
    db=tmp_path/'db';repo=ResultRepository(db);identity=repo.save(result)
    original=repo.detail(identity)['result_hash']
    append(db,identity,'admitted',{'domain':'test','monotonic_ns':10,'wall_time':'test'})
    append(db,identity,'admitted',{'domain':'test','monotonic_ns':20,'wall_time':'test'})
    batch=pending(db);acknowledge(db,batch['source_id'],batch['items'][-1]['revision'])
    detail=repo.detail(identity)
    assert detail['result_hash']==original
    assert {e['event'] for e in detail['lifecycle']}=={'admitted','persisted','uploaded_ack'}
    assert detail['end_to_end_ns'] is None  # Different clock domains.
    assert next(e for e in detail['lifecycle'] if e['event']=='admitted')['monotonic_ns']==10
    assert repo.save(result,attempt_id=identity)==identity


def test_processing_span_is_not_sum_or_kernel(tmp_path):
    from fuzz_fixtures import bundle
    from test_fuzz_precise_metrics import Adapter
    from autofusex.fuzz.client.execution import execute
    from autofusex.fuzz.client.storage.queries import ResultQueries
    result=execute(bundle('C16'),Adapter(),EvaluationConfig())
    metrics=json.loads(result['execution']['telemetry'])['shapes']['0']
    assert metrics['shape_processing']>=sum(v for k,v in metrics.items() if k!='shape_processing')
    repo=ResultRepository(tmp_path/'db');repo.save(result)
    assert ResultQueries(repo.path).case_page()['items']==[]
    from copy import deepcopy
    target=deepcopy(result);target['backend']='af-mlir';target['execution']['backend']='af-mlir'
    repo.save(target)
    member=ResultQueries(repo.path).case_page()['items'][0]['backends']['cpu'][0]
    assert member['processing_ns']==metrics['shape_processing']
    assert member['total_ns']!=member['processing_ns']+sum(v for k,v in metrics.items() if k!='shape_processing')
    assert member['end_to_end_ns'] is None


def test_cpu_refill_not_blocked_by_npu_backlog(tmp_path):
    ledger=TaskLedger(tmp_path/'queue')
    configs=[ExecutionConfig('cpu','cpu',1),ExecutionConfig('af-mlir','npu:0',1)]
    campaign.initialize(ledger,configs,EvaluationConfig(),8,['npu:0','npu:1'])
    for _ in range(3): ledger.enqueue(CURRENT_SEED,configs[1],EvaluationConfig())
    assert campaign.refill(ledger,target_pending=3)
    assert any(json.loads(t['execution'])['device']=='cpu' for t in ledger.page()['items'])
    # Even a permanently unavailable device cannot cause unbounded prefetch.
    for _ in range(30):campaign.refill(ledger,target_pending=3)
    assert len(ledger.page()['items']) <= 14


def test_lifecycle_write_failure_does_not_rewrite_execution(tmp_path, monkeypatch):
    from fuzz_fixtures import bundle
    from test_fuzz_precise_metrics import Adapter
    from autofusex.fuzz.client.execution import execute
    from autofusex.fuzz.client.storage import repository
    result=execute(bundle('C16'),Adapter(),EvaluationConfig())
    repo=ResultRepository(tmp_path/'db');identity=repo.save(result)
    before=repo.detail(identity)['result_hash']
    original=repository._connect
    def unavailable(*a, **kw):
        raise sqlite3.OperationalError('injected disk write failure')
    monkeypatch.setattr(repository, '_connect', unavailable)
    with pytest.warns(RuntimeWarning, match='observation not saved'):
        append(repo.path, identity, 'finished')
    monkeypatch.setattr(repository, '_connect', original)
    assert repo.detail(identity)['result_hash']==before
