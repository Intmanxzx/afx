from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import uuid

import pytest

from autofusex.contracts.execution import ExecutionConfig, EvaluationConfig
from autofusex.execution.scheduling.ledger import TaskLedger
from autofusex.execution.storage.transfer import check_database


EX = ExecutionConfig('cpu', 'cpu')
EV = EvaluationConfig()


def test_independent_case_backend_tasks_and_bound(tmp_path):
    ledger = TaskLedger(tmp_path/'tasks.sqlite', max_active=2)
    a = ledger.enqueue('14615016373650978276328604632813839116929039597610', EX, EV)
    b = ledger.enqueue('14615016373650978276328604632813839116929039597610', ExecutionConfig('ascendc', 'npu:0'), EV)
    assert a != b
    with pytest.raises(ValueError, match='budget'):
        ledger.enqueue('14615016373650978276328604632813839116929039597611', EX, EV)
    assert ledger.claim()['id'] == a
    assert ledger.claim()['id'] == b
    assert ledger.claim() is None
    with pytest.raises(ValueError, match='result database'):
        check_database(ledger.path)


def test_enqueue_idempotence_and_conflict(tmp_path):
    ledger = TaskLedger(tmp_path/'tasks.sqlite', max_active=1)
    identity = str(uuid.uuid4())
    assert ledger.enqueue('14615016373650978276328604632813839116929039597610', EX, EV, task_id=identity) == identity
    assert ledger.enqueue('14615016373650978276328604632813839116929039597610', EX, EV, task_id=identity) == identity
    with pytest.raises(ValueError, match='conflicts'):
        ledger.enqueue('14615016373650978276328604632813839116929039597611', EX, EV, task_id=identity)


def test_concurrent_initialization_budget_and_claims(tmp_path):
    path = tmp_path/'tasks.sqlite'
    def enqueue(i):
        try:
            return TaskLedger(path, max_active=8).enqueue(str(SamplingConfig.profile('small',max_nodes=8,dtype_mode='float32').seed(i).value), EX, EV)
        except ValueError as exc:
            assert 'budget' in str(exc)
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = [i for i in pool.map(enqueue, range(24)) if i]
        claims = list(pool.map(lambda _: TaskLedger(path).claim(), range(16)))
    assert len(ids) == 8
    assert sorted(c['id'] for c in claims if c) == sorted(ids)
    assert len({c['claim'] for c in claims if c}) == 8


def test_stop_resume_and_stale_completion_fencing(tmp_path):
    ledger = TaskLedger(tmp_path/'tasks.sqlite')
    identity = ledger.enqueue('14615016373650978276328604632813839116929039597610', EX, EV)
    claim = ledger.claim()
    ledger.stop(identity)
    assert ledger.get(identity)['state'] == 'stop_requested'
    with pytest.raises(ValueError, match='only interrupted'):
        ledger.resume(identity)
    ledger.interrupt(identity, claim['claim'])
    assert ledger.claim() is None  # Never automatically retry after interruption.
    ledger.resume(identity)
    second = ledger.claim()
    assert (second['seed'], second['execution'], second['evaluation']) == ('14615016373650978276328604632813839116929039597610', claim['execution'], claim['evaluation'])
    assert second['attempt_id'] != claim['attempt_id']
    with pytest.raises(ValueError, match='stale'):
        ledger.complete(identity, claim['claim'])
    ledger.complete(identity, second['claim'])
    assert ledger.get(identity)['state'] == 'completed'
    with pytest.raises(ValueError):
        ledger.resume(identity)


def test_pending_cancel_and_completed_stop_are_idempotent(tmp_path):
    ledger = TaskLedger(tmp_path/'tasks.sqlite')
    identity = ledger.enqueue('14615016373650978276328604632813839116929039597568', EX, EV)
    ledger.stop(identity); ledger.stop(identity)
    assert ledger.claim() is None
    ledger.resume(identity)
    claim = ledger.claim()
    ledger.stop(identity)
    ledger.complete(identity, claim['claim'])
    ledger.stop(identity)
    assert ledger.get(identity)['state'] == 'completed'


def test_process_crash_does_not_reclaim_until_explicit_recovery(tmp_path):
    ledger = TaskLedger(tmp_path/'tasks.sqlite')
    identity = ledger.enqueue('14615016373650978276328604632813839116929039597610', EX, EV)
    code = '''
import os,sys
from autofusex.execution.scheduling.ledger import TaskLedger
TaskLedger(sys.argv[1]).claim()
os._exit(79)
'''
    child = subprocess.run([sys.executable, '-c', code, str(ledger.path)])
    assert child.returncode == 79
    reopened = TaskLedger(ledger.path)
    assert reopened.claim() is None
    task = reopened.get(identity)
    assert task['state'] == 'running'
    # Child has exited and been reaped above; only now may supervisor interrupt.
    reopened.interrupt(identity, task['claim'])
    assert reopened.claim() is None
    reopened.resume(identity)
    assert reopened.claim()['id'] == identity


def test_cross_process_claim_is_unique(tmp_path):
    ledger = TaskLedger(tmp_path/'tasks.sqlite')
    ledger.enqueue('14615016373650978276328604632813839116929039597610', EX, EV)
    code = '''
import json,sys
from autofusex.execution.scheduling.ledger import TaskLedger
print(json.dumps(TaskLedger(sys.argv[1]).claim()))
'''
    children = [subprocess.Popen([sys.executable, '-c', code, str(ledger.path)], stdout=subprocess.PIPE, text=True)
                for _ in range(6)]
    results = [json.loads(child.communicate(timeout=20)[0]) for child in children]
    assert all(child.returncode == 0 for child in children)
    assert len([r for r in results if r]) == 1


def test_resume_cancelled_task_respects_capacity(tmp_path):
    ledger = TaskLedger(tmp_path/'tasks.sqlite', max_active=1)
    old = ledger.enqueue('14615016373650978276328604632813839116929039597569', EX, EV)
    ledger.stop(old)
    ledger.enqueue('14615016373650978276328604632813839116929039597570', EX, EV)
    with pytest.raises(ValueError, match='budget'):
        ledger.resume(old)
    assert ledger.get(old)['state'] == 'cancelled'


def test_scheduler_import_has_no_execution_dependency():
    code = '''
import sys
from autofusex.execution.scheduling.ledger import TaskLedger
assert not set(sys.modules) & {'torch','numpy','nnsmith','torch_npu'}
assert 'autofusex.workloads.fuzz.execution' not in sys.modules
assert 'autofusex.execution.backends' not in sys.modules
'''
    assert subprocess.run([sys.executable, '-c', code]).returncode == 0


def test_capacity_is_persisted_across_clients(tmp_path):
    path = tmp_path/'tasks.sqlite'
    TaskLedger(path, max_active=1).enqueue('14615016373650978276328604632813839116929039597610', EX, EV)
    with pytest.raises(ValueError, match='budget'):
        TaskLedger(path).enqueue('14615016373650978276328604632813839116929039597611', EX, EV)
    with pytest.raises(ValueError, match='persisted configuration'):
        TaskLedger(path, max_active=2).enqueue('14615016373650978276328604632813839116929039597612', EX, EV)
