from concurrent.futures import ThreadPoolExecutor
import sqlite3
import uuid

import pytest

from autofusex.contracts.execution import ExecutionConfig, EvaluationConfig
from autofusex.execution.scheduling.ledger import TaskLedger, DDL, APPLICATION_ID

EX, EV = ExecutionConfig('cpu','cpu',1), EvaluationConfig()


def test_batch_capacity_and_late_conflict_are_atomic(tmp_path):
    queue = TaskLedger(tmp_path/'queue', max_active=2)
    identity = queue.enqueue('14615016373650978276328604632813839116929039597610', EX, EV)
    with pytest.raises(ValueError, match='entire batch'):
        queue.enqueue_many([('14615016373650978276328604632813839116929039597611', EX, EV, None), ('14615016373650978276328604632813839116929039597612', EX, EV, None)])
    assert len(queue.page()['items']) == 1
    with pytest.raises(ValueError, match='conflicts'):
        queue.enqueue_many([('14615016373650978276328604632813839116929039597611', EX, EV, None), ('14615016373650978276328604632813839116929039597612', EX, EV, identity)])
    assert len(queue.page()['items']) == 1
    ids = queue.enqueue_many([('14615016373650978276328604632813839116929039597610', EX, EV, identity), ('14615016373650978276328604632813839116929039597611', EX, EV, None)])
    assert ids[0] == identity and len(queue.page()['items']) == 2


def test_late_invalid_spec_never_partially_enqueues(tmp_path):
    queue = TaskLedger(tmp_path/'queue')
    with pytest.raises(ValueError):
        queue.enqueue_many([('14615016373650978276328604632813839116929039597610', EX, EV, None), ('bad', EX, EV, None)])
    assert queue.page()['items'] == []


def test_concurrent_batch_capacity(tmp_path):
    path = tmp_path/'queue'
    TaskLedger(path, max_active=3).page()
    def enqueue(_):
        try:
            return TaskLedger(path).enqueue_many([('14615016373650978276328604632813839116929039597610', EX, EV, None), ('14615016373650978276328604632813839116929039597611', EX, EV, None)])
        except ValueError:
            return []
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(len(ids) for ids in pool.map(enqueue, range(2))) == [0, 2]
    assert len(TaskLedger(path).page()['items']) == 2










def test_reason_is_bounded_and_survives_reopen(tmp_path):
    path = tmp_path/'queue'
    queue = TaskLedger(path); identity = queue.enqueue('14615016373650978276328604632813839116929039597610', EX, EV)
    claim = queue.claim()
    with pytest.raises(ValueError, match='short text'):
        queue.interrupt(identity, claim['claim'], 'x'*501)
    queue.interrupt(identity, claim['claim'], 'timeout')
    assert TaskLedger(path).get(identity)['reason'] == 'timeout'
    queue.resume(identity)
    assert queue.get(identity)['reason'] == ''
