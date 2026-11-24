"""Exercise real staged admission/pause boundaries without Torch, CANN or NPU."""
from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
CURRENT_SEED = str(SamplingConfig.profile("small", max_nodes=1).seed(3))
import json
from pathlib import Path
import threading
import time
from autofusex.client.artifacts import atomic
from autofusex.contracts.execution import ExecutionConfig, EvaluationConfig
from autofusex.workloads.fuzz.scheduling import staged
from autofusex.workloads.fuzz.scheduling.ledger import TaskLedger
from autofusex.execution.scheduling.budgets import Budget,write_policy


def test_paused_queue_does_not_claim_and_graceful_stop_cancels_pending(tmp_path,monkeypatch):
    ledger=TaskLedger(tmp_path/'tasks.sqlite')
    identity=ledger.enqueue(CURRENT_SEED,ExecutionConfig('cpu','cpu',1,scheduling='staged',static_only=True),EvaluationConfig())
    write_policy(ledger,Budget())
    atomic(tmp_path/'control.json',dict(action='pause'))
    result=[];errors=[]
    def run():
        try:result.append(staged.drain(ledger,tmp_path/'results.sqlite'))
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=run);thread.start()
    try:
        for _ in range(100):
            state=tmp_path/'state.json'
            if state.exists() and json.loads(state.read_text())['state']=='paused':break
            if errors:raise errors[0]
            time.sleep(.02)
        else:raise AssertionError('scheduler did not pause')
        assert ledger.get(identity)['state']=='pending'
        atomic(tmp_path/'control.json',dict(action='stop'))
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert not errors
        assert ledger.get(identity)['state']=='cancelled'
        assert result[0]['completed']==0
    finally:
        atomic(tmp_path/'control.json',dict(action='stop'));thread.join(timeout=10)
