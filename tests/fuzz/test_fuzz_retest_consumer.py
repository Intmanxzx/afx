"""Consumer uses stable local IDs and a common scheduler, never shell commands."""
from autofusex.fuzz.generation.sampling import SamplingConfig
import copy
from dataclasses import asdict
import json
import uuid

import pytest
from autofusex.fuzz.sync.retests import accept, initialize, expected_workload
from autofusex.fuzz.scheduling.ledger import TaskLedger
from autofusex.fuzz.scheduling.budgets import Budget
from autofusex.fuzz.contracts import EvaluationConfig


def lease():
    return dict(task_id=str(uuid.uuid4()),lease_id=str(uuid.uuid4()),generation=1,expires=100,repetition=0,
        spec=dict(seed=str(SamplingConfig().seed(16, dynamic=True)),backend='af-mlir',workload_id='a'*64,oracle_id='cpu/1',
                  execution={'static_only':False},evaluation=asdict(EvaluationConfig())))


def test_local_admission_idempotence_and_fixed_seed(tmp_path):
    ledger=TaskLedger(tmp_path/'tasks.sqlite');initialize(ledger)
    item=lease();budget=Budget()
    identity=accept(ledger,item,['npu:0','npu:1'],budget)
    assert accept(ledger,item,['npu:0','npu:1'],budget)==identity
    tasks=ledger.page()['items'];assert len(tasks)==2
    ex=json.loads(next(t for t in tasks if json.loads(t['execution'])['backend']=='af-mlir')['execution'])
    assert ex['batch_id']==item['lease_id'] and ex['save_results'] is True
    assert ex['scheduling']=='staged' and ex['static_only'] is False
    assert expected_workload(ledger,tasks)=='a'*64
    with ledger._connection() as conn:
        assert json.loads(conn.execute('SELECT devices FROM task_devices').fetchone()[0])==['npu:0','npu:1']
    changed=copy.deepcopy(item);changed['spec']['seed']=str(SamplingConfig().seed(17, dynamic=True))
    with pytest.raises(ValueError):accept(ledger,changed,['npu:0'],budget)


def test_remote_parameters_do_not_become_commands(tmp_path):
    ledger=TaskLedger(tmp_path/'tasks.sqlite');initialize(ledger)
    item=lease();item['spec']['backend']='sh -c something'
    with pytest.raises(ValueError):accept(ledger,item,['npu:0'],Budget())
    assert ledger.page()['items']==[]


def test_reproduction_guard_precedes_backend_execution(tmp_path,monkeypatch):
    from autofusex.fuzz.scheduling.stage_worker import perform
    from autofusex.fuzz import generation
    # Import consumers before patching: execute imports workload_id by value.
    from autofusex.fuzz import execution
    monkeypatch.setattr(generation,'materialize',lambda _:object())
    monkeypatch.setattr(generation,'workload_id',lambda _:'wrong')
    with pytest.raises(ValueError,match='fingerprint mismatch'):
        perform(dict(kind='generate',seed='D00000000G',expected_workload_id='expected',directory=str(tmp_path)))


def test_consumer_does_not_start_unrelated_pending_queue(tmp_path,monkeypatch):
    from autofusex.fuzz.sync.retests import Consumer
    from autofusex.fuzz.contracts import ExecutionConfig
    from autofusex.fuzz.config import setup
    from autofusex.fuzz.scheduling import hosting
    home=tmp_path/'home';setup(home,cpu_workers=1,compile_threads=1)
    consumer=Consumer(home,home/'results/results.sqlite',devices=['npu:0'])
    consumer.ledger.enqueue(str(2*2**128+16),ExecutionConfig('cpu','cpu',1,scheduling='staged'),EvaluationConfig())
    calls=[]
    monkeypatch.setattr(consumer,'api',lambda route,payload:(calls.append(route) or {'items':[]}))
    monkeypatch.setattr(hosting,'launch',lambda *a,**kw:pytest.fail('must not start unrelated queue'))
    consumer.tick()
    assert calls==['register','claim']


def test_consumer_waits_for_legacy_queue_before_claiming(tmp_path,monkeypatch):
    from autofusex.fuzz.sync.retests import Consumer
    from autofusex.fuzz.contracts import ExecutionConfig
    from autofusex.fuzz.config import setup
    home=tmp_path/'home';setup(home,cpu_workers=1,compile_threads=1)
    consumer=Consumer(home,home/'results/results.sqlite',devices=['npu:0'])
    consumer.ledger.enqueue(str(2*2**128+16),ExecutionConfig('cpu','cpu',1,scheduling='combined'),EvaluationConfig())
    monkeypatch.setattr(consumer,'api',lambda *args:pytest.fail('must not claim into legacy queue'))
    assert consumer.tick()['state']=='waiting_for_legacy_queue'
