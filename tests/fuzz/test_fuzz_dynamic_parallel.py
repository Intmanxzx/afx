import json
import sys
import time
from dataclasses import replace

import pytest

from autofusex.fuzz.client.contracts import Seed, ExecutionConfig, EvaluationConfig
from autofusex.fuzz.client.generation.nnsmith_dynamic import variants, output_shapes
from autofusex.fuzz.client.generation.nnsmith_sampled import materialize_spec
from test_fuzz_current_contract import build_spec
from autofusex.fuzz.client.generation.sampling import SamplingConfig
import random
CURRENT_SEED=str(SamplingConfig.profile("small").seed(3))


def test_dynamic_variants_are_reproducible():
    original=build_spec(3)
    spec=variants(original,65536,3,random.Random(42))
    assert spec==variants(original,65536,3,random.Random(42))
    assert len(spec.shape_samples)==3
    assert len({tuple(t.shape for t in sample) for sample in spec.shape_samples})==3
    graph=json.loads(spec.graph_json)
    for inputs,outputs in zip(spec.shape_samples,spec.output_samples):
        assert output_shapes(graph,inputs,spec.weights,65536)==[t.shape for t in outputs]
    materialize_spec(spec)


def test_dynamic_no_legal_extent_is_not_silently_static():
    spec=build_spec()
    scalar=replace(spec.inputs[0],shape=(),stride=())
    with pytest.raises(ValueError,match='no 3 legal'):
        variants(replace(spec,inputs=(scalar,)),65536,3,random.Random(42))


def test_campaign_mix_and_round_robin(tmp_path,monkeypatch):
    from autofusex.fuzz.client.scheduling import campaign
    from autofusex.fuzz.client.scheduling.ledger import TaskLedger
    ledger=TaskLedger(tmp_path/'queue')
    configs=[ExecutionConfig('cpu','cpu',1),ExecutionConfig('af-mlir','npu:0',1)]
    campaign.initialize(ledger,configs,EvaluationConfig(),2,['npu:0','npu:1'])
    values=iter([42,43]);monkeypatch.setattr(campaign.secrets,'randbelow',lambda _:next(values))
    assert campaign.refill(ledger,target_pending=4)
    assert campaign.refill(ledger,target_pending=4)
    tasks=ledger.page()['items']
    assert [Seed.parse(t['seed']).version for t in tasks]==[11,11,12,12]
    assert [json.loads(t['execution'])['device'] for t in tasks]==['cpu','npu:0','cpu','npu:1']
    assert [json.loads(t['execution'])['static_only'] for t in tasks]==[True,True,False,False]


def test_parallel_cpu_and_distinct_devices_not_same_device(tmp_path,monkeypatch):
    from autofusex.fuzz.client.scheduling import supervisor
    from autofusex.fuzz.client.scheduling.ledger import TaskLedger
    from autofusex.fuzz.client.scheduling.budgets import Budget,write_policy
    ledger=TaskLedger(tmp_path/'queue')
    configs=[ExecutionConfig('cpu','cpu',1,save_results=False),
             ExecutionConfig('af-mlir','npu:0',1,save_results=False),
             ExecutionConfig('af-mlir','npu:0',1,save_results=False),
             ExecutionConfig('af-mlir','npu:1',1,save_results=False)]
    for c in configs:ledger.enqueue(CURRENT_SEED,c,EvaluationConfig())
    write_policy(ledger,Budget(cpu_workers=3))
    monkeypatch.setattr(Budget,'admit',lambda *a:None)
    def command(l,d,t):
        # Synchronize the first wave rather than assuming process startup takes
        # less than .5 s on a shared NPU host running other regression work.
        peers=[x['id'] for x in l.page()['items']]
        gate="" if t['id']==peers[2] else ("p=pathlib.Path("+repr(str(tmp_path/(t['id']+'.ready')))+");p.touch();deadline=time.monotonic()+10\nwhile len(list(p.parent.glob('*.ready')))<3 and time.monotonic()<deadline: time.sleep(.01)\n")
        code="import time,json,pathlib;start=time.monotonic();"+gate+"time.sleep(.5);pathlib.Path("+repr(str(tmp_path/t['id']))+").write_text(json.dumps([start,time.monotonic()]))"
        return [sys.executable,'-c',code]
    monkeypatch.setattr(supervisor,'_worker_command',command)
    result=supervisor.drain(ledger,tmp_path/'results')
    assert result['completed']==4
    intervals=[json.loads((tmp_path/t['id']).read_text()) for t in ledger.page()['items']]
    assert max(intervals[i][0] for i in [0,1,3]) < min(intervals[i][1] for i in [0,1,3])
    assert intervals[2][0]>=intervals[1][1]
    assert not list(tmp_path.glob('.fuzz-worker-*'))


def test_device_discovery_honors_logical_count(monkeypatch):
    from autofusex.fuzz.client.scheduling.devices import select_devices
    from types import SimpleNamespace
    monkeypatch.setattr('subprocess.run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='warning\n{"count":2}\n'))
    assert select_devices()==['npu:0','npu:1']
    assert select_devices(['npu:1','npu:1'])==['npu:1']
    with pytest.raises(ValueError):select_devices(['12'])


def test_canonical_values_repeat_exactly():
    spec=variants(build_spec(),65536,3,random.Random(42))
    assert materialize_spec(spec)==materialize_spec(spec)


def test_generation_failure_is_completed_without_regeneration(tmp_path,monkeypatch):
    from autofusex.fuzz.client.scheduling.ledger import TaskLedger
    from autofusex.fuzz.client.scheduling.worker import run
    from autofusex.fuzz.client.scheduling.supervisor import _has_result
    ledger=TaskLedger(tmp_path/'queue');identity=ledger.enqueue(CURRENT_SEED,ExecutionConfig('cpu','cpu',1),EvaluationConfig())
    task=ledger.claim();db=tmp_path/'results'
    def failed(*a):raise ValueError('no legal dynamic shapes')
    monkeypatch.setattr('autofusex.fuzz.client.generation.materialize',failed)
    assert run(ledger.path,db,identity,task['claim'])==0
    assert _has_result(db,task)
