import json
import sys
import time
from dataclasses import replace

import pytest

from autofusedebugger.fuzz.contracts import Seed, ExecutionConfig, EvaluationConfig
from autofusedebugger.fuzz.generation.nnsmith_dynamic import variants, output_shapes
from autofusedebugger.fuzz.generation.nnsmith_v3 import materialize_spec
from fuzz_fixtures import bundle


def test_dynamic_seed_and_legal_variants_preserve_c():
    original=bundle().spec
    spec=variants(replace(original,seed=Seed.parse('D16')))
    assert Seed.parse(str(spec.seed)).short=='D000000016'
    assert len(original.shape_samples)==1 and len(spec.shape_samples)==3
    assert len({tuple(t.shape for t in sample) for sample in spec.shape_samples})==3
    graph=json.loads(spec.graph_json)
    for inputs,outputs in zip(spec.shape_samples,spec.output_samples):
        assert output_shapes(graph,inputs,spec.weights)==[t.shape for t in outputs]
    materialize_spec(spec)  # All views fit canonical storage.
    from autofusedebugger.fuzz.case_spec import decode,encode
    assert decode(encode(spec))==spec


def test_dynamic_no_legal_extent_is_not_silently_static():
    spec=bundle().spec
    scalar=replace(spec.inputs[0],shape=(),stride=())
    with pytest.raises(ValueError,match='no three legal'):
        variants(replace(spec,inputs=(scalar,)))


def test_campaign_mix_and_round_robin(tmp_path,monkeypatch):
    from autofusedebugger.fuzz.scheduling import campaign
    from autofusedebugger.fuzz.scheduling.ledger import TaskLedger
    ledger=TaskLedger(tmp_path/'queue')
    configs=[ExecutionConfig('cpu','cpu',1),ExecutionConfig('af-mlir','npu:0',1)]
    campaign.initialize(ledger,configs,EvaluationConfig(),2,['npu:0','npu:1'])
    values=iter([42,43]);monkeypatch.setattr(campaign.secrets,'randbelow',lambda _:next(values))
    assert campaign.refill(ledger,target_pending=4)
    assert campaign.refill(ledger,target_pending=4)
    tasks=ledger.page()['items']
    assert [Seed.parse(t['seed']).version for t in tasks]==[3,3,4,4]
    assert [json.loads(t['execution'])['device'] for t in tasks]==['cpu','npu:0','cpu','npu:1']
    assert [json.loads(t['execution'])['static_only'] for t in tasks]==[True,True,False,False]


def test_parallel_cpu_and_distinct_devices_not_same_device(tmp_path,monkeypatch):
    from autofusedebugger.fuzz.scheduling import supervisor
    from autofusedebugger.fuzz.scheduling.ledger import TaskLedger
    from autofusedebugger.fuzz.scheduling.budgets import Budget,write_policy
    ledger=TaskLedger(tmp_path/'queue')
    configs=[ExecutionConfig('cpu','cpu',1,save_results=False),
             ExecutionConfig('af-mlir','npu:0',1,save_results=False),
             ExecutionConfig('af-mlir','npu:0',1,save_results=False),
             ExecutionConfig('af-mlir','npu:1',1,save_results=False)]
    for c in configs:ledger.enqueue('C16',c,EvaluationConfig())
    write_policy(ledger,Budget(cpu_workers=3))
    monkeypatch.setattr(Budget,'admit',lambda *a:None)
    def command(l,d,t):
        code="import time,json,pathlib;start=time.monotonic();time.sleep(.5);pathlib.Path("+repr(str(tmp_path/t['id']))+").write_text(json.dumps([start,time.monotonic()]))"
        return [sys.executable,'-c',code]
    monkeypatch.setattr(supervisor,'_worker_command',command)
    result=supervisor.drain(ledger,tmp_path/'results')
    assert result['completed']==4
    intervals=[json.loads((tmp_path/t['id']).read_text()) for t in ledger.page()['items']]
    assert max(intervals[i][0] for i in [0,1,3]) < min(intervals[i][1] for i in [0,1,3])
    assert intervals[2][0]>=intervals[1][1]
    assert not list(tmp_path.glob('.fuzz-worker-*'))


def test_device_discovery_honors_logical_count(monkeypatch):
    from autofusedebugger.fuzz.scheduling.devices import select_devices
    from types import SimpleNamespace
    monkeypatch.setattr('subprocess.run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='warning\n{"count":2}\n'))
    assert select_devices()==['npu:0','npu:1']
    assert select_devices(['npu:1','npu:1'])==['npu:1']
    with pytest.raises(ValueError):select_devices(['12'])


def test_canonical_c_bytes_unchanged():
    import hashlib,struct
    original=bundle().spec
    actual=materialize_spec(original)
    for tensor in (*original.inputs,*original.weights):
        values=[]
        import math
        for index in range(math.prod(tensor.shape)):
            raw=hashlib.sha256(f'nnsmith-v3/{original.seed}/{tensor.name}/{index}'.encode()).digest()
            values.append((int.from_bytes(raw[:4],'little')%65-32)/16)
        assert actual.storages[tensor.alias_group]==struct.pack('<'+'f'*len(values),*values)


def test_cpu_outputs_match_saved_dynamic_shapes():
    torch=pytest.importorskip('torch')
    from autofusedebugger.fuzz.generation.nnsmith_v3 import execute_graph
    spec=variants(replace(bundle().spec,seed=Seed.parse('D16')))
    forward=execute_graph(torch,json.loads(spec.graph_json),{},[t.name for t in spec.inputs])
    for sample,outputs in zip(spec.shape_samples,spec.output_samples):
        result=forward(*(torch.ones(t.shape) for t in sample))
        assert [tuple(t.shape) for t in result]==[t.shape for t in outputs]


def test_generation_failure_is_completed_without_regeneration(tmp_path,monkeypatch):
    from autofusedebugger.fuzz.scheduling.ledger import TaskLedger
    from autofusedebugger.fuzz.scheduling.worker import run
    from autofusedebugger.fuzz.scheduling.supervisor import _has_result
    ledger=TaskLedger(tmp_path/'queue');identity=ledger.enqueue('D16',ExecutionConfig('cpu','cpu',1),EvaluationConfig())
    task=ledger.claim();db=tmp_path/'results'
    def failed(*a):raise ValueError('no legal dynamic shapes')
    monkeypatch.setattr('autofusedebugger.fuzz.generation.materialize',failed)
    assert run(ledger.path,db,identity,task['claim'])==0
    assert _has_result(db,task)
