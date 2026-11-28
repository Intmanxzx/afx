import json
import pytest
@pytest.mark.parametrize('placement', ['same_card', 'same_model', 'different_model'])
@pytest.mark.parametrize('fallback_pass', [True, False])
@pytest.mark.parametrize('scenario', ['fallback', 'cpu_pass', 'existing_eager', 'eager_error'])
def test_eager_fallback_dag(tmp_path, monkeypatch, fallback_pass, scenario, placement):
    late_kind = "pass"
    """A fast peer must not commit before a slow peer's skip verdict."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from autofusex.workloads.fuzz.scheduling import staged
    from autofusex.workloads.fuzz.scheduling.ledger import TaskLedger
    from autofusex.execution.scheduling.budgets import Budget, write_policy
    from autofusex.contracts.execution import ExecutionConfig, EvaluationConfig
    recorded = []
    eager_runs = []
    reused_references = []
    class FakePool:
        def __init__(self,*args,**kw):
            self.executor=ThreadPoolExecutor(4); self.leases={}; self.workers=[]; self.devices=[]
            from types import SimpleNamespace
            if placement != 'same_card':
                self.devices = [SimpleNamespace(uuid='card0', model='910C', capacity=2**40),
                                SimpleNamespace(uuid='card1', model='910C' if placement == 'same_model' else '910B', capacity=2**40)]
            self.cpu_capacity=4; self.memory_capacity=2**40; self.cache_capacity=2**40
        def reserve(self,s):
            device = 'cpu'
            if placement != 'same_card' and s.requirements.devices:
                device = s.requirements.devices[0]
            self.leases[s.identity]={'device':device,'device_uuid':'cpu' if device == 'cpu' else 'card' + device[-1]};return self.leases[s.identity]
        def release(self,i): return dict(self.leases.pop(i),released=time.monotonic_ns())
        def submit(self,s,job,timeout):
            def run():
                k=job['kind']; backend=job['config']['backend']
                if k=='generate': value={'spec':{},'workload_id':'a'*64}
                elif k=='reference':
                    time.sleep(.08 if late_kind=='cpu_failure' else .01)
                    value={'outputs':{},'result':dict(status='error' if late_kind=='cpu_failure' else 'ok',execution={},shape_results=[])}
                elif k=='first_call_combined':
                    if backend == 'eager-npu':
                        eager_runs.append(job['config']['device'])
                    if backend == 'eager-npu' and scenario == 'eager_error':
                        return dict(ok=False,error='baseline worker failed',finished=time.monotonic_ns())
                    if backend=='cpu-compile': time.sleep(.12)
                    value={'case_skipped':late_kind=='noaf' and backend=='cpu-compile',
                           'skip_reason':'no_autofuse_coverage','outputs':{},'shape_starts':{},
                           'result':dict(status='ok',execution={},shape_results=[] if late_kind=='empty' else [dict(shape_index=0)])}
                elif k=='compare':
                    if job.get('reference_backend') == 'eager-npu':
                        reused_references.append(job['reference_exec_uuid'])
                    result = job['result']
                    result['status'] = 'ok' if scenario == 'cpu_pass' or backend == 'eager-npu' or (job.get('reference_backend') == 'eager-npu' and fallback_pass) else 'fail'
                    result['execution']['diagnostics'] = json.dumps({'version':1,'shapes':{}})
                    value={'result': result}
                else:
                    recorded.append((backend, job['result']['status'])); value={'status':'ok'}
                return dict(ok=True,value=value,finished=time.monotonic_ns())
            return self.executor.submit(run)
        def close(self): self.executor.shutdown(wait=True)
    monkeypatch.setattr(staged,'Pool',FakePool)
    monkeypatch.setattr(staged,'discover',lambda names: [])
    from autofusex.execution.storage import stage_facts
    monkeypatch.setattr(stage_facts,'save_case',lambda *args:None)
    monkeypatch.setattr(stage_facts,'finish_window',lambda *args:None)
    ledger=TaskLedger(tmp_path/'queue')
    backends = ['cpu','af-mlir','af-legacy'] + (['eager-npu'] if scenario == 'existing_eager' else [])
    for backend in backends:
        from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
        device = 'cpu' if backend == 'cpu' else 'npu:1' if backend == 'af-legacy' else 'npu:0'
        ledger.enqueue(str(SamplingConfig.profile('small').seed(3)),ExecutionConfig(backend,device,1,scheduling='staged',batch_id='test'),EvaluationConfig())
    write_policy(ledger,Budget(cpu_workers=4))
    summary=staged.drain(ledger,tmp_path/'results')
    status = 'ok' if scenario == 'cpu_pass' or (scenario != 'eager_error' and fallback_pass) else 'fail'
    expected = [('cpu','ok'), ('af-mlir',status), ('af-legacy',status)]
    if scenario != 'cpu_pass':
        expected.append(('eager-npu', 'error' if scenario == 'eager_error' else 'ok'))
    if placement == 'different_model' and scenario != 'cpu_pass':
        expected.append(('eager-npu', 'error' if scenario == 'eager_error' else 'ok'))
    assert sorted(recorded) == sorted(expected)
    assert len(eager_runs) == sum(backend == 'eager-npu' for backend, _ in expected)
    if scenario in {'fallback', 'existing_eager'}:
        assert len(reused_references) == 2
        assert len(set(reused_references)) == (2 if placement == 'different_model' else 1)
    assert summary['completed'] == len(backends)


def test_second_compare_preserves_cpu_evidence(tmp_path):
    from dataclasses import asdict
    import torch
    from autofusex.contracts.execution import ExecutionConfig, EvaluationConfig
    from autofusex.workloads.fuzz.scheduling.artifacts import write_arrays
    from autofusex.workloads.fuzz.scheduling.stage_worker import perform
    from autofusex.execution.failure_details import validate
    target = write_arrays(tmp_path/'target.npz', [(torch.ones(3),)])
    cpu = write_arrays(tmp_path/'cpu.npz', [(torch.zeros(3),)])
    eager = write_arrays(tmp_path/'eager.npz', [(torch.ones(3),)])
    result = dict(status='ok', execution=dict(telemetry=json.dumps({'shapes':{'0':{}}}),
        diagnostics=json.dumps({'version':1,'shapes':{}})),
        shape_results=[dict(shape_index=0,correctness='not_checked')])
    job = dict(kind='compare',directory=str(tmp_path),config=asdict(ExecutionConfig('af-mlir','npu:0',1)),
        evaluation=asdict(EvaluationConfig()),outputs=target,reference=cpu,result=result,shape_starts={'0':0})
    first=perform(job)['result']
    assert first['status']=='fail'
    second=perform(dict(job,result=first,reference=eager,reference_backend='eager-npu',reference_exec_uuid='test'))['result']
    assert second['status']=='ok'
    diag=json.loads(validate(second['execution']['diagnostics']))
    assert [c['passed'] for c in diag['accuracy_evidence']['comparisons']]==[False,True]
    assert diag['accuracy_evidence']['final_basis']=='eager-npu'
    assert diag['shapes']['0']['secondary_pass'] is True
    assert diag['shapes']['0']['max_abs'] == 0


@pytest.mark.parametrize('values,pattern', [([1,1,1,1,1], 'all'), ([0,1,1,0,0], 'contiguous'), ([1,0,1,0,1], 'periodic'), ([1,1,0,0,1], 'scattered')])
def test_accuracy_metrics_patterns(values, pattern):
    import torch
    from autofusex.execution.accuracy import summarize, classify
    from autofusex.workloads.fuzz.torch_backend import TorchAdapter
    from autofusex.contracts.execution import ExecutionConfig
    adapter = TorchAdapter(ExecutionConfig('cpu','cpu',1))
    adapter.torch = torch
    actual = (torch.tensor(values, dtype=torch.float32),)
    expected = (torch.zeros(5),)
    metrics = summarize(actual, expected, adapter._close)
    assert not adapter.compare(actual, expected)
    assert metrics['mismatches'] == sum(values)
    assert metrics['outputs'][0]['spatial_pattern'] == pattern
    evidence = {'comparisons': [{'shapes': [{'metrics': metrics}]}]}
    assert classify(evidence)['pattern'] == pattern
    json.dumps(metrics, allow_nan=False)


def test_accuracy_nonfinite_and_strict_single_failure():
    import torch
    from autofusex.execution.accuracy import summarize
    from autofusex.workloads.fuzz.torch_backend import TorchAdapter
    from autofusex.contracts.execution import ExecutionConfig
    adapter = TorchAdapter(ExecutionConfig('cpu','cpu',1))
    adapter.torch = torch
    actual = (torch.tensor([float('nan'), float('inf'), -float('inf'), 0.]),)
    expected = (torch.tensor([float('nan'), float('inf'), float('inf'), 0.]),)
    metrics = summarize(actual, expected, adapter._close)
    assert metrics['mismatches'] == 1
    assert metrics['outputs'][0]['nan_mismatches'] == 0
    assert metrics['outputs'][0]['inf_mismatches'] == 1
    json.dumps(metrics, allow_nan=False)
    reference = torch.zeros(10000)
    target = reference.clone(); target[0] = 1
    assert not adapter.compare((target,), (reference,))


def test_complex_accuracy_uses_tolerance_for_pass_fail():
    import torch
    from autofusex.workloads.fuzz.torch_backend import TorchAdapter
    from autofusex.contracts.execution import ExecutionConfig
    adapter = TorchAdapter(ExecutionConfig('cpu', 'cpu', 1))
    adapter.torch = torch
    reference = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64)
    actual = reference.clone()
    actual[0] += 1e-6 + 1e-6j
    assert bool(adapter._close(actual, reference).all())
    assert adapter.outputs_equal((actual,), (reference,))


def test_accuracy_evidence_survives_local_record_and_upload_mapping(tmp_path):
    from dataclasses import asdict
    from autofusex.execution.storage import ResultRepository
    from autofusex.client.platform_transport import Mapper
    from autofusex.workloads.fuzz.generation import plan
    from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
    from autofusex.workloads.fuzz.case_spec import encode
    seed = SamplingConfig.profile('small', max_nodes=1, dtype_mode='float32').seed(3)
    spec = plan(seed)
    comparisons = [dict(reference=ref, reference_exec_uuid='baseline' if ref=='eager-npu' else None,
        passed=passed, shapes=[dict(shape_index=0, passed=passed, metrics=dict(max_abs=error,max_rel=error))])
        for ref, passed, error in [('cpu',False,1.0),('eager-npu',True,0.0)]]
    evidence = dict(version=1,policy='strict-v1',final_basis='eager-npu',comparisons=comparisons)
    repo = ResultRepository(tmp_path/'results.sqlite')
    identity = repo.save(dict(seed=str(seed),workload_id='a'*64,oracle_id='cpu/1',backend='af-mlir',status='ok',
        execution=dict(backend='af-mlir',device='npu:0',case_spec=encode(spec),
            diagnostics=json.dumps(dict(version=1,shapes={},accuracy_evidence=evidence))),
        evaluation=dict(correctness=True,performance=False),
        shape_results=[dict(shape_index=0,correctness='pass',performance='not_requested')]))
    mapper = Mapper(repo.path)
    try:
        row = mapper.conn.execute('SELECT * FROM attempts WHERE id=?',(identity,)).fetchone()
        item, _ = mapper.item(row)
        assert item['accuracy_status'] == 'pass'
        assert item['accuracy_ref'] == 'eager-npu'
        assert item['max_abs_err'] == 0
        assert item['shapes'][0]['max_abs_err'] == 0
        assert item['detail']['accuracy_evidence'] == evidence
        json.dumps(item, allow_nan=False)
    finally:
        mapper.conn.close()


def test_nonfinite_evidence_distinguishes_sides_and_alignment():
    import torch
    from autofusex.execution.accuracy import nonfinite_evidence
    a = torch.tensor([float('nan'), float('inf'), -float('inf'), float('nan'), 2., 3.])
    r = torch.tensor([float('nan'), -float('inf'), -float('inf'), 1., float('inf'), 3.])
    e = nonfinite_evidence(a, r)
    assert e == dict(actual=dict(nan=2, pos_inf=1, neg_inf=1),
                     reference=dict(nan=1, pos_inf=1, neg_inf=2),
                     actual_only=1, reference_only=1, aligned_nan=1, aligned=2, different=3)
    a = torch.tensor([complex(float('nan'), 1), complex(1, float('inf'))])
    r = torch.tensor([complex(1, float('nan')), complex(1, -float('inf'))])
    e = nonfinite_evidence(a, r)
    assert e['aligned'] == 0 and e['different'] == 2
    assert nonfinite_evidence(torch.tensor(True), torch.tensor(False))['different'] == 0
    assert nonfinite_evidence(torch.empty(0), torch.empty(0))['aligned'] == 0
