"""Full-range skips do not weaken the CPU oracle or effective denominator."""
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch
import pytest


def test_fixed_full_range_seed_replays_without_global_rng_changes():
    pytest.importorskip('nnsmith')
    import random
    from autofusex.fuzz.generation import materialize, workload_id, nnsmith_sampled
    from autofusex.fuzz.generation.sampling import SamplingConfig
    config = SamplingConfig.profile('small', dtype_mode='float32', max_nodes=3)
    for i in (0, 1, 2, 10):
        seed = config.seed(i)
        state = random.getstate()
        first = materialize(seed)
        assert random.getstate() == state
        nnsmith_sampled.plan.cache_clear()
        assert materialize(seed) == first
        assert workload_id(materialize(seed)) == workload_id(first)
        assert all('materializer' in n for n in json.loads(first.spec.graph_json)['nodes'])


def test_probe_noaf_failure_and_unknown_restore(monkeypatch):
    pytest.importorskip('torch_npu')
    from torch._inductor.scheduler import Scheduler
    from torch_npu._inductor.ascendc import codegen
    from autofusex.fuzz.backends.autofuse_coverage import observe_autofuse, NoAutofuseCoverage
    from autofusex.fuzz.failure_details import validate
    with patch.object(Scheduler, 'codegen', lambda self: None):
        original = Scheduler.codegen
        with observe_autofuse() as c:
            with pytest.raises(NoAutofuseCoverage): Scheduler.codegen(None)
        assert c.no_coverage and c.candidates == 0
        assert Scheduler.codegen is original
        validate(json.dumps(dict(version=1, shapes={}, autofuse_coverage=asdict(c))))
    def af_failure(graph): raise RuntimeError('real AF failure')
    with patch.object(codegen, 'codegen_kernel_def', af_failure), \
         patch.object(Scheduler, 'codegen', lambda self: codegen.codegen_kernel_def(None)):
        with observe_autofuse() as c:
            with pytest.raises(RuntimeError, match='real AF failure'): Scheduler.codegen(None)
        assert c.candidates == 1 and not c.no_coverage
    with observe_autofuse() as c:
        pass  # A cache hit/empty graph does not prove zero AF coverage.
    assert not c.no_coverage and c.scheduler_codegen_entered == 0


@pytest.mark.parametrize('late_kind', ['noaf', 'cpu_failure', 'empty', 'pass'])
def test_all_target_publish_barrier(tmp_path, monkeypatch, late_kind):
    """A fast peer must not commit before a slow peer's skip verdict."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from autofusex.fuzz.scheduling import staged
    from autofusex.fuzz.scheduling.ledger import TaskLedger
    from autofusex.fuzz.scheduling.budgets import Budget, write_policy
    from autofusex.fuzz.contracts import ExecutionConfig, EvaluationConfig
    recorded = []
    class FakePool:
        def __init__(self,*args,**kw):
            self.executor=ThreadPoolExecutor(4); self.leases={}; self.workers=[]; self.devices=[]
            self.cpu_capacity=4; self.memory_capacity=2**40; self.cache_capacity=2**40
        def reserve(self,s):
            self.leases[s.identity]={'device':'cpu','device_uuid':'cpu'};return self.leases[s.identity]
        def release(self,i): return dict(self.leases.pop(i),released=time.monotonic_ns())
        def submit(self,s,job,timeout):
            def run():
                k=job['kind']; backend=job['config']['backend']
                if k=='generate': value={'spec':{},'workload_id':'a'*64}
                elif k=='reference':
                    time.sleep(.08 if late_kind=='cpu_failure' else .01)
                    value={'outputs':{},'result':dict(status='error' if late_kind=='cpu_failure' else 'ok',execution={},shape_results=[])}
                elif k=='first_call_combined':
                    if backend=='cpu-compile': time.sleep(.12)
                    value={'case_skipped':late_kind=='noaf' and backend=='cpu-compile',
                           'skip_reason':'no_autofuse_coverage','outputs':{},'shape_starts':{},
                           'result':dict(status='ok',execution={},shape_results=[] if late_kind=='empty' else [dict(shape_index=0)])}
                elif k=='compare': value={'result':job['result']}
                else:
                    recorded.append(backend); value={'status':'ok'}
                return dict(ok=True,value=value,finished=time.monotonic_ns())
            return self.executor.submit(run)
        def close(self): self.executor.shutdown(wait=True)
    monkeypatch.setattr(staged,'Pool',FakePool)
    monkeypatch.setattr(staged,'discover',lambda names: [])
    from autofusex.fuzz.storage import stage_facts
    monkeypatch.setattr(stage_facts,'save_case',lambda *args:None)
    monkeypatch.setattr(stage_facts,'finish_window',lambda *args:None)
    ledger=TaskLedger(tmp_path/'queue')
    for backend in ['cpu','eager-npu','cpu-compile']:
        from autofusex.fuzz.generation.sampling import SamplingConfig
        ledger.enqueue(str(SamplingConfig.profile('small').seed(3)),ExecutionConfig(backend,'npu:0' if backend=='eager-npu' else 'cpu',1,scheduling='staged',batch_id='test'),EvaluationConfig())
    write_policy(ledger,Budget(cpu_workers=4))
    summary=staged.drain(ledger,tmp_path/'results')
    if late_kind in {'noaf','cpu_failure'}:
        assert recorded==[] and summary['skipped']==3 and summary['completed']==0
    else:
        assert sorted(recorded)==['cpu','cpu-compile','eager-npu'] and summary['completed']==3


def test_generation_failure_reuses_worker(tmp_path):
    pytest.importorskip('nnsmith')
    from autofusex.fuzz.scheduling.stage_worker import perform
    from autofusex.fuzz.generation import materialize
    from autofusex.fuzz.generation.sampling import SamplingConfig
    config = SamplingConfig.profile('small', dtype_mode='float32', max_nodes=3)
    job = dict(kind='generate', directory=str(tmp_path), seed=str(config.seed(0)), config={})
    with patch('autofusex.fuzz.generation.materialize', side_effect=ValueError('unsupported op parameters')):
        result = perform(job)
    assert result['skipped'] and result['reason'] == 'generation_failed'
    assert perform(job)['workload_id']  # same process succeeds on next job
    with patch('autofusex.fuzz.generation.runtime.source_evidence', side_effect=ValueError('source mismatch')):
        with pytest.raises(ValueError, match='source mismatch'): perform(job)


def test_noaf_stops_remaining_dynamic_samples():
    from test_fuzz_precise_metrics import Adapter
    from autofusex.fuzz.generation import materialize as bundle
    from autofusex.fuzz.generation.sampling import SamplingConfig
    current=str(SamplingConfig.profile('small',max_nodes=1,dtype_mode='float32').seed(3))
    from autofusex.fuzz.execution import execute
    from autofusex.fuzz.contracts import EvaluationConfig
    from autofusex.fuzz.backends.autofuse_coverage import NoAutofuseCoverage
    adapter = Adapter()
    calls = []
    def invoke(*args):
        calls.append(1)
        try:
            raise NoAutofuseCoverage('no candidate')
        except NoAutofuseCoverage as cause:
            raise RuntimeError('wrapped backend failure') from cause
    adapter.invoke = invoke
    from dataclasses import replace
    original = bundle(current)
    dynamic = replace(original, spec=replace(original.spec,
        shape_samples=original.spec.shape_samples * 3, output_samples=original.spec.output_samples * 3))
    result = execute(dynamic, adapter, EvaluationConfig())
    assert len(calls) == len(result['shape_results']) == 1


def test_coverage_diagnostics_repository_roundtrip(tmp_path):
    from test_fuzz_precise_metrics import Adapter
    from autofusex.fuzz.generation import materialize as bundle
    from autofusex.fuzz.generation.sampling import SamplingConfig
    current=str(SamplingConfig.profile('small',max_nodes=1,dtype_mode='float32').seed(3))
    from autofusex.fuzz.execution import execute
    from autofusex.fuzz.contracts import EvaluationConfig
    from autofusex.fuzz.backends.autofuse_coverage import Coverage
    from autofusex.fuzz.storage import ResultRepository
    result = execute(bundle(current), Adapter(), EvaluationConfig())
    coverage = asdict(Coverage(candidates=1, installed=True, events=['AF_CANDIDATE']))
    diag = json.loads(result['execution']['diagnostics'])
    diag['autofuse_coverage'] = coverage
    result['execution']['diagnostics'] = json.dumps(diag)
    repo = ResultRepository(tmp_path/'results')
    identity = repo.save(result)
    assert json.loads(repo.detail(identity)['contexts']['execution']['diagnostics'])['autofuse_coverage'] == coverage


def test_retired_graph_never_falls_back_to_interpreter():
    from autofusex.fuzz.generation.torch_dialect import execute_graph
    with pytest.raises(ValueError,match='retired graph dialect'):
        execute_graph(None,{'dialect':'nnsmith-sampled/5'}, {}, [], 'cpu')


def test_skips_have_bounded_audit_without_attempts(tmp_path):
    import sqlite3
    from autofusex.fuzz.storage.stage_facts import save_case
    from autofusex.fuzz.storage.lifecycle import stamp
    from autofusex.fuzz.scheduling.stages import Case, StageTask, Kind
    from autofusex.fuzz.contracts import ExecutionConfig
    case = Case('id', 'C16', stamp(), [{'attempt_id':'not-saved', 'execution':json.dumps(asdict(ExecutionConfig('cpu','cpu',1)))}], str(tmp_path))
    case.skip_reason = 'cpu_reference_failed'; case.skip_finalized = True
    stage = StageTask(Kind.REFERENCE, 'id')
    stage.state = 'failed'; stage.facts['error'] = 'unsupported dtype'
    case.stages[stage.identity] = stage
    save_case(tmp_path/'results', case, {}, 'window', stamp(), SimpleNamespace())
    with sqlite3.connect(tmp_path/'results') as db:
        assert db.execute('select count(*) from attempts').fetchone()[0] == 0
        reason, payload = db.execute('select reason,payload from skipped_cases_v1').fetchone()
        assert reason == 'cpu_reference_failed'
        assert json.loads(payload)['stages'][0]['error'] == 'unsupported dtype'
