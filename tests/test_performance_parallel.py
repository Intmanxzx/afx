"""Dynamic performance dispatch: distinct cards, out-of-order commits, stop/resume."""
import json
import random
import signal
from types import SimpleNamespace

import pytest

from autofusedebugger.nnsmith_integration.performance import parallel,session
from autofusedebugger.nnsmith_integration.performance.lifecycle import SessionStopped
from autofusedebugger.nnsmith_integration.performance.schema import ProfileConfig
from autofusedebugger.nnsmith_integration.generation import GenerationConfig


def case(task):
    return dict(index=task['index'],seed=task['seed'],case_id=f"{task['index']:05d}-{task['seed']}",
        status='compared',performance_status='pass',correctness_status='pass',ascendc_speedup=2,
        backends={})


@pytest.fixture
def harness(monkeypatch,tmp_path):
    import nnsmith.materialize as materialize
    from nnsmith.abstract.op import Add
    monkeypatch.setattr(materialize.Model,'init',lambda *a,**k:SimpleNamespace(add_seed_setter=lambda:None))
    def run(**kwargs):
        return session.run_performance_session(out_dir=tmp_path,generation=GenerationConfig(),
            profile=ProfileConfig(),opset=[Add],seed=42,devices=[0,1],jobs=2,**kwargs)
    return run


def test_distinct_cards_and_jobs_cap():
    assert parallel.worker_targets(2,None,'npu:0')==['npu:0']
    assert parallel.worker_targets(4,[0,1],'npu')==['npu:0','npu:1']
    assert parallel.worker_targets(1,[0,1],'npu')==['npu:0']
    with pytest.raises(ValueError):parallel.worker_targets(2,[0,0],'npu')
    with pytest.raises(ValueError):parallel.worker_targets(0,[0,1],'npu')


def test_out_of_order_stop_resume_keeps_completed_higher_index(tmp_path,monkeypatch,harness):
    assigned=[]
    def first(tasks,**kwargs):
        iterator=iter(tasks)
        slow=next(iterator);fast=next(iterator)
        assigned.extend([slow,fast])
        yield case(fast)
        next_task=next(iterator);assigned.append(next_task)
        yield case(next_task)
        # Index 0 remains in-flight while indices 1/2 completed.
        raise SessionStopped(signal.SIGTERM)
    monkeypatch.setattr(parallel,'dynamic_results',first)
    stopped=harness(n_cases=6)
    assert stopped['total']==2 and [c['index'] for c in stopped['cases']]==[1,2]
    assert stopped['session_status']=='stopped'
    assert stopped['checkpoint']['pending_tasks']==[assigned[0]]
    assert stopped['checkpoint']['next_index']==3
    retry=[]
    def second(tasks,**kwargs):
        for task in tasks:
            retry.append(task)
            yield case(task)
    monkeypatch.setattr(parallel,'dynamic_results',second)
    resumed=harness(resume=True,n_cases=6)
    assert [t['index'] for t in retry]==[0,3,4,5]
    assert resumed['total']==6 and resumed['session_status']=='completed'
    assert not resumed['checkpoint']['pending_tasks']
    rng=random.Random(42)
    assert [c['seed'] for c in resumed['cases']]==[rng.getrandbits(32) for _ in range(6)]
    assert json.loads((tmp_path/'summary.json').read_text())['total']==6


def test_continuous_parallel_stops_with_multiple_pending_seeds(tmp_path,monkeypatch,harness):
    def dispatch(tasks,**kwargs):
        iterator=iter(tasks)
        a,b=next(iterator),next(iterator)
        yield case(b)
        next(iterator)
        raise SessionStopped(signal.SIGINT)
    monkeypatch.setattr(parallel,'dynamic_results',dispatch)
    stopped=harness()
    assert stopped['requested_cases'] is None and stopped['stop_signal']==signal.SIGINT
    assert stopped['total']==1 and len(stopped['checkpoint']['pending_tasks'])==2
    assert [t['index'] for t in stopped['checkpoint']['pending_tasks']]==[0,2]


def test_parallel_worker_uses_one_card_for_both_backends(monkeypatch):
    from nnsmith.abstract.op import Add
    received={}
    def measure(**kwargs):
        received.update(kwargs)
        return {'index':3,'seed':42}
    monkeypatch.setattr(session,'_measure_case',measure)
    result=parallel.execute_case(('model',[Add],'npu:1'),{'index':3,'seed':42},
        dict(out_dir='/tmp/example',generation=GenerationConfig().__dict__,profile=ProfileConfig().__dict__))
    assert received['target']=='npu:1' and received['devices'] is None
    assert result['target']=='npu:1'


def test_parallel_runner_exception_preserves_pending_and_report(tmp_path,monkeypatch,harness):
    def dispatch(tasks,**kwargs):
        iterator=iter(tasks)
        yield case(next(iterator))
        next(iterator)
        raise RuntimeError('worker died')
    monkeypatch.setattr(parallel,'dynamic_results',dispatch)
    with pytest.raises(RuntimeError,match='worker died'):harness(n_cases=5)
    saved=json.loads((tmp_path/'summary.json').read_text())
    assert saved['total']==1 and saved['session_status']=='error'
    assert saved['checkpoint']['pending_tasks'][0]['index']==1
    assert (tmp_path/'summary.html').exists()


def spawn_initialize(index,config):
    config['barrier'].wait(timeout=20)
    return index


def spawn_execute(worker,task,config):
    import time
    time.sleep(1 if task['index']==0 else .02)
    result=case(task)
    result['target']=config['targets'][worker]
    return result


def test_real_spawn_pool_rebalances_complete_performance_cases(tmp_path,monkeypatch,harness):
    import multiprocessing
    from collections import Counter
    from autofusedebugger.nnsmith_integration.task_pool import dynamic_results
    def dispatch(tasks,**kwargs):
        kwargs['config']['barrier']=multiprocessing.get_context('spawn').Barrier(2)
        kwargs['initialize']=spawn_initialize
        kwargs['execute']=spawn_execute
        return dynamic_results(tasks,**kwargs)
    monkeypatch.setattr(parallel,'dynamic_results',dispatch)
    summary=harness(n_cases=10)
    assert summary['total']==10 and summary['session_status']=='completed'
    counts=Counter(c['target'] for c in summary['cases'])
    slow=summary['cases'][0]['target']
    assert counts[slow]==1 and sorted(counts.values())==[1,9]
    assert not summary['checkpoint']['pending_tasks']
