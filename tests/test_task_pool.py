"""Spawn-process checks for dynamic case assignment and crash detection."""
import multiprocessing
import os
import time
import json
import random
from collections import Counter

import pytest
from autofusedebugger.nnsmith_integration.task_pool import dynamic_results


def initialize(index, config):
    config['barrier'].wait(timeout=20)
    return index


def execute(worker, task, config):
    if task == 0:
        time.sleep(1)
    else:
        time.sleep(.02)
    return worker, task


def crash(worker, task, config):
    os._exit(7)


def fail(worker, task, config):
    raise ValueError('test case failure')


def test_idle_worker_claims_remaining_tasks_instead_of_waiting_for_half_batch():
    config={'barrier':multiprocessing.get_context('spawn').Barrier(2)}
    results=list(dynamic_results(range(12),workers=2,config=config,initialize=initialize,execute=execute))
    assert sorted(task for worker,task in results)==list(range(12))
    counts=Counter(worker for worker,task in results)
    slow=next(worker for worker,task in results if task==0)
    assert counts[slow]==1
    assert counts[1-slow]==11


@pytest.mark.parametrize('callback,match',[(crash,'exited'),(fail,'test case failure')])
def test_worker_crash_or_exception_stops_pool_without_hanging(callback,match):
    config={'barrier':multiprocessing.get_context('spawn').Barrier(2)}
    with pytest.raises(RuntimeError,match=match):
        list(dynamic_results(range(4),workers=2,config=config,initialize=initialize,execute=callback))


def test_parallel_parent_uses_global_seeds_and_saves_in_index_order(tmp_path,monkeypatch):
    from autofusedebugger.nnsmith_integration import fuzz_loop,task_pool
    captured={}
    def dispatch(tasks,**kwargs):
        rows=list(tasks);captured.update(kwargs);captured['tasks']=rows
        for index,seed in reversed(rows):
            yield fuzz_loop.CaseResult(index=index,seed=seed,status='pass',duration_s=.1)
    monkeypatch.setattr(task_pool,'dynamic_results',dispatch)
    summary=fuzz_loop.run_fuzz_session_parallel(out_dir=tmp_path,n_cases=7,jobs=2,devices=[0,1],seed=42)
    rng=random.Random(42)
    expected=[rng.getrandbits(32) for _ in range(7)]
    assert [case.seed for case in summary.cases]==expected
    assert captured['workers']==2 and captured['config']['devices']==[0,1]
    assert captured['tasks']==list(enumerate(expected))
    assert json.loads((tmp_path/'summary.json').read_text())['total']==7
    assert (tmp_path/'summary.html').exists()
    assert fuzz_loop._shard_target('npu',[0,1],1)=='npu:1'
    assert fuzz_loop._shard_target('npu:0',[0,1],1)=='npu:1'
