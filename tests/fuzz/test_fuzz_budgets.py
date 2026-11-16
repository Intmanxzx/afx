from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import pytest

from autofusex.fuzz.client.config import setup
from autofusex.fuzz.client.contracts import ExecutionConfig, EvaluationConfig
from autofusex.fuzz.client.scheduling.budgets import Budget, group_rss, scratch_size, write_policy, read_policy
from autofusex.fuzz.client.scheduling.ledger import TaskLedger
from autofusex.fuzz.client.scheduling import supervisor, hosting, budgets
from fuzz_fixtures import CURRENT_SEED

EX, EV = ExecutionConfig('cpu','cpu',1), EvaluationConfig()


def inventory(**overrides):
    return dict({'cpu_worker_ceiling': 2, 'effective_memory_bytes': 2**30, 'disk_free_bytes': 2**30}, **overrides)


def test_configured_budget_resolves_cli_and_rejects_override():
    config = {'compile_threads':2,'cpu_workers':4,'memory_budget_bytes':1000000,'cache_budget_bytes':2000000}
    assert Budget.configured(config).compile_threads == 2
    assert Budget.configured(config,1).compile_threads == 1
    with pytest.raises(ValueError, match='setup budget'):
        Budget.configured(config,3)
    assert Budget.configured(None).compile_threads == 1
    assert Budget.configured(config).cpu_workers == 4  # shared worker concurrency limit


@pytest.mark.parametrize('options,values,message', [
    ({'compile_threads':3}, {}, 'CPU capacity'),
    ({'memory_budget_bytes':2**31}, {}, 'memory capacity'),
    ({'cache_budget_bytes':2**30}, {}, 'scratch budget'),
    ({}, {'disk_free_bytes':1}, 'disk reserve'),
    ({'memory_budget_bytes':1}, {'effective_memory_bytes':None}, 'memory capacity'),
])
def test_admission_fails_closed(monkeypatch,tmp_path,options,values,message):
    monkeypatch.setattr(budgets,'resource_inventory',lambda _:inventory(**values))
    with pytest.raises(ValueError, match=message):
        Budget(**options).admit(tmp_path)


def test_queued_spec_is_not_rewritten_by_lower_budget(monkeypatch,tmp_path):
    monkeypatch.setattr(budgets,'resource_inventory',lambda _:inventory())
    spec = {'execution':json.dumps({'compile_threads':2})}
    with pytest.raises(ValueError,match='queued task'):
        Budget(compile_threads=1).admit(tmp_path,spec)
    assert json.loads(spec['execution'])['compile_threads'] == 2


def test_proc_parser_sums_only_own_group_and_handles_comm(tmp_path):
    for pid,group,rss in [(1,42,10),(2,42,20),(3,99,100)]:
        folder=tmp_path/str(pid); folder.mkdir()
        fields=['S','0',str(group)]+['0']*18+[str(rss)]
        (folder/'stat').write_text(f'{pid} (worker (name)) '+' '.join(fields))
    assert group_rss(42,tmp_path)==30*os.sysconf('SC_PAGE_SIZE')


def test_scratch_counter_does_not_follow_external_symlink(tmp_path):
    outside=tmp_path/'external';outside.write_bytes(b'x'*100000)
    scratch=tmp_path/'scratch';scratch.mkdir()
    (scratch/'link').symlink_to(outside)
    (scratch/'small').write_bytes(b'a'*10)
    assert scratch_size(scratch)<100000


def test_policy_roundtrip_and_no_credentials(tmp_path):
    ledger=TaskLedger(tmp_path/'queue'); ledger.page()
    budget=Budget(memory_budget_bytes=1000000)
    write_policy(ledger,budget)
    assert read_policy(ledger)==budget
    assert set(json.loads(budgets.policy_path(ledger).read_text()))==set(asdict(budget))


@pytest.mark.parametrize('kind', ['memory','scratch'])
def test_real_worker_budget_kill_and_scratch_cleanup(tmp_path,monkeypatch,kind):
    ledger=TaskLedger(tmp_path/'queue');identity=ledger.enqueue(CURRENT_SEED,EX,EV)
    limit=Budget(memory_budget_bytes=1) if kind=='memory' else Budget(cache_budget_bytes=4096)
    write_policy(ledger,limit)
    code="import os,pathlib,time;pathlib.Path(os.environ['TMPDIR'],'data').write_bytes(b'x'*100000);time.sleep(30)"
    monkeypatch.setattr(supervisor,'_worker_command',lambda *a:[sys.executable,'-c',code])
    result=supervisor.drain(ledger,tmp_path/'results',timeout=10)
    assert result['interrupted']==1
    assert ledger.get(identity)['reason']==('memory_budget_exceeded' if kind=='memory' else 'scratch_budget_exceeded')
    assert not list(tmp_path.glob('.fuzz-worker-*'))
    assert ledger.get(identity)['seed']==CURRENT_SEED


def test_cache_environment_isolated_and_removed(tmp_path,monkeypatch):
    ledger=TaskLedger(tmp_path/'queue');identity=ledger.enqueue(CURRENT_SEED,EX,EV)
    output=tmp_path/'observed.json'
    code="import json,os,pathlib,sys;pathlib.Path(sys.argv[1]).write_text(json.dumps({k:os.environ[k] for k in ['TMPDIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','OMP_NUM_THREADS']}))"
    monkeypatch.setattr(supervisor,'_worker_command',lambda *a:[sys.executable,'-c',code,str(output)])
    supervisor.drain(ledger,tmp_path/'results')
    observed=json.loads(output.read_text())
    assert observed['OMP_NUM_THREADS']=='1'
    scratch=Path(observed['TMPDIR'])
    assert Path(observed['TORCHINDUCTOR_CACHE_DIR']).parent==scratch
    assert Path(observed['TRITON_CACHE_DIR']).parent==scratch
    assert not scratch.exists()


def test_launch_admission_failure_has_no_partial_batch(tmp_path,monkeypatch):
    ledger=TaskLedger(tmp_path/'queue')
    monkeypatch.setattr(budgets,'resource_inventory',lambda _:inventory(disk_free_bytes=1))
    with pytest.raises(ValueError,match='disk reserve'):
        hosting.launch(ledger,tmp_path/'results',[(CURRENT_SEED,EX,EV,None)],foreground=True)
    assert ledger.page()['items']==[]


def test_setup_budget_reaches_cli_and_background_policy(tmp_path,monkeypatch):
    from autofusex.fuzz.client.cli import main
    setup(tmp_path,compile_threads=2,memory_budget_bytes=10000000,cache_budget_bytes=20000000)
    calls=[]
    monkeypatch.setattr(hosting,'launch',lambda l,d,s,**kw: calls.append((list(s),kw)) or {})
    assert main(['run',CURRENT_SEED,'--data-dir',str(tmp_path)])==0
    tasks,options=calls[0]
    assert tasks[0][1].compile_threads==2
    assert options['budget'].memory_budget_bytes==10000000
    assert main(['run',CURRENT_SEED,'--compile-threads','3','--data-dir',str(tmp_path)])==2
    assert len(calls)==1


def test_monitor_failure_cleans_worker_and_preserves_reason(tmp_path,monkeypatch):
    ledger=TaskLedger(tmp_path/'queue'); identity=ledger.enqueue(CURRENT_SEED,EX,EV)
    monkeypatch.setattr(supervisor,'_worker_command',lambda *a:[sys.executable,'-c','import time;time.sleep(30)'])
    def failed(*args):
        raise OSError('injected unavailable sampler')
    monkeypatch.setattr(budgets,'violation',failed)
    with pytest.raises(OSError):
        supervisor.drain(ledger,tmp_path/'results')
    assert ledger.get(identity)['state']=='interrupted'
    assert ledger.get(identity)['reason']=='supervisor_error:OSError'
    assert not list(tmp_path.glob('.fuzz-worker-*'))


def test_disk_reserve_exhaustion_terminates_worker(tmp_path,monkeypatch):
    ledger=TaskLedger(tmp_path/'queue');identity=ledger.enqueue(CURRENT_SEED,EX,EV)
    monkeypatch.setattr(supervisor,'_worker_command',lambda *a:[sys.executable,'-c','import time;time.sleep(30)'])
    # Keep admission realistic, inject only the sampled disk observation.
    monkeypatch.setattr(budgets,'violation',lambda *a:'disk_reserve_exhausted')
    supervisor.drain(ledger,tmp_path/'results')
    assert ledger.get(identity)['reason']=='disk_reserve_exhausted'
    assert not list(tmp_path.glob('.fuzz-worker-*'))
