"""Behavioral checks for disk retention, stop handling and exact seed continuation."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from autofusedebugger.nnsmith_integration.performance import artifacts, session
from autofusedebugger.nnsmith_integration.performance.lifecycle import SessionStopped
from autofusedebugger.nnsmith_integration.performance.schema import BackendResult, ProfileConfig
from autofusedebugger.nnsmith_integration.generation import GenerationConfig


def capture(tmp_path):
    run = tmp_path / 'cases' / 'run-owned'
    raw = run / 'profiling' / 'capture' / 'ASCEND_PROFILER_OUTPUT'
    raw.mkdir(parents=True)
    csv = raw / 'op_statistic.csv'
    csv.write_text('OP Type,Count,Total Time(us)\nAdd,10,25\n')
    (raw / 'trace_view.json').write_text('[]')
    cache = tmp_path / 'compile_cache' / run.name
    cache.mkdir(parents=True)
    (cache / 'kernel.so').write_bytes(b'compiled')
    result = BackendResult('ascendc', status='ok', correctness='pass', mean_kernel_us=2.5,
                           measured_steps=10, csv_path=str(csv))
    return run, cache, result


def test_compaction_preserves_csv_and_durable_metrics_before_deletion(tmp_path, monkeypatch):
    run, cache, result = capture(tmp_path)
    unrelated = cache.parent / 'other-worker'
    unrelated.mkdir(); (unrelated / 'keep').touch()
    real_delete = artifacts.shutil.rmtree
    def delete(path):
        saved = json.loads((run/'result.json').read_text())
        assert saved['mean_kernel_us'] == 2.5
        assert Path(saved['csv_path']).exists()
        real_delete(path)
    monkeypatch.setattr(artifacts.shutil, 'rmtree', delete)
    artifacts.finalize_artifacts(result, run, cache)
    assert not (run/'profiling').exists() and not cache.exists()
    assert Path(result.csv_path) == run/'op_statistic.csv'
    assert (unrelated/'keep').exists()
    assert result.mean_kernel_us == 2.5


def test_keep_artifacts_is_explicit(tmp_path):
    run, cache, result = capture(tmp_path)
    artifacts.finalize_artifacts(result, run, cache, keep=True)
    assert (run/'profiling').exists() and cache.exists()
    assert result.runtime['artifact_policy'] == 'keep'


def test_failed_capture_is_compacted_too(tmp_path):
    run, cache, result = capture(tmp_path)
    result.status, result.error, result.csv_path = 'error', 'parse failed', None
    artifacts.finalize_artifacts(result, run, cache)
    saved=json.loads((run/'result.json').read_text())
    assert saved['error']=='parse failed'
    assert Path(saved['csv_path']).name=='op_statistic.csv' and Path(saved['csv_path']).exists()
    assert not (run/'profiling').exists() and not cache.exists()


def test_failed_cleanup_stops_instead_of_accumulating_data(tmp_path, monkeypatch):
    run, cache, result = capture(tmp_path)
    monkeypatch.setattr(artifacts.shutil, 'rmtree', lambda path: (_ for _ in ()).throw(PermissionError('denied')))
    with pytest.raises(artifacts.ArtifactCleanupError):
        artifacts.finalize_artifacts(result, run, cache)
    saved=json.loads((run/'result.json').read_text())
    assert saved['mean_kernel_us']==2.5 and saved['cleanup_error']=='denied'


def test_external_csv_is_never_deleted(tmp_path):
    run, cache, result = capture(tmp_path)
    external=tmp_path/'external.csv'; external.write_text('keep')
    result.csv_path=str(external)
    with pytest.raises(artifacts.ArtifactCleanupError):
        artifacts.finalize_artifacts(result, run, cache)
    assert external.read_text()=='keep' and (run/'profiling').exists()


@pytest.fixture
def fake_benchmark(monkeypatch):
    import nnsmith.materialize as materialize
    from nnsmith.abstract.op import Add
    from autofusedebugger.nnsmith_integration import generation
    model = SimpleNamespace(add_seed_setter=lambda:None)
    monkeypatch.setattr(materialize.Model, 'init', lambda *a,**k:model)
    monkeypatch.setattr(generation, 'generate_case', lambda **k:object())
    monkeypatch.setattr(session, 'case_fingerprint', lambda case:'same')
    monkeypatch.setattr(session, 'describe_case', lambda case:{'n_nodes':2})
    calls=[]
    control={'stop_at':None, 'stop_signal':signal.SIGTERM, 'fatal':False}
    def backend(**kwargs):
        name=kwargs['spec'].name
        if control['stop_at'] is not None and len(calls)==control['stop_at']:
            if control['fatal']:raise OSError('cannot launch worker')
            signal.raise_signal(control['stop_signal'])
        calls.append((kwargs['request']['seed'],name))
        return BackendResult(name,status='ok',correctness='pass',fingerprint='same',mean_kernel_us=2 if name=='ascendc' else 3)
    monkeypatch.setattr(session,'run_backend',backend)
    def run(path,**kwargs):
        return session.run_performance_session(out_dir=path,generation=GenerationConfig(),
            profile=ProfileConfig(),opset=[Add],seed=42,**kwargs)
    return run,calls,control


@pytest.mark.parametrize('signum',[signal.SIGTERM,signal.SIGINT])
def test_continuous_session_stops_and_counts_only_completed_pairs(tmp_path,fake_benchmark,signum):
    run,calls,control=fake_benchmark
    control.update(stop_at=5,stop_signal=signum)
    old_handler=signal.getsignal(signum)
    summary=run(tmp_path)
    assert len(calls)==5 and summary['total']==2
    assert summary['session_status']=='stopped' and summary['stop_signal']==signum
    assert summary['checkpoint']['pending_seed'] is not None
    assert signal.getsignal(signum)==old_handler
    saved=json.loads((tmp_path/'summary.json').read_text())
    assert saved['total']==2 and saved['session_status']=='stopped'
    assert '"index": 2' not in (tmp_path/'summary.html').read_text()
    assert '持续运行' in (tmp_path/'summary.html').read_text()


def test_resume_retries_pending_seed_then_continues_identically(tmp_path,fake_benchmark):
    run,calls,control=fake_benchmark
    control['stop_at']=4
    first=run(tmp_path)
    pending=first['checkpoint']['pending_seed']
    control['stop_at']=None
    second=run(tmp_path,resume=True,n_cases=5)
    rng=random.Random(42)
    expected=[rng.getrandbits(32) for _ in range(5)]
    assert [c['seed'] for c in second['cases']]==expected
    assert second['cases'][2]['seed']==pending
    assert len(calls)==10 and second['total']==5
    assert [c['index'] for c in second['cases']]==list(range(5))
    assert second['session_status']=='completed'
    assert second['checkpoint']['pending_seed'] is None


def test_resuming_finite_session_uses_total_limit_not_additional_cases(tmp_path,fake_benchmark):
    run,calls,control=fake_benchmark
    run(tmp_path,n_cases=2)
    resumed=run(tmp_path,resume=True)
    assert resumed['total']==2 and len(calls)==4
    with pytest.raises(ValueError,match='below completed'):
        run(tmp_path,resume=True,n_cases=1)


def test_fatal_runner_error_keeps_prior_cases_and_final_report(tmp_path,fake_benchmark):
    run,calls,control=fake_benchmark
    control.update(stop_at=4,fatal=True)
    with pytest.raises(OSError,match='cannot launch'):
        run(tmp_path)
    saved=json.loads((tmp_path/'summary.json').read_text())
    assert saved['total']==2 and saved['session_status']=='error'
    assert 'cannot launch' in saved['session_error']
    assert 'error' in (tmp_path/'summary.html').read_text()


def test_new_run_does_not_overwrite_existing_results(tmp_path,fake_benchmark):
    run,calls,control=fake_benchmark
    run(tmp_path,n_cases=1)
    before=(tmp_path/'summary.json').read_bytes()
    with pytest.raises(ValueError,match='--resume'):
        run(tmp_path,n_cases=2)
    assert (tmp_path/'summary.json').read_bytes()==before


def test_resume_rejects_changed_measurement_settings(tmp_path,fake_benchmark):
    run,calls,control=fake_benchmark
    run(tmp_path,n_cases=1)
    with pytest.raises(ValueError,match='profile.steps'):
        session.run_performance_session(out_dir=tmp_path,generation=GenerationConfig(),
            profile=ProfileConfig(steps=20),seed=42,resume=True)


def test_report_write_is_atomic_if_replace_fails(tmp_path,monkeypatch):
    from autofusedebugger.nnsmith_integration.performance.storage import atomic_write_text
    path=tmp_path/'summary.json'; path.write_text('{"old":true}')
    monkeypatch.setattr(Path,'replace',lambda *args: (_ for _ in ()).throw(OSError('write interrupted')))
    with pytest.raises(OSError):atomic_write_text(path,'{"new":true}')
    assert json.loads(path.read_text())=={'old':True}


def test_output_lock_prevents_two_writers(tmp_path):
    from autofusedebugger.nnsmith_integration.performance.storage import session_lock
    with session_lock(tmp_path):
        with pytest.raises(ValueError,match='Another performance session'):
            with session_lock(tmp_path):pass


def test_replay_does_not_accept_legacy_retention_flags():
    from autofusedebugger.fuzz.cli import replay_parser
    for flag in ('--resume', '--keep-perf-artifacts'):
        with pytest.raises(SystemExit):
            replay_parser().parse_args(['42', flag])


def test_replay_default_is_one_complete_case_not_legacy_batch():
    from autofusedebugger.fuzz.cli import replay_parser
    args = replay_parser().parse_args(['42'])
    assert args.seed == '42' and not args.static_only
    assert not hasattr(args, 'n_cases')


def test_runner_interrupt_cleans_worker_capture_and_cache(tmp_path,monkeypatch):
    from autofusedebugger.nnsmith_integration.performance import runner
    from autofusedebugger.nnsmith_integration.performance.backends import BACKENDS
    class Process:
        pid=12345
        waits=0
        def wait(self,timeout=None):
            self.waits+=1
            if self.waits==1:raise KeyboardInterrupt()
            return -15
    def popen(command,**kwargs):
        request=json.loads(Path(command[command.index('--request')+1]).read_text())
        raw=Path(request['profile_dir']); raw.mkdir();(raw/'trace.bin').write_bytes(b'large')
        cache=Path(kwargs['env']['TORCHINDUCTOR_CACHE_DIR']);cache.mkdir(parents=True)
        (cache/'kernel.so').touch()
        return Process()
    signals=[]
    monkeypatch.setattr(runner.subprocess,'Popen',popen)
    monkeypatch.setattr(runner.os,'killpg',lambda pid,sig:signals.append(sig))
    with pytest.raises(KeyboardInterrupt):
        runner.run_backend(spec=BACKENDS['ascendc'],request={},output_dir=tmp_path/'case',
            cache_dir=tmp_path/'cache',profile=ProfileConfig())
    assert signals==[signal.SIGTERM,signal.SIGKILL]
    assert not list((tmp_path/'case').rglob('trace.bin'))
    assert not list((tmp_path/'cache').rglob('kernel.so'))
    result=json.loads(next((tmp_path/'case').rglob('result.json')).read_text())
    assert result['status']=='interrupted'


def test_compaction_is_idempotent_and_resume_removes_partial_capture(tmp_path):
    from autofusedebugger.nnsmith_integration.performance.artifacts import compact_pending_case
    run=tmp_path/'cases'/'00000-42'/'ascendc'/'run-abandoned'
    (run/'profiling').mkdir(parents=True)
    (run/'profiling'/'partial.bin').touch()
    cache=tmp_path/'compile_cache'/'ascendc'/run.name
    cache.mkdir(parents=True);(cache/'kernel.so').touch()
    compact_pending_case(tmp_path,'00000-42')
    assert not (run/'profiling').exists() and not cache.exists()
    compact_pending_case(tmp_path,'00000-42')
    assert json.loads((run/'result.json').read_text())['status']=='interrupted'


@pytest.mark.skipif(sys.platform!='linux',reason='Linux parent-death signal')
def test_kill9_of_parent_terminates_worker_and_its_child(tmp_path):
    # Directly load the stdlib-only module, avoiding Torch/NPU startup in this OS test.
    import autofusedebugger.nnsmith_integration.performance.lifecycle as lifecycle
    marker=tmp_path/'ready.json'
    worker_code='''import importlib.util,os,subprocess,sys,time,json
spec=importlib.util.spec_from_file_location('lifecycle',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.guard_worker_parent(sys.argv[2])
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
open(sys.argv[3],'w').write(json.dumps([os.getpid(),p.pid]))
time.sleep(60)
'''
    parent_code='''import subprocess,sys,time,os
subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],str(os.getpid()),sys.argv[3]],start_new_session=True)
time.sleep(60)
'''
    parent=subprocess.Popen([sys.executable,'-c',parent_code,worker_code,lifecycle.__file__,str(marker)])
    pids=[]
    def active(pid):
        try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z'
        except FileNotFoundError:return False
    try:
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if marker.exists():
                try:pids=json.loads(marker.read_text());break
                except ValueError:pass
            time.sleep(.02)
        assert len(pids)==2
        parent.kill();parent.wait(timeout=5)
        deadline=time.monotonic()+5
        while any(active(p) for p in pids) and time.monotonic()<deadline:time.sleep(.02)
        assert not any(active(p) for p in pids)
    finally:
        if parent.poll() is None:parent.kill();parent.wait()
        if pids:
            try:os.killpg(pids[0],signal.SIGKILL)
            except ProcessLookupError:pass


def test_signal_during_commit_keeps_counts_and_checkpoint_consistent(tmp_path,fake_benchmark,monkeypatch):
    run,calls,control=fake_benchmark
    original=session.write_reports
    triggered=[]
    def write(summary,out):
        if summary['total']==1 and summary['session_status']=='running' and not triggered:
            triggered.append(True)
            signal.raise_signal(signal.SIGTERM)
        original(summary,out)
    monkeypatch.setattr(session,'write_reports',write)
    summary=run(tmp_path,n_cases=5)
    saved=json.loads((tmp_path/'summary.json').read_text())
    assert summary['session_status']=='stopped'
    assert saved['total']==len(saved['cases'])==saved['comparable']==1
    assert saved['checkpoint']['pending_seed'] is None
    assert saved['performance_passed']==1
