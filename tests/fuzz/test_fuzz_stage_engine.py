from autofusex.fuzz.generation.sampling import SamplingConfig
CURRENT_SEED = str(SamplingConfig.profile('small', max_nodes=1, dtype_mode='float32').seed(3))
"""Stage scheduling, independent fact sync and bounded artifact regressions."""
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from dataclasses import asdict
import pytest
from autofusex.fuzz.scheduling.stages import StageTask,Kind,Requirements
from autofusex.fuzz.scheduling.pool import Pool,Device
from autofusex.fuzz.scheduling.budgets import Budget
from autofusex.fuzz.contracts import ExecutionConfig,EvaluationConfig


def test_dag_waits_for_both_branches():
    a=StageTask(Kind.REFERENCE,'case');b=StageTask(Kind.COMBINED,'case')
    compare=StageTask(Kind.COMPARE,'case',(a.identity,b.identity))
    stages={s.identity:s for s in (a,b,compare)}
    a.state='succeeded'
    assert not compare.ready(stages,10)
    b.state='succeeded'
    assert compare.ready(stages,20) and compare.ready_ns==20


def test_atomic_resources_do_not_hold_cpu_waiting_for_device(tmp_path):
    pool=Pool(Budget(cpu_workers=2),tmp_path,0,[Device('npu:0','uuid0',100),Device('npu:1','uuid1',100)])
    try:
        a=StageTask(Kind.COMBINED,'case',requirements=Requirements(devices=('npu:0',),device_bytes=100))
        b=StageTask(Kind.COMBINED,'case',requirements=Requirements(devices=('npu:0',),device_bytes=100))
        c=StageTask(Kind.COMBINED,'case',requirements=Requirements(devices=('npu:0','npu:1'),device_bytes=100))
        assert pool.reserve(a)['device_uuid']=='uuid0'
        assert pool.reserve(b) is None and len(pool.leases)==1
        assert pool.reserve(c)['device_uuid']=='uuid1'
        assert pool.release(a.identity)['released']>0
        pool.release(c.identity)
    finally:pool.close()


def test_contract_staged_does_not_enable_compile_split():
    assert ExecutionConfig('cpu','cpu',1,scheduling='staged').scheduling=='staged'
    with pytest.raises(ValueError):ExecutionConfig('cpu','cpu',1,scheduling='split')
    from autofusex.fuzz.backends.torch import TorchAdapter
    assert not TorchAdapter.capabilities.independent_compile
    assert not TorchAdapter.capabilities.transferable_compilation


def test_safe_host_artifact_digest(tmp_path):
    torch=pytest.importorskip('torch')
    from autofusex.fuzz.scheduling.artifacts import write_arrays,read_arrays
    ref=write_arrays(tmp_path/'out.npz',[(torch.arange(3),)])
    assert torch.equal(read_arrays(tmp_path,ref)[0][0],torch.arange(3))
    (tmp_path/'out.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='digest'):read_arrays(tmp_path,ref)
    with pytest.raises(ValueError,match='reference'):read_arrays(tmp_path,dict(ref,file='../out.npz'))


def test_fact_sync_idempotent_owned_and_immutable(tmp_path):
    from autofusex.fuzz.storage import ResultRepository
    from autofusex.fuzz.storage.lifecycle import append
    from autofusex.fuzz.sync import outbox,facts
    from autofusex.fuzz.central.repository import create_user,ingest
    from test_fuzz_precise_metrics import Adapter
    from fuzz_fixtures import bundle
    from autofusex.fuzz.execution import execute
    local=tmp_path/'local';central=tmp_path/'central'
    result=execute(bundle('C16'),Adapter(),EvaluationConfig())
    identity=ResultRepository(local).save(result)
    user=create_user(central,'alice');batch=outbox.pending(local)
    ack=ingest(central,user['token'],batch);outbox.acknowledge(local,batch['source_id'],ack['ack'])
    batch=facts.pending(local)
    ack=facts.ingest(central,user['token'],batch)
    assert facts.ingest(central,user['token'],batch)==ack
    facts.acknowledge(local,batch,ack)
    assert not facts.pending(local)['items']
    detail=ResultRepository(central).detail(identity)
    assert {e['event'] for e in detail['lifecycle']}=={'persisted','uploaded_ack'}
    evil=json.loads(json.dumps(batch));evil['items'][0]['payload']['monotonic_ns']+=1
    with pytest.raises(ValueError,match='revision conflict'):facts.ingest(central,user['token'],evil)
    other=create_user(central,'bob')
    with pytest.raises(PermissionError):facts.ingest(central,other['token'],batch)


def test_interval_union_and_percentiles():
    from autofusex.fuzz.reporting.efficiency import union,percentile
    assert union([(1,5),(3,7),(10,12)])==8
    assert percentile([1,2,3,4],.95)==4
    assert percentile([],.5) is None


@pytest.mark.skipif(sys.platform!='linux',reason='Linux managed runtime')
def test_real_staged_cpu_shared_generation_and_runtime(tmp_path):
    pytest.importorskip('torch')
    from autofusex.fuzz.scheduling.ledger import TaskLedger
    from autofusex.fuzz.scheduling.supervisor import drain
    from autofusex.fuzz.scheduling.budgets import write_policy
    ledger=TaskLedger(tmp_path/'tasks')
    for backend in ['cpu','cpu-compile']:
        ledger.enqueue(CURRENT_SEED,ExecutionConfig(backend,'cpu',1,scheduling='staged',static_only=True,batch_id='test'),EvaluationConfig())
    write_policy(ledger,Budget(cpu_workers=2))
    # CPU compile is a real distinct backend; no self-comparison shortcut.
    result=drain(ledger,tmp_path/'db',timeout=120)
    assert result['completed']==2
    with sqlite3.connect(tmp_path/'db') as conn:
        kinds=[json.loads(r[0])['kind'] for r in conn.execute('SELECT payload FROM stage_facts_v1')]
        assert kinds.count('generate')==1 and kinds.count('reference')==1
        assert kinds.count('compare')==1 and kinds.count('record')==2
        assert conn.execute("SELECT count(*) FROM attempts WHERE status='ok'").fetchone()[0]==2


@pytest.mark.skipif(sys.platform!='linux',reason='Linux managed runtime')
def test_staged_cancel_reaps_workers_and_preserves_retry(tmp_path,monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from autofusex.fuzz.scheduling.ledger import TaskLedger
    from autofusex.fuzz.scheduling.supervisor import drain
    from autofusex.fuzz.scheduling.budgets import write_policy
    from autofusex.fuzz.scheduling import pool as module
    ledger=TaskLedger(tmp_path/'tasks')
    identity=ledger.enqueue(CURRENT_SEED,ExecutionConfig('cpu','cpu',1,scheduling='staged'),EvaluationConfig())
    write_policy(ledger,Budget(cpu_workers=1))
    original=module.subprocess.Popen
    children=[]
    def slow(args,**kwargs):
        args=[sys.executable,'-c','import time;time.sleep(60)']
        child=original(args,**kwargs);children.append(child);return child
    monkeypatch.setattr(module.subprocess,'Popen',slow)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(drain,ledger,tmp_path/'db',timeout=10)
        until=time.monotonic()+5
        while not children and time.monotonic()<until:time.sleep(.02)
        ledger.stop(identity)
        assert future.result(timeout=15)['interrupted']==1
    assert all(p.poll() is not None for p in children)
    assert ledger.get(identity)['state']=='interrupted'
    assert not list(tmp_path.glob('.stage-window-*'))


@pytest.mark.skipif(sys.platform!='linux',reason='Linux managed runtime')
def test_staged_worker_timeout_no_shape_and_no_auto_retry(tmp_path,monkeypatch):
    from autofusex.fuzz.scheduling.ledger import TaskLedger
    from autofusex.fuzz.scheduling.supervisor import drain
    from autofusex.fuzz.scheduling.budgets import write_policy
    from autofusex.fuzz.scheduling import pool as module
    ledger=TaskLedger(tmp_path/'tasks');ledger.enqueue(CURRENT_SEED,ExecutionConfig('cpu','cpu',1,scheduling='staged'),EvaluationConfig())
    write_policy(ledger,Budget(cpu_workers=1))
    original=module.subprocess.Popen;count=0
    def once(args,**kwargs):
        nonlocal count
        count+=1
        if count==1:args=[sys.executable,'-c','import time;time.sleep(60)']
        return original(args,**kwargs)
    monkeypatch.setattr(module.subprocess,'Popen',once)
    # A worker timeout is infrastructure interruption, not a generated-case
    # rejection. Neither outcome creates a synthetic attempt or auto-retry.
    result=drain(ledger,tmp_path/'db',timeout=10)
    assert result['completed']==0 and result['interrupted']==1 and result['skipped']==0
    with sqlite3.connect(tmp_path/'db') as conn:
        assert conn.execute('SELECT count(*) FROM shapes').fetchone()[0]==0
        assert conn.execute('SELECT count(*) FROM attempts').fetchone()[0]==0
    assert count==1


def test_snapshot_keeps_lifecycle_but_not_private_fact_outbox(tmp_path):
    from autofusex.fuzz.storage import ResultRepository
    from test_fuzz_precise_metrics import Adapter
    from fuzz_fixtures import bundle
    from autofusex.fuzz.execution import execute
    repo=ResultRepository(tmp_path/'db');identity=repo.save(execute(bundle('C16'),Adapter(),EvaluationConfig()))
    repo.snapshot(tmp_path/'copy')
    assert ResultRepository(tmp_path/'copy').detail(identity)['lifecycle']
    with sqlite3.connect(tmp_path/'copy') as conn:
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='fact_outbox_v1'").fetchone()


@pytest.mark.skipif(sys.platform!='linux',reason='Linux managed runtime')
def test_record_failure_is_resumable_not_fake_success(tmp_path,monkeypatch):
    from autofusex.fuzz.scheduling.ledger import TaskLedger
    from autofusex.fuzz.scheduling.supervisor import drain
    from autofusex.fuzz.scheduling.budgets import write_policy
    from autofusex.fuzz.scheduling.pool import Pool
    ledger=TaskLedger(tmp_path/'tasks')
    identity=ledger.enqueue(CURRENT_SEED,ExecutionConfig('cpu','cpu',1,scheduling='staged'),EvaluationConfig())
    write_policy(ledger,Budget(cpu_workers=2))
    original=Pool._run
    def disk_failure(self, identity, key, job, timeout):
        if job['kind']=='record':raise OSError('injected read-only result volume')
        return original(self,identity,key,job,timeout)
    monkeypatch.setattr(Pool,'_run',disk_failure)
    result=drain(ledger,tmp_path/'db',timeout=90)
    assert result['completed']==0 and result['record_failed']==1
    assert ledger.get(identity)['state']=='interrupted'
    if (tmp_path/'db').exists():
        with sqlite3.connect(tmp_path/'db') as conn:
            assert conn.execute('SELECT count(*) FROM attempts').fetchone()[0]==0


def test_oversized_device_request_fails_instead_of_hanging(tmp_path):
    pool=Pool(Budget(cpu_workers=1),tmp_path,0,[Device('npu:0','uuid',32)])
    try:
        task=StageTask(Kind.COMBINED,'case',requirements=Requirements(devices=('npu:0',),device_bytes=64))
        with pytest.raises(ValueError,match='capacity'):pool.reserve(task)
        assert not pool.leases
    finally:pool.close()


def test_shared_bundle_reuses_exact_input_bytes_without_regeneration(tmp_path,monkeypatch):
    from fuzz_fixtures import bundle
    from autofusex.fuzz.scheduling.artifacts import write_bundle,bundle_from
    case=bundle('C16');ref=write_bundle(tmp_path,case)
    monkeypatch.setattr('autofusex.fuzz.generation.nnsmith_sampled.materialize_spec',lambda *a:pytest.fail('must not regenerate input bytes'))
    assert bundle_from(ref,tmp_path)==case
    (tmp_path/'inputs.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='digest'):bundle_from(ref,tmp_path)


def test_fact_http_disconnect_retry_and_ack_loss(tmp_path):
    import threading
    from urllib.request import urlopen
    from autofusex.fuzz.central.server import create_server
    from autofusex.fuzz.central.repository import create_user
    from autofusex.fuzz.storage import ResultRepository
    from autofusex.fuzz.sync.client import once
    from autofusex.fuzz.sync.facts import pending
    from test_fuzz_precise_metrics import Adapter
    from fuzz_fixtures import bundle
    from autofusex.fuzz.execution import execute
    server=create_server(tmp_path/'central',0)
    user=create_user(tmp_path/'central','test')
    key=tmp_path/'key';key.write_text(user['token'])
    (tmp_path/'sync.json').write_text(json.dumps({'endpoint':f'http://127.0.0.1:{server.server_port}','key_file':str(key),'interval':1}))
    identity=ResultRepository(tmp_path/'local').save(execute(bundle('C16'),Adapter(),EvaluationConfig()))
    thread=threading.Thread(target=server.serve_forever);thread.start()
    try:
        def disconnected(*a,**k):raise OSError('offline')
        with pytest.raises(OSError):once(tmp_path,tmp_path/'local',opener=disconnected)
        assert once(tmp_path,tmp_path/'local',opener=urlopen)['uploaded']==1
        def lost_ack(request,**kwargs):
            assert request.get_header('User-agent') == 'AutofuseX/0.1.2 (+https://esay.app)'
            response=urlopen(request,**kwargs);response.read();response.close()
            raise OSError('ACK lost after central commit')
        with pytest.raises(OSError):once(tmp_path,tmp_path/'local',opener=lost_ack)
        assert pending(tmp_path/'local')['items']
        assert once(tmp_path,tmp_path/'local',opener=urlopen)['fact_uploaded']==2
        assert not pending(tmp_path/'local')['items']
        assert len(ResultRepository(tmp_path/'central').detail(identity)['lifecycle'])==2
    finally:
        server.shutdown();thread.join();server.server_close()


def test_scratch_sampling_tolerates_concurrent_retirement(tmp_path,monkeypatch):
    from autofusex.fuzz.scheduling.budgets import scratch_size
    import os
    def vanished(root,followlinks,onerror):
        onerror(FileNotFoundError('retired cache'))
        return iter([])
    monkeypatch.setattr(os,'walk',vanished)
    assert scratch_size(tmp_path)==0


def test_fact_epoch_reset_preserves_existing_observations(tmp_path):
    from autofusex.fuzz.storage import ResultRepository
    from autofusex.fuzz.sync import outbox,facts
    from autofusex.fuzz.central.repository import create_user,ingest
    from test_fuzz_precise_metrics import Adapter
    from fuzz_fixtures import bundle
    from autofusex.fuzz.execution import execute
    local=tmp_path/'local';central=tmp_path/'central'
    ResultRepository(local).save(execute(bundle('C16'),Adapter(),EvaluationConfig()))
    user=create_user(central,'alice')
    for _ in range(2):
        b=outbox.pending(local);a=ingest(central,user['token'],b);outbox.acknowledge(local,b['source_id'],a['ack'])
        b=facts.pending(local);a=facts.ingest(central,user['token'],b);facts.acknowledge(local,b,a)
        outbox.reset_epoch(local)
    with sqlite3.connect(central) as c:assert c.execute('SELECT count(*) FROM lifecycle_v1').fetchone()[0]==2


@pytest.mark.parametrize("samples", [1, 3])
def test_reference_is_once_per_shape_without_self_comparison(tmp_path, monkeypatch, samples):
    torch = pytest.importorskip('torch')
    from fuzz_fixtures import bundle
    from autofusex.fuzz.scheduling import artifacts
    from autofusex.fuzz.scheduling.stage_worker import perform
    from autofusex.fuzz.backends import torch as backend
    calls = []
    original = backend.TorchAdapter.invoke

    def invoke(self, prepared, inputs):
        calls.append(self.name)
        return original(self, prepared, inputs)

    def forbidden(*args):
        raise AssertionError('reference must not execute a second CPU oracle')

    monkeypatch.setattr(backend.TorchAdapter, 'invoke', invoke)
    monkeypatch.setattr(backend.TorchAdapter, 'cpu_reference', forbidden)
    monkeypatch.setattr(backend.TorchAdapter, 'compare', forbidden)
    from dataclasses import replace
    from autofusex.fuzz.generation import materialize
    shared = materialize(CURRENT_SEED)
    shared = replace(shared, spec=replace(shared.spec, shape_samples=shared.spec.shape_samples*samples,
                                          output_samples=shared.spec.output_samples*samples))
    monkeypatch.setattr(artifacts, 'bundle_from', lambda *args: shared)
    result = perform(dict(kind='reference', directory=str(tmp_path), bundle={},
                         config=asdict(ExecutionConfig('cpu', 'cpu', 1, scheduling='staged')),
                         evaluation=asdict(EvaluationConfig()), output_file='reference.npz'))
    assert calls == ['cpu'] * samples
    assert result['result']['status'] == 'ok'
    assert all(row['correctness'] == 'not_checked' for row in result['result']['shape_results'])
    assert len(artifacts.read_arrays(tmp_path, result['outputs'])) == samples


def test_cancel_before_io_dispatch_never_starts_worker(tmp_path, monkeypatch):
    pool = Pool(Budget(cpu_workers=1), tmp_path, 0, [])
    task = StageTask(Kind.GENERATE, 'case')
    pool.reserve(task)
    monkeypatch.setattr(pool, '_worker', lambda key: pytest.fail('cancelled worker launched'))
    pool.cancel(task.identity)
    try:
        with pytest.raises(RuntimeError, match='cancelled'):
            pool.submit(task, {'kind': 'generate'}, 1).result(timeout=5)
        pool.release(task.identity)
    finally:
        pool.close()


def test_compressed_artifact_bound_checked_before_loading(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    import hashlib
    import zipfile
    import numpy as np
    from autofusex.fuzz.scheduling import artifacts
    path = tmp_path/'out.npz'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('s0_0.npy', b'x'*4096)
    monkeypatch.setattr(artifacts, 'MAX_BYTES', 1024)
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('allocated before checking bound'))
    with pytest.raises(ValueError, match='unpacked'):
        artifacts.read_arrays(tmp_path, dict(version=1, file=path.name, counts=[1],
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def test_ops_whitelist_filters_generated_graph_without_changing_seed(tmp_path, monkeypatch):
    from fuzz_fixtures import bundle
    from autofusex.fuzz import generation
    from autofusex.fuzz.scheduling.stage_worker import perform
    monkeypatch.setattr(generation, 'materialize', lambda _: bundle())
    # C16 contains relu/amin/add/mul/amax/transpose; an add-only whitelist
    # must skip it after generation, not alter the generated graph.
    assert perform({'kind':'generate','seed':'C16','directory':str(tmp_path),
                    'ops':['add']}) == {'excluded':True}
    assert not list(tmp_path.iterdir())


def test_excluded_graph_never_runs_backend(tmp_path, monkeypatch):
    from fuzz_fixtures import bundle
    from autofusex.fuzz import generation
    from autofusex.fuzz.scheduling.stage_worker import perform
    monkeypatch.setattr(generation, 'materialize', lambda _: bundle())
    assert perform({'kind':'generate','seed':'C16','directory':str(tmp_path),
                    'exclude_ops':['transpose']}) == {'excluded':True}
    assert not list(tmp_path.iterdir())


def test_npu_first_resource_budget_respects_limits(monkeypatch):
    from autofusex.fuzz import resources
    monkeypatch.setattr(resources, 'resource_inventory', lambda _: {
        'cpu_worker_ceiling':40,'effective_memory_bytes':200*1024**3})
    monkeypatch.setattr(sys, 'platform', 'linux')
    assert resources.npu_first_defaults('.',2,Budget()) == {'cpu_workers':4,'compile_threads':8}
    assert resources.npu_first_defaults('.',2,Budget(memory_budget_bytes=4*1024**3))['cpu_workers']==2


def test_mlir_runtime_reuse_is_explicit_and_not_compile_split():
    from autofusex.fuzz.backends.torch import TorchAdapter
    for backend, reusable in [('cpu', True), ('eager-npu', True), ('af-mlir', True),
                              ('af-legacy', False), ('triton-npu', False)]:
        adapter = TorchAdapter(ExecutionConfig(backend, 'cpu' if backend == 'cpu' else 'npu:0', 1))
        assert adapter.capabilities.persistent_runtime is reusable
        assert not adapter.capabilities.independent_compile
        assert not adapter.capabilities.transferable_compilation


@pytest.mark.parametrize('reusable,status,count,rss,device_bytes,cache_bytes,retired', [
    (True, 'ok', 0, 1024, 0, 0, False),
    (False, 'ok', 0, 1024, 0, 0, True),
    (None, 'ok', 0, 1024, 0, 0, True),
    (True, 'error', 0, 1024, 0, 0, True),
    (True, 'unsupported', 0, 1024, 0, 0, True),
    (True, 'ok', 15, 1024, 0, 0, True),
    (True, 'ok', 0, 300*1024**2, 0, 0, True),
    (True, 'ok', 0, 1024, 17*1024**2, 0, True),
    (True, 'ok', 0, 1024, 0, 3*1024**3, True),
])
def test_compiled_runtime_reuse_preserves_recycling_guards(tmp_path, monkeypatch,
        reusable, status, count, rss, device_bytes, cache_bytes, retired):
    from types import SimpleNamespace
    from autofusex.fuzz.scheduling import pool as module, budgets
    pool = Pool(Budget(cpu_workers=1), tmp_path, 0, [])
    worker = dict(socket=SimpleNamespace(settimeout=lambda _: None), identity='worker',
                  child=SimpleNamespace(pid=123), count=count, rss_floor=1024,
                  scratch=SimpleNamespace(name=str(tmp_path)), busy=False)
    pool.workers.append(worker)
    monkeypatch.setattr(pool, '_worker', lambda key: worker)
    removed = []
    monkeypatch.setattr(pool, '_retire', lambda w: removed.append(w))
    monkeypatch.setattr(module, 'send', lambda *a: None)
    value = dict(result={'status': status}, device_memory_bytes=device_bytes)
    if reusable is not None:
        value['persistent_runtime'] = reusable
    monkeypatch.setattr(module, 'receive', lambda *a: dict(ok=True, value=value))
    monkeypatch.setattr(budgets, 'group_rss', lambda pid: rss)
    monkeypatch.setattr(budgets, 'scratch_size', lambda path: cache_bytes)
    try:
        result = pool._run('stage', ('af-mlir', 'npu:0', 1),
                           {'kind': 'first_call_combined', 'config': {'backend': 'af-mlir'}}, 30)
        assert result['ok']
        assert bool(removed) is retired
        assert (worker in pool.workers) is not retired
        if not retired:
            assert not worker['busy']
    finally:
        pool.close()


def test_worker_key_includes_compile_threads(tmp_path, monkeypatch):
    from types import SimpleNamespace
    pool = Pool(Budget(cpu_workers=1), tmp_path, 0, [])
    keys = []
    monkeypatch.setattr(pool.executor, 'submit', lambda fn, identity, key, job, timeout: keys.append(key))
    stage = SimpleNamespace(identity='stage')
    pool.leases['stage'] = {'device': 'npu:1'}
    try:
        for threads in (1, 2):
            pool.submit(stage, {'kind': 'first_call_combined',
                'config': {'backend': 'af-mlir', 'compile_threads': threads}}, 30)
        assert keys == [('af-mlir', 'npu:1', 1), ('af-mlir', 'npu:1', 2)]
    finally:
        pool.leases.clear()
        pool.close()
