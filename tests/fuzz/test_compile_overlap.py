import threading
import time
import pytest
from autofusex.workloads.fuzz.scheduling.pool import Pool, Device
from autofusex.workloads.fuzz.scheduling.stages import StageTask, Kind, Requirements
from autofusex.execution.scheduling.budgets import Budget


def stage():
    return StageTask(Kind.COMBINED, 'case', requirements=Requirements(devices=('npu:0',), device_bytes=100))


def test_compile_yield_admits_next_session_and_resumes_exclusively(tmp_path):
    pool = Pool(Budget(cpu_workers=3), tmp_path, 0, [Device('npu:0','uuid',300)])
    a,b,c = stage(),stage(),stage()
    try:
        pool.reserve(a)
        assert pool.reserve(b) is None
        pool._device_transition(a.identity, 'compile_wait', time.monotonic()+2)
        assert pool.reserve(b)
        resumed = threading.Event()
        def resume():
            pool._device_transition(a.identity, 'device_resume', time.monotonic()+2)
            resumed.set()
        t=threading.Thread(target=resume);t.start()
        assert not resumed.wait(.05)
        pool._device_transition(b.identity, 'compile_wait', time.monotonic()+2)
        assert resumed.wait(1)
        assert pool.reserve(c) is None
        t.join()
        lease=pool.release(a.identity)
        assert len(lease['device_intervals'])==2
        assert lease['device_intervals'][0][1] <= lease['device_intervals'][1][0]
        pool.release(b.identity)
    finally:pool.close()


def test_suspended_session_keeps_memory_budget_and_resume_times_out(tmp_path):
    pool=Pool(Budget(cpu_workers=2),tmp_path,0,[Device('npu:0','uuid',100)])
    a,b=stage(),stage()
    try:
        pool.reserve(a)
        pool._device_transition(a.identity,'compile_wait',time.monotonic()+1)
        assert pool.reserve(b) is None
        with pytest.raises(TimeoutError):
            pool._device_transition(a.identity,'device_resume',time.monotonic()-1)
        assert not pool.device_waiters
        pool.release(a.identity)
    finally:pool.close()


def test_cancelled_resume_does_not_reenter_device(tmp_path):
    pool=Pool(Budget(cpu_workers=2),tmp_path,0,[Device('npu:0','uuid',300)])
    a=stage()
    try:
        pool.reserve(a)
        pool._device_transition(a.identity,'compile_wait',time.monotonic()+1)
        pool.cancel(a.identity)
        with pytest.raises(RuntimeError,match='cancelled'):
            pool._device_transition(a.identity,'device_resume',time.monotonic()+1)
        assert not pool.leases[a.identity]['device_held']
        pool.release(a.identity)
    finally:pool.close()


def test_only_known_cpu_futures_yield_and_failures_reacquire(monkeypatch):
    import sys
    import types
    from concurrent.futures import Future
    from autofusex.execution.backends.compile_overlap import overlap_compile
    events=[]
    class AsyncCompile:
        def wait(self, scope):
            for future in scope.values():future.result()
    class CodeCacheFuture:pass
    compiler=types.ModuleType('compiler')
    compiler._start_parallel_jit_coordinator=lambda future:future
    modules={
        'torch._inductor.async_compile':types.SimpleNamespace(AsyncCompile=AsyncCompile),
        'torch._inductor.codecache':types.SimpleNamespace(CodeCacheFuture=CodeCacheFuture),
        'torch_npu._inductor.ascendc':types.SimpleNamespace(compiler=compiler),
    }
    for name,module in modules.items():monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.delenv('AUTOFUSE_DFX_FLAGS',raising=False)
    class Lease:
        def close(self):events.append('close')
        def acquire(self,timeout):events.append('acquire')
    adapter=types.SimpleNamespace(name='af-mlir',device_transition=events.append,
                                  synchronize=lambda:events.append('sync'),lease=Lease())
    unknown=Future();unknown.set_result(None)
    with overlap_compile(adapter):
        AsyncCompile().wait({'unknown':unknown})
        assert not events
        known=Future();known.set_exception(ValueError('compiler failure'))
        compiler._start_parallel_jit_coordinator(known)
        with pytest.raises(ValueError,match='compiler failure'):
            AsyncCompile().wait({'known':known})
    assert events==['sync','close','compile_wait','device_resume','acquire']


def test_pgo_retains_exclusive_lease(monkeypatch):
    from types import SimpleNamespace
    from autofusex.execution.backends.compile_overlap import overlap_compile
    monkeypatch.setenv('AUTOFUSE_DFX_FLAGS','--autofuse_enable_pgo=true')
    with overlap_compile(SimpleNamespace(name='af-mlir',device_transition=lambda event:pytest.fail(event))):
        pass


def test_resume_failure_aborts_instead_of_running_shape_recovery(monkeypatch):
    import sys
    from types import SimpleNamespace
    from concurrent.futures import Future
    from autofusex.execution.backends.compile_overlap import overlap_compile, DeviceOwnershipLost
    class AsyncCompile:
        def wait(self, scope):
            for f in scope.values():f.result()
    compiler=SimpleNamespace(_start_parallel_jit_coordinator=lambda future:future)
    monkeypatch.setitem(sys.modules,'torch._inductor.async_compile',SimpleNamespace(AsyncCompile=AsyncCompile))
    monkeypatch.setitem(sys.modules,'torch._inductor.codecache',SimpleNamespace(CodeCacheFuture=type('CodeCacheFuture',(),{})))
    monkeypatch.setitem(sys.modules,'torch_npu._inductor.ascendc',SimpleNamespace(compiler=compiler))
    monkeypatch.delenv('AUTOFUSE_DFX_FLAGS',raising=False)
    def transition(event):
        if event=='device_resume':raise TimeoutError('cancelled')
    adapter=SimpleNamespace(name='af-mlir',device_transition=transition,synchronize=lambda:None,
                            lease=SimpleNamespace(close=lambda:None))
    with overlap_compile(adapter):
        future=Future();future.set_result(None);compiler._start_parallel_jit_coordinator(future)
        with pytest.raises(DeviceOwnershipLost):AsyncCompile().wait({'kernel':future})
    assert not issubclass(DeviceOwnershipLost,Exception)


def test_topn_build_yields_but_runtime_load_still_has_device(monkeypatch):
    import sys
    from types import SimpleNamespace
    from concurrent.futures import Future
    from autofusex.execution.backends.compile_overlap import overlap_compile
    events=[]
    class Feature(Future):pass
    class AsyncCompile:
        def wait(self, scope):
            for f in scope.values():f.result()
    compiler=SimpleNamespace(_start_parallel_jit_coordinator=lambda f:f,_AscendcFeature=Feature)
    monkeypatch.setitem(sys.modules,'torch._inductor.async_compile',SimpleNamespace(AsyncCompile=AsyncCompile))
    monkeypatch.setitem(sys.modules,'torch._inductor.codecache',SimpleNamespace(CodeCacheFuture=type('CodeCacheFuture',(),{})))
    monkeypatch.setitem(sys.modules,'torch_npu._inductor.ascendc',SimpleNamespace(compiler=compiler))
    monkeypatch.delenv('AUTOFUSE_DFX_FLAGS',raising=False)
    adapter=SimpleNamespace(name='af-mlir',device_transition=events.append,synchronize=lambda:None,
                            lease=SimpleNamespace(close=lambda:events.append('close'),acquire=lambda timeout:events.append('acquire')))
    with overlap_compile(adapter):
        future=Feature();future.set_result(None)
        AsyncCompile().wait({'topn_build':future})
        events.append('runtime_load_and_autotune')
    assert events==['close','compile_wait','device_resume','acquire','runtime_load_and_autotune']
