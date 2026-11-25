"""Yield a staged device reservation while AscendC build futures finish.

No compiler output crosses processes. Top-N generates candidates on the host;
device autotuning runs in NpuInductorAutoTuneKernel.load(), after wait returns
and ownership has been restored. PGO/unknown futures retain device ownership.
Runtime patches live only in one worker invocation.
"""
from contextlib import contextmanager
from concurrent.futures import Future
from unittest.mock import patch


class DeviceOwnershipLost(BaseException):
    """Abort the worker; ordinary per-shape recovery cannot run without a lease."""


@contextmanager
def overlap_compile(adapter):
    transition = getattr(adapter, 'device_transition', None)
    if transition is None or adapter.name not in {'af-mlir', 'af-legacy'}:
        yield
        return
    import os
    flags = os.environ.get('AUTOFUSE_DFX_FLAGS', '').lower()
    if 'pgo' in flags:
        yield
        return
    try:
        from torch._inductor.async_compile import AsyncCompile
        from torch._inductor.codecache import CodeCacheFuture
        from torch_npu._inductor.ascendc import compiler
        coordinator = compiler._start_parallel_jit_coordinator
        # Both the JIT coordinator and Top-N build return this CPU-build future.
        feature_type = getattr(compiler, '_AscendcFeature', ())
    except (ImportError, AttributeError):
        yield
        return
    safe = set()
    original_wait = AsyncCompile.wait

    def submit(*args, **kwargs):
        future = coordinator(*args, **kwargs)
        safe.add(future)
        return future

    def wait(instance, scope):
        futures = [v for v in scope.values() if isinstance(v, (Future, CodeCacheFuture))]
        if not futures or any(f not in safe and not isinstance(f, feature_type) for f in futures):
            return original_wait(instance, scope)
        # Inputs may have asynchronous copies queued before tracing.
        adapter.synchronize()
        adapter.lease.close()
        transition('compile_wait')
        try:
            return original_wait(instance, scope)
        finally:
            # The scheduler grants the slot before the OS lease is reacquired.
            # Even compilation failure must regain ownership before cleanup.
            try:
                transition('device_resume')
                adapter.lease.acquire(timeout=60)
            except Exception as exc:
                raise DeviceOwnershipLost('device ownership could not be restored') from exc
            safe.difference_update(futures)

    with patch.object(compiler, '_start_parallel_jit_coordinator', submit), patch.object(AsyncCompile, 'wait', wait):
        yield
