"""Serial foreground supervisor with process-group cleanup/reconciliation.

No throughput claims or CPU/compile-stage splitting. NPU correctness uses UUID leases.
"""
from contextlib import closing
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from .ownership import own_queue
from ..contracts import ExecutionConfig, EvaluationConfig


def _bind_results(ledger, database):
    """Keep recovery pointed at the original result archive, including on restart."""
    path = ledger.path.with_name(ledger.path.name + '.results.json')
    value = {'version': 1, 'database': str(database)}
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('queue is bound to a different result database')
        return
    # Called only while holding queue ownership; atomic metadata publication.
    fd, name = tempfile.mkstemp(prefix='.fuzz-binding-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(value, out)
            out.flush(); os.fsync(out.fileno())
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _tasks(ledger):
    after = 0
    while True:
        page = ledger.page(after=after)
        yield from page['items']
        if page['next_cursor'] is None:
            return
        after = page['next_cursor']


def _validate_task(task):
    config = ExecutionConfig(**json.loads(task['execution']))
    evaluation = EvaluationConfig(**json.loads(task['evaluation']))
    if config.backend in {'cpu', 'cpu-eager', 'cpu-compile'}:
        if config.device != 'cpu':
            raise ValueError('backend/device mismatch')
    elif config.backend in {'ascendc', 'triton', 'npu-eager', 'eager-npu', 'triton-npu', 'af-legacy', 'af-mlir'}:
        import re
        if not re.fullmatch(r'npu:[0-9]+', config.device):
            raise ValueError('NPU requires explicit logical npu:index')
        if evaluation.performance:
            raise ValueError('strict NPU performance requires site-wide external workload coordination')
    else:
        raise ValueError('unsupported managed backend')
    if type(config.compile_threads) is not int or not 1 <= config.compile_threads <= (os.cpu_count() or 1):
        raise ValueError('compile thread budget exceeds available CPU count')


def _has_result(database, task):
    from ..storage.repository import _connect
    from ..storage.transfer import _record
    from ..generation import materialize, workload_id, oracle_id
    import sqlite3
    if not database.exists():
        return False
    with closing(_connect(database)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('BEGIN')
        row = conn.execute('SELECT * FROM attempts WHERE id=?', (task['attempt_id'],)).fetchone()
        if row is None:
            return False
        contexts, _ = _record(conn, row)
        bundle = materialize(task['seed'])
        if (row['seed'] != task['seed'] or row['workload_id'] != workload_id(bundle) or
                row['oracle_id'] not in {oracle_id(bundle), 'torch-eager-cpu/1'} or
                row['backend'] != json.loads(task['execution'])['backend'] or
                {k:v for k,v in contexts['execution'].items() if k not in {'telemetry','diagnostics'}} != json.loads(task['execution']) or
                contexts['evaluation'] != json.loads(task['evaluation'])):
            raise ValueError('saved attempt conflicts with task; refusing acknowledgement')
        return True


def _reconcile(ledger, database):
    recovered = interrupted = 0
    for task in _tasks(ledger):
        if task['state'] not in {'running', 'stop_requested'}:
            continue
        # Caller owns queue; inherited worker lock prevents recovery while alive.
        if _has_result(database, task):
            ledger.complete(task['id'], task['claim']); recovered += 1
        else:
            ledger.interrupt(task['id'], task['claim'], 'worker_lost_before_ack'); interrupted += 1
    return {'recovered': recovered, 'interrupted': interrupted}


def _terminate_group(child):
    # Even a reaped group leader can have descendants; kill the entire new session
    # group before acknowledging/releasing ownership. Worker helpers must not detach.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=5)


def _worker_command(ledger, database, task):
    return [sys.executable, '-m', 'autofusedebugger.fuzz.scheduling.worker',
            str(ledger.path), str(database), task['id'], task['claim']]


def drain(ledger, database, *, timeout=120, _owned_fd=None, _ready=None):
    """Drain pending tasks; stale claims are recovered, never auto-retried."""
    if not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    database = Path(database).expanduser().resolve()
    if database == ledger.path:
        raise ValueError('task and result databases must differ')
    from contextlib import nullcontext
    ownership = own_queue(ledger) if _owned_fd is None else nullcontext(_owned_fd)
    with ownership as fd:
        _bind_results(ledger, database)
        for task in _tasks(ledger):
            if task['state'] in {'pending', 'running', 'stop_requested'}:
                _validate_task(task)
        from .budgets import Budget, read_policy, violation
        budget = read_policy(ledger) or Budget(compile_threads=max(
            [json.loads(t['execution'])['compile_threads'] for t in _tasks(ledger)
             if t['state'] == 'pending'] or [1]))
        budget.admit(ledger.path.parent)
        budget.admit(database.parent)
        for candidate in _tasks(ledger):
            if candidate['state'] == 'pending':
                budget.admit(ledger.path.parent, candidate)
        summary = dict(_reconcile(ledger, database), completed=0)
        if _ready is not None:
            _ready()
        while True:
            task = ledger.claim()
            if task is None:
                from .campaign import refill
                if refill(ledger):continue
                return summary
            child = None
            scratch = None
            reason = 'worker_exited_without_result'
            try:
                _validate_task(task)
                budget.admit(ledger.path.parent, task)
                budget.admit(database.parent)
                database.parent.mkdir(parents=True, exist_ok=True)
                scratch = tempfile.TemporaryDirectory(prefix='.fuzz-worker-', dir=ledger.path.parent)
                env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                env.update(TMPDIR=scratch.name, TORCHINDUCTOR_CACHE_DIR=str(Path(scratch.name)/'inductor'),
                           TRITON_CACHE_DIR=str(Path(scratch.name)/'triton'),
                           TORCHINDUCTOR_COMPILE_THREADS=str(json.loads(task['execution'])['compile_threads']))
                child = subprocess.Popen(_worker_command(ledger, database, task),
                    start_new_session=True, pass_fds=(fd,), env=env,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                deadline = time.monotonic() + timeout
                while child.poll() is None:
                    exceeded = violation(budget, child.pid, scratch.name, database.parent)
                    if exceeded:
                        reason = exceeded
                        break
                    if ledger.get(task['id'])['state'] == 'stop_requested':
                        reason = 'user_stop'
                        break
                    if time.monotonic() >= deadline:
                        reason = 'timeout'
                        break
                    time.sleep(.05)
                if child.poll() is not None:
                    reason = f'worker_exit:{child.returncode}:without_result'
            except KeyboardInterrupt:
                reason = 'supervisor_interrupted'
                raise
            except Exception as exc:
                reason = f'supervisor_error:{type(exc).__name__}'
                raise
            finally:
                if child is not None:
                    _terminate_group(child)
                if scratch is not None:
                    scratch.cleanup()
                if _has_result(database, task):
                    ledger.complete(task['id'], task['claim'])
                    summary['completed'] += 1
                else:
                    ledger.interrupt(task['id'], task['claim'], reason)
                    summary['interrupted'] += 1


def resume(ledger, database, identity):
    """Reconcile a saved result before allowing an interrupted task to retry."""
    database = Path(database).expanduser().resolve()
    with own_queue(ledger):
        _bind_results(ledger, database)
        _reconcile(ledger, database)
        if ledger.get(identity)['state'] == 'completed':
            return {'task_id': identity, 'state': 'completed', 'already_saved': True}
        ledger.resume(identity)
        return {'task_id': identity, 'state': 'pending', 'execution': 'not_started'}
