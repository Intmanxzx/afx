"""Bounded concurrent workers with process-group cleanup and per-device exclusion."""
from concurrent.futures import ThreadPoolExecutor
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
    from ..contracts import Seed
    if 'seed' in task and not Seed.parse(task['seed']).current:
        raise ValueError('retired template task; cannot resume with NNSmith')
    if type(config.compile_threads) is not int or not 1 <= config.compile_threads <= (os.cpu_count() or 1):
        raise ValueError('compile thread budget exceeds available CPU count')


def _has_result(database, task):
    from ..storage.repository import _connect
    from ..storage.transfer import _record
    from ..generation import materialize, workload_id, oracle_id
    import sqlite3
    if not json.loads(task['execution']).get('save_results', True):
        return False
    if not database.exists():
        return False
    with closing(_connect(database)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('BEGIN')
        row = conn.execute('SELECT * FROM attempts WHERE id=?', (task['attempt_id'],)).fetchone()
        if row is None:
            return False
        contexts, _ = _record(conn, row)
        worker_failed=row['oracle_id']=='worker-failed/1' and row['error_stage']=='worker' and row['status']=='error'
        generation_failed=row['oracle_id']=='generation-failed/1' and row['error_stage']=='generation' and row['status']=='error'
        if worker_failed:
            from ..contracts import fingerprint
            expected_workload=fingerprint({'worker_failed':task['seed']})
            expected_oracles={'worker-failed/1'}
        elif generation_failed:
            from ..contracts import fingerprint
            expected_workload=fingerprint({'generation_failed':task['seed']})
            expected_oracles={'generation-failed/1'}
        elif contexts['execution'].get('case_spec'):
            from ..case_spec import decode
            from ..generation import materialize_spec
            spec=decode(contexts['execution']['case_spec'])
            if str(spec.seed)!=task['seed']:
                raise ValueError('saved case seed conflicts with task')
            bundle=materialize_spec(spec)
        else:
            bundle = materialize(task['seed'])
        if not (generation_failed or worker_failed):
            expected_workload=workload_id(bundle)
            expected_oracles={oracle_id(bundle),'torch-eager-cpu/1'}
        saved_execution = dict(contexts['execution'])
        task_execution = json.loads(task['execution'])
        saved_execution.setdefault('save_results', True)
        task_execution.setdefault('save_results', True)
        if (row['seed'] != task['seed'] or row['workload_id'] != expected_workload or
                row['oracle_id'] not in expected_oracles or
                row['backend'] != json.loads(task['execution'])['backend'] or
                {k:v for k,v in saved_execution.items() if k not in {'telemetry','diagnostics','case_spec'}} != task_execution or
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
    return [sys.executable, '-m', 'autofusex.fuzz.scheduling.worker',
            str(ledger.path), str(database), task['id'], task['claim']]


def _save_worker_failure(ledger, database, task, reason, log):
    """Persist a terminated attempt, but keep its task explicitly resumable."""
    if not json.loads(task['execution']).get('save_results', True):
        return
    from ..storage import ResultRepository
    from ..environment import collect
    from ..contracts import fingerprint
    # No graph generation in finalization. An unassigned worker failure has no
    # Shape and uses its own sentinel, never masquerades as generation failure.
    workload = fingerprint({'worker_failed': task['seed']})
    execution = json.loads(task['execution'])
    execution['diagnostics'] = json.dumps({'version': 1, 'error': (reason+'\n'+log)[-1600:], 'shapes': {}})
    result = {'seed': task['seed'], 'workload_id': workload,
              'oracle_id': 'worker-failed/1', 'backend': execution['backend'],
              'execution': execution, 'evaluation': json.loads(task['evaluation']),
              'status': 'error', 'error_stage': 'worker', 'error_summary': reason,
              'shape_results': []}
    ResultRepository(database).save(result, environment=collect(ledger.path.parent.parent),
                                    attempt_id=task['attempt_id'])


def drain(ledger, database, *, timeout=120, _owned_fd=None, _ready=None):
    """Drain pending tasks; stale claims are recovered, never auto-retried."""
    if not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    staged_tasks = [json.loads(t['execution']).get('scheduling') for t in _tasks(ledger) if t['state']=='pending']
    if not staged_tasks:
        with ledger._connection() as conn:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='campaign'").fetchone():
                row=conn.execute('SELECT payload FROM campaign WHERE id=1').fetchone()
                if row:staged_tasks=[c.get('scheduling') for c in json.loads(row[0])['configs']]
    if 'staged' in staged_tasks:
        if any(x != 'staged' for x in staged_tasks):
            raise ValueError('cannot mix staged and legacy queued tasks; drain or stop existing queue first')
        from .staged import drain as drain_staged
        return drain_staged(ledger,database,timeout=timeout,_owned_fd=_owned_fd,_ready=_ready)
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
        from ..storage.lifecycle import append, stamp
        active = {}
        finalizing = {}
        finalizer = ThreadPoolExecutor(max_workers=budget.cpu_workers, thread_name_prefix="fuzz-finalize")
        def finish(state, reason):
            task=state['task']; child=state['child']
            if child is not None: _terminate_group(child)
            log=state['tail'].finish() if state['tail'] is not None else ''
            state['scratch'].cleanup()
            ephemeral_done=(not json.loads(task['execution']).get('save_results',True)
                            and reason=='worker_exit:0:without_result')
            if ephemeral_done or _has_result(database,task):
                ledger.complete(task['id'],task['claim']); return 'completed'
            else:
                if reason=='timeout' or reason.startswith('worker_exit:'):
                    _save_worker_failure(ledger,database,task,reason,log)
                ledger.interrupt(task['id'],task['claim'],reason); return 'interrupted'

        try:
            while True:
                for identity, (state, future) in list(finalizing.items()):
                    if future.done():
                        summary[future.result()] += 1
                        del finalizing[identity]
                from .campaign import refill
                for _ in range(budget.cpu_workers):
                    if not refill(ledger, target_pending=budget.cpu_workers): break
                busy={json.loads(state['task']['execution'])['device'] for state in list(active.values()) + [s for s, _ in finalizing.values()]}
                from .placement import candidates, claim_on, admission
                for candidate in _tasks(ledger):
                    if len(active) + len(finalizing)>=budget.cpu_workers: break
                    if candidate['state']!='pending': continue
                    execution=json.loads(candidate['execution'])
                    device=next((d for d in candidates(ledger, candidate) if d=='cpu' or d not in busy), None)
                    if device is None: continue
                    _validate_task(candidate)
                    budget.admit(ledger.path.parent,candidate); budget.admit(database.parent)
                    scratch=tempfile.TemporaryDirectory(prefix='.fuzz-worker-',dir=ledger.path.parent)
                    task=claim_on(ledger,candidate,device)
                    if task is None:
                        scratch.cleanup(); continue
                    state={'task':task,'scratch':scratch,'child':None,'tail':None,'deadline':time.monotonic()+timeout}
                    active[task['id']]=state
                    database.parent.mkdir(parents=True,exist_ok=True)
                    if execution.get('save_results', True):
                        observed = admission(ledger, task['id'])
                        if observed is not None:
                            append(database, task['attempt_id'], 'admitted', observed)
                            append(database, task['attempt_id'], 'ready', observed)
                        append(database, task['attempt_id'], 'dispatched')
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                    env.update(TMPDIR=scratch.name,TORCHINDUCTOR_CACHE_DIR=str(Path(scratch.name)/'inductor'),
                               TRITON_CACHE_DIR=str(Path(scratch.name)/'triton'),
                               TORCHINDUCTOR_COMPILE_THREADS=str(execution['compile_threads']))
                    state['child']=subprocess.Popen(_worker_command(ledger,database,task),start_new_session=True,
                        pass_fds=(fd,),env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                    from ..core_logs import ProcessTail
                    state['tail']=ProcessTail(state['child'].stdout)
                    busy.add(device)
                if not active and not finalizing:
                    if refill(ledger,target_pending=budget.cpu_workers): continue
                    return summary
                # Budgets cover the complete active pool, not one budget per worker.
                from .budgets import group_rss, scratch_size
                pool_reason=None
                sampled = list(active.values()) + [s for s, _ in finalizing.values()]
                if budget.memory_budget_bytes and sum(group_rss(s['child'].pid) for s in sampled)>budget.memory_budget_bytes:
                    pool_reason='memory_budget_exceeded'
                if budget.cache_budget_bytes and sum(scratch_size(s['scratch'].name) for s in active.values())>budget.cache_budget_bytes:
                    pool_reason='scratch_budget_exceeded'
                for identity,state in list(active.items()):
                    child=state['child']; task=state['task']; reason=pool_reason
                    if not reason: reason=violation(budget,child.pid,state['scratch'].name,database.parent)
                    if not reason and ledger.get(identity)['state']=='stop_requested': reason='user_stop'
                    if not reason and time.monotonic()>=state['deadline']: reason='timeout'
                    if not reason and child.poll() is not None: reason=f'worker_exit:{child.returncode}:without_result'
                    if reason:
                        active.pop(identity)
                        finalizing[identity]=(state,finalizer.submit(finish,state,reason))
                time.sleep(.05)
        except BaseException as exc:
            reason='supervisor_interrupted' if isinstance(exc,KeyboardInterrupt) else f'supervisor_error:{type(exc).__name__}'
            # Reap every worker even if one acknowledgement fails.
            for identity in list(active):
                try: finish(active.pop(identity),reason)
                except Exception:
                    pass
            raise
        finally:
            finalizer.shutdown(wait=True)


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
