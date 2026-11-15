"""Finite detached queue drain, not a permanent API/service or restart daemon.

POSIX inherited queue lock + bounded socket handshake. Startup never claims tasks
before the submitting process authorizes execution. Acknowledgement means started,
not that every task will finish. Status metadata is advisory, never a PID kill key.
"""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import uuid

from .ledger import TaskLedger
from .ownership import own_queue
from .supervisor import _bind_results, _tasks, _validate_task, drain


def _status_path(ledger):
    return ledger.path.with_name(ledger.path.name+'.supervisor.json')


def _write_status(ledger, value):
    value = dict(value, updated_at=datetime.now(timezone.utc).isoformat())
    fd, name = tempfile.mkstemp(prefix='.fuzz-supervisor-', dir=ledger.path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, _status_path(ledger))
    finally:
        Path(name).unlink(missing_ok=True)


def status(ledger):
    path = _status_path(ledger)
    value = json.loads(path.read_text()) if path.exists() else {'state': 'not_started'}
    if os.name != 'posix':
        return dict(value, ownership='unavailable')
    import fcntl
    if not ledger.path.exists():
        return dict(value, ownership='unowned')
    with ledger.path.open('rb') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(value, ownership='owned_by_supervisor_or_worker')
    if value['state'] in {'starting', 'running'}:
        value['state'] = 'exited_without_final_status'
    return dict(value, ownership='unowned')


def _send(sock, value):
    sock.sendall(json.dumps(value).encode()+b'\n')


def _receive(sock):
    data = bytearray()
    while len(data) < 8192:
        byte = sock.recv(1)
        if not byte:
            raise ValueError('supervisor handshake closed before acknowledgement')
        if byte == b'\n':
            return json.loads(data)
        data.extend(byte)
    raise ValueError('supervisor handshake exceeded bound')


def _command(ledger, database, timeout, fd, connection, run_id):
    return [sys.executable, '-m', 'autofusex.fuzz.scheduling.hosting',
            str(ledger.path), str(database), str(timeout), str(fd), str(connection), run_id]


def launch(ledger, database, specifications, *, timeout=120, foreground=False, startup_timeout=15, budget=None, campaign=None, devices=None, preserve_campaign=False):
    """Bind and enqueue under ownership before handing lock to a child supervisor."""
    if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(startup_timeout) or startup_timeout <= 0:
        raise ValueError('timeouts must be finite and positive')
    database = Path(database).expanduser().resolve()
    if database == ledger.path:
        raise ValueError('task and result databases must differ')
    from dataclasses import asdict
    from ..contracts import canonical_json
    prepared = []
    for specification in specifications:
        if len(prepared) >= 10000:
            raise ValueError('batch exceeds 10000 tasks')
        seed, execution, evaluation, identity = specification
        _validate_task({'execution': canonical_json(asdict(execution)),
                        'evaluation': canonical_json(asdict(evaluation))})
        prepared.append(specification)
    from .budgets import Budget, read_policy, write_policy
    with own_queue(ledger) as fd:
        effective_budget = budget or read_policy(ledger) or Budget()
        effective_budget.admit(ledger.path.parent)
        effective_budget.admit(database.parent)
        for seed, execution, evaluation, identity in prepared:
            effective_budget.admit(ledger.path.parent, {'execution': canonical_json(asdict(execution))})
        _bind_results(ledger, database)
        # Validate existing queued work before accepting any additional tasks.
        for task in _tasks(ledger):
            if task['state'] in {'pending', 'running', 'stop_requested'}:
                _validate_task(task)
                effective_budget.admit(ledger.path.parent, task)
        write_policy(ledger, effective_budget)
        from .campaign import initialize
        if campaign is not None:
            initialize(ledger, *campaign)
        elif not preserve_campaign:
            # A finite explicit-seed run must not resume a prior infinite feeder.
            with ledger._connection() as conn:
                if conn.execute("SELECT 1 FROM sqlite_master WHERE name='campaign'").fetchone():
                    conn.execute('UPDATE campaign SET stopped=1')
        ids = ledger.enqueue_many(prepared)
        if devices:
            from .placement import initialize as init_placement
            with ledger._connection() as conn:
                init_placement(conn)
                for identity,spec in zip(ids,prepared):
                    if spec[1].device!='cpu':
                        conn.execute('INSERT OR REPLACE INTO task_devices VALUES (?,?)',(identity,json.dumps(devices)))
        if foreground:
            return dict(drain(ledger, database, timeout=timeout, _owned_fd=fd),
                        enqueued=ids, queue=str(ledger.path), mode='bounded_foreground',
                        budget=asdict(effective_budget), active_workers=0, worker_limit=effective_budget.cpu_workers, enforcement='sampled_not_hard_isolation')
        run_id = str(uuid.uuid4())
        _write_status(ledger, {'run_id': run_id, 'state': 'starting'})
        parent, child_socket = socket.socketpair()
        child = None
        accepted = False
        try:
            parent.settimeout(startup_timeout)
            child = subprocess.Popen(_command(ledger, database, timeout, fd, child_socket.fileno(), run_id),
                pass_fds=(fd, child_socket.fileno()), start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            child_socket.close()
            message = _receive(parent)
            if message.get('state') != 'ready' or message.get('run_id') != run_id:
                raise ValueError('supervisor did not become ready')
            _send(parent, {'action': 'start', 'run_id': run_id})
            message = _receive(parent)
            if message.get('state') != 'running' or message.get('run_id') != run_id:
                raise ValueError('supervisor failed before startup acknowledgement')
            accepted = True
            # Reap in long-lived embedding callers; ordinary CLI exits immediately.
            threading.Thread(target=child.wait, daemon=True).start()
            return {'run_id': run_id, 'state': 'running', 'supervisor_pid': child.pid,
                    'enqueued': ids, 'queue': str(ledger.path), 'mode': 'bounded_background',
                    'budget': asdict(effective_budget), 'worker_limit': effective_budget.cpu_workers, 'enforcement': 'sampled_not_hard_isolation'}
        finally:
            parent.close(); child_socket.close()
            if not accepted:
                if child is not None:
                    # Before start authorization no worker exists. After authorization
                    # child handles SIGTERM and cleans its separate worker group.
                    child.terminate()
                    try:
                        child.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                        # Surviving workers retain queue lock: never force takeover.
                _write_status(ledger, {'run_id': run_id, 'state': 'startup_failed',
                                      'reason': 'startup_not_acknowledged; inspect tasks before retry'})


def _child(queue, database, timeout, fd, connection, run_id):
    ledger = TaskLedger(queue)
    fd, connection = int(fd), int(connection)
    status_value = {'run_id': run_id, 'pid': os.getpid(), 'state': 'starting'}
    # Supervisor termination follows the same cleanup path as interactive interrupt.
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    sock = socket.socket(fileno=connection)
    try:
        if os.fstat(fd).st_ino != ledger.path.stat().st_ino or os.fstat(fd).st_dev != ledger.path.stat().st_dev:
            raise ValueError('inherited ownership does not match queue')
        sock.settimeout(15)
        _send(sock, {'state': 'ready', 'run_id': run_id})
        if _receive(sock) != {'action': 'start', 'run_id': run_id}:
            raise ValueError('execution was not authorized')
        def ready():
            status_value['state'] = 'running'
            _write_status(ledger, status_value)
            _send(sock, {'state': 'running', 'run_id': run_id})
            sock.close()
        summary = drain(ledger, database, timeout=float(timeout), _owned_fd=fd, _ready=ready)
        _write_status(ledger, dict(status_value, state='finished', summary=summary))
        return 0
    except KeyboardInterrupt:
        _write_status(ledger, dict(status_value, state='interrupted', reason='supervisor_signal'))
        return 130
    except Exception as exc:
        _write_status(ledger, dict(status_value, state='error', reason=type(exc).__name__))
        return 1
    finally:
        sock.close()
        os.close(fd)


if __name__ == '__main__':
    raise SystemExit(_child(*sys.argv[1:]))
