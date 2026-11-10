import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from autofusedebugger.fuzz.contracts import ExecutionConfig, EvaluationConfig
from autofusedebugger.fuzz.scheduling.ledger import TaskLedger
from autofusedebugger.fuzz.scheduling.ownership import own_queue
from autofusedebugger.fuzz.scheduling import hosting
from autofusedebugger.fuzz.storage import ResultRepository

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='POSIX process hosting')
EX, EV = ExecutionConfig('cpu-eager','cpu',1), EvaluationConfig()


def wait_finished(ledger, timeout=30):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        value = hosting.status(ledger)
        if value['ownership'] == 'unowned' and value['state'] not in {'starting', 'running'}:
            return value
        time.sleep(.05)
    raise AssertionError('background supervisor did not exit')


def test_detached_run_finishes_after_launcher_exits(tmp_path):
    pytest.importorskip("torch")
    output = subprocess.run([sys.executable, '-m', 'autofusedebugger.fuzz', 'run', '42',
                             '--data-dir', str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert output.returncode == 0, output.stderr
    response = json.loads(output.stdout)
    assert response['mode'] == 'serial_background' and response['state'] == 'running'
    ledger = TaskLedger(response['queue'])
    final = wait_finished(ledger)
    assert final['state'] == 'finished', final
    assert final['summary']['completed'] == 2  # mandatory CPU + AF-MLIR
    task = ledger.get(response['enqueued'][0])
    assert task['state'] == 'completed'
    result = ResultRepository(tmp_path/'results/results.sqlite').detail(task['attempt_id'])
    assert result['status'] == 'ok' and len(result['shape_results']) == 3


def test_failed_spawn_keeps_batch_pending(tmp_path, monkeypatch):
    ledger = TaskLedger(tmp_path/'queue')
    monkeypatch.setattr(hosting, '_command', lambda *a: ['/nonexistent/supervisor'])
    with pytest.raises(OSError):
        hosting.launch(ledger, tmp_path/'results', [('42',EX,EV,None)])
    assert ledger.page()['items'][0]['state'] == 'pending'
    assert hosting.status(ledger)['state'] == 'startup_failed'
    with own_queue(ledger):
        pass


def test_startup_timeout_does_not_claim_and_releases_lock(tmp_path, monkeypatch):
    ledger = TaskLedger(tmp_path/'queue')
    monkeypatch.setattr(hosting, '_command', lambda *a: [sys.executable,'-c','import time;time.sleep(60)'])
    with pytest.raises(TimeoutError):
        hosting.launch(ledger, tmp_path/'results', [('42',EX,EV,None)], startup_timeout=.2)
    assert ledger.page()['items'][0]['state'] == 'pending'
    with own_queue(ledger):
        pass


def test_wrong_binding_or_owned_queue_does_not_enqueue(tmp_path):
    ledger = TaskLedger(tmp_path/'queue')
    hosting.launch(ledger, tmp_path/'results', [], foreground=True)
    with pytest.raises(ValueError, match='different result database'):
        hosting.launch(ledger, tmp_path/'wrong', [('42',EX,EV,None)])
    assert ledger.page()['items'] == []
    with own_queue(ledger):
        with pytest.raises(ValueError, match='still owned'):
            hosting.launch(ledger, tmp_path/'results', [('42',EX,EV,None)])
    assert ledger.page()['items'] == []


def test_stale_metadata_never_claims_process_liveness(tmp_path):
    ledger = TaskLedger(tmp_path/'queue'); ledger.page()
    hosting._write_status(ledger, {'run_id':'old', 'state':'running', 'pid': os.getpid()})
    assert hosting.status(ledger)['state'] == 'exited_without_final_status'
    assert hosting.status(ledger)['ownership'] == 'unowned'


def test_child_never_claims_without_start_authorization(tmp_path):
    import socket
    ledger = TaskLedger(tmp_path/'queue')
    identity = ledger.enqueue('42',EX,EV)
    with own_queue(ledger) as fd:
        parent, child_socket = socket.socketpair()
        child = subprocess.Popen(hosting._command(ledger, tmp_path/'results',120,fd,child_socket.fileno(),'test'),
                                 pass_fds=(fd,child_socket.fileno()), start_new_session=True)
        child_socket.close()
        try:
            parent.settimeout(5)
            assert hosting._receive(parent)['state'] == 'ready'
            parent.close()  # submitting process disappears before authorizing execution
            assert child.wait(timeout=10) == 1
        finally:
            parent.close()
            if child.poll() is None:
                child.kill();child.wait()
    assert ledger.get(identity)['state'] == 'pending'


def test_background_sigterm_cleans_active_worker(tmp_path, monkeypatch):
    ledger = TaskLedger(tmp_path/'queue')
    pidfile = tmp_path/'worker.pid'
    # Injection exists only in this test's child process, not a production CLI hook.
    code = '''
import sys
from autofusedebugger.fuzz.scheduling import hosting,supervisor
worker="import pathlib,os,sys,time;pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(60)"
supervisor._worker_command=lambda *a:[sys.executable,'-c',worker,sys.argv[-1]]
sys.exit(hosting._child(*sys.argv[1:-1]))
'''
    def command(q, db, timeout, fd, sock, run_id):
        return [sys.executable,'-c',code,str(q.path),str(db),str(timeout),str(fd),str(sock),run_id,str(pidfile)]
    monkeypatch.setattr(hosting, '_command', command)
    started = hosting.launch(ledger,tmp_path/'results',[('42',EX,EV,None)])
    try:
        deadline = time.monotonic()+10
        while not pidfile.exists() and time.monotonic()<deadline:
            time.sleep(.05)
        assert pidfile.exists()
        worker_pid = int(pidfile.read_text())
        os.kill(started['supervisor_pid'],signal.SIGTERM)
        assert wait_finished(ledger)['state'] == 'interrupted'
        task = ledger.get(started['enqueued'][0])
        assert task['state'] == 'interrupted' and task['reason'] == 'supervisor_interrupted'
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid,0)
    finally:
        try:
            os.kill(started['supervisor_pid'],signal.SIGTERM)
        except ProcessLookupError:
            pass


def test_background_rejects_npu_before_enqueue(tmp_path):
    ledger = TaskLedger(tmp_path/'queue')
    with pytest.raises(ValueError, match='strict NPU performance'):
        hosting.launch(ledger,tmp_path/'results',[('42',ExecutionConfig('ascendc','npu:0'),EvaluationConfig(performance=True),None)])
    assert ledger.page()['items'] == []


def test_count_generates_self_contained_v2_seeds_without_execution(monkeypatch, tmp_path, capsys):
    import json
    from autofusedebugger.fuzz.cli import main
    from autofusedebugger.fuzz.contracts import Seed
    def launch(ledger, database, specs, **kwargs):
        assert list(specs) == []
        configs, evaluation, count = kwargs['campaign']
        return {'count': count, 'backends': [c.backend for c in configs]}
    monkeypatch.setattr(hosting, 'launch', launch)
    assert main(['run', '--data-dir', str(tmp_path), '--count', '4']) == 0
    value = json.loads(capsys.readouterr().out)
    assert value == {'count': 4, 'backends': ['cpu', 'af-mlir']}
    assert main(['run', '42', '--data-dir', str(tmp_path), '--count', '4']) == 2


def test_background_infinite_campaign_stops_without_refilling(tmp_path):
    from autofusedebugger.fuzz.scheduling.campaign import stop
    ledger = TaskLedger(tmp_path/'queue')
    hosting.launch(ledger, tmp_path/'results', [], campaign=([EX], EV, None))
    deadline = time.monotonic() + 20
    while not ledger.page()['items'] and time.monotonic() < deadline:
        time.sleep(.05)
    assert ledger.page()['items']
    stop(ledger)
    assert wait_finished(ledger)['state'] == 'finished'
    count = len(ledger.page()['items'])
    time.sleep(.1)
    assert len(ledger.page()['items']) == count
    assert all(t['state'] not in {'running', 'pending', 'stop_requested'} for t in ledger.page()['items'])
