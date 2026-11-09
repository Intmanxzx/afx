from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
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
from autofusedebugger.fuzz.scheduling import supervisor
from autofusedebugger.fuzz.storage import ResultRepository

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='managed execution currently POSIX only')
EX = ExecutionConfig('cpu-eager', 'cpu', 1)
EV = EvaluationConfig()


def saved_result(task):
    from autofusedebugger.fuzz.generation import materialize, workload_id, oracle_id
    bundle = materialize(task['seed'])
    return {'seed': task['seed'], 'workload_id': workload_id(bundle), 'oracle_id': oracle_id(bundle),
            'backend': 'cpu-eager', 'status': 'ok', 'execution': json.loads(task['execution']),
            'evaluation': json.loads(task['evaluation']), 'shape_results': []}


def setup(tmp_path):
    ledger = TaskLedger(tmp_path/'tasks.sqlite')
    identity = ledger.enqueue('42', EX, EV)
    return ledger, identity, tmp_path/'results.sqlite'


def test_real_cpu_worker_uses_same_bundle_and_persists(tmp_path):
    ledger, identity, database = setup(tmp_path)
    summary = supervisor.drain(ledger, database)
    assert summary['completed'] == 1 and summary['interrupted'] == 0
    task = ledger.get(identity)
    assert task['state'] == 'completed'
    result = ResultRepository(database).detail(task['attempt_id'])
    assert result['status'] == 'ok' and len(result['shape_results']) == 3
    assert result['workload_id'] == '610516f468114ce745c01ab2d1cb78129c87e4395d60e19e034aee9a714e9daa'
    assert supervisor.drain(ledger, database)['completed'] == 0
    assert len(ResultRepository(database).list()['items']) == 1


def test_result_saved_before_ack_reconciles_without_rerun(tmp_path, monkeypatch):
    ledger, identity, database = setup(tmp_path)
    claim = ledger.claim()
    ResultRepository(database).save(saved_result(claim), attempt_id=claim['attempt_id'])
    monkeypatch.setattr(supervisor, '_worker_command', lambda *args: pytest.fail('must not rerun'))
    assert supervisor.drain(ledger, database)['recovered'] == 1
    assert ledger.get(identity)['state'] == 'completed'
    assert supervisor.resume(ledger, database, identity)['already_saved']


def test_unsaved_claim_interrupted_and_requires_explicit_resume(tmp_path):
    ledger, identity, database = setup(tmp_path)
    first = ledger.claim()
    assert supervisor.drain(ledger, database)['interrupted'] == 1
    assert ledger.get(identity)['state'] == 'interrupted'
    assert supervisor.resume(ledger, database, identity)['state'] == 'pending'
    assert ledger.get(identity)['attempt_id'] is None
    assert ledger.claim()['attempt_id'] != first['attempt_id']


def test_conflicting_saved_attempt_not_acknowledged(tmp_path):
    ledger, identity, database = setup(tmp_path)
    claim = ledger.claim()
    value = saved_result(claim); value['workload_id'] = 'a'*64
    ResultRepository(database).save(value, attempt_id=claim['attempt_id'])
    with pytest.raises(ValueError, match='conflicts'):
        supervisor.drain(ledger, database)
    assert ledger.get(identity)['state'] == 'running'


def test_result_binding_prevents_recovery_against_wrong_db(tmp_path):
    ledger, identity, database = setup(tmp_path)
    ledger.stop(identity)
    supervisor.drain(ledger, database)
    with pytest.raises(ValueError, match='different result database'):
        supervisor.resume(ledger, tmp_path/'other.sqlite', identity)


def test_queue_ownership_rejects_second_supervisor(tmp_path):
    ledger, _, database = setup(tmp_path)
    with own_queue(ledger):
        with pytest.raises(ValueError, match='still owned'):
            supervisor.drain(ledger, database)


def test_timeout_terminates_process_group(tmp_path, monkeypatch):
    ledger, identity, database = setup(tmp_path)
    pidfile = tmp_path/'child.pid'
    code = '''
import pathlib,subprocess,sys,time
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
pathlib.Path(sys.argv[1]).write_text(str(p.pid))
time.sleep(120)
'''
    monkeypatch.setattr(supervisor, '_worker_command', lambda *a: [sys.executable, '-c', code, str(pidfile)])
    assert supervisor.drain(ledger, database, timeout=1)['interrupted'] == 1
    assert ledger.get(identity)['state'] == 'interrupted'
    pid = int(pidfile.read_text())
    # A killed orphan can remain a zombie until init reaps it, but cannot execute.
    deadline = time.monotonic()+5
    while Path(f'/proc/{pid}/stat').exists() and time.monotonic() < deadline:
        if Path(f'/proc/{pid}/stat').read_text().split()[2] == 'Z':
            break
        time.sleep(.05)
    assert not Path(f'/proc/{pid}/stat').exists() or Path(f'/proc/{pid}/stat').read_text().split()[2] == 'Z'


def test_stop_running_worker_preserves_explicit_resume(tmp_path, monkeypatch):
    ledger, identity, database = setup(tmp_path)
    monkeypatch.setattr(supervisor, '_worker_command', lambda *a: [sys.executable, '-c', 'import time;time.sleep(120)'])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(supervisor.drain, ledger, database, timeout=10)
        deadline = time.monotonic()+5
        while ledger.get(identity)['state'] != 'running' and time.monotonic() < deadline:
            time.sleep(.02)
        ledger.stop(identity)
        assert future.result(timeout=10)['interrupted'] == 1
    assert ledger.get(identity)['state'] == 'interrupted'


def test_worker_spawn_failure_releases_claim(tmp_path, monkeypatch):
    ledger, identity, database = setup(tmp_path)
    monkeypatch.setattr(supervisor, '_worker_command', lambda *a: ['/nonexistent/fuzz-worker'])
    with pytest.raises(OSError):
        supervisor.drain(ledger, database)
    assert ledger.get(identity)['state'] == 'interrupted'


def test_npu_and_invalid_timeout_never_claim(tmp_path):
    ledger = TaskLedger(tmp_path/'tasks.sqlite')
    identity = ledger.enqueue('42', ExecutionConfig('ascendc','npu:0'), EvaluationConfig(performance=True))
    with pytest.raises(ValueError, match='strict NPU performance'):
        supervisor.drain(ledger, tmp_path/'result.sqlite')
    assert ledger.get(identity)['state'] == 'pending'
    for timeout in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='timeout'):
            supervisor.drain(ledger, tmp_path/'result.sqlite', timeout=timeout)


def test_inherited_lock_survives_supervisor_death(tmp_path):
    ledger, _, _ = setup(tmp_path)
    pidfile = tmp_path/'worker.pid'
    code = '''
import os,pathlib,subprocess,sys,time
from autofusedebugger.fuzz.scheduling.ledger import TaskLedger
from autofusedebugger.fuzz.scheduling.ownership import own_queue
with own_queue(TaskLedger(sys.argv[1])) as fd:
 p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],pass_fds=(fd,),start_new_session=True)
 pathlib.Path(sys.argv[2]).write_text(str(p.pid))
 os._exit(81)
'''
    parent = subprocess.run([sys.executable, '-c', code, str(ledger.path), str(pidfile)])
    assert parent.returncode == 81
    pid = int(pidfile.read_text())
    try:
        with pytest.raises(ValueError, match='still owned'):
            with own_queue(ledger):
                pytest.fail('inherited lock lost')
    finally:
        os.killpg(pid, signal.SIGKILL)
    deadline = time.monotonic()+5
    while True:
        try:
            with own_queue(ledger):
                break
        except ValueError:
            if time.monotonic() > deadline:
                raise
            time.sleep(.05)


def test_cli_rejects_npu_and_exposes_lightweight_status(tmp_path, capsys):
    from autofusedebugger.fuzz.cli import main
    with pytest.raises(SystemExit):
        main(['run', '42', '--backend', 'invalid'])
    ledger, identity, _ = setup(tmp_path)
    assert main(['status', identity, '--queue', str(ledger.path)]) == 0
    assert main(['stop', identity, '--queue', str(ledger.path)]) == 0
    assert ledger.get(identity)['state'] == 'cancelled'


def test_worker_commits_then_crashes_before_parent_ack(tmp_path, monkeypatch):
    ledger, identity, database = setup(tmp_path)
    code = '''
import os,sys
from autofusedebugger.fuzz.scheduling.worker import run
run(*sys.argv[1:])
os._exit(82)
'''
    monkeypatch.setattr(supervisor, '_worker_command', lambda l, d, t:
                        [sys.executable, '-c', code, str(l.path), str(d), t['id'], t['claim']])
    assert supervisor.drain(ledger, database)['completed'] == 1
    assert ledger.get(identity)['state'] == 'completed'
    assert len(ResultRepository(database).list()['items']) == 1


def test_task_pagination_and_validation(tmp_path):
    ledger, first, _ = setup(tmp_path)
    second = ledger.enqueue('43', EX, EV)
    page = ledger.page(limit=1)
    assert page['items'][0]['id'] == first
    assert ledger.page(limit=1, after=page['next_cursor'])['items'][0]['id'] == second
    assert ledger.page(limit=2)['next_cursor'] is None
    for kwargs in ({'limit': 201}, {'after': -1}):
        with pytest.raises(ValueError):
            ledger.page(**kwargs)


def test_keyboard_interrupt_cleans_worker_and_leaves_explicit_resume(tmp_path):
    ledger, identity, database = setup(tmp_path)
    pidfile = tmp_path/'worker.pid'
    code = '''
import pathlib,sys
from autofusedebugger.fuzz.scheduling import supervisor
from autofusedebugger.fuzz.scheduling.ledger import TaskLedger
worker="import os,pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(120)"
supervisor._worker_command=lambda *a:[sys.executable,'-c',worker,sys.argv[3]]
try:
 supervisor.drain(TaskLedger(sys.argv[1]),sys.argv[2])
except KeyboardInterrupt:
 sys.exit(130)
'''
    child = subprocess.Popen([sys.executable, '-c', code, str(ledger.path), str(database), str(pidfile)])
    try:
        deadline = time.monotonic()+5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert pidfile.exists()
        pid = int(pidfile.read_text())
        child.send_signal(signal.SIGINT)
        assert child.wait(timeout=10) == 130
        assert ledger.get(identity)['state'] == 'interrupted'
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        with own_queue(ledger):
            pass
    finally:
        if child.poll() is None:
            child.kill(); child.wait()


def test_management_commands_do_not_load_model_dependencies(tmp_path):
    code = '''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','numpy','nnsmith','torch_npu'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
from autofusedebugger.fuzz.cli import main
assert main(['status','--data-dir',sys.argv[1]]) == 0
'''
    assert subprocess.run([sys.executable, '-c', code, str(tmp_path)]).returncode == 0
