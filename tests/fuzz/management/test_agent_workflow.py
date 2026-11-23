"""Agent workflow through the real CLI/socket with isolated, saved result fixtures."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pytest

from autofusex.client import cli
from autofusex.contracts.configuration import canonical
from autofusex.client.daemon import request
from autofusex.client.results import read
from autofusex.client.store import Store
from autofusex.execution.storage import ResultRepository
from autofusex.workloads.fuzz.generation.sampling import SamplingConfig


def spec():
    return dict(schema_version=1, kind='fuzz_task', environment='fixture',
                workload=dict(mode='generate'), results=dict(save=True, upload=False))


def invoke(home, *args, input=None):
    proc = subprocess.run([sys.executable, '-m', 'autofusex', *args, '--data-dir', str(home)],
                          input=input, text=True, capture_output=True, timeout=20)
    assert not proc.stderr, proc.stderr
    return proc.returncode, json.loads(proc.stdout)


@pytest.fixture
def daemon():
    # Keep AF_UNIX paths short on macOS as well as Linux.
    with tempfile.TemporaryDirectory(prefix='afx-agent-', dir='/tmp') as path:
        home = Path(path)
        store = Store(home)
        with store.connect() as db:
            db.execute('INSERT INTO environments VALUES (?,?,?,?)',
                       ('fixture', 'fixture', canonical(dict(id='fixture', python=sys.executable, env={})), 'ready'))
        (home/'trust.json').write_text(canonical(dict(resource_limits=dict(allowed_devices=['npu:0'], max_workers=1, max_compile_threads=1))))
        (home/'trust.json').chmod(0o600)
        child = subprocess.Popen([sys.executable, '-m', 'autofusex', 'daemon', 'run', '--data-dir', path],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for _ in range(100):
                if child.poll() is not None:
                    raise AssertionError(child.communicate())
                try:
                    request(home, dict(action='status'))
                    break
                except OSError:
                    time.sleep(.05)
            else:
                raise AssertionError('daemon unavailable')
            yield home, store
        finally:
            request(home, dict(action='shutdown'))
            child.communicate(timeout=10)
            assert child.returncode == 0


def seed_run(store, identity='example', state='completed'):
    value = cli.validate(spec())
    value['execution']['resources'].update(workers=1, compile_threads=1)
    prepared = dict(spec=value, environment=dict(id='fixture'))
    with store.connect() as db:
        db.execute('INSERT INTO runs VALUES (?,?,?,?,?)', (identity, canonical(prepared), state, None, None))
    directory = store.home/'runs'/identity
    directory.mkdir(parents=True)
    return dict(id=identity, state=state, spec=prepared), directory


def test_validate_stdin_wait_and_idempotency(daemon):
    home, store = daemon
    args = ('task', 'validate', '-', '--wait', '--request-key', 'check-1')
    code, first = invoke(home, *args, input=canonical(spec()))
    assert code == 0 and first['data']['state'] == 'succeeded'
    assert first['data']['result']['spec']['environment'] == 'fixture'
    code, again = invoke(home, *args, input=canonical(spec()))
    assert code == 0 and again['data']['id'] == first['data']['id']
    changed = spec(); changed['name'] = 'different'
    code, conflict = invoke(home, *args, input=canonical(changed))
    assert code == 2 and not conflict['ok']


def test_submit_stdin_and_workers_override(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'stdin', io.StringIO(canonical(spec())))
    submitted = []
    def fake(home, args, kind, payload, **options):
        submitted.append((kind, payload))
        return dict(id='operation', state='queued')
    monkeypatch.setattr(cli, '_submit', fake)
    result = cli._run(cli.parser().parse_args(['task', 'submit', '-', '--workers', '2', '--data-dir', str(tmp_path)]))
    assert result['id'] == 'operation'
    assert submitted[0][1]['execution']['resources']['workers'] == 2
    assert submitted[0][1]['execution']['resources']['policy'] == 'manual'


def test_wait_and_result_paging_with_real_daemon(daemon):
    home, store = daemon
    run, directory = seed_run(store)
    seed = str(SamplingConfig.profile('small', max_nodes=1, dtype_mode='float32').seed(3))
    repo = ResultRepository(directory/'results.sqlite')
    for status in ('ok', 'fail'):
        repo.save(dict(seed=seed, workload_id='a'*64, oracle_id='ref/1', backend='af-mlir', status=status,
                       execution={'backend':'af-mlir','device':'npu:0'}, evaluation={'correctness':True,'performance':False},
                       shape_results=[dict(shape_index=0, correctness='pass' if status=='ok' else 'fail', performance='not_requested')]))
    code, waited = invoke(home, 'task', 'wait', run['id'])
    assert code == 0 and waited['data']['state'] == 'completed'
    code, page = invoke(home, 'task', 'results', run['id'], '--limit', '1')
    assert code == 0 and page['data']['unit'] == 'attempt'
    assert len(page['data']['items']) == 1
    cursor = page['data']['next_cursor']
    code, last = invoke(home, 'task', 'results', run['id'], '--after', str(cursor))
    assert code == 0 and last['data']['items'][0]['status'] == 'fail'
    assert last['data']['next_cursor'] is None
    identity = last['data']['items'][0]['id']
    code, detail = invoke(home, 'task', 'results', run['id'], '--attempt', identity)
    assert code == 0 and detail['data']['attempt']['seed'] == seed
    code, filtered = invoke(home, 'task', 'results', run['id'], '--status', 'fail')
    assert code == 0 and len(filtered['data']['items']) == 1
    code, missing = invoke(home, 'task', 'results', 'missing')
    assert code == 2 and not missing['ok']


def test_wait_failure_and_timeout_do_not_change_run(daemon):
    home, store = daemon
    for state in ('failed', 'cancelled', 'stopped', 'interrupted', 'needs_attention'):
        run, _ = seed_run(store, identity=state, state=state)
        code, result = invoke(home, 'task', 'wait', state)
        assert code == 1
        # The real daemon may reconcile an unowned attention run to interrupted.
        assert result['data']['state'] in ({'needs_attention', 'interrupted'} if state == 'needs_attention' else {state})
    seed_run(store, identity='running', state='running')
    code, result = invoke(home, 'task', 'wait', 'running', '--timeout', '0.01')
    assert code == 2 and result['error']['code'] == 'TimeoutError'
    assert request(home, dict(action='query', query='run', value='running'))['state'] == 'running'
    for timeout in ('nan', 'inf', '0'):
        code, result = invoke(home, 'task', 'wait', 'running', '--timeout', timeout)
        assert code == 2 and result['error']['code'] == 'ValueError'


def test_wait_observes_transition(monkeypatch):
    states = iter(['running', 'draining', 'stopped'])
    monkeypatch.setattr(cli, '_local_query', lambda *a: dict(id='run', state=next(states)))
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    assert cli._wait_run(None, 'run', 1)['state'] == 'stopped'


def test_replay_reuses_snapshot_without_mutating_original(monkeypatch, tmp_path):
    store = Store(tmp_path)
    original, _ = seed_run(store)
    before = canonical(original)
    monkeypatch.setattr(cli, '_local_query', lambda *a: original)
    captured = []
    monkeypatch.setattr(cli, '_submit', lambda home, args, kind, payload, **kw: captured.append(payload) or dict(id='op'))
    seed = str(SamplingConfig.profile('small', max_nodes=1).seed(3))
    cli._run(cli.parser().parse_args(['task', 'replay', seed, '--from-run', 'example', '--data-dir', str(tmp_path)]))
    assert canonical(original) == before
    assert captured[0]['workload'] == dict(mode='replay', seeds=[seed])
    assert 'termination' not in captured[0]
    assert captured[0]['execution']['resources']['policy'] == 'manual'
    assert captured[0]['evaluation'] == original['spec']['spec']['evaluation']


def test_empty_results_do_not_create_database_and_invalid_paths_rejected(tmp_path):
    store = Store(tmp_path)
    run, directory = seed_run(store)
    assert read(tmp_path, run)['available'] is False
    assert not (directory/'results.sqlite').exists()
    with pytest.raises(ValueError): read(tmp_path, run, limit=201)
    with pytest.raises(ValueError): read(tmp_path, dict(run, id='../other'))
    with pytest.raises(LookupError): read(tmp_path, run, attempt='missing')
