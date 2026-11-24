"""Retention tests use private synthetic trees, never an installation's data."""
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import time

import pytest
from autofusex.client.cleanup import sweep, access, debug_worker
from autofusex.client.store import Store

NOW = time.time()
OLD = NOW - 4 * 86400


def age(root, timestamp=OLD):
    for base, dirs, files in os.walk(root, topdown=False):
        for name in files + dirs:
            path = Path(base) / name
            if not path.is_symlink(): os.utime(path, (timestamp,timestamp))
    os.utime(root,(timestamp,timestamp))


def run(store, identity='one', state='completed', upload=False):
    directory=store.home/'runs'/identity;directory.mkdir(parents=True)
    (directory/'execution.log').write_text('log')
    spec=dict(spec=dict(kind='fuzz_task',results=dict(upload=upload)))
    with sqlite3.connect(directory/'results.sqlite') as db:
        db.execute('CREATE TABLE attempts(id TEXT,seed TEXT)')
        db.execute("INSERT INTO attempts VALUES ('attempt','seed')")
    with store.connect() as db:
        db.execute('INSERT INTO runs VALUES (?,?,?,?,?)',(identity,json.dumps(spec),state,None,json.dumps(dict(state=state,ended_at=OLD))))
    age(directory)
    return directory


def test_expired_run_preserves_history_and_count(tmp_path):
    store=Store(tmp_path);directory=run(store)
    result=sweep(store,NOW)
    assert result['runs_removed']==1 and result['bytes_removed']>0
    assert not directory.exists()
    with store.connect() as db: row=db.execute('SELECT * FROM runs').fetchone()
    summary=json.loads(row['result'])
    assert row['state']=='completed' and summary['result_count']==1
    assert summary['cleanup']['state']=='cleaned'


@pytest.mark.parametrize('state',['running','preparing','paused','draining','interrupted','needs_attention'])
def test_resumable_states_never_deleted(tmp_path,state):
    store=Store(tmp_path);directory=run(store,state=state)
    sweep(store,NOW)
    assert directory.exists()


def test_newest_file_and_end_time_protect_run(tmp_path):
    store=Store(tmp_path);directory=run(store)
    # Background uploader bookkeeping must not extend retention.
    (directory/'execution.log').touch()
    with store.connect() as db:
        db.execute('UPDATE runs SET result=?',(json.dumps(dict(ended_at=NOW)),))
    sweep(store,NOW);assert directory.exists()


def test_zero_disabled_invalid_fails_closed_and_independent_days(tmp_path):
    store=Store(tmp_path);directory=run(store)
    debug=store.home/'debug'/'autofuse_compile'/'old.txt';debug.parent.mkdir(parents=True);debug.write_text('dump');age(debug.parent)
    config=store.home/'task.json'
    config.write_text(json.dumps(dict(cleanup=dict(debug_days=0,runs_days=5))))
    sweep(store,NOW);assert directory.exists() and debug.exists()
    config.write_text(json.dumps(dict(cleanup=dict(debug_days=-1,runs_days=1))))
    assert sweep(store,NOW)['state']=='error';assert directory.exists()
    config.write_text(json.dumps(dict(cleanup=dict(debug_days=1,runs_days=0))))
    sweep(store,NOW);assert directory.exists() and not debug.exists()


def test_pending_and_rejected_uploads(tmp_path):
    store=Store(tmp_path);directory=run(store,upload=True)
    sweep(store,NOW);assert directory.exists()
    from autofusex.client.platform_transport import State
    state=State(directory/'upload.sqlite');state.set('fuzz:cursor',0);state.conn.close();age(directory)
    sweep(store,NOW);assert directory.exists()
    state=State(directory/'upload.sqlite');state.set('fuzz:cursor',1)
    state.conn.execute("INSERT INTO rejected VALUES ('attempt','map: failed',0)");state.conn.close();age(directory)
    sweep(store,NOW);assert directory.exists()
    state=State(directory/'upload.sqlite');state.conn.execute("UPDATE rejected SET reason='environment: unsupported'");state.conn.close();age(directory)
    sweep(store,NOW);assert not directory.exists()


@pytest.mark.parametrize('reference',['upgrade','retest','source'])
def test_unfinished_delivery_and_upgrade_references(tmp_path,reference):
    store=Store(tmp_path);directory=run(store)
    if reference=='upgrade':store.put('client_switch',dict(state='stage_failed',old_runs=['one']))
    elif reference=='retest':
        records=store.home/'retest';records.mkdir()
        (records/'task-1.json').write_text(json.dumps(dict(queue=str(directory/'tasks.sqlite'),database=str(directory/'results.sqlite'))))
    else:
        op=store.submit('task.submit',{},'source');store.state(op['id'],'succeeded',dict(run_id='one'))
        store.put('source_job',dict(operation_id=op['id']))
    sweep(store,NOW);assert directory.exists()


def test_shared_reader_and_live_queue_protect_run(tmp_path):
    store=Store(tmp_path);directory=run(store)
    with access(store.home):
        sweep(store,NOW);assert directory.exists()
    queue=directory/'tasks.sqlite';queue.touch();age(directory)
    lockpath=Path(str(queue)+'.owner.lock') if __import__('sys').platform=='darwin' else queue
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        sweep(store,NOW);assert directory.exists()


def test_active_debug_lease_and_symlink_safety(tmp_path):
    store=Store(tmp_path/'home');directory,lease=debug_worker(store.home)
    (directory/'dump.txt').write_text('active');age(directory)
    sweep(store,NOW);assert directory.exists()
    lease.close();sweep(store,NOW);assert not directory.exists()
    outside=tmp_path/'outside';outside.mkdir();(outside/'keep').write_text('safe');age(outside)
    debug=store.home/'debug'/'autofuse_compile';debug.mkdir(parents=True,exist_ok=True)
    (debug/'external').symlink_to(outside,target_is_directory=True)
    candidate=run(store);(candidate/'external').symlink_to(outside,target_is_directory=True);age(candidate)
    sweep(store,NOW)
    assert candidate.exists() and (outside/'keep').read_text()=='safe'


def test_detached_deletion_recovers_after_crash(tmp_path):
    store=Store(tmp_path);directory=run(store)
    summary=dict(result_count=1,cleanup=dict(state='deleting',at=OLD,bytes=3))
    with store.connect() as db:db.execute('UPDATE runs SET result=?',(json.dumps(summary),))
    trash=directory.with_name('.cleanup-one');directory.rename(trash)
    sweep(store,NOW)
    assert not trash.exists()
    with store.connect() as db:assert json.loads(db.execute('SELECT result FROM runs').fetchone()[0])['cleanup']['state']=='cleaned'


def test_legacy_run_gets_conservative_completion_floor(tmp_path):
    store=Store(tmp_path);directory=run(store)
    with store.connect() as db:db.execute("UPDATE runs SET result='{}'")
    sweep(store,NOW);assert directory.exists()
    sweep(store,NOW+3*86400);assert not directory.exists()


def test_worker_inherits_debug_lock_after_parent_releases(tmp_path):
    import subprocess,sys
    store=Store(tmp_path);directory,lease=debug_worker(store.home)
    (directory/'dump.txt').write_text('dump');age(directory)
    child=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.read()'],stdin=subprocess.PIPE,pass_fds=(lease.fileno(),))
    lease.close()
    try:
        sweep(store,NOW);assert directory.exists()
    finally:
        child.communicate(timeout=10)
    sweep(store,NOW);assert not directory.exists()


def test_background_sweep_keeps_ipc_responsive(tmp_path,monkeypatch):
    import threading
    from autofusex.client.daemon import Manager,request
    import autofusex.client.cleanup as cleanup
    entered=threading.Event();release=threading.Event()
    original=cleanup.sweep
    def slow(store):
        entered.set();release.wait(10)
        return original(store)
    monkeypatch.setattr(cleanup,'sweep',slow)
    manager=Manager(tmp_path)
    thread=threading.Thread(target=manager.run);thread.start()
    try:
        assert entered.wait(5)
        started=time.monotonic()
        assert request(tmp_path,dict(action='status'))['protocol']==1
        assert time.monotonic()-started<2
        request(tmp_path,dict(action='shutdown'))
    finally:
        release.set();manager.stop.set();thread.join(10)
    assert not thread.is_alive()


def test_cleaned_results_report_retention_status(tmp_path):
    from autofusex.client.results import read
    store=Store(tmp_path);run(store)
    sweep(store,NOW)
    with store.connect() as db:row=dict(db.execute('SELECT * FROM runs').fetchone())
    row['spec']=json.loads(row['spec']);row['result']=json.loads(row['result'])
    result=read(tmp_path,row)
    assert not result['available'] and result['cleanup']['state']=='cleaned'
    assert result['items']==[]
