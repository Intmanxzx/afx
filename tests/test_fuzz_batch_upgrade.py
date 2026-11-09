from concurrent.futures import ThreadPoolExecutor
import sqlite3
import uuid

import pytest

from autofusedebugger.fuzz.contracts import ExecutionConfig, EvaluationConfig
from autofusedebugger.fuzz.scheduling.ledger import TaskLedger, DDL, APPLICATION_ID
from autofusedebugger.fuzz.scheduling.migrations import upgrade_queue

EX, EV = ExecutionConfig('cpu-eager','cpu',1), EvaluationConfig()


def test_batch_capacity_and_late_conflict_are_atomic(tmp_path):
    queue = TaskLedger(tmp_path/'queue', max_active=2)
    identity = queue.enqueue('42', EX, EV)
    with pytest.raises(ValueError, match='entire batch'):
        queue.enqueue_many([('43', EX, EV, None), ('44', EX, EV, None)])
    assert len(queue.page()['items']) == 1
    with pytest.raises(ValueError, match='conflicts'):
        queue.enqueue_many([('43', EX, EV, None), ('44', EX, EV, identity)])
    assert len(queue.page()['items']) == 1
    ids = queue.enqueue_many([('42', EX, EV, identity), ('43', EX, EV, None)])
    assert ids[0] == identity and len(queue.page()['items']) == 2


def test_late_invalid_spec_never_partially_enqueues(tmp_path):
    queue = TaskLedger(tmp_path/'queue')
    with pytest.raises(ValueError):
        queue.enqueue_many([('42', EX, EV, None), ('bad', EX, EV, None)])
    assert queue.page()['items'] == []


def test_concurrent_batch_capacity(tmp_path):
    path = tmp_path/'queue'
    TaskLedger(path, max_active=3).page()
    def enqueue(_):
        try:
            return TaskLedger(path).enqueue_many([('42', EX, EV, None), ('43', EX, EV, None)])
        except ValueError:
            return []
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(len(ids) for ids in pool.map(enqueue, range(2))) == [0, 2]
    assert len(TaskLedger(path).page()['items']) == 2


def old_queue(path):
    with sqlite3.connect(path) as conn:
        conn.executescript(DDL.replace(",\n reason TEXT NOT NULL DEFAULT ''", ''))
        conn.execute('INSERT INTO queue_config VALUES (1,1000)')
        conn.execute(f'PRAGMA application_id={APPLICATION_ID}')
        conn.execute('PRAGMA user_version=1')
        conn.execute("INSERT INTO tasks(id,seed,execution,evaluation,state) VALUES (?,?,?,?,?)",
                     (str(uuid.uuid4()), '42', '{}', '{}', 'pending'))


def test_offline_upgrade_keeps_exact_v1_backup(tmp_path):
    path, backup = tmp_path/'queue', tmp_path/'backup'
    old_queue(path)
    with pytest.raises(ValueError, match='supported task ledger'):
        TaskLedger(path).page()
    assert upgrade_queue(path, backup)['schema_version'] == 2
    task = TaskLedger(path).page()['items'][0]
    assert task['seed'] == '42' and task['reason'] == ''
    with sqlite3.connect(backup) as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0] == 1
        assert len(conn.execute('PRAGMA table_info(tasks)').fetchall()) == 8
        assert conn.execute('SELECT id FROM tasks').fetchone()[0] == task['id']


def test_backup_publication_failure_leaves_original_schema(tmp_path):
    path, backup = tmp_path/'queue', tmp_path/'backup'
    old_queue(path); backup.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        upgrade_queue(path, backup)
    with sqlite3.connect(path) as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0] == 1
    assert backup.read_bytes() == b'preserve'
    assert not list(tmp_path.glob('.fuzz-queue-backup-*'))


def test_upgrade_live_queue_rejected(tmp_path):
    import fcntl
    path = tmp_path/'queue'; old_queue(path)
    with path.open('rb') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='still owned'):
            upgrade_queue(path, tmp_path/'backup')
    assert not (tmp_path/'backup').exists()


def test_reason_is_bounded_and_survives_reopen(tmp_path):
    path = tmp_path/'queue'
    queue = TaskLedger(path); identity = queue.enqueue('42', EX, EV)
    claim = queue.claim()
    with pytest.raises(ValueError, match='short text'):
        queue.interrupt(identity, claim['claim'], 'x'*501)
    queue.interrupt(identity, claim['claim'], 'timeout')
    assert TaskLedger(path).get(identity)['reason'] == 'timeout'
    queue.resume(identity)
    assert queue.get(identity)['reason'] == ''


def test_upgrade_failure_after_alter_rolls_back_and_keeps_backup(tmp_path, monkeypatch):
    from autofusedebugger.fuzz.scheduling import migrations
    path, backup = tmp_path/'queue', tmp_path/'backup'
    old_queue(path)
    original = sqlite3.connect
    class FailVersion(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == 'PRAGMA user_version=2':
                raise sqlite3.OperationalError('injected failure after ALTER')
            return super().execute(sql, *args, **kwargs)
    def connect(*args, **kwargs):
        if 'mode=rw' in str(args[0]):
            kwargs['factory'] = FailVersion
        return original(*args, **kwargs)
    monkeypatch.setattr(migrations.sqlite3, 'connect', connect)
    with pytest.raises(sqlite3.OperationalError, match='after ALTER'):
        upgrade_queue(path, backup)
    for database in (path, backup):
        with original(database) as conn:
            assert conn.execute('PRAGMA user_version').fetchone()[0] == 1
            assert len(conn.execute('PRAGMA table_info(tasks)').fetchall()) == 8
    assert not list(tmp_path.glob('.fuzz-queue-backup-*'))
