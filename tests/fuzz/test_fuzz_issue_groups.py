from concurrent.futures import ThreadPoolExecutor

import pytest
from test_fuzz_retests import setup

from autofusex.fuzz.central import issue_groups
from autofusex.fuzz.central.repository import create_user


def row(error, *, status='error', correctness='error', performance='skipped', function_status='unknown',
        seed='1', shape_index=0, op_types=()):
    return dict(seed=seed, shape_index=shape_index, id=f'{seed}-{shape_index}', status=status,
        function_status=function_status, accuracy_status='Unknown', core_error=error,
        error_summary=error, correctness=correctness, performance=performance,
        measurement_status='not_requested', op_types=op_types)


def install_rows(database, rows):
    from contextlib import closing
    from autofusex.fuzz.storage.repository import _connect
    import json
    with closing(_connect(database, writable=True)) as conn, conn:
        conn.execute('DELETE FROM issue_cases_v1')
        for row in rows:
            dimension, category, message = issue_groups._dimension(row)
            key, title = issue_groups._issue(dimension, category, message)
            conn.execute('INSERT INTO issue_cases_v1 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (row['seed'],row['shape_index'],row['id'],row['status'],row['function_status'],
                 row['core_error'],row['correctness'],row['performance'],row['error_summary'],
                 json.dumps(row['op_types']),dimension,category,title,key))


def groups(monkeypatch, tmp_path, rows):
    database = tmp_path / 'central.sqlite'
    create_user(database, 'Alice')
    install_rows(database, rows)
    return issue_groups.list_groups(database)['items']


def test_function_issues_cluster_by_exact_first_error_line(monkeypatch, tmp_path):
    items = groups(monkeypatch, tmp_path, [
        row('RuntimeError: compiler failed\n/tmp/a 123'),
        row('RuntimeError: compiler failed\n/tmp/b 456', seed='2'),
        row('RuntimeError: different failure\n/tmp/c 789', seed='3'),
    ])
    assert [item['count'] for item in items if item['dimension'] == 'function'] == [2, 1]
    assert items[0]['title'] == 'RuntimeError: compiler failed'


def test_accuracy_and_performance_group_by_operator_category(monkeypatch, tmp_path):
    items = groups(monkeypatch, tmp_path, [
        row('mismatch one', status='fail', correctness='fail', seed='1', op_types=('aten.matmul', 'add')),
        row('mismatch two', status='fail', correctness='fail', seed='2', op_types=('linear', 'relu')),
        row('mismatch three', status='fail', correctness='fail', seed='3', op_types=('add', 'relu')),
        row('perf failed', status='error', correctness='error', performance='error', seed='4', op_types=('layer_norm', 'add')),
    ])
    accuracy = [item for item in items if item['dimension'] == 'accuracy']
    performance = [item for item in items if item['dimension'] == 'performance']
    assert [(item['title'], item['count']) for item in accuracy] == [('MatMul', 2), ('Elementwise', 1)]
    assert [(item['title'], item['count']) for item in performance] == [('Norm', 1)]


def test_claim_is_exclusive_and_can_be_released(monkeypatch, tmp_path):
    database = tmp_path / 'central.sqlite'
    alice = create_user(database, 'Alice')
    bob = create_user(database, 'Bob')
    install_rows(database, [row('same failure')])
    key = issue_groups.list_groups(database)['items'][0]['issue_key']
    assert issue_groups.claim(database, alice['user_id'], key)['owner_name'] == 'Alice'
    with pytest.raises(PermissionError):
        issue_groups.claim(database, bob['user_id'], key)
    assert issue_groups.release(database, alice['user_id'], key)['released']
    assert issue_groups.claim(database, bob['user_id'], key)['owner_name'] == 'Bob'


def test_concurrent_claims_have_one_owner(monkeypatch, tmp_path):
    database = tmp_path / 'central.sqlite'
    users = [create_user(database, f'User {index}') for index in range(8)]
    install_rows(database, [row('same failure')])
    key = issue_groups.list_groups(database)['items'][0]['issue_key']
    def attempt(user):
        try:
            return issue_groups.claim(database, user['user_id'], key)
        except PermissionError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [item for item in pool.map(attempt, users) if item is not None]
    owners = {item['owner_id'] for item in results}
    assert len(results) == 1 and len(owners) == 1
    import sqlite3
    with sqlite3.connect(database) as conn:
        assert conn.execute('SELECT count(*) FROM issue_claims_v1').fetchone()[0] == 1


def test_materialized_issue_index_is_used(tmp_path, monkeypatch):
    from contextlib import closing

    from autofusex.fuzz.central import retests, sessions
    from autofusex.fuzz.central.repository import create_user, ingest
    from autofusex.fuzz.reporting import materialized
    from autofusex.fuzz.storage.repository import ResultRepository, _connect
    from autofusex.fuzz.sync.outbox import pending

    database = tmp_path / 'central.sqlite'
    alice = create_user(database, 'Alice')
    with closing(_connect(database, writable=True)) as conn, conn:
        sessions.initialize(conn)
        retests.initialize(conn)
    local = ResultRepository(tmp_path / 'local.sqlite')
    local.save(dict(seed=str(16076518010981881194532289465530122136584972140560), workload_id='a' * 64, oracle_id='cpu/1',
        backend='af-mlir', status='fail', execution=dict(backend='af-mlir', static_only=False),
        evaluation=dict(correctness=True, performance=False, warmup_steps=1, measured_steps=1),
        shape_results=[dict(shape_index=0, correctness='fail', performance='not_requested')]))
    ingest(database, alice['token'], pending(local.path))
    materialized.rebuild(database)
    with closing(_connect(database)) as conn:
        assert conn.execute('SELECT count(*) FROM issue_cases_v1').fetchone()[0] == 1

    def forbidden(*args, **kwargs):
        raise AssertionError('materialized issue index must be used')
    groups = issue_groups.list_groups(database, 'accuracy')['items']
    assert len(groups) == 1 and groups[0]['count'] == 1


def test_ingest_refreshes_materialized_issue_index(tmp_path, monkeypatch):
    from contextlib import closing

    from autofusex.fuzz.central import retests, sessions
    from autofusex.fuzz.central.repository import create_user, ingest
    from autofusex.fuzz.reporting import materialized
    from autofusex.fuzz.storage.repository import ResultRepository, _connect
    from autofusex.fuzz.sync.outbox import pending

    database = tmp_path / 'central.sqlite'
    alice = create_user(database, 'Alice')
    with closing(_connect(database, writable=True)) as conn, conn:
        sessions.initialize(conn)
        retests.initialize(conn)
    materialized.rebuild(database)
    local = ResultRepository(tmp_path / 'local.sqlite')
    local.save(dict(seed=str(16076518010981881194532289465530122136584972140560), workload_id='a' * 64, oracle_id='cpu/1',
        backend='af-mlir', status='fail', execution=dict(backend='af-mlir', static_only=False),
        evaluation=dict(correctness=True, performance=False, warmup_steps=1, measured_steps=1),
        shape_results=[dict(shape_index=0, correctness='fail', performance='not_requested')]))
    ingest(database, alice['token'], pending(local.path))
    with closing(_connect(database)) as conn:
        assert conn.execute('SELECT count(*) FROM issue_cases_v1').fetchone()[0] == 1

    def forbidden(*args, **kwargs):
        raise AssertionError('ingest must refresh the materialized issue index')
    assert len(issue_groups.list_groups(database, 'accuracy')['items']) == 1


def test_issue_retest_reuses_existing_control_plane(setup):
    from contextlib import closing
    import json as json_module
    import uuid

    from autofusex.fuzz.central import retests
    from autofusex.fuzz.reporting import materialized
    from autofusex.fuzz.storage.repository import _connect

    database, _, bob, _, client, _ = setup
    materialized.rebuild(database)
    group = issue_groups.list_groups(database, 'accuracy')['items'][0]
    issue_groups.claim(database, bob['user_id'], group['issue_key'])
    answer = issue_groups.retest(database, bob['user_id'], group['issue_key'], client, 1, str(uuid.uuid4()))
    assert answer['target_count'] == 1 and len(answer['batch_ids']) == 1
    detail = retests.detail(database, bob['user_id'], answer['batch_ids'][0])['items'][0]
    assert detail['state'] == 'queued'
    assert [target['shape_index'] for target in detail['targets'] if target['selected']] == [1]
    with closing(_connect(database)) as conn:
        claim = conn.execute('SELECT status,batch_id FROM issue_claims_v1 WHERE issue_key=?',
                             (group['issue_key'],)).fetchone()
        assert claim[0] == 'retesting'
        assert json_module.loads(claim[1]) == answer['batch_ids']
