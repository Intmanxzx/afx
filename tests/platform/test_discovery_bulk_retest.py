"""Bulk discovery selection and durable per-user retest assignment."""
import pytest
from test_issues import World, item, seed, INSTALLED
from autofusex.server import auth, queries, retests, regression
from autofusex.server.issues import IssueError


def test_selection_spans_pages_and_uses_same_filters(tmp_path):
    w = World(tmp_path)
    w.upload([item('bulk-exec-'+str(i), i) for i in range(1, 42)])
    assert len(queries.case_page(w.store, limit=30)['groups']) == 30
    assert len(queries.case_page(w.store, keys_only=True)['case_keys']) == 41
    assert queries.case_page(w.store, keys_only=True, results='none')['case_keys'] == []
    assert queries.case_page(w.store, keys_only=True, chips='a5')['case_keys'] == []
    filtered = dict(q='backend:af-mlir', results='pass', chips='a2a3')
    selected = queries.case_page(w.store, keys_only=True, **filtered)['case_keys']
    paged = queries.case_page(w.store, limit=200, **filtered)['groups']
    assert selected == [g['case']['case_key'] for g in paged]


def test_even_assignment_survives_client_replacement_and_failure(tmp_path):
    w = World(tmp_path)
    w.upload([item('assign-'+str(i), i) for i in range(1, 6)])
    body = dict(request_key='bulk', case_keys=[seed(i) for i in range(1, 6)],
                backends=['af-mlir'], user_ids=['dev', 'root'])
    assert retests.create_manual(w.store, w.admin, dict(body))['tasks'] == 5
    assert retests.create_manual(w.store, w.admin, dict(body))['tasks'] == 5
    conn = w.store.reader()
    counts = dict(conn.execute('SELECT assigned_user,COUNT(*) FROM retest_tasks GROUP BY assigned_user'))
    assert counts == {'dev': 3, 'root': 2}
    tasks = w.claim_tasks()
    assert len(tasks) == 3
    w.run_tasks(tasks[:1], lambda _: 'compile_fail')
    with w.store.write() as db:
        db.execute("UPDATE retest_tasks SET lease_expires_at=0 WHERE status='leased'")
    replacement = retests.claim(w.store, w.client, dict(request_key='new-client',
        machine_id='replacement-id', npu_arch='910b', installed=INSTALLED, limit=20))['tasks']
    assert {t['task_id'] for t in replacement} == {t['task_id'] for t in tasks[1:]}
    assert conn.execute('SELECT status,result FROM retest_tasks WHERE task_id=?',
                        (tasks[0]['task_id'],)).fetchone()[:] == ('done', 'fail')
    assert dict(conn.execute('SELECT assigned_user,COUNT(*) FROM retest_tasks GROUP BY assigned_user')) == counts
    batches = retests.list_manual(w.store, w.admin)
    assert batches[0]['tasks'] == 5
    assert {a['user_id']: a['tasks'] for a in batches[0]['assignments']} == counts
    assert not retests.list_manual(w.store, w.client)


def test_only_admin_can_assign_and_invalid_users_rollback(tmp_path):
    w = World(tmp_path)
    w.upload([item('permissions', 1)])
    body = dict(request_key='p', case_keys=[seed(1)], backends=['af-mlir'], user_ids=['dev'])
    with pytest.raises(auth.AuthError):
        retests.create_manual(w.store, w.client, dict(body))
    for users in ([], ['missing'], ['dev', 'missing']):
        with pytest.raises(IssueError):
            retests.create_manual(w.store, w.admin, dict(body, user_ids=users))
    assert w.store.reader().execute('SELECT COUNT(*) FROM retest_tasks').fetchone()[0] == 0


def test_bulk_over_old_limits_and_bounded_listing(tmp_path):
    w = World(tmp_path)
    for start in range(1, 602, 100):
        w.upload([item('many-exec-'+str(i), i) for i in range(start, min(start+100, 602))])
    keys = queries.case_page(w.store, keys_only=True)['case_keys']
    result = retests.create_manual(w.store, w.admin, dict(request_key='large', case_keys=keys,
        backends=['af-mlir'], user_ids=['dev']))
    assert result['tasks'] == 601
    batch = retests.list_manual(w.store, w.admin)[0]
    assert batch['items_truncated'] and len(batch['items']) == 100
    assert regression.add_pins(w.store, w.admin, dict(request_key='pins', scope='global', case_keys=keys))['pinned'] == 601


def test_http_bulk_selection_and_assignment(tmp_path):
    from fastapi.testclient import TestClient
    from autofusex.server.server import create_app
    w = World(tmp_path)
    w.upload([item('http-bulk', 1)])
    token = auth.issue_token(w.store, 'root')
    client = TestClient(create_app(w.store), base_url='http://127.0.0.1')
    headers = {'Authorization': 'Bearer ' + token}
    response = client.get('/api/cases/selection', params={'results': 'pass', 'chips': 'a2a3'}, headers=headers)
    assert response.status_code == 200, response.text
    keys = response.json()['data']['case_keys']
    assert keys == [seed(1)]
    # Above the old 64 KiB request limit, and duplicate keys still create one task.
    response = client.post('/api/retests', headers=headers, json=dict(request_key='http-large',
        case_keys=keys * 3000, backends=['af-mlir'], user_ids=['dev']))
    assert response.status_code == 200, response.text
    assert response.json()['data']['tasks'] == 1


def test_existing_tasks_migrate_to_requesting_user(tmp_path):
    import sqlite3
    from autofusex.server import schema
    conn = sqlite3.connect(tmp_path / 'old.sqlite', isolation_level=None)
    schema.apply(conn, 'main', [(v, sql) for v, sql in schema.CORE if v < 21])
    conn.execute("INSERT INTO retest_tasks(issue_id,round_no,case_id,npu_arch,backends,required_backend,required_version,required_sha256,status,created_at,requested_by) VALUES('',0,1,'910b','af-mlir','af-mlir','*','*','pending',1,'dev')")
    schema.apply(conn, 'main', schema.CORE)
    assert conn.execute('SELECT assigned_user FROM retest_tasks').fetchone()[0] == 'dev'
    conn.close()


def test_no_coverage_receipt_finishes_without_fabricating_execution(tmp_path):
    w = World(tmp_path)
    w.upload([item('coverage-retest', 1)])
    retests.create_manual(w.store, w.client, dict(request_key='no-coverage', case_keys=[seed(1)], backends=['af-mlir']))
    task = w.claim_tasks()[0]
    body = dict(request_key='no-coverage-done', lease_id=task['lease_id'], installed=INSTALLED,
                exec_uuids=[], terminal_reason='no_autofuse_coverage')
    with pytest.raises(IssueError):
        retests.complete(w.store, w.admin, task['task_id'], dict(body))
    assert retests.complete(w.store, w.client, task['task_id'], dict(body))['result'] == 'invalid'
    assert retests.complete(w.store, w.client, task['task_id'], dict(body))['result'] == 'invalid'
    assert w.claim_tasks() == []
    assert w.store.reader().execute('SELECT COUNT(*) FROM executions').fetchone()[0] == 1


def test_new_retest_backends_inherit_case_chip(tmp_path):
    w = World(tmp_path)
    w.upload([item('only-mlir-original', 1)])
    result = retests.create_manual(w.store, w.client, dict(request_key='all-backends',
        case_keys=[seed(1)], backends=['af-mlir', 'af-legacy', 'triton-npu', 'eager-npu']))
    assert result['tasks'] == 4 and result['skipped_without_arch'] == 0
    rows = w.store.reader().execute('SELECT required_backend,npu_arch FROM retest_tasks').fetchall()
    assert {(r[0], r[1]) for r in rows} == {(b, '910b') for b in ('af-mlir', 'af-legacy', 'triton-npu', 'eager-npu')}
    installed = {b: {'version':'1', 'sha256':'a'*64} for b in ('af-legacy','triton-npu','eager-npu')}
    tasks = w.claim_tasks(installed=installed)
    assert {t['required']['backend'] for t in tasks} == set(installed)


def test_new_backend_explicit_chip_overrides_history(tmp_path):
    w = World(tmp_path)
    w.upload([item('original-chip', 1)])
    retests.create_manual(w.store, w.client, dict(request_key='explicit-chip',
        case_keys=[seed(1)], backends=['triton-npu'], npu_arch='950'))
    assert w.store.reader().execute('SELECT npu_arch FROM retest_tasks').fetchone()[0] == '950'


def test_no_coverage_persists_and_skips_only_matching_target(tmp_path):
    w = World(tmp_path)
    w.upload([item('coverage-observation', 1)])
    def create(key, **extra):
        return retests.create_manual(w.store, w.client, dict(request_key=key,
            case_keys=[seed(1)], backends=['af-mlir'], **extra))
    create('first')
    task = w.claim_tasks()[0]
    retests.complete(w.store, w.client, task['task_id'], dict(request_key='noaf',
        lease_id=task['lease_id'], installed=INSTALLED, exec_uuids=[], terminal_reason='no_autofuse_coverage'))
    observations = queries.case_history(w.store, seed(1))['case']['coverage_exclusions']
    assert observations[0]['reason'] == 'no_autofuse_coverage'
    assert observations[0]['task_id'] == task['task_id']
    assert create('skip')['skipped_no_coverage'] == 1
    assert create('skip-again')['tasks'] == 0
    assert retests.create_manual(w.store, w.client, dict(request_key='other-backend',
        case_keys=[seed(1)], backends=['af-legacy']))['tasks'] == 1
    assert create('other-chip', npu_arch='950')['tasks'] == 1
    assert create('force', force=True)['tasks'] == 1
    # A later valid retest clears the observation, enabling ordinary replay again.
    tasks = [t for t in w.claim_tasks() if t['required']['backend'] == 'af-mlir']
    w.run_tasks(tasks, lambda _: 'pass')
    assert create('after-pass')['tasks'] == 1


def test_no_coverage_is_revision_scoped(tmp_path):
    w = World(tmp_path)
    w.upload([item('queued-coverage', 1)])
    for key in ('queued-one',):
        retests.create_manual(w.store, w.client, dict(request_key=key,
            case_keys=[seed(1)], backends=['af-mlir']))
    task = w.claim_tasks(limit=1)[0]
    retests.complete(w.store, w.client, task['task_id'], dict(request_key='queued-finish',
        lease_id=task['lease_id'], installed=INSTALLED, exec_uuids=[], terminal_reason='no_autofuse_coverage'))
    assert w.claim_tasks() == []
    assert w.store.reader().execute("SELECT COUNT(*) FROM retest_tasks WHERE status='invalid'").fetchone()[0] == 1
    with w.store.write() as db:
        db.execute('UPDATE cases SET case_revision=case_revision+1')
    assert retests.create_manual(w.store, w.client, dict(request_key='new-revision',
        case_keys=[seed(1)], backends=['af-mlir']))['tasks'] == 1


def test_environment_terminal_does_not_exclude_case(tmp_path):
    w = World(tmp_path)
    w.upload([item('environment-coverage', 1)])
    retests.create_manual(w.store, w.client, dict(request_key='env-first',
        case_keys=[seed(1)], backends=['af-mlir']))
    task = w.claim_tasks()[0]
    retests.complete(w.store, w.client, task['task_id'], dict(request_key='env-finish',
        lease_id=task['lease_id'], installed=INSTALLED, exec_uuids=[], terminal_reason='user_stop'))
    assert not queries.case_history(w.store, seed(1))['case']['coverage_exclusions']
    assert retests.create_manual(w.store, w.client, dict(request_key='env-retry',
        case_keys=[seed(1)], backends=['af-mlir']))['tasks'] == 1
