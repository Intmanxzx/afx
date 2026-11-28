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


ENV = dict(torch='2.10', torch_npu='post7', torch_npu_build='b'*64, cann='9.2', autofuse_binary='a'*64)


def test_no_coverage_skips_only_proven_matching_environment(tmp_path):
    w = World(tmp_path)
    w.upload([item('coverage-observation', 1)])
    def create(key, **extra):
        return retests.create_manual(w.store, w.client, dict(request_key=key,
            case_keys=[seed(1)], backends=['af-mlir'], **extra))
    def claim(key, environment):
        return retests.claim(w.store, w.client, dict(request_key=key, machine_id='one',
            npu_arch='910b', installed=INSTALLED, coverage_environment=environment))['tasks']
    create('first')
    task = claim('claim-first', ENV)[0]
    environment = dict(ENV, autofuse_binary=INSTALLED['af-mlir']['sha256'])
    evidence = [dict(backend='af-mlir', environment=environment,
        autofuse_coverage=dict(installed=True, no_coverage=True, candidates=0, scheduler_codegen_succeeded=1))]
    retests.complete(w.store, w.client, task['task_id'], dict(request_key='noaf',
        lease_id=task['lease_id'], installed=INSTALLED, exec_uuids=[], terminal_reason='no_autofuse_coverage',
        coverage_environment=environment, coverage_evidence=evidence))
    observations = queries.case_history(w.store, seed(1))['case']['coverage_exclusions']
    assert observations[0]['evidence'] == evidence
    assert observations[0]['environment_key']
    assert create('same')['tasks'] == 1
    assert claim('claim-same', environment) == []
    assert create('changed')['tasks'] == 1
    changed = claim('claim-changed', dict(environment, torch_npu_build='c'*64))
    assert len(changed) == 1
    assert create('force', force=True)['tasks'] == 1
    assert len(claim('claim-force', environment)) == 1
    assert create('unknown')['tasks'] == 1
    assert len(claim('claim-unknown', {})) == 1


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


def test_no_coverage_preserves_original_execution_and_statistics(tmp_path):
    w = World(tmp_path)
    w.upload([item('original-pass', 1)])
    before = queries.case_history(w.store, seed(1))['executions'][0]
    stats = queries.backend_stats(w.store, 'af-mlir')['groups']
    retests.create_manual(w.store, w.client, dict(request_key='observe', case_keys=[seed(1)], backends=['af-mlir']))
    task = w.claim_tasks()[0]
    retests.complete(w.store, w.client, task['task_id'], dict(request_key='observed',
        lease_id=task['lease_id'], installed=INSTALLED, exec_uuids=[], terminal_reason='no_autofuse_coverage'))
    assert queries.case_history(w.store, seed(1))['executions'][0] == before
    assert queries.backend_stats(w.store, 'af-mlir')['groups'] == stats
    assert retests.create_manual(w.store, w.client, dict(request_key='retry', case_keys=[seed(1)], backends=['af-mlir']))['tasks'] == 1



def test_existing_coverage_marks_heads_and_rebuilds_statistics(tmp_path):
    from autofusex.server import schema
    from autofusex.server.ingest import check_stats
    w = World(tmp_path)
    w.upload([item('historical-invalid-head', 1)])
    with w.store.write() as db:
        db.execute("INSERT INTO case_coverage VALUES(1,1,'af-mlir','910b','no_autofuse_coverage',1,'{}',1)")
        db.execute('ALTER TABLE executions DROP COLUMN invalid_reason')
        db.execute('ALTER TABLE executions DROP COLUMN pre_invalid_status')
        db.execute('DELETE FROM schema_migrations WHERE version IN (23,24,33)')
        db.execute('DROP TABLE coverage_observations')
        db.execute('ALTER TABLE retest_tasks DROP COLUMN force_coverage')
        db.execute('ALTER TABLE retest_tasks DROP COLUMN coverage_environment_key')
    with w.store.connect(readonly=False) as db:
        schema.apply(db, 'main', schema.CORE)
    assert check_stats(w.store)
    group = queries.backend_stats(w.store, 'af-mlir')['groups'][0]
    assert group['valid_total'] == 1 and group['pass'] == 1
    assert w.store.reader().execute('SELECT pre_invalid_status FROM executions').fetchone()[0] is None


def test_migrate_deployed_invalid_coverage_to_unsupported(tmp_path):
    from autofusex.server import schema
    from autofusex.server.ingest import check_stats
    w = World(tmp_path)
    w.upload([item('deployed-invalid-head', 1)])
    with w.store.write() as db:
        db.execute("UPDATE executions SET function_status='invalid', invalid_reason='no_autofuse_coverage', pre_invalid_status='pass'")
        db.execute('DELETE FROM schema_migrations WHERE version IN (24,33)')
        db.execute('DROP TABLE coverage_observations')
        db.execute('ALTER TABLE retest_tasks DROP COLUMN force_coverage')
        db.execute('ALTER TABLE retest_tasks DROP COLUMN coverage_environment_key')
    with w.store.connect(readonly=False) as db:
        schema.apply(db, 'main', schema.CORE)
    assert check_stats(w.store)
    row = queries.case_history(w.store, seed(1))['executions'][0]
    assert row['function_status'] == 'pass'
    assert row['invalid_reason'] is None
    group = queries.backend_stats(w.store, 'af-mlir')['groups'][0]
    assert group['results'] == {'pass': 1}
    assert group['valid_total'] == 1
    assert w.store.reader().execute('SELECT pre_invalid_status FROM executions').fetchone()[0] is None


@pytest.mark.parametrize('repetitions', [1, 3])
def test_case_backends_stay_with_one_user(tmp_path, repetitions):
    w = World(tmp_path)
    w.upload([item('grouped-'+str(i), i) for i in range(1, 6)])
    body = dict(request_key='grouped', case_keys=[seed(i) for i in range(1, 6)],
                backends=['af-mlir', 'af-legacy'], user_ids=['dev', 'root'],
                repetitions=repetitions)
    assert retests.create_manual(w.store, w.admin, dict(body))['tasks'] == 10 * repetitions
    assert retests.create_manual(w.store, w.admin, dict(body))['tasks'] == 10 * repetitions
    conn = w.store.reader()
    rows = conn.execute('SELECT case_id,COUNT(DISTINCT assigned_user),COUNT(*) FROM retest_tasks GROUP BY case_id').fetchall()
    assert len(rows) == 5
    assert all(r[1:] == (1, 2 * repetitions) for r in rows)
    assert dict(conn.execute('SELECT assigned_user,COUNT(*) FROM retest_tasks GROUP BY assigned_user')) == {
        'dev': 6 * repetitions, 'root': 4 * repetitions}


def test_case_assignment_balances_tasks_and_ignores_skipped_cases(tmp_path):
    w = World(tmp_path)
    w.upload([item('varying-'+str(i), i) for i in range(1, 5)])
    with w.store.write() as conn:
        conn.execute('DELETE FROM execution_heads WHERE case_id=(SELECT case_id FROM cases WHERE case_key=?)', (seed(1),))
        conn.execute("UPDATE execution_heads SET backend='af-legacy' WHERE case_id=(SELECT case_id FROM cases WHERE case_key=?)", (seed(2),))
    # Case 2 has two architectures across its requested backends; subsequent
    # single-architecture cases should both go to the other user to catch up.
    w.upload([dict(item('extra-chip', 2), npu_arch='950')])
    retests.create_manual(w.store, w.admin, dict(request_key='varying',
        case_keys=[seed(i) for i in range(1, 5)], backends=['af-mlir', 'triton-npu'],
        user_ids=['dev', 'root']))
    conn = w.store.reader()
    owners = {r[0]: r[1] for r in conn.execute('SELECT c.case_key,t.assigned_user FROM retest_tasks t JOIN cases c USING(case_id)')}
    assert owners == {seed(2): 'dev', seed(3): 'root', seed(4): 'root'}
