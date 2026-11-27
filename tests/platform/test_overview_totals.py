"""Global identities, actual shapes, and accepted executions."""
import pytest
from autofusex.server import schema
from test_platform import env, http, body, case, item, upload, SEED_B


def totals(store):
    return tuple(store.reader().execute('SELECT case_count,shape_count,record_count,execution_count FROM overview_totals').fetchone())


def test_totals_replays_shapes_arches_and_duplicates(env):
    store, ingestor, principal = env
    assert totals(store) == (0, 0, 0, 0)
    dynamic = dict(case(), is_dynamic=True, shape_count=3)
    payload = body([item('totals-first', c=dynamic), item('totals-replay', c=dynamic),
                    item('totals-chip', c=dynamic, npu_arch='910c'),
                    item('totals-cpu', backend='cpu', c=dynamic),
                    item('totals-static', c=case(SEED_B), status='runtime_fail')])
    upload(ingestor, principal, payload)
    assert totals(store) == (2, 4, 3, 5)
    upload(ingestor, principal, payload)
    assert totals(store) == (2, 4, 3, 5)
    with pytest.raises(RuntimeError):
        with store.write() as conn:
            conn.execute('UPDATE cases SET shape_count=8')
            raise RuntimeError('rollback')
    assert totals(store) == (2, 4, 3, 5)
    with store.write() as conn:
        conn.execute('UPDATE cases SET shape_count=4 WHERE is_dynamic=1')
        conn.execute("DELETE FROM executions WHERE exec_uuid='totals-cpu'")
    assert totals(store) == (2, 5, 2, 4)
    with store.write() as conn:
        conn.execute("UPDATE executions SET backend='new' WHERE exec_uuid='totals-replay'")
    assert totals(store) == (2, 5, 3, 4)


def test_totals_backfill(env):
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('totals-one'), item('totals-two'), item('totals-three', backend='cpu')]))
    with store.write() as conn:
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'overview_%'").fetchall():
            conn.execute('DROP TRIGGER ' + row[0])
        conn.execute('DROP TABLE overview_records')
        conn.execute('DROP TABLE overview_totals')
        conn.execute('DELETE FROM schema_migrations WHERE version=29')
    with store.connect(readonly=False) as conn:
        schema.apply(conn, 'main', schema.CORE)
    assert totals(store) == (1, 1, 2, 3)
    upload(ingestor, principal, body([item('totals-four')], key='next'))
    assert totals(store) == (1, 1, 2, 4)


def test_totals_api_ignores_filters(http):
    assert http('POST', '/api/client/executions', body([item('totals-one'), item('totals-two')]))[0] == 200
    code, result = http('GET', '/api/overview/totals?q=missing&chips=unknown&selected_backends=none&results=invalid')
    assert code == 200
    assert result['data'] == dict(case_count=1, shape_count=1, record_count=1, execution_count=2)
