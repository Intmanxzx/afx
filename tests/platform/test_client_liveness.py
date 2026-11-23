"""Uploads must not impersonate the live control agent."""
from test_platform_ops import world
from autofusex.platform import ops
from autofusex.platform.ingest import Ingestor, upsert_client
from autofusex.platform.store import now_ms


def test_upload_activity_does_not_keep_client_online_or_restore_deleted(world):
    store, user, admin = world
    cid = ops.heartbeat(store, user, {'machine_id': 'installation', 'afx_version': '0.2.3',
                                     'hostname': 'live-host', 'npu_arch': 'live-chip'})['client_id']
    with store.write() as conn:
        conn.execute('UPDATE clients SET last_heartbeat_at=? WHERE client_id=?',
                     (now_ms() - ops.ONLINE_MS - 1000, cid))
        Ingestor(store)._client(conn, user.user_id, {'machine_id': 'installation',
                                                  'afx_version': '0.2.1', 'hostname': 'old-host', 'npu_arch': 'old-chip'})
    row = next(c for c in ops.summary(store)['clients'] if c['client_id'] == cid)
    assert not row['online']
    assert row['afx_version'] == '0.2.3'
    assert row['hostname'] == 'live-host' and row['npu_arch'] == 'live-chip'
    assert now_ms() - row['last_seen_at'] < 5000
    ops.delete_client(store, admin, cid)
    with store.write() as conn:
        assert upsert_client(conn, user.user_id, {'machine_id': 'installation'}) == cid
    assert cid not in [c['client_id'] for c in ops.summary(store)['clients']]
    assert ops.heartbeat(store, user, {'machine_id': 'installation'})['client_id'] == cid
    assert next(c for c in ops.summary(store)['clients'] if c['client_id'] == cid)['online']


def test_upload_only_identity_is_offline(world):
    store, user, _ = world
    with store.write() as conn:
        cid = upsert_client(conn, user.user_id, {'machine_id': 'legacy-upload'})
    row = next(c for c in ops.summary(store)['clients'] if c['client_id'] == cid)
    assert row['last_heartbeat_at'] is None and not row['online']


def test_control_identity_does_not_read_archived_results(tmp_path):
    from autofusex.platform.client import Client
    client = object.__new__(Client)
    # A non-database archive proves identity never opens historical results.
    archive = tmp_path / 'archive.sqlite'
    archive.write_text('historical data is not control identity')
    client.results_db = str(archive)
    client.env = {'installation_id': 'current-installation', 'package_version': '0.2.3'}
    assert client.identity() == client.env
