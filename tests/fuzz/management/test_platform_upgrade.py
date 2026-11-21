import json
from types import SimpleNamespace
from autofusex.fuzz.daemon.management import platform_sync
from autofusex.fuzz.daemon.management.store import Store
from autofusex.fuzz.daemon.management.modern_cli import run


def test_new_homes_have_distinct_persistent_identities(tmp_path):
    from autofusex.fuzz.client.config import load_config
    homes = [tmp_path/'one', tmp_path/'two']
    for home in homes:
        run(['--data-dir', str(home), 'init', '--local'])
    assert load_config(homes[0])['source_id'] != load_config(homes[1])['source_id']
    existing = tmp_path/'existing'
    existing.mkdir()
    config = load_config(homes[0])
    (existing/'config.json').write_text(json.dumps(config))
    run(['--data-dir', str(existing), 'init', '--local'])
    assert load_config(existing) == config


def test_active_version_and_update_request_retry(tmp_path, monkeypatch):
    from autofusex.fuzz.client import environment
    from autofusex.fuzz.client.scheduling import devices
    monkeypatch.setattr(environment, 'collect', lambda home: {'installation_id':'test', 'package_version':'0.2.2'})
    monkeypatch.setattr(devices, 'device_inventory_info', lambda: {'devices': [], 'model': None})
    store = Store(tmp_path)
    (tmp_path/'client.json').write_text('{}')
    (tmp_path/'active-client.json').write_text(json.dumps({'version':'0.2.0'}))
    beats = []
    reply = {'update_to':'0.2.1', 'update_request_id':'first'}
    class API:
        def __init__(self, *a, **kw): pass
        def call(self, method, route, body=None, **kw):
            if route.endswith('heartbeat'):
                beats.append(body)
                return reply
            return {'config':{}, 'revision':'test', 'source':'default'}
    monkeypatch.setattr(platform_sync, 'API', API)
    manager = SimpleNamespace(store=store)
    client = {'endpoint':'https://example.invalid', 'token':'test'}
    platform_sync.synchronize(manager, client)
    first = store.pending()[0]
    store.state(first['id'], 'succeeded', {})
    platform_sync.synchronize(manager, client)
    assert not store.pending()  # repeated heartbeat must not repeat installation
    reply['update_request_id'] = 'second'
    platform_sync.synchronize(manager, client)
    assert store.pending()[0]['id'] != first['id']
    assert all(b['afx_version']=='0.2.0' for b in beats)

