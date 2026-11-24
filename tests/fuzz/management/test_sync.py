from autofusex.client.daemon import Manager


def test_retired_control_credentials_do_not_trigger_network(tmp_path):
    manager = Manager(tmp_path)
    manager.store.put('credential', {'endpoint':'https://obsolete.invalid','token':'old'})
    manager.store.put('pairing', {'endpoint':'https://obsolete.invalid'})
    try:
        assert manager.synchronize() is None
    finally:
        manager.pool.shutdown()


def test_platform_sync_module_imports():
    from autofusex.client import platform_sync
    assert callable(platform_sync.synchronize)


def test_restart_sync_uses_server_devices_and_parameters(tmp_path, monkeypatch):
    import json
    from autofusex.client import platform_sync, execution_runtime, source_jobs
    from autofusex.client.artifacts import atomic
    from autofusex.execution import environment
    from autofusex.execution.scheduling import devices
    manager = Manager(tmp_path)
    client = dict(mode='managed', endpoint='https://example.invalid', token='fixture', devices=['npu:0'])
    atomic(tmp_path/'client.json', client)
    from autofusex.client.modern_cli import _default_task
    config = _default_task()
    config['execution']['devices']=['npu:1']
    config['workload']['generation']['max_nodes']=23
    class API:
        def __init__(self, *args, **kwargs): pass
        def call(self, method, path, body=None, **kwargs):
            if path == '/api/client/heartbeat':
                return dict(daemon_state='restart', daemon_request_id='restart-new-config')
            assert path == '/api/config/me'
            return dict(config=config, revision='new', source='personal')
    monkeypatch.setattr(platform_sync, 'API', API)
    monkeypatch.setattr(environment, 'collect', lambda *args: dict(installation_id='fixture'))
    monkeypatch.setattr(devices, 'device_inventory_info', lambda: dict(devices=['npu:0','npu:1'], model='test'))
    monkeypatch.setattr(execution_runtime, 'facts', lambda *args: None)
    monkeypatch.setattr(source_jobs, 'synchronize', lambda *args: None)
    try:
        platform_sync.synchronize(manager, client)
        task=json.loads((tmp_path/'task.json').read_text())
        assert task['execution']['devices']==['npu:1']
        assert task['workload']['generation']['max_nodes']==23
        assert json.loads((tmp_path/'daemon-control.json').read_text())['request_id']=='restart-new-config'
        assert not (tmp_path/'trust.json').exists()
    finally:
        manager.pool.shutdown()
