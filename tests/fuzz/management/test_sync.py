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
