import sqlite3
import uuid

import autofusex.client as daemon
from autofusex.client import daemon as runtime
from autofusex.client.central import ControlStore


def create_user(path, display_name):
    """Minimal local stand-in for the owner identity ControlStore expects (see test_control.py)."""
    identity = str(uuid.uuid4())
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, display_name TEXT, active INTEGER NOT NULL DEFAULT 1)')
        db.execute('INSERT INTO users(id,display_name) VALUES (?,?)', (identity, display_name))
    return {'user_id': identity, 'display_name': display_name}


def test_manager_center_flow_retries_lost_ack_without_duplicate_effect(tmp_path,monkeypatch):
    central_path=tmp_path/'central.sqlite';user=create_user(central_path,'Docker')
    center=ControlStore(central_path);pair=center.pairing('fixture');center.approve(user['user_id'],pair['code'])
    credential=center.exchange(pair['id'],pair['secret'])
    manager=daemon.Manager(tmp_path/'client')
    manager.store.put('credential',dict(endpoint='https://fixture.invalid',**credential))
    identity=center.submit(user['user_id'],credential['device_id'],'environment.inspect',{},'once')['id']
    calls={'ack':0}
    def transport(endpoint,route,payload,token=None):
        device=center.authenticate(token)
        if route=='poll':return center.poll(device,payload['status'])
        if route=='ack':
            result=center.ack(device,payload['id'],payload['generation'],payload['state'],payload['result'])
            calls['ack']+=1
            if calls['ack']==1:raise OSError('simulated dropped ACK response')
            return result
        raise AssertionError(route)
    monkeypatch.setattr(runtime,'remote',transport)
    manager.synchronize();manager.synchronize()
    with manager.store.connect() as db:assert db.execute('SELECT count(*) FROM operations').fetchone()[0]==1
    manager.store.state(identity,'succeeded',dict(state='fixture-ok',secret='must not upload'))
    try:manager.synchronize()
    except OSError:pass
    assert manager.store.get('remote_operations')
    manager.synchronize()
    assert not manager.store.get('remote_operations')
    result=center.operations(user['user_id'],credential['device_id'])[0]
    assert result['result']==dict(state='fixture-ok')
    manager.pool.shutdown()


def test_platform_sync_module_imports():
    from autofusex.client import platform_sync

    assert callable(platform_sync.synchronize)
