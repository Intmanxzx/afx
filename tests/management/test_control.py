import json
import time
import pytest
from autofusex.management.central import ControlStore, Conflict
from autofusex.management.store import Store
from autofusex.fuzz.central.repository import create_user


@pytest.fixture
def control(tmp_path):
    path=tmp_path/'central.sqlite'
    alice=create_user(path,'Alice');bob=create_user(path,'Bob')
    store=ControlStore(path)
    pairing=store.pairing('docker')
    assert store.exchange(pairing['id'],pairing['secret'])['state']=='pending'
    store.approve(alice['user_id'],pairing['code'])
    credential=store.exchange(pairing['id'],pairing['secret'])
    return store,alice,bob,credential


def test_pairing_once_and_permissions(control):
    store,alice,bob,credential=control
    device=store.authenticate(credential['token'])
    assert device['user_id']==alice['user_id']
    assert store.devices(bob['user_id'])==[]
    with pytest.raises(PermissionError):store.operations(bob['user_id'],device['id'])
    with pytest.raises(PermissionError):store.authenticate('wrong')


def test_pairing_secret_expiration(control):
    store,*_=control
    p=store.pairing('new')
    with pytest.raises(PermissionError):store.exchange(p['id'],'wrong')
    with store.connect() as db:db.execute('UPDATE managed_pairings SET expires=0 WHERE id=?',(p['id'],))
    with pytest.raises(PermissionError):store.exchange(p['id'],p['secret'])


def test_operation_delivery_idempotency_fence(control):
    store,alice,bob,credential=control;device=store.authenticate(credential['token'])
    operation=store.submit(alice['user_id'],device['id'],'environment.inspect',{},'key')
    assert store.submit(alice['user_id'],device['id'],'environment.inspect',{},'key')==operation
    with pytest.raises(Conflict):store.submit(alice['user_id'],device['id'],'environment.apply',{'plan_id':'p'},'key')
    first=store.poll(device,{})['operations'][0]
    assert store.poll(device,{})['operations'][0]==first
    with pytest.raises(Conflict):store.ack(device,first['id'],99,'succeeded',{})
    store.ack(device,first['id'],first['generation'],'succeeded',{})
    store.ack(device,first['id'],first['generation'],'succeeded',{})
    with pytest.raises(Conflict):store.ack(device,first['id'],first['generation'],'failed',{})
    assert not store.poll(device,{})['operations']


def test_config_revision_conflict(control):
    store,alice,_,credential=control
    spec=dict(schema_version=1,kind='client_policy')
    assert store.configure(alice['user_id'],credential['device_id'],spec,0)['revision']==1
    with pytest.raises(Conflict):store.configure(alice['user_id'],credential['device_id'],spec,0)


def test_local_journal_recovery(tmp_path):
    store=Store(tmp_path/'home')
    op=store.submit('environment.inspect',{},'key')
    assert store.submit('environment.inspect',{},'key')['id']==op['id']
    with pytest.raises(ValueError):store.submit('environment.inspect',{'different':1},'key')
    store.state(op['id'],'running');store.recover()
    assert store.operation(op['id'])['state']=='needs_attention'
    for n in range(105):store.event(op['id'],str(n))
    assert len(store.events(op['id']))==100


def test_device_token_cannot_login_or_adopt_other_source(control):
    import uuid
    from autofusex.fuzz.central.repository import authenticate,authenticate_upload
    store,alice,_,credential=control
    with store.connect() as db:
        with pytest.raises(PermissionError):authenticate(db,credential['token'])
        source=str(uuid.uuid4())
        assert authenticate_upload(db,credential['token'],source)==alice['user_id']
    pair=store.pairing('second');store.approve(alice['user_id'],pair['code']);second=store.exchange(pair['id'],pair['secret'])
    with store.connect() as db:
        with pytest.raises(PermissionError):authenticate_upload(db,second['token'],source)


def test_pairing_exchange_retries_without_plaintext_secret(control):
    store,alice,_,_=control
    pair=store.pairing('retry');store.approve(alice['user_id'],pair['code'])
    first=store.exchange(pair['id'],pair['secret'])
    assert store.exchange(pair['id'],pair['secret'])==first
    with store.connect() as db:
        row=db.execute('SELECT * FROM managed_pairings WHERE id=?',(pair['id'],)).fetchone()
        assert row['token'] is None
        assert first['token'] not in dict(row).values()


def test_cancel_queued_and_running_operations(tmp_path):
    from autofusex.management.operations import execute
    store=Store(tmp_path/'home');target=store.submit('environment.inspect',{},'target')
    cancel=store.submit('operation.cancel',dict(operation_id=target['id']),'cancel')
    execute(store,cancel,{})
    assert store.operation(target['id'])['state']=='cancelled'
    target=store.submit('environment.inspect',{},'running');store.state(target['id'],'running')
    cancel=store.submit('operation.cancel',dict(operation_id=target['id']),'cancel2')
    execute(store,cancel,{})
    assert store.get('cancel:'+target['id']) is True
