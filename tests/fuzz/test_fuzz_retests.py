"""Isolated control-plane tests: no production identities, devices or network."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import copy
import json
import threading
import time
import uuid
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError

import pytest
from autofusex.fuzz.server.central import retests, sessions
from autofusex.fuzz.server.central.repository import create_user, ingest, rotate_key
from autofusex.fuzz.server.central.server import create_server
from autofusex.fuzz.client.storage.repository import ResultRepository, _connect
from autofusex.fuzz.client.sync.outbox import pending


def uid():
    return str(uuid.uuid4())


@pytest.fixture
def setup(tmp_path):
    database = tmp_path/'central.sqlite'
    alice = create_user(database, 'Alice')
    bob = create_user(database, 'Bob')
    with closing(_connect(database,writable=True)) as conn, conn:
        sessions.initialize(conn)
        retests.initialize(conn)
    local = ResultRepository(tmp_path/'local.sqlite')
    attempt = local.save(dict(seed=str(16076518010981881194532289465530122136584972140560),workload_id='a'*64,oracle_id='cpu/1',
        backend='af-mlir',status='fail',execution=dict(backend='af-mlir',static_only=False),
        evaluation=dict(correctness=True,performance=False,warmup_steps=1,measured_steps=1),
        shape_results=[dict(shape_index=i,correctness='fail' if i==1 else 'pass',performance='not_requested') for i in range(3)]))
    batch = pending(local.path)
    ingest(database,alice['token'],batch)
    # Bob's independently registered source is the execution client. It cannot
    # impersonate Alice's source even though it can retest her public result.
    client = uid()
    with closing(_connect(database,writable=True)) as conn,conn:
        conn.execute('INSERT INTO sources(id,user_id) VALUES (?,?)',(client,bob['user_id']))
    registration=dict(client_id=client,machine_id='machine-b',name='Bob NPU',backends=['af-mlir'])
    retests.register(database,bob['user_id'],registration)
    payload=dict(client_id=client,request_key=uid(),repetitions=1,
                 targets=[dict(attempt_id=attempt,shape_index=i) for i in (1,2)])
    return database,alice,bob,attempt,client,payload


def test_sessions_rotation_disable_logout(setup):
    database,alice,*_ = setup
    secret=sessions.issue(database,alice['token'])
    profile=sessions.identify(database,secret)
    assert profile['display_name']=='Alice'
    with pytest.raises(PermissionError):sessions.identify(database,secret,'wrong')
    assert sessions.identify(database,secret,profile['csrf'])==profile
    rotate_key(database,alice['user_id'])
    with pytest.raises(PermissionError):sessions.identify(database,secret)
    with pytest.raises(PermissionError):sessions.issue(database,'invalid')
    bob=setup[2]
    secret=sessions.issue(database,bob['token'])
    sessions.revoke(database,secret)
    with pytest.raises(PermissionError):sessions.identify(database,secret)
    secret=sessions.issue(database,bob['token'])
    with closing(_connect(database,writable=True)) as conn,conn:
        conn.execute('UPDATE users SET active=0 WHERE id=?',(bob['user_id'],))
    with pytest.raises(PermissionError):sessions.identify(database,secret)


def test_merge_shapes_cross_user_and_idempotency(setup):
    database,alice,bob,attempt,client,payload=setup
    answer=retests.create(database,bob['user_id'],payload)
    assert answer['task_count']==1
    assert retests.create(database,bob['user_id'],payload)['batch_id']==answer['batch_id']
    with closing(_connect(database)) as conn:
        assert conn.execute('SELECT shape_index,selected FROM retest_targets_v1 ORDER BY shape_index').fetchall()==[(0,0),(1,1),(2,1)]
        assert conn.execute('SELECT user_id FROM owners WHERE attempt_id=?',(attempt,)).fetchone()[0]==alice['user_id']
    changed=copy.deepcopy(payload);changed['repetitions']=2
    with pytest.raises(ValueError):retests.create(database,bob['user_id'],changed)
    with pytest.raises(PermissionError):retests.create(database,alice['user_id'],payload)
    assert retests.clients(database,alice['user_id'])['items']==[]


def test_bounded_concurrent_claims_and_lease_fencing(setup):
    database,alice,bob,attempt,client,payload=setup
    payload['repetitions']=10
    retests.create(database,bob['user_id'],payload)
    def claim():return retests.claim(database,bob['user_id'],dict(client_id=client,limit=8))['items']
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:claim(),range(2)))
    items=sum(results,[])
    assert len(items)==8 and len({i['task_id'] for i in items})==8
    assert claim()==[]
    old=items[0]
    with closing(_connect(database,writable=True)) as conn,conn:
        conn.execute('UPDATE retest_tasks_v1 SET expires=0 WHERE id=?',(old['task_id'],))
    new=claim()
    assert len(new)==1 and new[0]['task_id']==old['task_id'] and new[0]['generation']==2
    with pytest.raises(PermissionError):
        retests.renew(database,bob['user_id'],dict(client_id=client,lease_id=old['lease_id'],state='running'))
    with pytest.raises(PermissionError):
        retests.claim(database,alice['user_id'],dict(client_id=client,limit=1))
    assert retests.renew(database,bob['user_id'],dict(client_id=client,lease_id=new[0]['lease_id'],state='running'))['state']=='running'


def test_cancel_and_pagination(setup):
    database,alice,bob,attempt,client,payload=setup
    payload['repetitions']=2
    batch=retests.create(database,bob['user_id'],payload)['batch_id']
    lease=retests.claim(database,bob['user_id'],dict(client_id=client,limit=1))['items'][0]['lease_id']
    with pytest.raises(PermissionError):retests.cancel(database,alice['user_id'],batch)
    retests.cancel(database,bob['user_id'],batch)
    assert retests.renew(database,bob['user_id'],dict(client_id=client,lease_id=lease,state='running'))['state']=='cancel_requested'
    assert retests.renew(database,bob['user_id'],dict(client_id=client,lease_id=lease,state='cancelled'))['state']=='cancelled'
    assert retests.batches(database,bob['user_id'])['items'][0]['states']=={'cancelled':2}
    assert not retests.batches(database,alice['user_id'])['items']
    assert not retests.claim(database,bob['user_id'],dict(client_id=client,limit=1))['items']


@pytest.mark.parametrize('field,value', [('repetitions',True),('repetitions',11),('targets',[]),('targets',[{'attempt_id':'bad','shape_index':0}])])
def test_invalid_batch_is_atomic(setup,field,value):
    database,_,bob,_,_,payload=setup
    payload[field]=value
    with pytest.raises((ValueError,TypeError)):retests.create(database,bob['user_id'],payload)
    assert not retests.batches(database,bob['user_id'])['items']


def test_http_session_csrf_and_agent_routes(setup):
    database,alice,bob,attempt,client,payload=setup
    server=create_server(database,0,cache_interval=.1)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    opener=build_opener(ProxyHandler({})).open
    endpoint=f'http://127.0.0.1:{server.server_port}'
    # Simulate the trusted TLS reverse proxy preserving Host. Secure cookies are
    # explicitly supplied only in this loopback fixture, never production HTTP.
    headers={'Origin':'https://test.example','Host':'test.example','Content-Type':'application/json'}
    def call(route,method='GET',body=None,extra=None):
        return opener(Request(endpoint+route,data=None if body is None else json.dumps(body).encode(),
                              method=method,headers={**headers,**(extra or {})}),timeout=5)
    try:
        with pytest.raises(HTTPError) as error:call('/api/session')
        assert error.value.code==401
        with pytest.raises(HTTPError):call('/api/session','POST',{'token':bob['token']},{'Origin':'https://evil.example'})
        with call('/api/session','POST',{'token':bob['token']}) as response:
            cookie=response.headers['Set-Cookie'];profile=json.load(response)
        assert 'HttpOnly' in cookie and 'Secure' in cookie and 'SameSite=Strict' in cookie
        auth={'Cookie':cookie.split(';')[0],'X-CSRF-Token':profile['csrf']}
        with call('/api/session',extra=auth) as response:assert json.load(response)['user_id']==bob['user_id']
        with pytest.raises(HTTPError) as error:call('/api/issues?dimension=accuracy')
        assert error.value.code==401
        with call('/api/issues?dimension=accuracy',extra=auth) as response:
            assert json.load(response)['dimension']=='accuracy'
        with pytest.raises(HTTPError):call('/api/retest/batches','POST',payload,{'Cookie':auth['Cookie']})
        with call('/api/retest/batches','POST',payload,auth) as response:assert json.load(response)['task_count']==1
        with call('/api/retest/claim','POST',dict(client_id=client,limit=1),{'Authorization':'Bearer '+bob['token']}) as response:
            assert len(json.load(response)['items'])==1
        with call('/api/session','DELETE',extra=auth) as response:assert json.load(response)['signed_out']
        with pytest.raises(HTTPError):call('/api/session',extra=auth)
    finally:
        server.shutdown();server.server_close();thread.join()


def test_pass_target_and_full_rollback_on_bad_shape(setup):
    database,_,bob,attempt,client,payload=setup
    # S0 was already Pass; it has exactly the same retest permissions as S1.
    payload['targets']=[dict(attempt_id=attempt,shape_index=0)]
    assert retests.create(database,bob['user_id'],payload)['task_count']==1
    bad=copy.deepcopy(payload);bad['request_key']=uid()
    bad['targets'].append(dict(attempt_id=attempt,shape_index=999))
    with pytest.raises(ValueError):retests.create(database,bob['user_id'],bad)
    assert len(retests.batches(database,bob['user_id'])['items'])==1


def test_expiry_retry_limit_and_restart_preserve_state(setup):
    database,_,bob,_,client,payload=setup
    batch=retests.create(database,bob['user_id'],payload)['batch_id']
    for generation in range(1,retests.MAX_GENERATIONS+1):
        item=retests.claim(database,bob['user_id'],dict(client_id=client,limit=1))['items'][0]
        assert item['generation']==generation
        with closing(_connect(database,writable=True)) as conn,conn:
            conn.execute('UPDATE retest_tasks_v1 SET expires=0 WHERE id=?',(item['task_id'],))
            # Additive initialization is safe after a process restart.
            retests.initialize(conn)
            sessions.initialize(conn)
    assert not retests.claim(database,bob['user_id'],dict(client_id=client,limit=1))['items']
    assert retests.batches(database,bob['user_id'])['items'][0]['states']=={'interrupted':1}


def test_registration_cannot_take_over_source_or_move_host(setup):
    database,alice,bob,_,client,_=setup
    request=dict(client_id=client,machine_id='machine-b',name='NPU',backends=['af-mlir'])
    with pytest.raises(PermissionError):retests.register(database,alice['user_id'],request)
    request['machine_id']='different-machine'
    with pytest.raises(PermissionError):retests.register(database,bob['user_id'],request)
    request['client_id']=uid()
    assert retests.register(database,bob['user_id'],request)['client_id']==request['client_id']
    with pytest.raises(PermissionError):retests.register(database,alice['user_id'],request)


def test_offline_client_and_limits(setup):
    database,_,bob,_,client,payload=setup
    for invalid in (True,0,9,-1,'8'):
        with pytest.raises(ValueError):retests.claim(database,bob['user_id'],dict(client_id=client,limit=invalid))
    with closing(_connect(database,writable=True)) as conn,conn:
        conn.execute('UPDATE retest_clients_v1 SET seen=0 WHERE id=?',(client,))
    with pytest.raises(ValueError):retests.create(database,bob['user_id'],payload)


def test_login_limit_is_bounded_and_session_expiration(setup):
    from autofusex.fuzz.server.central.control_api import ControlAPI
    database,alice,*_=setup
    api=ControlAPI(database)
    assert all(api._login_limit('loopback') for _ in range(10))
    assert not api._login_limit('loopback')
    for i in range(1100):api._login_limit(str(i))
    assert len(api.login_attempts)==1024
    secret=sessions.issue(database,alice['token'])
    with closing(_connect(database,writable=True)) as conn,conn:
        conn.execute('UPDATE browser_sessions_v1 SET expires=0')
    with pytest.raises(PermissionError):sessions.identify(database,secret)


def test_summary_slot_is_nullable_and_private_snapshot_does_not_export_sessions(setup,tmp_path):
    database,alice,*_=setup
    sessions.issue(database,alice['token'])
    snapshot=ResultRepository(database).snapshot(tmp_path/'public.sqlite')
    import sqlite3
    with sqlite3.connect(snapshot) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'browser_sessions%' OR name LIKE 'retest_%'").fetchall()==[]
    with closing(_connect(database)) as conn:
        columns={row[1]:row for row in conn.execute('PRAGMA table_info(retest_result_annotations_v1)')}
        assert columns['summary_id'][3]==0


def test_batch_detail_owned_paginated_and_no_lease_leak(setup):
    database,alice,bob,_,client,payload=setup
    batch=retests.create(database,bob['user_id'],payload)['batch_id']
    lease=retests.claim(database,bob['user_id'],dict(client_id=client,limit=1))['items'][0]
    page=retests.detail(database,bob['user_id'],batch)
    assert page['items'][0]['state']=='leased'
    assert len(page['items'][0]['targets'])==3
    assert page['items'][0]['summaries']==[]
    assert lease['lease_id'] not in json.dumps(page)
    assert retests.detail(database,bob['user_id'],batch,page['items'][0]['sequence'])['items']==[]
    with pytest.raises(PermissionError):retests.detail(database,alice['user_id'],batch)


def test_preview_is_read_only_and_reports_merged_count(setup):
    database,_,bob,_,_,payload=setup
    payload['repetitions']=2
    assert retests.create(database,bob['user_id'],payload,preview=True)=={'task_count':2,'selected_count':2}
    assert retests.batches(database,bob['user_id'])['items']==[]
