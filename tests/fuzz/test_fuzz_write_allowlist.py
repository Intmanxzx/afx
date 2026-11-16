"""Maintenance admission, authentication, ACK preservation and recovery over HTTP."""
from contextlib import closing, contextmanager
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

import pytest
from test_fuzz_retests import setup, uid
from autofusex.fuzz.server.central.maintenance import WriteAllowlist, MaintenanceBlocked, ENVIRONMENT
from autofusex.fuzz.server.central.server import create_server
from autofusex.fuzz.server.central import sessions, retests
from autofusex.fuzz.client.storage.repository import _connect, ResultRepository
from autofusex.fuzz.client.sync.client import once
from autofusex.fuzz.client.sync.outbox import pending


def policy(tmp_path, monkeypatch, bob, client):
    path=tmp_path/'allowlist.json'
    value=dict(enabled=True,users=[bob['user_id']],clients=[dict(user_id=bob['user_id'],client_id=client)])
    path.write_text(json.dumps(value))
    monkeypatch.setenv(ENVIRONMENT,str(path))
    return path


@contextmanager
def running(database):
    server=create_server(database,0,cache_interval=.1)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    opener=build_opener(ProxyHandler({})).open
    base='http://127.0.0.1:'+str(server.server_port)
    def call(route,body=None,token=None,headers=None,method=None):
        h={'Content-Type':'application/json','Origin':'https://127.0.0.1:'+str(server.server_port)}
        if token:h['Authorization']='Bearer '+token
        h.update(headers or {})
        req=Request(base+route,None if body is None else json.dumps(body).encode(),headers=h,method=method or ('POST' if body is not None else 'GET'))
        try:
            response=opener(req,timeout=10)
        except HTTPError as e:response=e
        with response:return response.status,dict(response.headers),json.load(response)
    try:yield call,base,opener
    finally:server.shutdown();server.server_close();thread.join()



def test_issue_retest_respects_browser_and_client_allowlist(setup,tmp_path,monkeypatch):
    from autofusex.fuzz.server.central import issue_groups
    database,alice,bob,attempt,client,_=setup
    policy(tmp_path,monkeypatch,bob,client)
    from test_fuzz_issue_groups import install_rows
    install_rows(database,[dict(
        seed='1',shape_index=1,id=attempt,status='fail',function_status='Unknown',
        accuracy_status='Failed',core_error='mismatch',error_summary='mismatch',
        correctness='fail',performance='not_requested',measurement_status='not_requested',
        op_types=['aten.matmul'])])
    key=issue_groups.list_groups(database,'accuracy')['items'][0]['issue_key']
    issue_groups.claim(database,bob['user_id'],key)
    payload=dict(issue_key=key,client_id=client,repetitions=1)
    with running(database) as (call,_,__):
        _,headers,profile=call('/api/session',dict(token=alice['token']))
        alice_auth={'Cookie':headers['Set-Cookie'].split(';')[0],'X-CSRF-Token':profile['csrf']}
        _,headers,profile=call('/api/session',dict(token=bob['token']))
        bob_auth={'Cookie':headers['Set-Cookie'].split(';')[0],'X-CSRF-Token':profile['csrf']}
        assert call('/api/issues/retest',payload,headers=alice_auth)[0]==503
        assert call('/api/issues/retest',dict(payload,client_id=uid()),headers=bob_auth)[0]==503

def test_config_strict_and_disabled(tmp_path,monkeypatch):
    monkeypatch.delenv(ENVIRONMENT,raising=False)
    WriteAllowlist.environment().client('any','any')
    with pytest.raises(OSError):WriteAllowlist.load(tmp_path/'absent')
    path=tmp_path/'policy'
    for value in ({},dict(enabled='false',users=[],clients=[]),dict(enabled=True,users=['bad'],clients=[]),
                  dict(enabled=True,users=[],clients=[dict(user_id=uid(),client_id=uid())])):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):WriteAllowlist.load(path)
    path.write_text(json.dumps(dict(enabled=True,users=[],clients=[])))
    with pytest.raises(MaintenanceBlocked):WriteAllowlist.load(path).client(uid(),uid())
    monkeypatch.setenv(ENVIRONMENT,str(tmp_path/'missing'))
    with pytest.raises(OSError):create_server(tmp_path/'not-created',0)
    assert not (tmp_path/'not-created').exists()


def test_http_all_write_routes_block_without_mutations(setup,tmp_path,monkeypatch):
    database,alice,bob,attempt,client,payload=setup
    policy(tmp_path,monkeypatch,bob,client)
    batch=retests.create(database,bob['user_id'],payload)['batch_id']
    job=retests.claim(database,bob['user_id'],dict(client_id=client,limit=1))['items'][0]
    with running(database) as (call,_,__):
        def snapshot():
            with closing(_connect(database)) as conn:
                return {t:conn.execute('SELECT * FROM '+t).fetchall() for t in
                        ('attempts','sources','receipts','retest_tasks_v1','retest_clients_v1','retest_completions_v1')}
        before=snapshot()
        bodies={
            '/api/upload':dict(source_id=client), '/api/facts':dict(source_id=client),
            '/api/heartbeat':dict(source_id=client),
            '/api/retest/register':dict(client_id=client), '/api/retest/claim':dict(client_id=client,limit=1),
            '/api/retest/renew':dict(client_id=client,lease_id=job['lease_id'],state='running'),
            '/api/retest/complete':dict(client_id=client,lease_id=job['lease_id'],attempt_id=attempt)}
        for route,body in bodies.items():
            # Other user cannot bypass by claiming the allowed client ID.
            status,h,data=call(route,body,alice['token'])
            assert status==503 and h['Retry-After']=='60' and data['code']=='maintenance_write_blocked'
            # Same user, another client, is also blocked.
            key='client_id' if route.startswith('/api/retest/') else 'source_id'
            assert call(route,dict(body,**{key:uid()}),bob['token'])[0]==503
        assert snapshot()==before
        assert call('/api/upload',{},'not-a-valid-token')[0]==401
        assert call('/healthz')[0]==200
        assert call('/api/case-history?seed='+str(16076518010981881194532289465530122136584972140560)+'&shape=1')[0]==200
        assert call('/api/retest/renew',bodies['/api/retest/renew'],bob['token'])[0]==200
        registration=dict(client_id=client,machine_id='machine-b',name='Bob NPU',backends=['af-mlir'])
        assert call('/api/retest/register',registration,bob['token'])[0]==200
        assert call('/api/retest/claim',dict(client_id=client,limit=1),bob['token'])[0]==200
        # Login/read/logout are available even to users excluded from writes.
        status,h,profile=call('/api/session',dict(token=alice['token']))
        assert status==200
        auth={'Cookie':h['Set-Cookie'].split(';')[0],'X-CSRF-Token':profile['csrf']}
        assert call('/api/session',headers=auth)[0]==200
        for route in ('/api/retest/preview','/api/retest/batches','/api/retest/batches/'+batch+'/cancel'):
            assert call(route,payload if not route.endswith('/cancel') else {},headers=auth)[0]==503
        assert call('/api/session',headers=auth,method='DELETE')[0]==200
        secret=sessions.issue(database,bob['token']);profile=sessions.identify(database,secret)
        auth={'Cookie':sessions.COOKIE+'='+secret,'X-CSRF-Token':profile['csrf']}
        assert call('/api/retest/preview',payload,headers=auth)[0]==200
        assert call('/api/retest/preview',dict(payload,client_id=uid()),headers=auth)[0]==503
        assert call('/api/retest/batches/'+batch+'/cancel',{},headers=auth)[0]==200


def test_upload_retry_preserves_ack_then_recovers(setup,tmp_path,monkeypatch):
    database,alice,bob,_,client,_=setup
    path=policy(tmp_path,monkeypatch,bob,client)
    home=tmp_path/'worker';home.mkdir()
    key=home/'token';key.write_text(alice['token']);key.chmod(0o600)
    repo=ResultRepository(home/'results')
    from test_fuzz_case_summary import save
    save(repo,'ordinary','af-mlir','ok')
    before=pending(repo.path)
    with running(database) as (_,base,opener):
        (home/'sync.json').write_text(json.dumps(dict(endpoint=base,key_file=str(key),interval=1)))
        with pytest.raises(HTTPError) as error:once(home,repo.path,opener=opener)
        assert error.value.code==503
        assert pending(repo.path)==before
        with closing(_connect(database)) as conn:
            assert conn.execute('SELECT 1 FROM sources WHERE id=?',(before['source_id'],)).fetchone() is None
        # Config is a startup snapshot: editing it cannot partially reopen a live server.
        path.write_text(json.dumps(dict(enabled=False,users=[],clients=[])))
        with pytest.raises(HTTPError):once(home,repo.path,opener=opener)
    with running(database) as (_,base,opener):
        (home/'sync.json').write_text(json.dumps(dict(endpoint=base,key_file=str(key),interval=1)))
        assert once(home,repo.path,opener=opener)['uploaded']==1
        assert once(home,repo.path,opener=opener)['pending']==0


def test_allowed_result_facts_and_heartbeat(setup,tmp_path,monkeypatch):
    database,_,bob,_,client,_=setup
    policy(tmp_path,monkeypatch,bob,client)
    local=ResultRepository(tmp_path/'allowed')
    from test_fuzz_case_summary import save
    save(local,'allowed','af-mlir','ok')
    package=pending(local.path);package['source_id']=client
    with running(database) as (call,_,__):
        assert call('/api/upload',package,bob['token'])[0]==200
        assert call('/api/upload',package,bob['token'])[2]['ack']==1
        from autofusex.fuzz.client.storage.lifecycle import stamp
        facts=dict(version=1,source_id=client,items=[dict(revision=1,kind='lifecycle',
            identity=package['items'][0]['attempt_id']+':admitted',payload=stamp())])
        assert call('/api/facts',facts,bob['token'])[0]==200
        heartbeat=dict(version=1,machine_id='machine-b',machine_name='Bob NPU',source_id=client,runs=[])
        assert call('/api/heartbeat',heartbeat,bob['token'])[0]==200


def test_consumer_503_does_not_retire_or_cancel_lease(tmp_path,monkeypatch):
    from autofusex.fuzz.client.sync.retests import Consumer, accept, initialize
    from autofusex.fuzz.client.scheduling.budgets import Budget
    from autofusex.fuzz.client.config import setup as configure
    from test_fuzz_retest_consumer import lease
    from autofusex.fuzz.client.scheduling import hosting
    home=tmp_path/'consumer';configure(home,cpu_workers=1,compile_threads=1)
    consumer=Consumer(home,home/'results/results.sqlite',devices=['npu:0'])
    initialize(consumer.ledger)
    item=lease();identity=accept(consumer.ledger,item,['npu:0'],Budget())
    monkeypatch.setattr(hosting,'status',lambda *a:dict(ownership='owned'))
    def blocked(route,payload):
        if route=='register':return {}
        raise HTTPError('http://test',503,'maintenance',{'Retry-After':'60'},None)
    monkeypatch.setattr(consumer,'api',blocked)
    with pytest.raises(HTTPError):consumer.tick()
    assert consumer.ledger.get(identity)['state']=='pending'
    with consumer.ledger._connection() as conn:
        assert conn.execute('SELECT finished FROM remote_retests_v1').fetchone()[0]==0


def test_allowed_browser_create_and_client_complete(setup,tmp_path,monkeypatch):
    database,_,bob,_,client,payload=setup
    policy(tmp_path,monkeypatch,bob,client)
    from test_fuzz_retest_completion import upload
    with running(database) as (call,_,__):
        secret=sessions.issue(database,bob['token']);profile=sessions.identify(database,secret)
        auth={'Cookie':sessions.COOKIE+'='+secret,'X-CSRF-Token':profile['csrf']}
        assert call('/api/retest/batches',payload,headers=auth)[0]==200
        job=call('/api/retest/claim',dict(client_id=client,limit=1),bob['token'])[2]['items'][0]
        record=upload(setup,tmp_path,job)
        status,_,answer=call('/api/retest/complete',record,bob['token'])
        assert status==200 and answer['accepted']
