from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import uuid

import pytest
from test_fuzz_retests import setup
from autofusex.fuzz.central import repairs,issue_groups,retests
from autofusex.fuzz.management.central import Conflict
from autofusex.fuzz.storage.repository import _connect


def request(**values):return dict(request_key=str(uuid.uuid4()),**values)


def claim(database,user,agent='agent'):
    return repairs.mutate(database,user,'claim',request(agent_id=agent))


def test_exclusive_same_user_agents_and_retries(setup):
    database,_,bob,*_=setup
    payload=request(agent_id='one')
    first=repairs.mutate(database,bob['user_id'],'claim',payload)
    assert first and repairs.mutate(database,bob['user_id'],'claim',payload)==first
    def attempt(n):return claim(database,bob['user_id'],str(n))
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert all(x is None for x in pool.map(attempt,range(6)))
    assert claim(database,bob['user_id'],'one')['id']==first['id']
    with pytest.raises(PermissionError):issue_groups.claim(database,bob['user_id'],first['issue_key'])
    with pytest.raises(PermissionError):issue_groups.release(database,bob['user_id'],first['issue_key'])
    with pytest.raises(Conflict):repairs.mutate(database,bob['user_id'],'claim',dict(payload,agent_id='changed'))


def test_committed_claim_retry_does_not_rebuild_current_issues(setup, monkeypatch):
    database,_,bob,*_=setup
    payload=request(agent_id='retry')
    first=repairs.mutate(database,bob['user_id'],'claim',payload)
    def unavailable(*args, **kwargs):
        raise RuntimeError('current issue projection unavailable')
    monkeypatch.setattr(issue_groups,'_current',unavailable)
    assert repairs.mutate(database,bob['user_id'],'claim',payload)==first
    with pytest.raises(Conflict):
        repairs.mutate(database,bob['user_id'],'claim',dict(payload,agent_id='changed'))


def test_expiration_fences_old_owner_and_preserves_snapshot(setup):
    database,alice,bob,*_=setup
    first=claim(database,bob['user_id'])
    with closing(_connect(database,writable=True)) as db,db:
        db.execute('UPDATE repairs_v1 SET expires=0 WHERE id=?',(first['id'],))
    second=claim(database,alice['user_id'],'two')
    assert second['id']==first['id'] and second['lease_id']!=first['lease_id']
    assert second['snapshot']==first['snapshot']
    with pytest.raises(PermissionError):repairs.mutate(database,bob['user_id'],'renew',request(lease_id=first['lease_id']),first['id'])
    with pytest.raises(Conflict):repairs.mutate(database,alice['user_id'],'renew',request(lease_id=first['lease_id']),first['id'])


def test_atomic_retest_and_summary_requires_evidence(setup):
    database,alice,bob,attempt,client,_=setup
    repair=claim(database,bob['user_id'])
    summary=request(lease_id=repair['lease_id'],background='CPU mismatch',solution='fixed lowering',notes='tested')
    with pytest.raises(Conflict):repairs.mutate(database,bob['user_id'],'summary',summary,repair['id'])
    body=request(lease_id=repair['lease_id'],client_id=client,artifact=dict(sha256='a'*64,version='candidate',installed_at='2026-11-15T00:00:00Z'))
    tested=repairs.mutate(database,bob['user_id'],'retest',body,repair['id'])
    assert tested['state']=='retesting'
    assert repairs.mutate(database,bob['user_id'],'retest',body,repair['id'])==tested
    with pytest.raises(Conflict):repairs.mutate(database,bob['user_id'],'retest',dict(body,request_key=str(uuid.uuid4())),repair['id'])
    # Completion alone with a failed attempt is not verification.
    batch=tested['retest']['batch_ids'][0]
    with closing(_connect(database,writable=True)) as db,db:
        task=db.execute('SELECT id FROM retest_tasks_v1 WHERE batch_id=?',(batch,)).fetchone()[0]
        db.execute("UPDATE retest_tasks_v1 SET state='completed',attempt_id=? WHERE id=?",(attempt,task))
    assert not repairs.get(database,bob['user_id'],repair['id'])['retest']['passed']
    assert repairs.get(database,bob['user_id'],repair['id'])['state']=='fixing'
    with pytest.raises(Conflict):repairs.mutate(database,bob['user_id'],'summary',summary,repair['id'])


@pytest.mark.parametrize("performance", [False, True])
def test_verified_summary_is_unique_and_linked(setup, performance):
    from autofusex.fuzz.storage.repository import ResultRepository
    from autofusex.fuzz.sync.outbox import pending
    from autofusex.fuzz.central.repository import ingest
    database,alice,bob,attempt,client,_=setup
    if performance:
        with closing(_connect(database,writable=True)) as db,db:
            evaluation_id=db.execute('SELECT evaluation_id FROM attempts WHERE id=?',(attempt,)).fetchone()[0]
            evaluation=json.loads(db.execute('SELECT payload FROM contexts WHERE id=?',(evaluation_id,)).fetchone()[0])
            evaluation['performance']=True
            db.execute('UPDATE contexts SET payload=? WHERE id=?',(json.dumps(evaluation),evaluation_id))
    repair=claim(database,bob['user_id'])
    body=request(lease_id=repair['lease_id'],client_id=client,artifact=dict(sha256='a'*64,version='candidate',installed_at='2026-11-15T00:00:00Z'))
    tested=repairs.mutate(database,bob['user_id'],'retest',body,repair['id'])
    lease=retests.claim(database,bob['user_id'],dict(client_id=client,limit=1))['items'][0]
    # Build actual bound uploaded result via the same consumer completion protocol.
    spec=lease['spec']
    local=ResultRepository(database.parent/'repaired.sqlite')
    record=dict(seed=spec['seed'],workload_id=spec['workload_id'],oracle_id=spec['oracle_id'],backend='af-mlir',status='ok',
                execution=dict(backend='af-mlir',static_only=spec['execution']['static_only'],batch_id=lease['lease_id']),
                evaluation=spec['evaluation'],shape_results=[dict(shape_index=i,correctness='pass',performance='measured' if performance else 'not_requested') for i in range(3)])
    identity=local.save(record)
    payload=pending(local.path);payload['source_id']=client
    ingest(database,bob['token'],payload)
    result=retests.complete(database,bob['user_id'],dict(client_id=client,lease_id=lease['lease_id'],attempt_id=identity))
    assert result['accepted']
    assert repairs.get(database,bob['user_id'],repair['id'])['state']=='verified'
    summary=request(lease_id=repair['lease_id'],background='mismatch',solution='fixed',notes='candidate')
    answer=repairs.mutate(database,bob['user_id'],'summary',summary,repair['id'])
    assert answer['state']=='resolved'
    assert repairs.mutate(database,bob['user_id'],'summary',summary,repair['id'])==answer
    with pytest.raises(Conflict):repairs.mutate(database,bob['user_id'],'summary',dict(summary,request_key=str(uuid.uuid4())),repair['id'])
    with closing(_connect(database)) as db:
        assert db.execute('SELECT count(*) FROM repair_summaries_v1').fetchone()[0]==1
        assert db.execute('SELECT DISTINCT summary_id FROM retest_result_annotations_v1').fetchone()[0]==answer['summary']['id']


def test_http_repair_claim_renew_release_and_experience(setup):
    import threading
    from urllib.request import Request,build_opener,ProxyHandler
    from urllib.error import HTTPError
    from autofusex.fuzz.central.server import create_server
    database,_,bob,*_=setup
    server=create_server(database,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    opener=build_opener(ProxyHandler({}))
    def call(route,body=None):
        req=Request('http://127.0.0.1:'+str(server.server_port)+'/api/agent/'+route,
                    json.dumps(body).encode() if body else None,
                    headers={'Content-Type':'application/json','Authorization':'Bearer '+bob['token']})
        try:
            with opener.open(req,timeout=5) as response:return response.status,json.load(response)['data']
        except HTTPError as error:return error.code,json.load(error)
    try:
        code,value=call('repairs/claim',request(agent_id='http'))
        assert code==200 and value['lease_id']
        identity=value['id'];lease=value['lease_id']
        assert call('repairs/'+identity)[1]['state']=='claimed'
        assert call('repairs/'+identity+'/renew',request(lease_id=lease))[0]==200
        assert call('repairs/clients')[1]['items']
        assert call('experience')[1]['items']==[]
        assert call('repairs/'+identity+'/summary',request(lease_id=lease,background='b',solution='s'))[0]==409
        assert call('repairs/'+identity+'/release',request(lease_id=lease))[0]==200
        assert call('repairs/'+identity+'/renew',request(lease_id=lease))[0]==409
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)


def test_retest_transaction_rolls_back_all_chunks(setup,monkeypatch):
    database,_,bob,_,client,_=setup
    repair=claim(database,bob['user_id'])
    with closing(_connect(database,writable=True)) as db,db:
        snapshot=repair['snapshot'];snapshot['targets']=snapshot['targets']*201
        db.execute('UPDATE repairs_v1 SET snapshot=? WHERE id=?',(json.dumps(snapshot),repair['id']))
    original=retests.create
    count=0
    def fail_second(*args,**kwargs):
        nonlocal count
        count+=1
        if count==2:raise ValueError('injected second chunk failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(retests,'create',fail_second)
    body=request(lease_id=repair['lease_id'],client_id=client,artifact=dict(sha256='b'*64,version='candidate',installed_at='2026-11-15T00:00:00Z'))
    with pytest.raises(ValueError):repairs.mutate(database,bob['user_id'],'retest',body,repair['id'])
    with closing(_connect(database)) as db:
        assert db.execute('SELECT count(*) FROM retest_batches_v1').fetchone()[0]==0
        assert db.execute('SELECT count(*) FROM repair_retests_v1').fetchone()[0]==0


def test_first_claim_race_has_exactly_one_winner(setup):
    database,_,bob,*_=setup
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers=list(pool.map(lambda i:claim(database,bob['user_id'],'agent-'+str(i)),range(8)))
    assert len([a for a in answers if a is not None])==1
    with closing(_connect(database)) as db:
        assert db.execute('SELECT count(*) FROM repairs_v1').fetchone()[0]==1


def test_missing_issue_after_retest_can_be_reclaimed(setup,monkeypatch):
    database,alice,bob,*_=setup
    original=claim(database,bob['user_id'])
    repairs.mutate(database,bob['user_id'],'release',request(lease_id=original['lease_id']),original['id'])
    monkeypatch.setattr(issue_groups,'_current',lambda *a,**kw:[])
    taken=claim(database,alice['user_id'],'replacement')
    assert taken['id']==original['id'] and taken['snapshot']==original['snapshot']


def test_invalid_issue_filter_is_rejected(setup):
    database,_,bob,*_=setup
    with pytest.raises(ValueError):repairs.mutate(database,bob['user_id'],'claim',request(agent_id='agent',issue_key=[]))
