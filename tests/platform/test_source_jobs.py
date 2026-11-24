import pytest
from autofusex.server.store import Store
from autofusex.server import auth, source_jobs
from autofusex.workloads.imported.registry import register, export


def test_source_job_identity_claim_receipt_and_isolation(tmp_path):
    store=Store(tmp_path/'server')
    from autofusex.server.auth import add_user
    add_user(store,'alice',role='developer')
    add_user(store,'bob',role='developer')
    alice=auth.authenticate(store,'Bearer '+auth.issue_token(store,'alice'))
    bob=auth.authenticate(store,'Bearer '+auth.issue_token(store,'bob'))
    source=tmp_path/'case.py';source.write_text('print("fixture")')
    case=register(tmp_path/'client',source)
    bundle=export(tmp_path/'client',case['case_key'])
    assert source_jobs.register(store,alice,bundle)==case
    with pytest.raises(PermissionError):source_jobs.register(store,bob,bundle)
    spec=dict(schema_version=1,kind='imported_task',environment='local',
              workload=dict(case_key=case['case_key'],revision=case['revision']))
    job=source_jobs.submit(store,alice,dict(request_key='one',spec=spec))
    assert source_jobs.submit(store,alice,dict(request_key='one',spec=spec))['job_id']==job['job_id']
    assert source_jobs.claim(store,bob,'machine')['job'] is None
    lease=source_jobs.claim(store,alice,'machine')['job']
    assert source_jobs.claim(store,alice,'machine')['job']['lease_id']==lease['lease_id']
    assert source_jobs.claim(store,alice,'other')['job'] is None
    receipt=dict(lease_id=lease['lease_id'],result=dict(case=case,status='passed',artifacts=[]))
    assert source_jobs.complete(store,alice,job['job_id'],receipt)['state']=='completed'
    assert source_jobs.complete(store,alice,job['job_id'],receipt)['state']=='completed'
    assert source_jobs.get(store,alice,job['job_id'])['result']['status']=='passed'
    with pytest.raises(LookupError):source_jobs.get(store,bob,job['job_id'])


def test_source_http_roundtrip(tmp_path):
    from fastapi.testclient import TestClient
    from autofusex.server.server import create_app
    store=Store(tmp_path/'server');auth.add_user(store,'alice','developer')
    headers={'Authorization':'Bearer '+auth.issue_token(store,'alice')}
    source=tmp_path/'case.py';source.write_text('print("fixture")')
    case=register(tmp_path/'client',source,workload='model')
    bundle=export(tmp_path/'client',case['case_key'])
    with TestClient(create_app(store)) as client:
        assert client.post('/api/source-cases',json=bundle).status_code==401
        assert client.post('/api/source-cases',json=bundle,headers=headers).status_code==200
        spec=dict(schema_version=1,kind='model_task',environment='local',workload=dict(case_key=case['case_key'],revision=case['revision']))
        response=client.post('/api/source-jobs',json=dict(request_key='one',spec=spec),headers=headers)
        assert response.status_code==200,response.text
        job=response.json()['data']
        lease=client.post('/api/source-jobs/claim',json=dict(machine_id='fixture'),headers=headers).json()['data']['job']
        import base64, hashlib
        raw=b'<html>HTTP report</html>'
        item=dict(path='artifacts/report.html',size=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        result=dict(case=case,status='passed',artifacts=[item])
        assert client.post('/api/source-jobs/'+job['job_id']+'/complete',json=dict(lease_id=lease['lease_id'],result=result),headers=headers).status_code==200
        assert client.get('/api/source-jobs',headers=headers).json()['data']['items'][0]['result']==result
        endpoint='/api/source-jobs/'+job['job_id']
        body=dict(lease_id=lease['lease_id'],path=item['path'],data=base64.b64encode(raw).decode())
        assert client.post(endpoint+'/artifacts',json=body,headers=headers).status_code==200
        response=client.get(endpoint+'/artifact',params={'path':item['path']},headers=headers)
        assert response.status_code==200
        assert response.content==raw
        assert response.headers['content-disposition'].startswith('attachment;')
        assert client.get(endpoint+'/artifact',params={'path':item['path']}).status_code==401


def test_artifact_requires_receipt_digest_and_owner(tmp_path):
    import base64,hashlib
    store=Store(tmp_path/'server');auth.add_user(store,'alice','developer')
    alice=auth.authenticate(store,'Bearer '+auth.issue_token(store,'alice'))
    source=tmp_path/'case.py';source.write_text('pass')
    case=register(tmp_path/'client',source)
    source_jobs.register(store,alice,export(tmp_path/'client',case['case_key']))
    spec=dict(schema_version=1,kind='imported_task',environment='local',workload=dict(case_key=case['case_key'],revision=case['revision']))
    job=source_jobs.submit(store,alice,dict(request_key='artifact',spec=spec));lease=source_jobs.claim(store,alice,'m')['job']['lease_id']
    raw=b'<html>report</html>';item=dict(path='artifacts/report.html',size=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    source_jobs.complete(store,alice,job['job_id'],dict(lease_id=lease,result=dict(case=case,status='passed',artifacts=[item])))
    body=dict(lease_id=lease,path=item['path'],data=base64.b64encode(raw).decode())
    source_jobs.artifact(store,alice,job['job_id'],body)
    assert source_jobs.artifact(store,alice,job['job_id'],path=item['path'])==raw
    with pytest.raises(ValueError):source_jobs.artifact(store,alice,job['job_id'],dict(body,data='YQ=='))
