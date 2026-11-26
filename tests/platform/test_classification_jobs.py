import json
import time
import pytest
from test_platform import env, body, item, upload
from autofusex.server.classification_jobs import RefreshJobs
from autofusex.server.store import Store


def wait(jobs, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = jobs.get(job_id)
        if row['status'] in ('failed','completed'): return row
        time.sleep(.01)
    raise AssertionError('job did not finish')


def test_job_commit_and_idempotency(env):
    store, ingestor, principal = env
    upload(ingestor, principal, body([item('job-failure-0001', status='runtime_fail', log_summary='unique job marker')]))
    rules = [{'id':'job-rule','pattern':'unique job marker','result':{'function_status':'gen_fail','fail_stage':'inductor','failure_owner':'torch'}}]
    jobs = RefreshJobs(store)
    job = jobs.submit('admin','request-one',rules)
    finished = wait(jobs,job['job_id'])
    assert finished['status'] == 'completed', finished
    assert finished['result']['count'] == 1
    assert finished['result']['rules'] == {'job-rule':1}
    assert jobs.submit('admin','request-one',rules)['job_id'] == job['job_id']
    with pytest.raises(ValueError): jobs.submit('admin','request-one',[])
    assert store.reader().execute('SELECT count(*) FROM classification_changes WHERE batch_id=?',(job['job_id'],)).fetchone()[0] == 1
    assert RefreshJobs(Store.open_readonly(store.dir)).get(job['job_id'])['status'] == 'completed'


def test_restart_resumes_incomplete_job(env):
    store, _, _ = env
    with store.write() as conn:
        conn.execute("INSERT INTO classification_jobs(job_id,actor,request_key,rules,status,phase,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)", ('restart-job','admin','restart','[]','running','statistics',0,0))
    jobs=RefreshJobs(store); jobs.start()
    assert wait(jobs,'restart-job')['status'] == 'completed'


def test_failed_job_has_durable_error(env, monkeypatch):
    store, _, _ = env
    import autofusex.server.classification_jobs as module
    def fail(*args, **kwargs): raise RuntimeError('scan failed')
    monkeypatch.setattr(module,'plan',fail)
    jobs=RefreshJobs(store)
    job=jobs.submit('admin','failure',[])
    result=wait(jobs,job['job_id'])
    assert result['status'] == 'failed'
    assert result['error'] == 'scan failed'


from test_platform import http
from autofusex.server import auth


def test_admin_job_endpoints(http, env):
    store, _, _ = env
    assert http('GET','/api/admin/classification/jobs',token=None)[0] == 401
    assert http('GET','/api/admin/classification/jobs')[0] == 403
    token = auth.issue_token(store,'root')
    code, body = http('POST','/api/admin/classification/jobs',{'request_key':'api-job','rules':[]}, token=token)
    assert code == 200, body
    job_id = body['data']['job_id']
    deadline = time.monotonic()+5
    while time.monotonic()<deadline:
        code, result = http('GET','/api/admin/classification/jobs/'+job_id,token=token)
        if result['data']['status'] in ('completed','failed'): break
        time.sleep(.02)
    assert result['data']['status'] == 'completed', result
    assert http('GET','/api/admin/classification/jobs',token=token)[1]['data'][0]['job_id'] == job_id
    assert http('GET','/api/admin/classification/jobs/not-found',token=token)[0] == 404
