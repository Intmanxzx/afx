"""Fixed-source and whole-model task dispatch, separate from small-graph statistics."""
import json
import uuid
from autofusex.contracts.source_cases import validate
from autofusex.contracts.serialization import canonical_json, fingerprint
from autofusex.server.store import now_ms
from autofusex.workloads.imported.configuration import validate as validate_task


def register(store,principal,body):
    principal.require_write()
    record=body['case'];validate(record,body['files'])
    bundle=canonical_json(body)
    with store.write() as db:
        old=db.execute('SELECT bundle,owner FROM source_cases WHERE case_key=?',(record['case_key'],)).fetchone()
        if old:
            if old['owner']!=principal.user_id:raise PermissionError('case belongs to another user')
            if old['bundle']!=bundle:raise ValueError('immutable case conflict')
        else:
            db.execute('INSERT INTO source_cases VALUES (?,?,?,?,?,?)',(record['case_key'],record['revision'],principal.user_id,record['spec']['workload'],bundle,now_ms()))
    return record


def submit(store,principal,body):
    principal.require_write();spec=validate_task(body['spec'])
    key=spec['workload']['case_key'];identity=str(uuid.uuid5(uuid.NAMESPACE_URL,principal.user_id+':'+body['request_key']))
    with store.write() as db:
        case=db.execute('SELECT * FROM source_cases WHERE case_key=?',(key,)).fetchone()
        if not case or case['owner']!=principal.user_id:raise LookupError('case not found')
        old=db.execute('SELECT spec FROM source_jobs WHERE job_id=?',(identity,)).fetchone()
        if old and old['spec']!=canonical_json(spec):raise ValueError('request key reused with different task')
        if not old:
            now=now_ms();db.execute('INSERT INTO source_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (identity,principal.user_id,key,canonical_json(spec),'queued',None,None,None,None,now,now))
    return get(store,principal,identity)


def get(store,principal,identity):
    principal.require_write()
    row=store.reader().execute('SELECT * FROM source_jobs WHERE job_id=? AND owner=?',(identity,principal.user_id)).fetchone()
    if not row:raise LookupError('job not found')
    value=dict(row);value['spec']=json.loads(value['spec']);value['result']=json.loads(value['result']) if value['result'] else None
    return value


def claim(store,principal,machine):
    principal.require_write();now=now_ms()
    with store.write() as db:
        row=db.execute("SELECT * FROM source_jobs WHERE owner=? AND (state='queued' OR (state='leased' AND machine_id=?)) ORDER BY created_at LIMIT 1",(principal.user_id,machine)).fetchone()
        if not row:return {'job':None}
        lease=row['lease_id'] or str(uuid.uuid4())
        db.execute("UPDATE source_jobs SET state='leased',machine_id=?,lease_id=?,lease_until=?,updated_at=? WHERE job_id=?",(machine,lease,now+3600000,now,row['job_id']))
        bundle=db.execute('SELECT bundle FROM source_cases WHERE case_key=?',(row['case_key'],)).fetchone()[0]
    return {'job':dict(job_id=row['job_id'],lease_id=lease,spec=json.loads(row['spec']),bundle=json.loads(bundle))}


def complete(store,principal,identity,body):
    principal.require_write()
    result=body['result']
    if result.get('status') not in {'passed','accuracy_failed','execution_failed','admission_failed','failed','stopped','cancelled','interrupted'}:
        raise ValueError('terminal result required')
    if not isinstance(result,dict) or len(canonical_json(result).encode())>512*1024:raise ValueError('invalid result')
    with store.write() as db:
        row=db.execute('SELECT * FROM source_jobs WHERE job_id=? AND owner=?',(identity,principal.user_id)).fetchone()
        if not row:raise LookupError('job not found')
        if row['lease_id']!=body['lease_id']:raise ValueError('lease lost')
        if result.get('case',{}).get('case_key')!=row['case_key']:raise ValueError('result case mismatch')
        spec=json.loads(row['spec'])
        if result['case'].get('revision')!=spec['workload']['revision']:raise ValueError('result revision mismatch')
        encoded=canonical_json(result)
        if row['state']=='completed' and row['result']!=encoded:raise ValueError('receipt conflict')
        db.execute("UPDATE source_jobs SET state='completed',result=?,updated_at=? WHERE job_id=?",(encoded,now_ms(),identity))
    return {'job_id':identity,'state':'completed'}


def list_jobs(store,principal):
    principal.require_write()
    rows=store.reader().execute('SELECT job_id FROM source_jobs WHERE owner=? ORDER BY created_at DESC LIMIT 100',(principal.user_id,)).fetchall()
    return {'items':[get(store,principal,row['job_id']) for row in rows]}


def record_result(store,principal,body):
    """Idempotently publish an explicitly uploaded local run."""
    spec=body['spec'];result=body['result']
    identity='local:'+result['execution_id']
    job=submit(store,principal,dict(request_key=identity,spec=spec))
    lease=str(uuid.uuid5(uuid.NAMESPACE_URL,principal.user_id+':'+identity))
    with store.write() as db:
        db.execute("UPDATE source_jobs SET lease_id=?,state=CASE WHEN state='completed' THEN state ELSE 'leased' END WHERE job_id=?",(lease,job['job_id']))
    return complete(store,principal,job['job_id'],dict(lease_id=lease,result=result))


def artifact(store,principal,identity,body=None,path=None):
    """Store bounded report files by verified digest; never use supplied paths on disk."""
    import base64, hashlib
    from autofusex.server.store import put_content, get_content
    job=get(store,principal,identity)
    if not job['result']:raise ValueError('job result is not available')
    name=body['path'] if body else path
    expected=next((item for item in job['result'].get('artifacts',[]) if item['path']==name),None)
    if not expected:raise LookupError('artifact not found')
    with store.write() as db:
        if body:
            if body['lease_id']!=job['lease_id']:raise ValueError('lease lost')
            try:raw=base64.b64decode(body['data'],validate=True)
            except ValueError as exc:raise ValueError('invalid artifact encoding') from exc
            if len(raw)>1024*1024 or len(raw)!=expected['size'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
                raise ValueError('artifact size or digest mismatch')
            content=put_content(db,raw)
            db.execute('INSERT OR REPLACE INTO source_artifacts VALUES (?,?,?)',(identity,name,content))
            return {'path':name,'sha256':expected['sha256']}
        row=db.execute('SELECT content_id FROM source_artifacts WHERE job_id=? AND path=?',(identity,name)).fetchone()
        if not row:raise LookupError('artifact not uploaded')
        return get_content(db,row[0])
