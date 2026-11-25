import json
import pytest
from test_platform import env,http,item,body,upload
from autofusex.server.ingest import UploadError
from autofusex.server.store import get_content
from autofusex.server.reclassification import refresh,rollback

def test_old_and_new_capture_are_independent_of_transport(env):
    store,ingestor,principal=env
    payload=body([item('capture-old1'),item('capture-new1',capture_version=2)],key='p2')
    payload['protocol_version']=2
    assert upload(ingestor,principal,payload)['accepted']==2
    rows=store.reader().execute('SELECT * FROM executions ORDER BY exec_id').fetchall()
    assert [(r['upload_protocol_version'],r['capture_version']) for r in rows]==[(2,1),(2,2)]
    assert json.loads(get_content(store.reader(),rows[1]['raw_upload_content_id']))['capture_version']==2
    assert upload(ingestor,principal,payload)['accepted']==2

def test_future_protocol_does_not_write(env):
    store,ingestor,principal=env
    payload=body([item('future-0001')]);payload['protocol_version']=999
    with pytest.raises(UploadError) as e:upload(ingestor,principal,payload)
    assert e.value.status==426
    assert store.reader().execute('SELECT COUNT(*) FROM executions').fetchone()[0]==0

def test_legacy_default_and_invalid_boolean(env):
    store,ingestor,principal=env
    assert upload(ingestor,principal,body([item('legacy-0001')]))['accepted']==1
    assert tuple(store.reader().execute('SELECT upload_protocol_version,capture_version FROM executions').fetchone())==(1,1)
    payload=body([item('legacy-0002')],key='bad');payload['protocol_version']=True
    with pytest.raises(UploadError):upload(ingestor,principal,payload)

def test_protocol_capabilities(http):
    code,result=http('GET','/api/client/protocol')
    assert code==200 and result['data']['protocol_versions']==[1,2]

def test_reclassification_preview_apply_idempotence_and_rollback(env):
    store,ingestor,principal=env
    payload=body([item('refresh-0001',status='runtime_fail',fail_stage='first_call_combined',log_summary='aclnnSign failed')])
    upload(ingestor,principal,payload)
    with store.write() as c:
        c.execute("UPDATE executions SET fail_stage='first_call_combined',classifier_version=1")
    before=tuple(store.reader().execute('SELECT content_hash,exec_uuid,capture_version FROM executions').fetchone())
    assert len(refresh(store,'preview'))==1
    assert store.reader().execute('SELECT classifier_version FROM executions').fetchone()[0]==1
    assert len(refresh(store,'upgrade',apply=True))==1
    assert refresh(store,'second',apply=True)==[]
    assert tuple(store.reader().execute('SELECT content_hash,exec_uuid,capture_version FROM executions').fetchone())==before
    assert rollback(store,'upgrade')==1
    assert store.reader().execute('SELECT fail_stage FROM executions').fetchone()[0]=='first_call_combined'

def test_future_capture_rejected_and_no_upgrade_of_old_records(env):
    store,ingestor,principal=env
    payload=body([item('future-capture',capture_version=3)]);payload['protocol_version']=2
    result=upload(ingestor,principal,payload)
    assert result['results'][0]['code']=='unsupported_capture'
    assert store.reader().execute('SELECT count(*) FROM executions').fetchone()[0]==0

def test_http_unknown_version_is_upgrade_required(http):
    payload=body([item('http-future1')]);payload['protocol_version']=99
    code,result=http('POST','/api/client/executions',payload)
    assert code==426
