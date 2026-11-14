import json
import threading
from urllib.request import Request,build_opener,ProxyHandler
from urllib.error import HTTPError
import pytest
from autofusedebugger.fuzz.central.repository import create_user
from autofusedebugger.fuzz.central.server import create_server


def test_real_http_session_pairing_control_and_static_ui(tmp_path):
    path=tmp_path/'central.sqlite';user=create_user(path,'Docker User')
    server=create_server(path,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    host='127.0.0.1:'+str(server.server_port);base='http://'+host
    opener=build_opener(ProxyHandler({}))
    def call(route,payload=None,headers=None):
        h={'Content-Type':'application/json','Origin':'https://'+host};h.update(headers or {})
        req=Request(base+route,json.dumps(payload).encode() if payload is not None else None,headers=h)
        with opener.open(req,timeout=10) as r:return r.read(),dict(r.headers)
    try:
        page,_=call('/manage');assert '设备与任务管理'.encode() in page
        with pytest.raises(HTTPError) as error:call('/api/manage/devices')
        assert error.value.code==401
        raw,h=call('/api/session',{'token':user['token']});session=json.loads(raw)
        cookie=h['Set-Cookie'].split(';')[0];headers={'Cookie':cookie,'X-CSRF-Token':session['csrf']}
        raw,_=call('/api/manage/pair',{'name':'Docker fixture'});pair=json.loads(raw)
        raw,_=call('/api/manage/approve',{'code':pair['code']},headers);device=json.loads(raw)['device_id']
        raw,_=call('/api/manage/exchange',{'id':pair['id'],'secret':pair['secret']});credential=json.loads(raw)
        raw,_=call('/api/manage/submit',dict(device_id=device,kind='environment.inspect',payload={},request_key='e2e'),headers)
        identity=json.loads(raw)['id']
        raw,_=call('/api/manage/poll',dict(status={}),{'Authorization':'Bearer '+credential['token']})
        operation=json.loads(raw)['operations'][0];assert operation['id']==identity
        call('/api/manage/ack',dict(id=identity,generation=operation['generation'],state='succeeded',result={}),{'Authorization':'Bearer '+credential['token']})
        raw,_=call('/api/manage/operations',dict(device_id=device),headers)
        assert json.loads(raw)['items'][0]['state']=='succeeded'
        with pytest.raises(HTTPError):call('/api/manage/revoke',dict(device_id=device),{'Cookie':cookie})
        call('/api/manage/revoke',dict(device_id=device),headers)
        with pytest.raises(HTTPError):call('/api/manage/poll',dict(status={}),{'Authorization':'Bearer '+credential['token']})
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)


def test_management_maintenance_and_legacy_auth_stay_isolated(tmp_path,monkeypatch):
    from autofusedebugger.fuzz.central.maintenance import ENVIRONMENT
    policy=tmp_path/'allowlist.json';policy.write_text(json.dumps(dict(enabled=True,users=[],clients=[])))
    monkeypatch.setenv(ENVIRONMENT,str(policy))
    server=create_server(tmp_path/'central.sqlite',0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        req=Request('http://127.0.0.1:'+str(server.server_port)+'/api/manage/pair',b'{"name":"test"}',headers={'Content-Type':'application/json'})
        with pytest.raises(HTTPError) as exc:build_opener(ProxyHandler({})).open(req)
        assert exc.value.code==503
    finally:server.shutdown();server.server_close();thread.join(timeout=5)
