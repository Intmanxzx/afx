import json
import threading
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError

import pytest

from autofusex.fuzz.central.repository import create_user, ingest
from autofusex.fuzz.central.server import create_server
from autofusex.fuzz.management.central import ControlStore
from autofusex.fuzz.storage import ResultRepository
from autofusex.fuzz.sync.outbox import pending
from autofusex.fuzz.generation.sampling import SamplingConfig


@pytest.fixture
def api(tmp_path):
    path=tmp_path/'central.sqlite'
    user=create_user(path,'agent'); other=create_user(path,'other')
    store=ControlStore(path)
    pair=store.pairing('npu');store.approve(user['user_id'],pair['code'])
    credential=store.exchange(pair['id'],pair['secret']);device=store.authenticate(credential['token'])
    server=create_server(path,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    opener=build_opener(ProxyHandler({}))
    def call(route,body=None,token=None):
        headers={'Authorization':'Bearer '+(token or user['token']),'Content-Type':'application/json'}
        req=Request('http://127.0.0.1:'+str(server.server_port)+'/api/agent/'+route,
                    json.dumps(body).encode() if body is not None else None,headers=headers)
        try:
            with opener.open(req,timeout=5) as response:return response.status,json.load(response)
        except HTTPError as exc:return exc.code,json.load(exc)
    yield call,store,device,credential,user,other,path
    server.shutdown();server.server_close();thread.join(timeout=5)


def submit_body(device,key='once'):
    return dict(device_id=device['id'],request_key=key,spec=dict(schema_version=1,kind='fuzz_task',
                environment='env',workload=dict(mode='generate'),termination=dict(mode='finite',max_cases=1)))


def test_http_task_lifecycle_results_and_replay(api,tmp_path):
    call,store,device,credential,user,other,path=api
    store.poll(device,dict(environments=[dict(id='env',name='NPU')],runs=[]))
    code,devices=call('devices');assert code==200
    assert devices['data']['items'][0]['environments'][0]['id']=='env'
    body=submit_body(device)
    code,response=call('tasks',body);assert code==202
    identity=response['data']['id']
    assert call('tasks',body)[1]['data']['id']==identity
    changed=submit_body(device);changed['spec']['name']='changed'
    assert call('tasks',changed)[0]==409
    operation=store.poll(device,{})['operations'][0]
    assert operation['payload']['results']==dict(save=True,upload=True)
    store.ack(device,identity,operation['generation'],'succeeded',dict(run_id='run-1'))
    store.poll(device,dict(runs=[dict(id='run-1',state='completed',result_count=1)]))
    task=call('tasks/'+identity)[1]['data']
    assert task['terminal'] and not task['results']['complete']
    seed=str(SamplingConfig.profile('small',max_nodes=1).seed(3))
    repo=ResultRepository(tmp_path/'local.sqlite')
    repo.save(dict(seed=seed,workload_id='a'*64,oracle_id='ref/1',backend='af-mlir',status='fail',
                   execution=dict(backend='af-mlir',device='npu:0',batch_id='run-1:0'),
                   evaluation=dict(correctness=True,performance=False),
                   shape_results=[dict(shape_index=0,correctness='fail',performance='not_requested')]))
    ingest(path,credential['token'],pending(tmp_path/'local.sqlite'))
    task=call('tasks/'+identity)[1]['data'];assert task['results']['complete']
    code,results=call('tasks/'+identity+'/results?status=fail&limit=1')
    assert code==200 and len(results['data']['items'])==1
    attempt=results['data']['items'][0]['id']
    assert call('tasks/'+identity+'/results?attempt='+attempt)[1]['data']['attempt']['status']=='fail'
    assert call('tasks/'+identity,token=other['token'])[0]==401
    assert call('tasks/'+identity+'/results',token=other['token'])[0]==401
    assert call('devices',token=credential['token'])[0]==401
    code,replay=call('tasks/'+identity+'/replay',dict(seeds=[seed],request_key='replay'))
    assert code==202 and replay['data']['id']!=identity
    replay_op=store.poll(device,{})['operations'][0]
    assert replay_op['payload']['workload']==dict(mode='replay',seeds=[seed])
    assert 'termination' not in replay_op['payload']


def test_stop_queued_inflight_and_running(api):
    call,store,device,*_=api
    identity=call('tasks',submit_body(device))[1]['data']['id']
    stopped=call('tasks/'+identity+'/stop',dict(request_key='stop'))
    assert stopped[0]==202 and stopped[1]['data']['state']=='cancelled'
    assert not store.poll(device,{})['operations']
    identity=call('tasks',submit_body(device,'second'))[1]['data']['id']
    op=store.poll(device,{})['operations'][0]
    assert call('tasks/'+identity+'/stop',dict(request_key='stop-second'))[0]==409
    store.ack(device,identity,op['generation'],'succeeded',dict(run_id='running'))
    assert call('tasks/'+identity+'/stop',dict(request_key='stop-second'))[1]['data']['stop_requested']
    actions=store.poll(device,{})['operations']
    assert len(actions)==1 and actions[0]['payload']==dict(run_id='running',action='stop')
    assert call('tasks/'+identity+'/stop',dict(request_key='stop-second'))[0]==202


def test_invalid_requests_and_stale_status(api):
    call,store,device,*_=api
    assert call('tasks',{})[0]==400
    assert call('devices?unexpected=1')[0]==404
    identity=call('tasks',submit_body(device))[1]['data']['id']
    assert call('tasks/'+identity)[1]['data']['stale']
    for query in ('limit=0','after=-1','status=bad','limit=1&limit=2'):
        assert call('tasks/'+identity+'/results?'+query)[0]==400
    assert call('tasks/'+identity+'/stop',dict(request_key=''))[0]==400


def test_maintenance_rejects_writes_but_keeps_reads(api):
    from autofusex.fuzz.management.agent_api import AgentAPI
    from autofusex.fuzz.central.maintenance import WriteAllowlist, MaintenanceBlocked
    call,store,device,credential,user,*_=api
    restricted=AgentAPI(store,WriteAllowlist(enabled=True))
    with pytest.raises(MaintenanceBlocked): restricted.submit(user['user_id'],submit_body(device))
    assert call('devices')[0]==200
