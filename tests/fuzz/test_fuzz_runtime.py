"""Live fleet: authenticated ownership, deduplication, freshness and rate safety."""
import json
import uuid
from contextlib import closing

import pytest

from autofusex.fuzz.central.repository import initialize, create_user
from autofusex.fuzz.central.runtime import initialize as init_runtime, ingest, snapshot, connect
from autofusex.fuzz.storage.repository import _connect


def sample(source=None, run=None, machine='host-a', seq=1, elapsed=0, count=0):
    return dict(version=1,machine_id=machine,machine_name='machine-a',source_id=source or str(uuid.uuid4()),runs=[dict(run_id=run or str(uuid.uuid4()),sequence=seq,age=0,elapsed=elapsed,state='running',devices=[dict(uuid='physical-card',model='Ascend910',logical='npu:0',busy=True)],stages={'first_call_combined':1},queued=2,active_cases=1,completed=count,success=count,case_errors=0,record_failed=0,backend=['af-mlir'],versions={'torch':'2.10'})])


@pytest.fixture
def central(tmp_path):
    path=tmp_path/'central.sqlite';initialize(path)
    with closing(connect(path)) as conn,conn:init_runtime(conn)
    return path,create_user(path,'Alice'),create_user(path,'Bob')


def test_identity_dedup_rates_and_disconnect(central):
    db,a,b=central;p=sample();ingest(db,a['token'],p,now=100)
    p['runs'][0].update(sequence=2,elapsed=10,completed=20,success=15)
    ingest(db,a['token'],p,now=110)
    # Retry must not refresh run presence or duplicate its throughput.
    ingest(db,a['token'],p,now=120)
    d=snapshot(db,now=120);assert d['summary']['busy_cards']==1
    assert d['summary']['completed_per_min']==20/5
    assert d['machines'][0]['user_name']=='Alice'
    assert d['machines'][0]['runs'][0]['age_seconds']==10
    # Same physical machine/card, another user/process: one global resource.
    ingest(db,b['token'],sample(),now=120)
    d=snapshot(db,now=120);assert d['summary']['busy_cards']==1
    assert d['summary']['online_machines']==1
    assert d['summary']['active_users']==2
    assert len(d['machines'])==2
    assert len(snapshot(db,user_id=b['user_id'],now=120)['machines'])==1
    with pytest.raises(PermissionError):ingest(db,b['token'],p,now=130)
    assert snapshot(db,now=190)['summary']['busy_cards']==0
    assert snapshot(db,now=190)['summary']['online_machines']==0
    # Catch-up after a gap must not be counted as a huge live burst.
    p['runs'][0].update(sequence=3,elapsed=1000,completed=10000,success=9000)
    ingest(db,a['token'],p,now=1100)
    assert snapshot(db,now=1100)['summary']['completed_per_min']==0


def test_validation_and_stale_observation(central):
    db,a,b=central;p=sample();p['user_id']=b['user_id']
    with pytest.raises(ValueError):ingest(db,a['token'],p,now=100)
    p.pop('user_id');p['runs'][0]['age']=90;ingest(db,a['token'],p,now=100)
    d=snapshot(db,now=100)
    assert d['summary']['online_machines']==1 and d['summary']['busy_cards']==0
    assert d['machines'][0]['cards'][0]['status']=='unknown'
    p['runs'][0]['age']=float('nan')
    with pytest.raises(ValueError):ingest(db,a['token'],p)
    with pytest.raises(ValueError):snapshot(db,minutes='2')


def test_filters_and_counter_regression(central):
    db,a,b=central;p=sample();ingest(db,a['token'],p,now=100)
    assert snapshot(db,backend='af-mlir',model='Ascend910',version='2.10',now=100)['summary']['busy_cards']==1
    assert snapshot(db,backend='other',now=100)['summary']['busy_cards']==0
    p['runs'][0].update(sequence=2,elapsed=10,completed=5,success=5);ingest(db,a['token'],p,now=110)
    p['runs'][0].update(sequence=3,elapsed=20,completed=4,success=4)
    with pytest.raises(ValueError):ingest(db,a['token'],p,now=120)


def test_client_auth_and_stale_samples(tmp_path):
    from autofusex.fuzz.runtime import heartbeat, atomic, boot_id
    import time
    token='private-test-key-not-an-identity'
    (tmp_path/'token').write_text(token)
    (tmp_path/'sync.json').write_text(json.dumps(dict(endpoint='https://example.invalid',key_file=str(tmp_path/'token'))))
    p=sample();(tmp_path/'config.json').write_text(json.dumps(dict(source_id=p['source_id'])))
    atomic(tmp_path/'runtime/a.json',dict(boot=boot_id(),monotonic=time.monotonic()-5,sample=p['runs'][0]))
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return b'{"accepted":true}'
    def opener(request,timeout):
        assert request.get_header('Authorization')=='Bearer '+token
        body=json.loads(request.data)
        assert 'user_id' not in body and token not in request.data.decode()
        assert 4<=body['runs'][0]['age']<20
        return Response()
    heartbeat(tmp_path,opener=opener)


def test_runtime_http_routes_and_credential(central):
    from autofusex.fuzz.central.server import create_server
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
    import threading
    db,a,b=central
    with create_server(db,port=0) as server:
        thread=threading.Thread(target=server.serve_forever);thread.start()
        root='http://127.0.0.1:'+str(server.server_port)
        try:
            req=Request(root+'/api/heartbeat',json.dumps(sample()).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+a['token']})
            with urlopen(req) as r:assert json.load(r)['accepted']
            with urlopen(root+'/api/runtime') as r:assert json.load(r)['summary']['busy_cards']==1
            with urlopen(root+'/runtime.js') as r:assert b'fleetState' in r.read()
            req=Request(root+'/api/heartbeat',b'{}',headers={'Content-Type':'application/json'})
            with pytest.raises(HTTPError) as error:urlopen(req)
            assert error.value.code==401
        finally:server.shutdown();thread.join()


def test_concurrent_heartbeats_and_server_names(central):
    from concurrent.futures import ThreadPoolExecutor
    db,a,b=central
    payloads=[sample(machine='host-'+str(i)) for i in range(16)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda p:ingest(db,a['token'],p,now=100),payloads))
    result=snapshot(db,now=100)
    assert {m['display_name'] for m in result['machines']}=={f'Alice-npu{i}' for i in range(16)}
    assert result['summary']['online_machines']==16
    assert result['summary']['busy_cards']==16
    assert {m['user_name'] for m in result['machines']}=={'Alice'}


def test_reporter_counts_each_shape_not_whole_attempt(tmp_path):
    from autofusex.fuzz.runtime import Reporter
    reporter=Reporter(tmp_path,[])
    reporter.record({'status':'fail','shape_results':[{'correctness':'pass'},{'correctness':'fail'}]},'af-mlir')
    assert reporter.completed==2 and reporter.success==1
    reporter.record({'status':'ok','shape_results':[{'correctness':'pass'}]},'cpu')
    assert reporter.completed==2
    reporter.record({'status':'error','shape_results':[]},'af-mlir')
    assert reporter.case_errors==1


def test_retired_sampled_metadata_cannot_be_attached_to_current_seed():
    from dataclasses import replace, asdict
    from autofusex.fuzz.case_spec import decode, encode
    from autofusex.fuzz.generation.sampling import SamplingConfig
    from fuzz_fixtures import three_shapes_spec
    spec=three_shapes_spec();config=SamplingConfig();graph=json.loads(spec.graph_json)
    graph.update(dialect='nnsmith-sampled/5',sampling=asdict(config))
    spec=replace(spec,seed=config.seed(42,True),graph_json=json.dumps(graph,sort_keys=True,separators=(',',':')))
    with pytest.raises(ValueError,match='current graph metadata differs from seed'):
        decode(encode(spec))
    graph['sampling']['dim_max']=2
    with pytest.raises(ValueError):decode(encode(replace(spec,graph_json=json.dumps(graph,sort_keys=True,separators=(',',':')))))


def test_machine_display_names_are_persistent_and_options_deduplicated(central):
    db,a,b=central
    first=sample(machine='host-z');ingest(db,a['token'],first,now=100)
    second=sample(machine='host-a');ingest(db,a['token'],second,now=110)
    # Same host, new installation and hostname: not a new machine number.
    replacement=sample(machine='host-z');replacement['machine_name']='new-container'
    ingest(db,a['token'],replacement,now=120)
    ingest(db,b['token'],sample(machine='host-z'),now=125)
    expected={'host-z':'Alice-npu0','host-a':'Alice-npu1'}
    def names(**filters):
        return {m['machine_id']:m['display_name'] for m in snapshot(db,now=130,**filters)['machines']}
    assert names(user_id=a['user_id'])==expected
    assert names(user_id=a['user_id'],machine_id='host-a',backend='af-mlir')=={'host-a':'Alice-npu1'}
    assert names(user_id=b['user_id'])=={'host-z':'Bob-npu0'}
    options=snapshot(db,now=130)['machine_options']
    assert len(options)==2
    assert {o['id']:o['name'] for o in options}=={'host-z':'Alice-npu0 / Bob-npu0','host-a':'Alice-npu1'}
    with closing(connect(db)) as conn,conn:init_runtime(conn)
    assert names(user_id=a['user_id'])==expected
    ingest(db,a['token'],sample(machine='host-0'),now=140)
    assert names(user_id=a['user_id'])==dict(expected,**{'host-0':'Alice-npu2'})




def test_terminal_runs_keep_known_end_state_after_observations_expire(central):
    db,a,b=central
    for state in ['running','idle','error']:
        p=sample();p['runs'][0]['state']=state
        ingest(db,a['token'],p,now=100)
    data=snapshot(db,now=1000)
    assert {r['state'] for r in data['machines'][0]['runs']}=={'unknown','idle','error'}
    assert all(not r['stages'] for r in data['machines'][0]['runs'])
    assert data['summary']['busy_cards']==0
    assert data['summary']['active_cases']==0
    assert data['machines'][0]['cards'][0]['status']=='unknown'


def test_stale_old_run_does_not_add_live_work_to_new_session(central):
    db,a,b=central
    old=sample();ingest(db,a['token'],old,now=100)
    new=sample();ingest(db,a['token'],new,now=1000)
    data=snapshot(db,now=1001)
    assert data['summary']['busy_cards']==1
    assert data['summary']['active_cases']==1
    assert data['summary']['queued']==2
    assert {r['state'] for r in data['machines'][0]['runs']}=={'unknown','running'}


def test_user_summary_counts_people_not_machines(central):
    db,a,b=central
    def counts(**filters):
        s=snapshot(db,now=110,**filters)['summary']
        return s['total_users'],s['online_users'],s['active_users']
    assert counts()==(2,0,0)  # Registered accounts without a heartbeat still count.
    ingest(db,a['token'],sample(machine='host-a'),now=100)
    ingest(db,a['token'],sample(machine='host-b'),now=100)
    idle=sample(machine='host-a');idle['runs'][0]['state']='idle'
    ingest(db,b['token'],idle,now=100)
    assert counts()==(2,2,1)
    assert counts(user_id=a['user_id'])==(1,1,1)
    assert counts(user_id=b['user_id'])==(1,1,0)
    assert counts(machine_id='host-b')==(1,1,1)
    assert counts(backend='missing')==(0,0,0)
    assert counts(user_id='missing')==(0,0,0)
    stale=snapshot(db,now=200)['summary']
    assert (stale['total_users'],stale['online_users'],stale['active_users'])==(2,0,0)
    with closing(_connect(db,writable=True)) as conn,conn:
        conn.execute('UPDATE users SET active=0 WHERE id=?',(b['user_id'],))
    assert counts()==(1,1,1)
