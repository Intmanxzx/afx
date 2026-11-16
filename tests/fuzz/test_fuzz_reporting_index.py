import json
import sqlite3

import pytest

from autofusex.fuzz.reporting.index import rebuild
from autofusex.fuzz.storage import ResultRepository
from autofusex.fuzz.storage.queries import ResultQueries
from autofusex.fuzz.storage.transfer import restore
from autofusex.fuzz.telemetry import validate


def record(backend='cpu', batch='run1'):
    return {'seed':'14615016373650978276328604632813839116929039597610','workload_id':'a'*64,'oracle_id':'ref/1','backend':backend,'status':'ok',
            'execution':{'backend':backend,'device':'cpu' if backend=='cpu' else 'npu:0','batch_id':batch,
                         'telemetry':json.dumps({'version':2,'stages':{'generation':10,'execution_total':100},'shapes':{'0':{'correctness':15}}})},
            'shape_results':[{'shape_index':0,'correctness':'pass','performance':'not_requested'}]}


def test_group_backend_retries_and_separate_provenance(tmp_path):
    repo=ResultRepository(tmp_path/'db')
    env={'installation_id':'device-a','device_name':'测试机','torch_npu':'2.10'}
    ids=[repo.save(record(b),environment=env) for b in ('cpu','af-mlir','cpu')]
    q=ResultQueries(repo.path)
    item=q.case_page()['items'][0]
    assert len(item['backends'])==2 and len(item['backends']['cpu'])==1
    assert item['backends']['cpu'][0]['total_ns']==15
    assert item['versions']['cann'] is None
    for e,b in [(dict(env,installation_id='device-b'),'run1'),(dict(env,torch_npu='2.11'),'run1'),(env,'run2')]:
        repo.save(record('af-mlir',batch=b),environment=e)
    assert len(q.case_page()['items'])==1
    first=q.case_page()['items'][0]
    second=q.case_page(seed=first['seed'],shape=0,after=first['cursor'],_history=True)
    assert len(second['items'])==3 and first['history_count']==4
    rebuild(repo.path);rebuild(repo.path)
    assert len(q.case_page()['items'])==1
    snapshot=tmp_path/'snapshot';repo.snapshot(snapshot)
    restore(snapshot,tmp_path/'restored')
    assert ResultQueries(tmp_path/'restored').case_page()==q.case_page()


@pytest.mark.parametrize('raw',['[]','null','{"version":1,"stages":{"bad":2}}','{"version":1,"stages":{"prepare":-1}}','{"version":1,"stages":{"prepare":true}}'])
def test_invalid_telemetry(raw):
    with pytest.raises(ValueError):validate(raw)


def test_environment_is_lightweight_and_never_reads_key(tmp_path):
    from autofusex.fuzz.config import setup
    from autofusex.fuzz.environment import collect
    setup(tmp_path,display_name='设备A',user_id='owner')
    result=collect(tmp_path)
    assert result['device_name']=='设备A' and result['user_id']=='owner'
    assert 'key_file' not in result


def test_failed_filter_keeps_peer_backend(tmp_path):
    repo=ResultRepository(tmp_path/'db'); repo.save(record())
    failed=record('af-mlir');failed['status']='fail';failed['shape_results'][0]['correctness']='fail';repo.save(failed)
    page=ResultQueries(repo.path).case_page(status='fail')
    assert set(page['items'][0]['backends'])=={'cpu','af-mlir'}


def test_statistics_uses_persisted_facts_and_stage_rows(tmp_path, monkeypatch):
    from autofusex.fuzz.reporting.statistics import statistics
    from autofusex.fuzz.reporting import report
    repo=ResultRepository(tmp_path/'db');repo.save(record());repo.save(record('af-mlir'))
    monkeypatch.setattr(report,'spec_metadata',lambda *args:pytest.fail('must not regenerate during query'))
    result=statistics(repo.path)
    assert result['attempts']==2 and result['case_groups']==1
    assert {r['label'] for r in result['backend']}=={'af-mlir'}
    assert sum(r['count'] for r in result['overall'])==1
    assert len(result['stages'])==6
    assert result['kernel_profiler']['kernel_count'] is None
    assert result['performance'][0]['performance']=='not_requested'


def test_new_http_endpoints_and_snapshot(tmp_path):
    import threading
    from urllib.request import urlopen
    from autofusex.fuzz.view.server import create_server
    repo=ResultRepository(tmp_path/'db');repo.save(record())
    with create_server(repo.path) as server:
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            for route in ('api/cases','api/statistics'):
                with urlopen(server.viewer_url+route) as response: assert response.status==200
        finally:server.shutdown();thread.join()


def test_central_keeps_authenticated_device_and_telemetry(tmp_path):
    from autofusex.fuzz.central.repository import create_user, ingest
    from autofusex.fuzz.sync.outbox import pending
    central=tmp_path/'central';key=tmp_path/'key';create_user(central,'管理员命名的设备',key)
    repo=ResultRepository(tmp_path/'local');repo.save(record(),environment={'device_name':'untrusted','torch_npu':'2.10'})
    repo.save(record('af-mlir'),environment={'device_name':'untrusted','torch_npu':'2.10'})
    payload=pending(repo.path);ingest(central,key.read_text().strip(),payload)
    first=ResultQueries(central).case_page()['items'][0]
    assert first['device_name']=='管理员命名的设备'
    assert first['backends']['cpu'][0]['total_ns']==15
    rebuild(central)
    assert ResultQueries(central).case_page()['items'][0]==first
