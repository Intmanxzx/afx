from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.storage.comparison import compare


def result(backend='cpu-eager',duration=100):
    return {'seed':'680564733841876926926749214863536422954','workload_id':'a'*64,'oracle_id':'ref/1','backend':backend,'status':'ok',
            'execution':{'backend':backend,'device':'cpu','static_only':False},
            'evaluation':{'correctness':True,'performance':True},
            'shape_results':[{'shape_index':0,'correctness':'pass','performance':'measured',
                              'mean_wall_ns':duration,'metric':'wall','measured_steps':10}]}


def test_pair_only_computes_speedup_for_compatible_measured_results(tmp_path):
    repo=ResultRepository(tmp_path/'results')
    left=repo.save(result(),environment={'machine':'aarch64'})
    right=repo.save(result('cpu-compile',50),environment={'machine':'aarch64'})
    value=compare(repo.path,left,right)
    assert value['same_workload'] and value['performance_comparable']
    assert value['shapes'][0]['speedup']==2


def test_cross_environment_and_shapes_never_invent_speedup(tmp_path):
    repo=ResultRepository(tmp_path/'results')
    left=repo.save(result(),environment={'machine':'aarch64'})
    raw=result('ascendc',1);raw['execution']['device']='npu:0';raw['shape_results'][0]['shape_index']=1
    right=repo.save(raw,environment={'machine':'x86_64'})
    value=compare(repo.path,left,right)
    assert not value['performance_comparable']
    assert len(value['incomparability_reasons'])==3
    assert all(row['speedup'] is None for row in value['shapes'])


def test_unmeasured_or_failed_shapes_have_no_speedup(tmp_path):
    repo = ResultRepository(tmp_path/'results')
    left = repo.save(result())
    raw = result('cpu-compile', 50)
    raw['shape_results'][0]['correctness'] = 'fail'
    right = repo.save(raw)
    assert compare(repo.path, left, right)['shapes'][0]['speedup'] is None


def test_public_and_offline_compare_routes(tmp_path):
    import json
    import threading
    from urllib.request import urlopen
    from autofusedebugger.fuzz.central.server import create_server as central
    from autofusedebugger.fuzz.view.server import create_server as offline
    repo = ResultRepository(tmp_path/'results')
    left = repo.save(result())
    right = repo.save(result('cpu-compile', 50))
    for factory in (central, offline):
        with factory(repo.path, port=0) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                base = getattr(server, 'viewer_url', f'http://127.0.0.1:{server.server_port}/')
                with urlopen(base+f'api/compare?left={left}&right={right}') as response:
                    assert json.load(response)['shapes'][0]['speedup'] == 2
            finally:
                server.shutdown(); thread.join()


def test_newest_pages_do_not_duplicate_or_skip(tmp_path):
    from autofusedebugger.fuzz.storage.queries import ResultQueries
    repo = ResultRepository(tmp_path/'results')
    identities = [repo.save(result()) for _ in range(5)]
    query = ResultQueries(repo.path)
    seen = []; cursor = 0
    while True:
        page = query.page(limit=2, after=cursor, newest=True)
        seen.extend(row['id'] for row in page['items'])
        if page['next_cursor'] is None:
            break
        cursor = page['next_cursor']
    assert seen == identities[::-1]
