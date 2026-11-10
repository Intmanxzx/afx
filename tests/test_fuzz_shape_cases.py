import json
from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.storage.queries import ResultQueries
from autofusedebugger.fuzz.reporting.statistics import statistics
from autofusedebugger.fuzz.generation import materialize,workload_id


def record(backend, states):
    return {'seed':str(2**128+42),'workload_id':workload_id(materialize(str(2**128+42))),
      'oracle_id':'ref/1','backend':backend,'status':'error' if 'error' in states else 'ok',
      'execution':{'backend':backend,'batch_id':'batch','telemetry':json.dumps({'version':1,'stages':{'execution_total':900},'shapes':{str(i):{'first_call_combined':100+i} for i in range(len(states))}}),
        'diagnostics':json.dumps({'version':1,'error':'error: native lowering failed','shapes':{'1':{'error':'shape one only'}}})},
      'shape_results':[{'shape_index':i,'correctness':state,'performance':'not_requested','error_summary':'bad' if state=='error' else ''} for i,state in enumerate(states)]}


def test_shape_rows_pagination_summary_and_details(tmp_path):
    repo=ResultRepository(tmp_path/'db')
    repo.save(record('cpu',['pass']*3))
    identity=repo.save(record('af-mlir',['pass','error','pass']))
    q=ResultQueries(repo.path)
    assert q.case_summary()=={'total':3,'by_status':{'ok':2,'error':1},'unit':'shape_case'}
    rows=[];after=0
    while True:
        page=q.case_page(limit=1,after=after);rows+=page['items']
        if page['next_cursor'] is None:break
        after=page['next_cursor']
    assert [r['shape_index'] for r in rows]==[0,1,2]
    assert len({r['case_id'] for r in rows})==3
    assert all(set(r['backends'])=={'cpu','af-mlir'} for r in rows)
    assert rows[1]['backends']['af-mlir'][0]['total_ns']==101
    assert rows[0]['sample']['input_shapes']!=rows[1]['sample']['input_shapes']
    assert rows[0]['metadata']['graph']['nodes']
    assert rows[0]['metadata']['weights']
    detail=q.detail(identity,shape_index=1)
    assert len(detail['shape_results'])==1 and detail['shape_results'][0]['shape_index']==1
    assert set(json.loads(detail['contexts']['execution']['diagnostics'])['shapes'])=={'1'}
    assert sum(r['count'] for r in statistics(repo.path)['overall'])==3


def test_optional_backend_is_not_hardcoded(tmp_path):
    repo=ResultRepository(tmp_path/'db')
    for b in ('cpu','af-mlir','triton-npu'):repo.save(record(b,['pass','pass','pass']))
    page=ResultQueries(repo.path).case_page()
    assert len(page['items'])==3
    assert all('triton-npu' in item['backends'] for item in page['items'])


def test_early_failure_remains_one_unassigned_record(tmp_path):
    repo=ResultRepository(tmp_path/'db');row=record('af-mlir',[]);row['status']='error'
    repo.save(row)
    q=ResultQueries(repo.path)
    assert q.case_page()['items'][0]['shape_index']==-1
    assert q.case_summary()['total']==1
    assert sum(r['count'] for r in statistics(repo.path)['overall'])==1


def test_split_error_line_retains_context():
    import os
    from autofusedebugger.fuzz.core_logs import capture_tail
    with capture_tail() as tail:
        os.write(2,b'error: Unsupport')
        os.write(2,b'edOperation: missing lowering\nsource location detail\n')
    assert 'UnsupportedOperation: missing lowering' in tail()
    assert 'source location detail' in tail()
