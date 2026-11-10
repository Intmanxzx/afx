import json
from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.storage.queries import ResultQueries
from autofusedebugger.fuzz.reporting.statistics import statistics
from autofusedebugger.fuzz.generation import workload_id
from fuzz_fixtures import bundle, three_shapes_spec
from autofusedebugger.fuzz.case_spec import encode


def record(backend, states):
    return {'seed':str(2*2**128+42),'workload_id':workload_id(bundle()),
      'oracle_id':'ref/1','backend':backend,'status':'error' if 'error' in states else 'ok',
      'execution':{'case_spec':encode(three_shapes_spec()),'backend':backend,'batch_id':'batch','telemetry':json.dumps({'version':1,'stages':{'execution_total':900},'shapes':{str(i):{'first_call_combined':100+i} for i in range(len(states))}}),
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
    assert len(rows[0]['metadata']['input_shapes']) > 1
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


def test_shape_search_and_sql_consistent_across_summary_charts_and_rows(tmp_path):
    repo=ResultRepository(tmp_path/'db')
    repo.save(record('cpu',['pass']*3))
    repo.save(record('af-mlir',['pass','error','pass']))
    q=ResultQueries(repo.path)
    for term, indices in [('pass',[0,2]),('Failed',[1]),
        ("WHERE status = 'Pass' AND backend = 'af-mlir'",[0,2]),
        ("WHERE backend = 'af-mlir' AND function_status = 'Failed'",[1]),
        ("WHERE shape_index = 1 OR (shape_index = 2 AND status = 'Pass')",[1,2]),
        ("WHERE op LIKE '%transpose%'",[0,1,2])]:
        rows=q.case_page(q=term)['items']
        assert [r['shape_index'] for r in rows]==indices
        assert q.case_summary(q=term)['total']==len(indices)
        stat=statistics(repo.path,q=term)
        assert sum(r['count'] for r in stat['overall'])==len(indices)
        assert sum(r['count'] for r in stat['backend'])==len(indices)
    assert q.case_summary(status='error')['total']==1


def test_sql_rejects_arbitrary_statements():
    import pytest
    from autofusedebugger.fuzz.reporting.search import compile_where
    for expression in ["status='Pass'; DELETE FROM attempts", "secret='x'", "status IN (SELECT status FROM attempts)",
                       "status='Pass' -- comment", "random() > 0", "status=Pass", "("*20+"status='Pass'"+")"*20]:
        with pytest.raises(ValueError): compile_where(expression)
    sql,params=compile_where("user = 'O''Brien' AND shape_index >= 0")
    assert "O'Brien" not in sql and params==["O'Brien",0]


def test_short_seed_search_and_current_shape_metadata(tmp_path):
    from autofusedebugger.fuzz.contracts import Seed
    repo=ResultRepository(tmp_path/'db')
    value=record('af-mlir',['pass','error','pass']);repo.save(value)
    q=ResultQueries(repo.path);short=Seed.parse(value['seed']).short
    assert q.case_summary(q=short)['total']==3
    assert q.case_summary(q=f"WHERE seed = '{short}'")['total']==3
    assert [r['shape_index'] for r in q.case_page(q='[2,1,1,1,1]')['items']]==[1]
    assert q.case_summary(q="WHERE shape_index = -1")['total']==0


def test_detail_deduplicates_and_scopes_diagnostics():
    from autofusedebugger.fuzz.failure_details import diagnostic_view,core_error
    data={'error':"error: '\nAF_MLIR_STRICT gate=lowering shortfall=yes\nAF_MLIR_STRICT gate=lowering shortfall=yes",'shapes':{'0':{'error':'generated source\nCodegen generate kernel failed or abort'}}}
    view=diagnostic_view(data,0)
    assert view['summary']=='Codegen generate kernel failed or abort'
    assert len(view['shared_lines'])==1
    assert '不能' in view['shared_explanation']
    assert core_error(data,0)==view['summary']
    assert diagnostic_view({'error':'same','shapes':{'0':{'error':'same'}}},0)['shared_log']==''


def test_run_log_preview_matches_detail_without_shape_fallback(tmp_path):
    repo = ResultRepository(tmp_path/'db')
    value = record('af-mlir', ['pass', 'error', 'pass'])
    diagnostics = json.loads(value['execution']['diagnostics'])
    diagnostics['error'] = 'saved runtime context\nerror: native lowering failed token=private-value'
    value['execution']['diagnostics'] = json.dumps(diagnostics)
    identity = repo.save(value)
    queries = ResultQueries(repo.path)
    rows = queries.case_page()['items']
    view = queries.detail(identity, shape_index=1)['diagnostic_view']
    assert view['shape_log'] == 'shape one only'
    assert view['run_log'] == view['shared_log']
    assert '[REDACTED]' in view['run_log'] and 'private-value' not in view['run_log']
    for row in rows:
        assert row['backends']['af-mlir'][0]['run_log'] == view['run_log']


def test_missing_or_duplicate_run_log_remains_explicit():
    from autofusedebugger.fuzz.failure_details import diagnostic_view
    assert diagnostic_view({'shapes': {'0': {'error': 'shape only'}}}, 0)['run_log'] == ''
    view = diagnostic_view({'error': 'same', 'shapes': {'0': {'error': 'same'}}}, 0)
    assert view['run_log'] == 'same'
    assert view['shared_log'] == ''
