from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.storage.queries import ResultQueries
from autofusedebugger.fuzz.reporting.statistics import statistics


def save(repo, batch, backend, status, *, user='alice'):
    return repo.save({'seed':'680564733841876926926749214863536422954','workload_id':'a'*64,'oracle_id':'ref/1','backend':backend,
        'status':status,'execution':{'backend':backend,'batch_id':batch},'shape_results':[{'shape_index':0,'correctness':{'ok':'pass','error':'error','fail':'fail'}[status],'performance':'not_requested','error_summary':'Codegen generate kernel failed' if status=='error' else ''}]},
        environment={'installation_id':user,'user_id':user,'device_name':user})


def test_case_summary_excludes_cpu_and_backend_retries(tmp_path):
    repo=ResultRepository(tmp_path/'results')
    for i in range(100):
        save(repo,str(i),'cpu','ok')
        save(repo,str(i),'af-mlir','ok' if i<9 else 'error')
    q=ResultQueries(repo.path)
    assert q.summary()['total']==200  # internal attempt count remains explicit
    assert q.case_summary()=={'total':100,'by_status':{'ok':9,'error':91},'mlir_failed':0,'unit':'shape_case'}
    stats=statistics(repo.path)
    assert sum(r['count'] for r in stats['overall'])==100
    assert stats['backend']==[{'label':'af-mlir','status':'error','count':91},{'label':'af-mlir','status':'ok','count':9}]
    save(repo,'99','af-mlir','ok')
    save(repo,'99','eager-npu','ok')
    assert q.case_summary()['by_status']=={'ok':10,'error':90}
    assert q.case_summary(q='alice')['total']==100
    assert q.case_summary(user_id='bob')['total']==0


def test_cpu_only_unknown_and_cpu_match_filter_keeps_mlir_failure(tmp_path):
    repo=ResultRepository(tmp_path/'results')
    save(repo,'1','cpu','ok')
    q=ResultQueries(repo.path)
    assert q.case_summary()['by_status']=={'unknown':1}
    save(repo,'1','af-mlir','error')
    assert q.case_summary(q='cpu')['by_status']=={'error':1}
    save(repo,'2','cpu','ok',user='bob')
    save(repo,'2','af-mlir','fail',user='bob')
    assert q.case_summary()['by_status']=={'error':1,'fail':1}


def test_unfiltered_rollup_does_not_sort_full_diagnostics():
    from autofusedebugger.fuzz.reporting.cases import rollup
    sql, values = rollup()
    assert 'SELECT a.*' not in sql
    assert 'cases AS (SELECT c.* FROM unfiltered_cases c WHERE 1=1)' in sql
    assert not values
    filtered, values = rollup(q='relu')
    assert 'SELECT p.group_id,p.shape_index FROM projection p' in filtered
    assert values


def test_bounded_page_keys_match_full_rollup_with_retries_and_sparse_shapes(tmp_path):
    import sqlite3
    from autofusedebugger.fuzz.reporting.cases import page_keys, rollup
    from test_fuzz_shape_cases import record
    repo = ResultRepository(tmp_path/'pages')
    # More than one rowid window, with old backend peers and fresh retries.
    for batch in range(85):
        for backend in ('cpu', 'af-mlir'):
            value = record(backend, ['pass', 'error', 'pass'])
            value['execution']['batch_id'] = str(batch)
            repo.save(value)
    value = record('af-mlir', ['pass'])
    value['execution']['batch_id'] = '0'
    repo.save(value)
    value = record('af-mlir', [])
    value['execution']['batch_id'] = 'early'
    value['status'] = 'error'
    repo.save(value)
    from contextlib import closing
    from autofusedebugger.fuzz.storage.repository import _connect
    with closing(_connect(repo.path)) as conn:
        conn.row_factory = sqlite3.Row
        cte, args = rollup()
        expected = [dict(r) for r in conn.execute(cte+'SELECT * FROM cases ORDER BY cursor DESC', args)]
        for limit in (1, 7, 50, 200):
            actual = []; after = 0
            while True:
                rows = page_keys(conn, limit=limit, after=after)
                actual.extend(dict(r) for r in rows[:limit])
                if len(rows) <= limit:
                    break
                after = rows[limit-1]['cursor']
            assert actual == expected
        # A legal client cursor well beyond the newest row must also work.
        assert [dict(r) for r in page_keys(conn, limit=7, after=2**63-1)] == expected[:8]
