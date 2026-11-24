"""Exact read-model equivalence and transactional invalidation."""
from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
from contextlib import closing
import sqlite3

from autofusex.execution.reporting.materialized import rebuild, ready, refresh
from autofusex.execution.storage.repository import ResultRepository, _connect
from autofusex.execution.storage.queries import ResultQueries
from autofusex.execution.reporting.statistics import statistics
from test_fuzz_global_rounds import save


def test_model_matches_legacy_and_incremental_updates(tmp_path):
    repo=ResultRepository(tmp_path/'db')
    save(repo,'a','alice','cpu');save(repo,'a','alice','af-mlir',('fail','pass','error'))
    save(repo,'b','bob','cpu');save(repo,'b','bob','af-mlir')
    q=ResultQueries(repo.path)
    filters=[{}, {'status':'ok'}, {'user_id':'alice'}, {'q':'F0F8A5E5JE7A4QBAOWXZCIT10Y8'}, {'q':"WHERE backend='af-mlir' AND status='Pass'"}]
    before=[(q.case_page(**kw),q.case_summary(**kw),statistics(repo.path,**kw)) for kw in filters]
    history=q.case_page(seed='F0F8A5E5JE7A4QBAOWXZCIT10Y8',shape=0,_history=True)
    rebuild(repo.path)
    assert before==[(q.case_page(**kw),q.case_summary(**kw),statistics(repo.path,**kw)) for kw in filters]
    assert q.case_page(seed='F0F8A5E5JE7A4QBAOWXZCIT10Y8',shape=0,_history=True)==history
    save(repo,'c','carol','cpu')
    assert q.case_summary()['by_status']=={'ok':3}
    save(repo,'c','carol','af-mlir',('fail','pass','fail'))
    assert q.case_summary()['by_status']=={'fail':2,'ok':1}
    with closing(_connect(repo.path)) as conn:
        assert ready(conn)
        conn.create_function('failure_status',5,lambda *a: (_ for _ in ()).throw(AssertionError('unexpected classification')))
        from autofusex.execution.reporting.materialized import rollup
        sql,args=rollup(conn)
        assert 'failure_status(' not in sql and 'json_extract(' not in sql
        assert len(conn.execute(sql+'SELECT * FROM cases',args).fetchall())==3


def test_dirty_model_falls_back_and_rebuild_is_atomic(tmp_path):
    repo=ResultRepository(tmp_path/'db');save(repo,'a','alice','cpu');identity=save(repo,'a','alice','af-mlir')
    rebuild(repo.path)
    with closing(_connect(repo.path,writable=True)) as conn,conn:
        conn.execute("UPDATE shapes SET correctness='fail' WHERE attempt_id=? AND shape_index=0",(identity,))
        assert not ready(conn)
    assert ResultQueries(repo.path).case_summary()['by_status']=={'fail':1,'ok':2}
    with closing(_connect(repo.path,writable=True)) as conn:
        conn.execute('BEGIN IMMEDIATE');refresh(conn);assert ready(conn);conn.rollback()
        assert not ready(conn)
    rebuild(repo.path)
    assert ResultQueries(repo.path).case_summary()['by_status']=={'fail':1,'ok':2}


def test_indexed_reads_do_not_reclassify_logs(tmp_path, monkeypatch):
    from test_fuzz_shape_cases import record
    repo=ResultRepository(tmp_path/'db')
    row=record('af-mlir',['error','pass'])
    row['shape_results'][0]['error_summary']='Codegen generate kernel failed'
    repo.save(row);rebuild(repo.path)
    q=ResultQueries(repo.path)
    before=q.case_page()
    def forbidden(*args,**kwargs):raise AssertionError('read must not classify text')
    monkeypatch.setattr('autofusex.execution.reporting.failure_origin.classify',forbidden)
    monkeypatch.setattr('autofusex.execution.reporting.failure_origin.mlir_failure_evidence',forbidden)
    monkeypatch.setattr('autofusex.execution.failure_details.core_error',forbidden)
    assert q.case_page()==before
    assert q.case_summary()['total']==2
    assert statistics(repo.path)['unit']=='shape_case'


def test_classification_version_change_rebuilds_without_stale_reads(tmp_path,monkeypatch):
    from autofusex.execution.reporting import materialized as model
    repo=ResultRepository(tmp_path/'db');save(repo,'a','alice','af-mlir')
    rebuild(repo.path)
    monkeypatch.setattr(model,'CLASSIFICATION_VERSION',model.CLASSIFICATION_VERSION+1)
    with closing(_connect(repo.path)) as c:assert not ready(c)
    assert rebuild(repo.path)['full']
    with closing(_connect(repo.path)) as c:assert ready(c)


def test_merge_into_model_rebuilds_derived_data(tmp_path):
    from autofusex.execution.storage.transfer import merge, restore
    source=ResultRepository(tmp_path/'src');target=ResultRepository(tmp_path/'dst')
    save(source,'a','alice','cpu');save(source,'a','alice','af-mlir')
    save(target,'b','bob','cpu');save(target,'b','bob','af-mlir')
    rebuild(target.path)
    merge(source.path,target.path)
    with closing(_connect(target.path)) as conn:assert ready(conn)
    assert ResultQueries(target.path).case_page()['items'][0]['history_count']==2
    restore(target.path,tmp_path/'restored')
    assert ResultQueries(tmp_path/'restored').case_summary()==ResultQueries(target.path).case_summary()


def test_per_seed_refresh_keeps_other_seeds_and_global_filters(tmp_path):
    from test_fuzz_shape_cases import record
    from autofusex.workloads.fuzz.contracts import Seed
    from autofusex.execution.reporting.cases import rollup
    repo=ResultRepository(tmp_path/'db')
    for n in range(8):
        row=record('af-mlir',['pass','error'])
        row['execution'].pop('case_spec',None)
        row['seed']=str(SamplingConfig.profile('small').seed(n,dynamic=True))
        row['workload_id']=str(n)*64
        repo.save(row)
    rebuild(repo.path)
    row=record('af-mlir',['fail','pass']);row['execution'].pop('case_spec',None);row['seed']=str(SamplingConfig.profile('small').seed(3,dynamic=True));row['workload_id']='3'*64
    repo.save(row)
    with closing(_connect(repo.path)) as conn:
        conn.row_factory=sqlite3.Row
        sql,args=rollup()
        assert [dict(r) for r in conn.execute(sql+'SELECT * FROM cases ORDER BY cursor',args)]==[dict(r) for r in conn.execute('SELECT * FROM case_heads_v1 ORDER BY cursor')]
        assert ready(conn)
    assert ResultQueries(repo.path).case_summary()['total']==16
