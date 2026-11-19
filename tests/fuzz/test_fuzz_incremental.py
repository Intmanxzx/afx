from autofusex.fuzz.client.generation.sampling import SamplingConfig
import re
import json
import sqlite3
from contextlib import closing

import pytest

from contextlib import closing as _closing
from autofusex.fuzz.client.storage.repository import _connect as _connect_db
from autofusex.fuzz.server.reporting.incremental import IncrementalStatistics
from autofusex.fuzz.server.reporting.statistics import statistics
from autofusex.fuzz.client.storage import ResultRepository
from autofusex.fuzz.client.storage.repository import _connect
from test_fuzz_case_summary import save
from test_fuzz_shape_cases import record


def initialize(path):
    """Local stand-in for the old central repository's bootstrap: pre-creates
    the shared reporting schema that ResultRepository.save() would otherwise
    only create lazily on first write."""
    from autofusex.fuzz.server.reporting.index import initialize as initialize_reporting
    with _closing(_connect_db(path, writable=True)) as conn, conn:
        initialize_reporting(conn)


def same(db, inc):
    actual = inc.refresh()
    assert actual == statistics(db)
    return actual


def test_incremental_retry_restart_delete_and_metadata(tmp_path):
    db=tmp_path/'db'; initialize(db); repo=ResultRepository(db)
    for batch in range(8):
        save(repo,str(batch),'cpu','ok')
        save(repo,str(batch),'af-mlir','fail')
    inc=IncrementalStatistics(db)
    same(db,inc); assert inc.last_groups==1
    same(db,inc); assert inc.last_groups==0
    identity=save(repo,'3','af-mlir','ok')
    same(db,inc); assert inc.last_groups==1
    inc=IncrementalStatistics(db)
    same(db,inc); assert inc.last_groups==0
    with sqlite3.connect(db) as conn:
        conn.execute('DELETE FROM stage_metrics WHERE attempt_id=?',(identity,))
        conn.execute('DELETE FROM shapes WHERE attempt_id=?',(identity,))
        conn.execute('DELETE FROM attempt_facts WHERE attempt_id=?',(identity,))
        conn.execute('DELETE FROM attempts WHERE id=?',(identity,))
    same(db,inc); assert inc.last_groups==1
    # Whole-group deletion removes old counts; reindex/change of group uses old+new.
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE attempt_facts SET group_id='moved' WHERE batch_id='2'")
    same(db,inc); assert inc.last_groups==1
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO case_dimensions VALUES (?, 'op', 'test-op')",('a'*64,))
    same(db,inc); assert inc.last_groups==1
    with sqlite3.connect(db) as conn:
        conn.execute('DELETE FROM attempt_facts')
    same(db,inc)
    assert inc.refresh()['attempts']==0


def test_shapes_nullable_means_extremes_and_duplicate(tmp_path):
    db=tmp_path/'db'; initialize(db); repo=ResultRepository(db)
    ids=[]
    for batch, ns in [('a',10),('b',100),('c',None)]:
        value=record('af-mlir',['pass','error','pass']);value['execution']['batch_id']=batch
        for i,row in enumerate(value['shape_results']):
            row['mean_wall_ns']=ns if i==0 else None
        ids.append(repo.save(value))
    inc=IncrementalStatistics(db);same(db,inc)
    # Update/remove former extrema and a NULL-valued timing without averaging means.
    with sqlite3.connect(db) as conn:
        conn.execute('UPDATE shapes SET mean_wall_ns=20 WHERE attempt_id=?',(ids[1],))
        conn.execute('UPDATE stage_metrics SET duration_ns=17 WHERE attempt_id=?',(ids[1],))
    same(db,inc);assert inc.last_groups==1
    with sqlite3.connect(db) as conn:
        conn.execute('DELETE FROM shapes WHERE attempt_id=?',(ids[0],))
    same(db,inc)
    value=record('af-mlir',['pass']);value['execution']['batch_id']='a'
    identity=repo.save(value);same(db,inc)
    repo.save(value,attempt_id=identity)
    same(db,inc);assert inc.last_groups==0


def test_failed_batch_keeps_checkpoint_and_retries(tmp_path,monkeypatch):
    import autofusex.fuzz.server.reporting.incremental as module
    db=tmp_path/'db'; initialize(db); repo=ResultRepository(db)
    save(repo,'a','af-mlir','error'); inc=IncrementalStatistics(db);same(db,inc)
    save(repo,'a','af-mlir','ok')
    original=module.replace_group
    def fail(*args):
        original(*args)
        raise RuntimeError('interrupted')
    monkeypatch.setattr(module,'replace_group',fail)
    with pytest.raises(RuntimeError):inc.refresh()
    monkeypatch.setattr(module,'replace_group',original)
    same(db,IncrementalStatistics(db))


def test_concurrent_commit_is_not_lost(tmp_path,monkeypatch):
    import autofusex.fuzz.server.reporting.incremental as module
    db=tmp_path/'db'; initialize(db); repo=ResultRepository(db)
    save(repo,'a','af-mlir','error');inc=IncrementalStatistics(db);same(db,inc)
    save(repo,'a','af-mlir','ok')
    original=module.replace_group; written=[]
    def write_during_read(*args):
        original(*args)
        if not written:
            written.append(save(repo,'b','af-mlir','error'))
    monkeypatch.setattr(module,'replace_group',write_during_read)
    inc.refresh()
    same(db,inc);assert inc.last_groups==1


def test_replaced_source_rebuilds_sidecar(tmp_path):
    db=tmp_path/'db'; initialize(db);repo=ResultRepository(db)
    save(repo,'a','af-mlir','error');inc=IncrementalStatistics(db);same(db,inc)
    other=tmp_path/'other';initialize(other);save(ResultRepository(other),'b','af-mlir','ok')
    # Close all handles before replacing a SQLite file.
    with sqlite3.connect(other) as conn:conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    other.replace(db)
    same(db,inc);assert inc.last_groups==1


def test_no_group_rescan_for_unrelated_writes(tmp_path,monkeypatch):
    import autofusex.fuzz.server.reporting.incremental as module
    db=tmp_path/'db';initialize(db);save(ResultRepository(db),'a','af-mlir','ok')
    inc=IncrementalStatistics(db);same(db,inc)
    with sqlite3.connect(db) as conn:conn.execute('CREATE TABLE unrelated(value)')
    monkeypatch.setattr(module,'statistics',lambda *a,**k:pytest.fail('unexpected group query'))
    inc.refresh();assert inc.last_groups==0


def test_window_late_facts_and_cached_aggregate(tmp_path):
    db=tmp_path/'db';initialize(db);repo=ResultRepository(db)
    identity=repo.save(record('af-mlir',['pass','pass','pass']))
    inc=IncrementalStatistics(db);same(db,inc)
    window={'start':{'domain':'test','monotonic_ns':0},'finished':{'domain':'test','monotonic_ns':1000},'devices':[], 'cpu_tokens':1}
    with sqlite3.connect(db) as conn:
        conn.execute('INSERT INTO stage_windows_v1 VALUES (?,?)',('w',json.dumps(window)))
        conn.execute('INSERT INTO stage_attempts_v1 VALUES (?,?,?)',(identity,'case','w'))
        conn.execute('INSERT INTO stage_facts_v1 VALUES (?,?,?,?,?,?)',('s','w','case',None,identity,json.dumps({'kind':'record','retry_index':1})))
    same(db,inc);assert inc.last_groups==0
    with sqlite3.connect(db) as conn:
        conn.execute('UPDATE stage_facts_v1 SET payload=? WHERE stage_id=?',(json.dumps({'kind':'record','retry_index':0}),'s'))
        conn.execute('INSERT OR REPLACE INTO lifecycle_v1 VALUES (?,?,?,?,?)',(identity,'admitted','test',10,'test'))
        conn.execute('INSERT OR REPLACE INTO lifecycle_v1 VALUES (?,?,?,?,?)',(identity,'persisted','test',900,'test'))
    same(db,inc);assert inc.last_groups==0
    snapshot=statistics(db)['tool_efficiency']['windows']
    with sqlite3.connect(db) as conn:
        conn.execute('INSERT INTO stage_window_aggregates_v1 VALUES (?,?)',('w',json.dumps(snapshot)))
        conn.execute('DELETE FROM stage_facts_v1')
    same(db,inc)
    with sqlite3.connect(db) as conn:conn.execute('DELETE FROM stage_windows_v1')
    same(db,inc)


def test_bootstrap_batches_resume(tmp_path,monkeypatch):
    import autofusex.fuzz.server.reporting.incremental as module
    db=tmp_path/'db';initialize(db);repo=ResultRepository(db)
    for i in range(5):
        value=record('af-mlir',['pass']);value['seed']=str(SamplingConfig.profile('small',max_nodes=8,dtype_mode='float32').seed(42+i).value)
        value['execution'].pop('case_spec');value['execution']['batch_id']=str(i)
        repo.save(value)
    monkeypatch.setattr(module,'BATCH',2)
    original=module.replace_group;calls=[]
    def fail_third(*args):
        calls.append(args[1])
        if len(calls)==3:raise RuntimeError('crash')
        original(*args)
    monkeypatch.setattr(module,'replace_group',fail_third)
    with pytest.raises(RuntimeError):IncrementalStatistics(db).refresh()
    monkeypatch.setattr(module,'replace_group',original)
    inc=IncrementalStatistics(db);same(db,inc)
    assert inc.last_groups==3


def test_backend_aliases_preserve_full_statistics_rows(tmp_path):
    db=tmp_path/'db';initialize(db);repo=ResultRepository(db)
    for b in ('triton','triton-npu','eager-npu','eager-npu'):
        save(repo,b,b,'ok')
    same(db,IncrementalStatistics(db))


def test_replace_group_and_transaction_rollback(tmp_path):
    db=tmp_path/'db';initialize(db);repo=ResultRepository(db)
    identity=save(repo,'a','af-mlir','error');inc=IncrementalStatistics(db);same(db,inc)
    with sqlite3.connect(db) as conn:
        row=list(conn.execute('SELECT * FROM attempt_facts WHERE attempt_id=?',(identity,)).fetchone())
        row[1]='replacement'
        conn.execute('INSERT OR REPLACE INTO attempt_facts VALUES ('+','.join('?' for _ in row)+')',row)
    same(db,inc);assert inc.last_groups==1
    conn=sqlite3.connect(db)
    try:
        conn.execute("UPDATE shapes SET correctness='pass' WHERE attempt_id=?",(identity,))
        conn.rollback()
    finally:conn.close()
    same(db,inc);assert inc.last_groups==0


def test_group_query_plan_uses_group_index(tmp_path):
    from autofusex.fuzz.server.reporting.cases import rollup
    db=tmp_path/'db';initialize(db);save(ResultRepository(db),'a','af-mlir','ok')
    from autofusex.fuzz.client.storage.repository import _connect
    from contextlib import closing
    with closing(_connect(db)) as conn:
        sql,args=rollup()
        sql=sql.replace('JOIN attempt_arrivals_v2 r ON r.attempt_id=a.id','JOIN attempt_arrivals_v2 r ON r.attempt_id=a.id AND a.seed=?')
        plans=[re.sub(r'(?:TABLE \w+ AS |SUBQUERY \d+ AS )', '', r[3]) for r in conn.execute('EXPLAIN QUERY PLAN '+sql+'SELECT * FROM latest',['missing'])]
    assert any('SEARCH a USING' in p and 'seed=?' in p for p in plans)
    # Later SCAN a entries refer to materialized CTE rows, not attempts.
    assert any('SEARCH r USING' in p for p in plans)
