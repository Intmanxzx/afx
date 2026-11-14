"""Result publication must preserve original facts and reject stale evidence."""
import copy
from contextlib import closing
import json

import pytest
from test_fuzz_retests import setup, uid
from autofusex.fuzz.central import retests
from autofusex.fuzz.central.repository import ingest
from autofusex.fuzz.storage.repository import ResultRepository, _connect
from autofusex.fuzz.storage.queries import ResultQueries
from autofusex.fuzz.sync.outbox import pending, acknowledge


def execute(setup,tmp_path,statuses=('pass','pass','pass'),payload=None):
    database,_,bob,original,client,request=setup
    request=copy.deepcopy(payload or request);request['request_key']=uid()
    batch=retests.create(database,bob['user_id'],request)
    job=retests.claim(database,bob['user_id'],dict(client_id=client,limit=1))['items'][0]
    return job


def upload(setup,tmp_path,job,statuses=('pass','pass','pass'),**changes):
    database,_,bob,_,client,_=setup
    repo=ResultRepository(tmp_path/('result-'+uid()))
    spec=job['spec']
    result=dict(seed=spec['seed'],workload_id=spec['workload_id'],oracle_id=spec['oracle_id'],
        backend=spec['backend'],status='fail' if 'fail' in statuses else 'ok',
        execution=dict(backend=spec['backend'],static_only=False,batch_id=job['lease_id']),evaluation=spec['evaluation'],
        shape_results=[dict(shape_index=i,correctness=s,performance='not_requested') for i,s in enumerate(statuses)])
    result.update(changes)
    identity=repo.save(result,environment={'cann':'9.3.0','user_id':bob['user_id']})
    package=pending(repo.path);package['source_id']=client
    with closing(_connect(database)) as conn:ack=conn.execute('SELECT ack FROM sources WHERE id=?',(client,)).fetchone()[0]
    package['after']=ack
    for item in package['items']:item['revision']+=ack
    ingest(database,bob['token'],package)
    return dict(client_id=client,lease_id=job['lease_id'],attempt_id=identity)


def test_complete_updates_all_shapes_without_rewriting_owner(setup,tmp_path):
    database,alice,bob,original,client,_=setup
    job=execute(setup,tmp_path)
    payload=upload(setup,tmp_path,job)
    answer=retests.complete(database,bob['user_id'],payload)
    assert answer['accepted']
    assert retests.complete(database,bob['user_id'],payload)['reused']
    with closing(_connect(database)) as conn:
        assert conn.execute('SELECT user_id FROM owners WHERE attempt_id=?',(original,)).fetchone()[0]==alice['user_id']
        assert conn.execute('SELECT correctness FROM shapes WHERE attempt_id=? AND shape_index=1',(original,)).fetchone()[0]=='fail'
    queries=ResultQueries(database)
    assert queries.case_summary()['by_status']=={'ok':3}
    page=queries.case_page()
    assert len(page['items'])==3
    for item in page['items']:
        current=item['backends']['af-mlir'][0]
        assert current['id']==payload['attempt_id'] and current['original_attempt_id']==payload['attempt_id']
        assert current['executor']=='Bob'
    history=retests.history(database,original)
    assert history['items'][0]['shapes'][1]['summary_id'] is None


def test_late_uploaded_old_task_becomes_current(setup,tmp_path):
    database,_,bob,original,_,_=setup
    older=execute(setup,tmp_path)
    newer=execute(setup,tmp_path)
    new=upload(setup,tmp_path,newer,('pass','fail','fail'))
    assert retests.complete(database,bob['user_id'],new)['accepted']
    old=upload(setup,tmp_path,older)
    assert retests.complete(database,bob['user_id'],old)['accepted']
    assert ResultQueries(database).case_summary()['by_status']=={'ok':3}
    latest=execute(setup,tmp_path)
    retests.complete(database,bob['user_id'],upload(setup,tmp_path,latest))
    assert ResultQueries(database).case_summary()['by_status']=={'ok':3}


def test_mismatch_and_missing_result_do_not_publish(setup,tmp_path):
    database,_,bob,_,client,_=setup
    job=execute(setup,tmp_path)
    with pytest.raises(LookupError):
        retests.complete(database,bob['user_id'],dict(client_id=client,lease_id=job['lease_id'],attempt_id=uid()))
    payload=upload(setup,tmp_path,job,workload_id='b'*64)
    assert retests.complete(database,bob['user_id'],payload)['reason']=='reproduction_mismatch'
    assert ResultQueries(database).case_summary()['by_status']=={'fail':1,'ok':2}


def test_expired_lease_does_not_block_uploaded_result(setup,tmp_path):
    database,_,bob,original,client,_=setup
    job=execute(setup,tmp_path)
    with closing(_connect(database,writable=True)) as conn,conn:
        conn.execute('UPDATE retest_tasks_v1 SET expires=0')
    retests.claim(database,bob['user_id'],dict(client_id=client,limit=1))
    answer=retests.complete(database,bob['user_id'],upload(setup,tmp_path,job))
    assert not answer['accepted'] and answer['reason']=='late_result'
    assert ResultQueries(database).case_summary()['by_status']=={'ok':3}


def test_source_and_lease_binding(setup,tmp_path):
    database,alice,bob,original,client,_=setup
    job=execute(setup,tmp_path)
    with pytest.raises(PermissionError):
        retests.complete(database,bob['user_id'],dict(client_id=client,lease_id=job['lease_id'],attempt_id=original))
    payload=upload(setup,tmp_path,job,execution=dict(backend='af-mlir',static_only=False,batch_id=uid()))
    with pytest.raises(ValueError):retests.complete(database,bob['user_id'],payload)


def test_partial_shapes_preserve_unverified_and_statistics(setup,tmp_path):
    from autofusex.fuzz.reporting.statistics import statistics
    database,_,bob,_,_,_=setup
    job=execute(setup,tmp_path)
    payload=upload(setup,tmp_path,job,('fail',))
    assert retests.complete(database,bob['user_id'],payload)['accepted']
    assert ResultQueries(database).case_summary()['by_status']=={'fail':2,'ok':1}
    answer=statistics(database)
    assert sum(r['count'] for r in answer['overall'])==3
    assert {r['status']:r['count'] for r in answer['overall']}=={'fail':2,'ok':1}


def test_cache_refresh_after_publication(setup,tmp_path):
    from autofusex.fuzz.central.cache import PublicCache
    database,_,bob,_,_,_=setup
    cache=PublicCache(database,interval=30)
    first=cache.build()
    assert json.loads(first['summary'].body)['by_status']=={'fail':1,'ok':2}
    job=execute(setup,tmp_path)
    retests.complete(database,bob['user_id'],upload(setup,tmp_path,job))
    second=cache.build()
    assert json.loads(second['summary'].body)['by_status']=={'ok':3}
    assert len(json.loads(second['cases:10'].body)['items'])==3


def test_shared_failure_does_not_fabricate_shapes(setup,tmp_path):
    database,_,bob,original,_,_=setup
    job=execute(setup,tmp_path)
    payload=upload(setup,tmp_path,job,(),status='error',error_stage='prepare',
        error_summary='RuntimeError: MLIR codegen failed')
    assert retests.complete(database,bob['user_id'],payload)['accepted']
    page=ResultQueries(database).case_page()['items']
    assert len(page)==4
    shared=[r for r in page if r['shape_index']==-1]
    assert len(shared)==1 and shared[0]['backends']['af-mlir'][0]['id']==payload['attempt_id']
    assert all(r['shape_index']>=0 for r in page if r not in shared)
    # Original per-shape facts, including its failure, remain intact.
    assert ResultQueries(database).case_summary()['by_status'].get('fail')==1


def test_cancelled_and_facility_results_do_not_clear_failure(setup,tmp_path):
    database,_,bob,original,client,_=setup
    job=execute(setup,tmp_path)
    with closing(_connect(database)) as conn:
        batch=conn.execute('SELECT batch_id FROM retest_tasks_v1 WHERE id=?',(job['task_id'],)).fetchone()[0]
    retests.cancel(database,bob['user_id'],batch)
    answer=retests.complete(database,bob['user_id'],upload(setup,tmp_path,job))
    assert not answer['accepted']
    assert ResultQueries(database).case_summary()['by_status']=={'ok':3}
    job=execute(setup,tmp_path)
    answer=retests.complete(database,bob['user_id'],upload(setup,tmp_path,job,(),status='error',
        error_stage='capability',error_summary='ValueError: device busy: another managed test owns this physical NPU'))
    assert answer['reason']=='unverified'
    assert ResultQueries(database).case_summary()['by_status']=={'unknown':1,'ok':3}


def test_late_result_remains_visible_in_history(setup,tmp_path):
    database,_,bob,original,client,_=setup
    old=execute(setup,tmp_path)
    with closing(_connect(database,writable=True)) as conn,conn:
        conn.execute('UPDATE retest_tasks_v1 SET expires=0')
    new=retests.claim(database,bob['user_id'],dict(client_id=client,limit=1))['items'][0]
    retests.complete(database,bob['user_id'],upload(setup,tmp_path,old))
    retests.complete(database,bob['user_id'],upload(setup,tmp_path,new))
    history=retests.history(database,original)['items']
    assert len(history)==1 and len(history[0]['late_results'])==1
    assert history[0]['late_results'][0]['reason']=='late_result'


def test_same_seed_elsewhere_not_updated_and_current_version_filter(setup,tmp_path):
    database,_,bob,original,client,_=setup
    job=execute(setup,tmp_path)
    record=upload(setup,tmp_path,job)
    # Create an unrelated ordinary result of the same seed under another group.
    unrelated=copy.deepcopy(job);unrelated['lease_id']=uid()
    another=upload(setup,tmp_path,unrelated,('fail','fail','fail'))
    retests.complete(database,bob['user_id'],record)
    query=ResultQueries(database)
    assert query.case_summary()['by_status']=={'fail':3}
    assert len(query.case_page(q="WHERE status = 'Pass' AND cann = '9.3.0'")['items'])==0


def test_shared_failure_cleared_by_whole_success_not_partial(setup,tmp_path):
    database,_,bob,original,_,_=setup
    job=execute(setup,tmp_path)
    retests.complete(database,bob['user_id'],upload(setup,tmp_path,job,(),status='error',error_stage='prepare',
        error_summary='RuntimeError: MLIR codegen failed'))
    job=execute(setup,tmp_path)
    retests.complete(database,bob['user_id'],upload(setup,tmp_path,job,('pass',)))
    assert any(r['shape_index']==-1 for r in ResultQueries(database).case_page()['items'])
    # Fixture metadata has no spec, so explicitly supply known shape cardinality.
    with closing(_connect(database,writable=True)) as conn,conn:
        conn.execute('UPDATE case_facts SET shape_count=3')
    job=execute(setup,tmp_path)
    retests.complete(database,bob['user_id'],upload(setup,tmp_path,job))
    assert ResultQueries(database).case_summary()['by_status']=={'ok':3}


def test_original_shared_failure_can_become_three_shape_passes(setup,tmp_path):
    database,_,bob,original,client,request=setup
    # Create an independent real shared-stage failure with the same fixed seed.
    job=execute(setup,tmp_path)
    ordinary=copy.deepcopy(job);ordinary['lease_id']=uid()
    failure=upload(setup,tmp_path,ordinary,(),status='error',error_stage='prepare',
        error_summary='RuntimeError: MLIR codegen failed')
    payload=copy.deepcopy(request);payload['targets']=[dict(attempt_id=failure['attempt_id'],shape_index=-1)]
    replay=execute(setup,tmp_path,payload=payload)
    retests.complete(database,bob['user_id'],upload(setup,tmp_path,replay))
    query=ResultQueries(database)
    # Original group is unchanged; former shared failure now contributes 3 Pass.
    assert query.case_summary()['by_status']=={'ok':3}
    assert not any(i['shape_index']==-1 for i in query.case_page()['items'])
    # New S2 can be selected using the stable original group anchor.
    payload['targets']=[dict(attempt_id=query.case_page()['items'][0]['backends']['af-mlir'][0]['id'],shape_index=2)]
    assert execute(setup,tmp_path,payload=payload)
