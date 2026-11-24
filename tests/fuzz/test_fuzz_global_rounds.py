"""Latest complete multi-backend round per global seed+Shape."""
from contextlib import closing
import json
import uuid

from autofusex.execution.storage.repository import ResultRepository,_connect
from autofusex.execution.storage.queries import ResultQueries
from autofusex.execution.reporting.statistics import statistics


def save(repo,batch,user,backend,states=('pass','pass','pass'),seed='F0F8A5E5JE7A4QBAOWXZCIT10Y8'):
    from autofusex.workloads.fuzz.contracts import Seed
    return repo.save(dict(seed=str(Seed.parse(seed)),workload_id='a'*64,oracle_id='cpu/1',backend=backend,
        status='fail' if 'fail' in states else 'ok',execution=dict(backend=backend,batch_id=batch,static_only=False,
        expected_backends=json.dumps(['cpu','af-mlir'])),evaluation=dict(correctness=True,performance=False,warmup_steps=3,measured_steps=10),
        shape_results=[dict(shape_index=i,correctness=s,performance='not_requested') for i,s in enumerate(states)]),
        environment=dict(installation_id=user,user_id=user,device_name=user,cann='9.3.0'))


def test_latest_whole_round_before_filter_and_no_mixing(tmp_path):
    repo=ResultRepository(tmp_path/'db')
    save(repo,'a','alice','cpu');old=save(repo,'a','alice','af-mlir',('fail','pass','fail'))
    q=ResultQueries(repo.path)
    save(repo,'b','bob','cpu')
    assert q.case_summary()['by_status']=={'fail':2,'ok':1}
    new=save(repo,'b','bob','af-mlir')
    assert q.case_summary()['by_status']=={'ok':3}
    assert q.case_page(user_id='alice')['items']==[]
    assert q.case_page(q="WHERE status='accuracy_failed'")['items']==[]
    page=q.case_page()['items'];assert len(page)==3
    for item in page:
        assert item['history_count']==2 and item['user_name']=='bob'
        assert item['backends']['af-mlir'][0]['id']==new
        assert item['time_basis']=='server_received'
        assert len(item['backends']['cpu'])==1
    parent=page[0]
    history=q.case_page(seed=parent['seed'],shape=parent['shape_index'],after=parent['cursor'],_history=True)
    assert len(history['items'])==1 and history['items'][0]['backends']['af-mlir'][0]['id']==old
    assert sum(x['count'] for x in statistics(repo.path)['overall'])==3


def test_history_bounded_and_current_unknown_no_old_pass(tmp_path):
    repo=ResultRepository(tmp_path/'db')
    for i in range(28):
        save(repo,str(i),'user','cpu');save(repo,str(i),'user','af-mlir',('pass',))
    save(repo,'last','other','cpu');save(repo,'last','other','af-mlir',('not_checked',))
    q=ResultQueries(repo.path)
    assert q.case_summary()['by_status']=={'unknown':1}
    item=q.case_page()['items'][0];assert item['history_count']==29
    history=q.case_page(seed=item['seed'],shape=0,after=item['cursor'],limit=20,_history=True)
    assert len(history['items'])==20 and history['next_cursor']
    rest=q.case_page(seed=item['seed'],shape=0,after=history['next_cursor'],limit=20,_history=True)
    assert len(rest['items'])==8
    assert not set(r['record_id'] for r in history['items'])&set(r['record_id'] for r in rest['items'])


def test_local_retry_same_id_does_not_move_arrival(tmp_path):
    repo=ResultRepository(tmp_path/'db')
    save(repo,'a','alice','cpu');identity=save(repo,'a','alice','af-mlir')
    with closing(_connect(repo.path)) as conn:
        arrival=conn.execute('SELECT received_at FROM attempt_arrivals_v2 WHERE attempt_id=?',(identity,)).fetchone()[0]
    from autofusex.execution.reporting.index import index_attempt
    with closing(_connect(repo.path,writable=True)) as conn,conn:index_attempt(conn,identity)
    with closing(_connect(repo.path)) as conn:
        assert conn.execute('SELECT received_at FROM attempt_arrivals_v2 WHERE attempt_id=?',(identity,)).fetchone()[0]==arrival


def test_new_round_id_prevents_crossed_backend_arrivals(tmp_path):
    repo=ResultRepository(tmp_path/'db')
    def put(round_id,backend,states):
        from test_fuzz_shape_cases import record
        value=record(backend,states)
        value['execution'].update(round_id=round_id,expected_backends=json.dumps(['cpu','af-mlir']))
        return repo.save(value)
    first=put('first','af-mlir',['fail'])
    put('second','cpu',['pass'])
    q=ResultQueries(repo.path)
    assert q.case_page()['items']==[]
    second=put('second','af-mlir',['pass'])
    assert q.case_page()['items'][0]['backends']['af-mlir'][0]['id']==second
    put('first','cpu',['pass'])
    row=q.case_page()['items'][0]
    assert row['backends']['af-mlir'][0]['id']==first and row['history_count']==2
    assert row['status']=='fail'


def test_arrival_delete_never_reuses_sequence(tmp_path):
    from autofusex.execution.reporting.index import initialize
    repo=ResultRepository(tmp_path/'db')
    first=save(repo,'first','alice','af-mlir',('pass',))
    second=save(repo,'second','bob','af-mlir',('pass',))
    with closing(_connect(repo.path,writable=True)) as conn,conn:
        high=conn.execute('SELECT max(sequence) FROM attempt_arrivals_v2').fetchone()[0]
        # Directly delete the derived row to test sequence behavior independently
        # of the administrative deletion routine's other foreign keys.
        conn.execute('DELETE FROM attempt_arrivals_v2 WHERE attempt_id=?',(second,))
    save(repo,'third','carol','cpu',('pass',))
    third=save(repo,'third','carol','af-mlir',('pass',))
    with closing(_connect(repo.path)) as conn:
        assert conn.execute('SELECT sequence FROM attempt_arrivals_v2 WHERE attempt_id=?',(third,)).fetchone()[0]>high
    assert ResultQueries(repo.path).case_page()['items'][0]['time_basis']=='server_received'



def test_exact_short_seed_search_is_pushed_before_rollup(tmp_path):
    from autofusex.execution.reporting.cases import rollup
    from autofusex.workloads.fuzz.contracts import Seed
    repo=ResultRepository(tmp_path/'db')
    save(repo,'a','alice','cpu');save(repo,'a','alice','af-mlir')
    q=ResultQueries(repo.path)
    assert q.case_page(q='F0F8A5E5JE7A4QBAOWXZCIT10Y8')==q.case_page(seed='F0F8A5E5JE7A4QBAOWXZCIT10Y8')
    sql,values=rollup(q='F0F8A5E5JE7A4QBAOWXZCIT10Y8')
    assert 'WHERE a.seed=? AND NOT EXISTS' in sql
    assert values[0]==str(Seed.parse('F0F8A5E5JE7A4QBAOWXZCIT10Y8'))
    assert 'WHERE a.seed=? AND NOT EXISTS' not in rollup(q='0000')[0]
