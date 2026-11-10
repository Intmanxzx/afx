from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.storage.queries import ResultQueries
from autofusedebugger.fuzz.reporting.statistics import statistics


def save(repo, batch, backend, status, *, user='alice'):
    return repo.save({'seed':'42','workload_id':'a'*64,'oracle_id':'ref/1','backend':backend,
        'status':status,'execution':{'backend':backend,'batch_id':batch},'shape_results':[]},
        environment={'installation_id':user,'user_id':user,'device_name':user})


def test_case_summary_excludes_cpu_and_backend_retries(tmp_path):
    repo=ResultRepository(tmp_path/'results')
    for i in range(100):
        save(repo,str(i),'cpu','ok')
        save(repo,str(i),'af-mlir','ok' if i<9 else 'error')
    q=ResultQueries(repo.path)
    assert q.summary()['total']==200  # internal attempt count remains explicit
    assert q.case_summary()=={'total':100,'by_status':{'ok':9,'error':91},'unit':'case'}
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
