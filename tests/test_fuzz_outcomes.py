from autofusedebugger.fuzz.reporting.outcomes import outcomes
from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.storage.queries import ResultQueries
from autofusedebugger.fuzz.reporting.statistics import statistics


def test_accuracy_survives_measurement_failure():
    row={'status':'error','shapes':[{'correctness':'pass','performance':'error'}]}
    assert outcomes(row,{'performance':True})=={'correctness_status':'pass','measurement_status':'error'}
    row['shapes'][0]['performance']='not_requested'
    assert outcomes(row,{})['measurement_status']=='not_requested'
    row['shapes'][0]['performance']='measured'
    assert outcomes(row,{'performance':True})['measurement_status']=='measured'


def test_user_filter_applies_to_rows_counts_statistics(tmp_path):
    repo=ResultRepository(tmp_path/'db')
    for user in ['alice','bob']:
        repo.save({'seed':'680564733841876926926749214863536422954','workload_id':'a'*64,'oracle_id':'ref/1','backend':'cpu','status':'ok','shape_results':[]},environment={'user_id':user})
    query=ResultQueries(repo.path)
    assert query.summary(user_id='alice')['total']==1
    assert len(query.page(user_id='alice')['items'])==1
    assert query.case_page(user_id='alice')['items'][0]['user_id']=='alice'
    assert statistics(repo.path,user_id='alice')['attempts']==1
    assert query.summary(user_id="' OR 1=1 --")['total']==0
