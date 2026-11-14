import json
from autofusedebugger.fuzz.execution import execute
from autofusedebugger.fuzz.contracts import EvaluationConfig
from fuzz_fixtures import bundle as materialize
from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.storage.queries import ResultQueries
from autofusedebugger.fuzz.reporting.statistics import statistics


class Adapter:
    name='cpu'
    def check(self,*args):pass
    def prepare(self,*args):return None
    def inputs(self,*args):return None
    def invoke(self,*args):return None
    def synchronize(self):pass
    def compare(self,*args):return True
    def cpu_reference(self,*args):return None
    def measure_kernel(self,*args):raise AssertionError('CPU must not measure kernels')
    def compile_duration_ns(self):return None


def test_cpu_no_wall_or_kernel_measurement():
    result=execute(materialize('680564733841876926926749214863536422954'),Adapter(),EvaluationConfig(performance=True))
    assert result['status']=='ok'
    assert all(s['performance']=='skipped' and 'mean_wall_ns' not in s for s in result['shape_results'])
    assert 'compile' not in json.loads(result['execution']['telemetry'])['stages']


def test_metric_hooks_and_search(tmp_path):
    class Profiled(Adapter):
        name='eager-npu'
        def measure_kernel(self,*args):return 1234
        def compile_duration_ns(self):return 5678
    result=execute(materialize('680564733841876926926749214863536422954'),Profiled(),EvaluationConfig(performance=True))
    repo=ResultRepository(tmp_path/'db');repo.save(result,environment={'user_id':'uuid','device_name':'niuyuhu','cann':'9.3'})
    # Public Shape cases require a complete primary AF-MLIR round.
    repo.save(dict(result, backend='af-mlir'), environment={'user_id':'uuid','device_name':'niuyuhu','cann':'9.3'})
    query=ResultQueries(repo.path)
    for text in ['niu','9.3','C000000016','pass','eager-npu']:
        assert query.summary(q=text)['total']==(1 if text=='eager-npu' else 2)
        assert len(query.case_page(q=text)['items'])==1
        assert statistics(repo.path,q=text)['attempts']==2  # A matching Shape retains peer backends.
    assert query.summary(q='%')['total']==0
    item=query.case_page()['items'][0]
    assert item['user_name']=='niuyuhu'
    member=item['backends']['eager-npu'][0]
    assert member['compile_ns'] is None and len(member['kernel_ns'])==1
    assert {r['stage'] for r in statistics(repo.path)['precise_metrics']}=={'compile','kernel_mean'}
