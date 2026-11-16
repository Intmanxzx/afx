from autofusex.fuzz.client.generation.sampling import SamplingConfig
CURRENT_SEED = str(SamplingConfig.profile('small', max_nodes=1, dtype_mode='float32').seed(3))
from dataclasses import asdict
import json
import pytest
from autofusex.fuzz.client.scheduling.ledger import TaskLedger
from autofusex.fuzz.client.scheduling.campaign import initialize,refill,stop
from autofusex.fuzz.client.contracts import ExecutionConfig,EvaluationConfig


def test_finite_campaign_bounded_queue(tmp_path):
    ledger=TaskLedger(tmp_path/'queue');configs=[ExecutionConfig(b,'cpu' if b=='cpu' else 'npu:0') for b in ['cpu','af-mlir']]
    initialize(ledger,configs,EvaluationConfig(),2)
    for _ in range(2):
        assert refill(ledger)
        assert not refill(ledger)
        for b in ['cpu','af-mlir']:
            task=ledger.claim();assert json.loads(task['execution'])['backend']==b
            ledger.complete(task['id'],task['claim'])
    assert not refill(ledger)
    assert len(ledger.page()['items'])==4


def test_infinite_refill_stopped_atomically(tmp_path):
    ledger=TaskLedger(tmp_path/'queue');initialize(ledger,[ExecutionConfig('cpu','cpu')],EvaluationConfig(),None)
    assert refill(ledger);task=ledger.claim();stop(ledger)
    assert ledger.get(task['id'])['state']=='stop_requested'
    assert not refill(ledger)


def test_cpu_disagreement_never_triggers_secondary_or_changes_to_pass():
    from autofusex.fuzz.client.execution import execute
    from fuzz_fixtures import bundle as materialize
    class Adapter:
        name='af-mlir'
        def check(self,*a):pass
        def prepare(self,*a):return None
        def inputs(self,*a):return None
        def invoke(self,*a):return None
        def synchronize(self):pass
        def cpu_reference(self,*a):return None
        def compare(self,*a):return False
        def diagnose_accuracy(self,*a):return {'max_abs':1.0,'mismatches':1,'elements':2}
        def secondary_output(self,*a):pytest.fail('automatic secondary forbidden')
        def outputs_equal(self,*a):return True
    result=execute(materialize('14615016373650978276328604632813839116929039597610'),Adapter(),EvaluationConfig())
    assert result['status']=='fail'
    detail=json.loads(result['execution']['diagnostics'])['shapes']['0']
    assert detail['cpu_pass'] is False and detail['secondary_used'] is False and detail['basis'] == 'cpu' and 'secondary_pass' not in detail
    adapter=Adapter();adapter.compare=lambda *a:True
    adapter.secondary_output=lambda *a:pytest.fail('secondary should not execute')
    assert execute(materialize('14615016373650978276328604632813839116929039597610'),adapter,EvaluationConfig())['status']=='ok'


def test_bounded_redacted_failure_excerpt():
    from autofusex.fuzz.failure_details import excerpt,validate
    try:raise RuntimeError('token=secretvalue '+'x'*2000)
    except RuntimeError as exc:text=excerpt(exc)
    assert len(text)<=1600 and 'secretvalue' not in text
    validate(json.dumps({'version':1,'error':text,'shapes':{}}))


def test_native_core_log_is_bounded_and_redacted():
    import os
    from autofusex.fuzz.client.core_logs import capture_tail
    with capture_tail() as tail:
        os.write(2, b'x' * 40000 + b'\ncompiler error token=private-value\n')
    assert len(tail()) <= 1600
    assert 'compiler error' in tail() and 'private-value' not in tail()


def test_campaign_capacity_is_enforced(tmp_path):
    import pytest
    from autofusex.fuzz.client.scheduling.ledger import TaskLedger
    from autofusex.fuzz.client.scheduling.campaign import initialize
    from autofusex.fuzz.client.contracts import ExecutionConfig, EvaluationConfig
    with pytest.raises(ValueError, match='capacity'):
        initialize(TaskLedger(tmp_path/'queue', max_active=1),
                   [ExecutionConfig('cpu','cpu'), ExecutionConfig('af-mlir','npu:0')],
                   EvaluationConfig(), None)


def test_torch_cpu_reference_and_error_metrics():
    pytest.importorskip("torch")
    from autofusex.fuzz.client.backends.torch import TorchAdapter
    from fuzz_fixtures import bundle as materialize
    from autofusex.fuzz.client.generation import materialize
    adapter=TorchAdapter(ExecutionConfig('cpu','cpu'))
    bundle=materialize(CURRENT_SEED)
    try:
        adapter.check(bundle, EvaluationConfig())
        expected=adapter.cpu_reference(bundle, 0)
        assert adapter.compare(expected, expected)
        changed=tuple(t + 1 for t in expected)
        assert not adapter.compare(changed, expected)
        metrics=adapter.diagnose_accuracy(changed, expected)
        assert metrics['max_abs']==1.0
        assert metrics['mismatches']==metrics['elements']
    finally:
        adapter.close()


def test_default_replay_uses_both_mandatory_backends(monkeypatch):
    from autofusex.fuzz import cli
    seen=[]
    monkeypatch.setattr(cli,'execute_args',lambda args: seen.append((args.backend,args.device)) or 0)
    assert cli.replay([CURRENT_SEED,'--scheduling','combined'])==0
    assert seen==[('cpu','cpu'),('af-mlir','npu:0')]


def test_secondary_disagreement_stays_failed_and_cpu_uses_actual_reference():
    from autofusex.fuzz.client.execution import execute
    from fuzz_fixtures import bundle as materialize
    class Adapter:
        name = 'af-mlir'
        def check(self, *args): pass
        def prepare(self, *args): return None
        def inputs(self, *args): return None
        def invoke(self, *args): return 'af-output'
        def synchronize(self): pass
        def cpu_reference(self, *args): return 'actual-cpu-output'
        def compare(self, output, expected):
            assert expected == 'actual-cpu-output'
            return False
        def diagnose_accuracy(self, *args): return {'max_abs': 2.0}
        def secondary_output(self, *args): pytest.fail('automatic secondary forbidden')
        def outputs_equal(self, left, right):
            assert (left, right) == ('af-output', 'eager-output')
            return False
    result = execute(materialize('14615016373650978276328604632813839116929039597610'), Adapter(), EvaluationConfig())
    assert result['status'] == 'fail'
    assert result['oracle_id'] == 'torch-eager-cpu/1'
    assert all(row['correctness'] == 'fail' for row in result['shape_results'])
    assert all(not item['secondary_used'] and item['basis'] == 'cpu' and 'secondary_pass' not in item
               for item in json.loads(result['execution']['diagnostics'])['shapes'].values())
