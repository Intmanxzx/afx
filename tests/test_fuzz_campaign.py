from dataclasses import asdict
import json
import pytest
from autofusedebugger.fuzz.scheduling.ledger import TaskLedger
from autofusedebugger.fuzz.scheduling.campaign import initialize,refill,stop
from autofusedebugger.fuzz.contracts import ExecutionConfig,EvaluationConfig


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


def test_secondary_reference_only_on_disagreement():
    from autofusedebugger.fuzz.execution import execute
    from autofusedebugger.fuzz.generation import materialize
    class Adapter:
        name='af-mlir'
        def check(self,*a):pass
        def prepare(self,*a):return None
        def inputs(self,*a):return None
        def invoke(self,*a):return None
        def synchronize(self):pass
        def compare(self,*a):return False
        def diagnose_accuracy(self,*a):return {'max_abs':1.0,'mismatches':1,'elements':2}
        def secondary_output(self,*a):return None
        def outputs_equal(self,*a):return True
    result=execute(materialize('42'),Adapter(),EvaluationConfig())
    assert result['status']=='ok'
    detail=json.loads(result['execution']['diagnostics'])['shapes']['0']
    assert detail['cpu_pass'] is False and detail['secondary_pass'] is True
    adapter=Adapter();adapter.compare=lambda *a:True
    adapter.secondary_output=lambda *a:pytest.fail('secondary should not execute')
    assert execute(materialize('42'),adapter,EvaluationConfig())['status']=='ok'


def test_bounded_redacted_failure_excerpt():
    from autofusedebugger.fuzz.failure_details import excerpt,validate
    try:raise RuntimeError('token=secretvalue '+'x'*2000)
    except RuntimeError as exc:text=excerpt(exc)
    assert len(text)<=1600 and 'secretvalue' not in text
    validate(json.dumps({'version':1,'error':text,'shapes':{}}))


def test_native_core_log_is_bounded_and_redacted():
    import os
    from autofusedebugger.fuzz.core_logs import capture_tail
    with capture_tail() as tail:
        os.write(2, b'x' * 40000 + b'\ncompiler error token=private-value\n')
    assert len(tail()) <= 1600
    assert 'compiler error' in tail() and 'private-value' not in tail()


def test_campaign_capacity_is_enforced(tmp_path):
    import pytest
    from autofusedebugger.fuzz.scheduling.ledger import TaskLedger
    from autofusedebugger.fuzz.scheduling.campaign import initialize
    from autofusedebugger.fuzz.contracts import ExecutionConfig, EvaluationConfig
    with pytest.raises(ValueError, match='capacity'):
        initialize(TaskLedger(tmp_path/'queue', max_active=1),
                   [ExecutionConfig('cpu','cpu'), ExecutionConfig('af-mlir','npu:0')],
                   EvaluationConfig(), None)


def test_torch_cpu_reference_and_error_metrics():
    from autofusedebugger.fuzz.backends.torch import TorchAdapter
    from autofusedebugger.fuzz.generation import materialize
    adapter=TorchAdapter(ExecutionConfig('cpu','cpu'))
    bundle=materialize('B000000016')
    try:
        adapter.check(bundle, EvaluationConfig())
        expected=adapter.cpu_reference(bundle, 0)
        assert adapter.compare(expected, expected)
        changed=(expected[0] + 1,)
        assert not adapter.compare(changed, expected)
        metrics=adapter.diagnose_accuracy(changed, expected)
        assert metrics['max_abs']==1.0
        assert metrics['mismatches']==metrics['elements']
    finally:
        adapter.close()


def test_default_replay_uses_both_mandatory_backends(monkeypatch):
    from autofusedebugger.fuzz import cli
    seen=[]
    monkeypatch.setattr(cli,'execute_args',lambda args: seen.append((args.backend,args.device)) or 0)
    assert cli.replay(['B000000016'])==0
    assert seen==[('cpu','cpu'),('af-mlir','npu:0')]


def test_secondary_disagreement_stays_failed_and_cpu_uses_actual_reference():
    from autofusedebugger.fuzz.execution import execute
    from autofusedebugger.fuzz.generation import materialize
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
        def secondary_output(self, *args): return 'eager-output'
        def outputs_equal(self, left, right):
            assert (left, right) == ('af-output', 'eager-output')
            return False
    result = execute(materialize('42'), Adapter(), EvaluationConfig())
    assert result['status'] == 'fail'
    assert result['oracle_id'] == 'torch-eager-cpu/1'
    assert all(row['correctness'] == 'fail' for row in result['shape_results'])
    assert all(not item['secondary_pass'] and item['basis'] == 'cpu+eager-npu'
               for item in json.loads(result['execution']['diagnostics'])['shapes'].values())
