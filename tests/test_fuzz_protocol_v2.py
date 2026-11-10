import pytest
from autofusedebugger.fuzz.generation import materialize, workload_id
from autofusedebugger.fuzz.generation.categories import categories, OP_CATEGORY
from autofusedebugger.fuzz.contracts import ExecutionConfig, EvaluationConfig
from autofusedebugger.fuzz.execution import execute
from autofusedebugger.fuzz.backends.torch import TorchAdapter

@pytest.mark.parametrize('entropy',range(12))
def test_explicit_transpose_protocol_cpu(entropy):
    pytest.importorskip("torch")
    bundle=materialize(str(2**128+entropy))
    assert set(categories(bundle.spec)) <= {'Elementwise','Reduce','Transpose'}
    assert 'Transpose' in categories(bundle.spec)
    assert OP_CATEGORY['transpose']=='Transpose'
    result=execute(bundle,TorchAdapter(ExecutionConfig('cpu-eager','cpu')),EvaluationConfig())
    assert result['status']=='ok', result
    assert len(result['shape_results'])==3
    assert workload_id(materialize(str(2**128+entropy)))==workload_id(bundle)


@pytest.mark.parametrize('entropy,expected', [
    (0,'24182e954f68eba4e62ac45d6719a8d54d61aa228d5b701b1eb9a7db5685825f'),
    (1,'2ea5d81a3fe857fe8c02b26176126ab2843ea1c473624e89df9b5585c0280ae9'),
    (42,'5a07553942a30eaa848b71a7e1bbae0aef80130b471338cf3c1ac130aff883de'),
    (389609433,'a54adb0e88c191c50db14cb026b5f1f7c22784f7d4c629f527267afd332a6027'),
    (2**128-1,'9b8f6497acdaa9a90c7f0e13a8b6cf0fa404b9ce813a5ae13ce0c058ecd1bb82'),
])
def test_v2_frozen_workload_vectors(entropy,expected):
    assert workload_id(materialize(str(2**128+entropy)))==expected
