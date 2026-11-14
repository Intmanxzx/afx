import json
import pytest
from autofusedebugger.fuzz.contracts import Seed


def test_c_seed_is_self_contained():
    assert Seed.parse('C000000016').value==2*2**128+42
    assert Seed.parse('C16').short=='C000000016'


def test_actual_nnsmith_graph_and_cpu(tmp_path):
    pytest.importorskip('nnsmith')
    pytest.importorskip('torch')
    from autofusedebugger.fuzz.generation import materialize, workload_id
    from autofusedebugger.fuzz.generation.runtime import source_evidence
    from autofusedebugger.fuzz.execution import execute
    from autofusedebugger.fuzz.backends import create_backend
    from autofusedebugger.fuzz.contracts import ExecutionConfig,EvaluationConfig
    from autofusedebugger.fuzz.storage import ResultRepository
    from autofusedebugger.fuzz.case_spec import encode,decode
    evidence=source_evidence()
    from autofusedebugger.fuzz.generation.sampling import SamplingConfig
    config=SamplingConfig.profile('small',max_nodes=3,dtype_mode='float32')
    seed=config.seed(0)
    bundle=materialize(seed); materialize(config.seed(1)); replay=materialize(seed)
    assert bundle==replay and workload_id(bundle)==workload_id(replay)
    assert json.loads(bundle.spec.graph_json)['generator']==evidence['generator']
    assert bundle.spec.inputs and bundle.spec.output_samples[0]
    assert decode(encode(bundle.spec))==bundle.spec
    result=execute(bundle,create_backend(ExecutionConfig('cpu','cpu',1,static_only=True)),EvaluationConfig())
    assert result['status']=='ok'
    repo=ResultRepository(tmp_path/'results');repo.save(result)
    from autofusedebugger.fuzz.storage.queries import ResultQueries
    page=ResultQueries(repo.path).page()
    assert page['items']
