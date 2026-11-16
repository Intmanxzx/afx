import json
import pytest
from autofusex.fuzz.client.contracts import Seed


def test_retired_c_seed_is_rejected():
    with pytest.raises(ValueError):
        Seed.parse('C16')


def test_actual_nnsmith_graph_and_cpu(tmp_path):
    pytest.importorskip('nnsmith')
    pytest.importorskip('torch')
    from autofusex.fuzz.client.generation import materialize, workload_id
    from autofusex.fuzz.client.generation.runtime import source_evidence
    from autofusex.fuzz.client.execution import execute
    from autofusex.fuzz.client.backends import create_backend
    from autofusex.fuzz.client.contracts import ExecutionConfig,EvaluationConfig
    from autofusex.fuzz.client.storage import ResultRepository
    from autofusex.fuzz.client.case_spec import encode,decode
    evidence=source_evidence()
    from autofusex.fuzz.client.generation.sampling import SamplingConfig
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
    from autofusex.fuzz.client.storage.queries import ResultQueries
    page=ResultQueries(repo.path).page()
    assert page['items']
