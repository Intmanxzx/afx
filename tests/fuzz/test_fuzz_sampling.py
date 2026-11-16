"""Current sampling factors and explicit retirement of internal-test seeds."""
import json
import random
import struct
import pytest
from autofusex.fuzz.contracts import Seed,TensorSpec,CaseSpec,canonical_json
from autofusex.fuzz.generation.sampling import SamplingConfig,decode,PROFILES,METHODS


@pytest.mark.parametrize('dynamic',[False,True])
@pytest.mark.parametrize('count',range(2,9))
@pytest.mark.parametrize('method',METHODS)
def test_sampling_roundtrip(dynamic,count,method):
    config=SamplingConfig.profile('small',dynamic_variants=count,method=method)
    seed=config.seed(0xffffffff,dynamic)
    assert Seed.parse(seed.short)==Seed.parse(str(seed))==seed
    assert decode(seed)==(config,0xffffffff)
    assert seed.dynamic==dynamic


@pytest.mark.parametrize('short',['C16','D16','E16','F16','G16','H16','I16','J16'])
def test_retired_seed_is_rejected(short):
    with pytest.raises(ValueError):
        Seed.parse(short)


@pytest.mark.parametrize('kwargs',[{'max_nodes':65},{'dynamic_variants':1},{'dynamic_variants':9},{'method':'auto'}, {'dim_max':2**24},{'max_elem_per_tensor':2**25},{'timeout_seconds':512}])
def test_config_limits_fail_before_execution(kwargs):
    with pytest.raises(ValueError):SamplingConfig(**kwargs)


@pytest.mark.parametrize('mode',['uniform','log_scale','llm','recommendation'])
def test_values_are_reproducible_and_complex_has_imaginary_part(mode):
    from autofusex.fuzz.generation.nnsmith_sampled import materialize_spec
    seed=SamplingConfig(value_dist_mode=mode).seed(42)
    t=TensorSpec('input','complex64',(64,),(1,))
    spec=CaseSpec(seed,canonical_json({'nodes':[]}),(t,),(),((t,),),())
    before=random.getstate()
    a=materialize_spec(spec);b=materialize_spec(spec)
    assert a==b and random.getstate()==before
    values=struct.unpack('<'+'f'*128,a.storages[0])
    assert any(v!=0 for v in values[1::2])
    assert values[::2]!=values[1::2]


def test_integer_rounding_matches_old_rule_not_truncation():
    from autofusex.fuzz.generation.nnsmith_sampled import _canonical_storage
    assert struct.unpack('<iii',_canonical_storage('int32',[3.2,3.8,6.8]))==(3,4,7)
    assert struct.unpack('<ii',_canonical_storage('int32',[.2,.8]))==(0,1)
    assert struct.unpack('<iiii',_canonical_storage('int32',[-1.5,-2.5,1.5,2.5]))==(-2,-2,2,2)


def test_real_integer_and_complex_input_distributions():
    from autofusex.fuzz.generation.nnsmith_sampled import materialize_spec
    for mode,expected in [('uniform',{3,4,5,6,7}),('recommendation',{0,1})]:
        seed=SamplingConfig(value_dist_mode=mode).seed(42)
        t=TensorSpec('input','int32',(1024,),(1,))
        spec=CaseSpec(seed,canonical_json({'nodes':[]}),(t,),(),((t,),),())
        assert set(struct.unpack('<'+'i'*1024,materialize_spec(spec).storages[0]))==expected


def test_current_campaign_embeds_all_factors(tmp_path):
    from autofusex.fuzz.scheduling.ledger import TaskLedger
    from autofusex.fuzz.scheduling.campaign import initialize,refill
    from autofusex.fuzz.contracts import ExecutionConfig,EvaluationConfig
    from dataclasses import asdict
    config=SamplingConfig.profile('small',dynamic_variants=8,method='symbolic')
    ledger=TaskLedger(tmp_path/'queue')
    initialize(ledger,[ExecutionConfig('cpu','cpu')],EvaluationConfig(),1,sampling=asdict(config))
    assert refill(ledger)
    seed=Seed.parse(ledger.claim()['seed'])
    assert decode(seed)[0]==config


def test_ops_post_filter_does_not_change_seed():
    from autofusex.fuzz.generation.operator_catalog import normalize_ops,matches_ops
    seed=SamplingConfig().seed(42)
    assert normalize_ops('Add,core.ReduceMean,aten.add.Tensor')==('add','mean','aten.add.Tensor')
    assert matches_ops('aten.add.Tensor',{'add'})
    assert not matches_ops('aten.add.Scalar',{'aten.add.Tensor'})
    assert SamplingConfig().seed(42)==seed


def test_legacy_space_cannot_alias_current_protocol():
    for version in (11,12):
        with pytest.raises(ValueError,match='unsupported generation'):
            Seed((version-1)*2**128+42)


@pytest.mark.parametrize('scheduling',['staged','combined'])
def test_cli_retired_replay_is_clear_error_without_runtime(scheduling,capsys):
    from autofusex.fuzz.cli import main
    assert main(['replay','C16','--scheduling',scheduling])==2
    assert 'seed must be canonical decimal digits' in capsys.readouterr().err
