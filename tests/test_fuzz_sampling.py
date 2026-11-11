"""Sampling regression: protocol, persisted campaigns, actual final signatures/bytes."""
from dataclasses import asdict, replace
import json
import math
import struct
import pytest
from autofusedebugger.fuzz.contracts import Seed, TensorSpec, CaseSpec, canonical_json
from autofusedebugger.fuzz.generation.sampling import SamplingConfig, decode, PROFILES, VALUE_MODES


def test_packed_config_replay_and_bounds():
    for profile in PROFILES:
        for mode in VALUE_MODES:
            config = SamplingConfig.profile(profile, value_dist_mode=mode)
            for dynamic in (False, True):
                seed = config.seed(2**32-1, dynamic)
                assert Seed.parse(seed.short) == Seed.parse(str(seed)) == seed
                assert decode(seed) == (config, 2**32-1)
    custom = SamplingConfig(dim_max=1000000, max_elem_per_tensor=10000000)
    assert decode(custom.seed(42))[0] == custom
    for kwargs in ({'dim_min': 0}, {'dim_max': 2**24}, {'dim_min': 10, 'dim_max': 2},
                   {'max_nodes': 0}, {'dim_dist_mode': 'normal'}, {'timeout_seconds': 512}):
        with pytest.raises(ValueError): SamplingConfig(**kwargs)
    with pytest.raises(ValueError): decode(Seed(4*2**128))
    assert Seed.parse('C16').short == 'C000000016'
    assert Seed.parse('D16').entropy == 42


def test_role_aware_values_and_captured_metadata():
    from autofusedebugger.fuzz.generation import materialize_spec
    for mode in VALUE_MODES:
        seed = SamplingConfig(value_dist_mode=mode).seed(42)
        tensors = tuple(TensorSpec(n, 'float32', (20000,), (1,), alias_group=i)
                        for i,n in enumerate(('input', 'weight')))
        spec = CaseSpec(seed, canonical_json({'nodes': []}), tensors[:1], tensors[1:])
        bundle = materialize_spec(spec)
        assert bundle == materialize_spec(spec)
        x,w = [struct.unpack('<20000f', b) for b in bundle.storages]
        if mode == 'uniform':
            assert all(3 <= v <= 7 for v in (*x,*w))
        elif mode == 'log_scale':
            assert min(x) < -100 and max(x) > 100 and min(map(abs,x)) < .001
        elif mode == 'llm':
            assert .018 < math.sqrt(sum(v*v for v in w)/len(w)) < .022
            assert .025 < sum(abs(v)>10 for v in x)/len(x) < .065
        else:
            assert all(0 <= v <= 1 for v in x)
            assert .9 < sum(v*v for v in w) < 1.1


def test_campaign_config_is_persisted_and_legacy_campaign_kept(tmp_path):
    from autofusedebugger.fuzz.scheduling.campaign import initialize, refill
    from autofusedebugger.fuzz.scheduling.ledger import TaskLedger
    from autofusedebugger.fuzz.contracts import ExecutionConfig, EvaluationConfig
    ledger = TaskLedger(tmp_path/'queue')
    config = SamplingConfig.profile('large', value_dist_mode='llm')
    initialize(ledger, [ExecutionConfig('cpu','cpu',static_only=True)], EvaluationConfig(), 2,
               sampling=asdict(config))
    assert refill(ledger)
    task = ledger.claim()
    assert decode(task['seed'])[0] == config
    assert Seed.parse(task['seed']).version == 5
    ledger.complete(task['id'], task['claim'])
    with ledger._connection() as conn:
        payload = json.loads(conn.execute('SELECT payload FROM campaign').fetchone()[0])
        del payload['sampling']
        conn.execute('UPDATE campaign SET payload=?', (canonical_json(payload),))
    assert refill(ledger)
    assert Seed.parse(ledger.claim()['seed']).version == 3


def test_actual_final_shapes_and_replay():
    pytest.importorskip('nnsmith')
    from autofusedebugger.fuzz.generation import plan, materialize, materialize_spec, workload_id
    from autofusedebugger.fuzz.generation import nnsmith_sampled, nnsmith_v3, nnsmith_dynamic
    old = {s: materialize(s) for s in ('C16', 'D16')}
    specs = [plan(SamplingConfig().seed(i)) for i in range(16)]
    axes = [d for p in specs for t in p.inputs for d in t.shape]
    assert max(axes) > 8192  # final inputs, not merely the initial placeholder
    assert len({len(json.loads(p.graph_json)['nodes']) for p in specs}) > 1
    assert all(math.prod(t.shape) <= 524288 for p in specs
               for t in (*p.inputs,*p.weights,*p.output_samples[0]))
    seed = SamplingConfig(value_dist_mode='llm').seed(3)
    first = materialize(seed)
    nnsmith_sampled.plan.cache_clear()
    assert materialize(seed) == first == materialize_spec(first.spec)
    for s in old:
        nnsmith_v3.plan.cache_clear(); nnsmith_dynamic.plan.cache_clear()
        assert materialize(s) == old[s]
        assert workload_id(materialize(s)) == workload_id(old[s])


def test_actual_dtype_modes_dynamic_and_cpu():
    pytest.importorskip('nnsmith')
    pytest.importorskip('torch')
    from autofusedebugger.fuzz.generation import materialize
    from autofusedebugger.fuzz.backends import create_backend
    from autofusedebugger.fuzz.execution import execute
    from autofusedebugger.fuzz.contracts import ExecutionConfig, EvaluationConfig
    for dtype in ('float16', 'mixed'):
        config = SamplingConfig.profile('small', dtype_mode=dtype, value_dist_mode='recommendation')
        bundle = materialize(config.seed(3, dynamic=True))
        assert len(bundle.spec.shape_samples) == 3
        if dtype == 'float16':
            assert {t.dtype for t in (*bundle.spec.inputs,*bundle.spec.weights)} == {'float16'}
        result = execute(bundle, create_backend(ExecutionConfig('cpu','cpu',1)), EvaluationConfig())
        assert result['status'] == 'ok'
        assert result['execution']['case_spec']


def test_historical_profiles_rank_and_axis_policy():
    pytest.importorskip('nnsmith')
    import random
    from nnsmith.abstract.op import weighted_axis_choice, weighted_axis_sample, ReduceSum, MultiAxisReduceBase
    from nnsmith.graph_gen import BaseGen
    from autofusedebugger.nnsmith_integration.generation import TENSOR_SCALE_PROFILES
    for name, (lo, hi, budget, timeout, mode) in PROFILES.items():
        old = TENSOR_SCALE_PROFILES[name]
        assert (lo,hi) == old['concr_ph_dim_rng']
        assert (budget,timeout*1000,mode) == (old['max_elem_per_tensor'],old['timeout_ms'],old['dim_dist_mode'])
    state = random.getstate()
    try:
        random.seed(42)
        axes = [weighted_axis_choice(5) for _ in range(10000)]
        assert .63 < sum(a in {0,4} for a in axes)/len(axes) < .70
        assert all(len(set(weighted_axis_sample(5,3))) == 3 for _ in range(100))
        assert issubclass(ReduceSum, MultiAxisReduceBase)
        dummy = type('Ranks', (), {'rank_choices': tuple(range(6))})()
        ranks = [BaseGen.random_rank(dummy) for _ in range(6000)]
        assert all(800 < ranks.count(rank) < 1200 for rank in range(6))
    finally:
        random.setstate(state)


@pytest.mark.parametrize('mode', ('uniform', 'half_normal', 'log_normal'))
def test_explicit_million_axis_reaches_final_inputs(mode):
    pytest.importorskip('nnsmith')
    from autofusedebugger.fuzz.generation import plan
    config = SamplingConfig(dim_min=1000000, dim_max=1000000,
                            max_elem_per_tensor=10000000, dim_dist_mode=mode,
                            dtype_mode='float32')
    spec = plan(config.seed(3))
    assert 1000000 in {d for t in spec.inputs for d in t.shape}
    assert all(math.prod(t.shape) <= config.max_elem_per_tensor for t in spec.inputs)
