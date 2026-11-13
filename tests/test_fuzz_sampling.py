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


def test_all_dtype_mode_restores_pre_refactor_dtype_surface():
    from autofusedebugger.fuzz.generation.sampling import DTYPE_MODES
    from autofusedebugger.fuzz.generation.nnsmith_sampled import _canonical_storage
    from autofusedebugger.fuzz.tensors import decode_storage

    assert DTYPE_MODES[:3] == ('float32', 'float16', 'mixed')
    assert 'all' in DTYPE_MODES
    config = SamplingConfig(dtype_mode='all')
    assert decode(config.seed(17))[0] == config
    for dtype, value in {
        'bool': 1, 'uint8': 255, 'uint16': 65535, 'uint32': 2**32 - 1,
        'int8': -1, 'int16': -1, 'int32': -1, 'int64': -1,
        'float16': 1.5, 'bfloat16': 1.5, 'float32': 1.5, 'float64': 1.5,
        'complex64': 1 + 2j, 'complex128': 1 + 2j,
    }.items():
        raw = _canonical_storage(dtype, [value])
        assert decode_storage(raw, dtype)[0] == value


@pytest.mark.parametrize('mode', ('uniform', 'log_scale', 'llm', 'recommendation'))
def test_legacy_numeric_stream_is_unchanged(mode):
    import hashlib
    import random
    from autofusedebugger.fuzz.generation.nnsmith_sampled import materialize_spec
    for dtype, fmt in [('float16', 'e'), ('float32', 'f')]:
        seed = SamplingConfig(dtype_mode='mixed', value_dist_mode=mode).seed(42)
        tensor = TensorSpec('input', dtype, (100,), (1,))
        spec = CaseSpec(seed, canonical_json({'nodes': []}), (tensor,), ())
        rng = random.Random(int.from_bytes(hashlib.sha256(
            f'nnsmith-sampled/5/{seed}/input'.encode()).digest(), 'little'))
        def old_value():
            if mode == 'uniform': return 3 + 4 * rng.random()
            if mode == 'log_scale':
                sign = -1 if rng.random() < .5 else 1
                return sign * 10 ** (8 * rng.random() - 4)
            if mode == 'llm':
                base = rng.gauss(0, .2)
                outlier = rng.random() < .05
                amplification = 100 + 9900 * rng.random()
                return base * (amplification if outlier else 1)
            return rng.random()
        expected = b''.join(struct.pack('<' + fmt, old_value()) for _ in range(100))
        assert materialize_spec(spec).storages[0] == expected


def test_generation_operator_catalog_matches_pre_refactor_allowlist():
    """The refactor must not shrink the historical operator factor surface."""
    from autofusedebugger.fuzz.generation.operator_catalog import (
        LEGACY_OPS, A3_OPS, A5_ONLY_OPS, ALL_OPS)

    assert len(LEGACY_OPS) == 9
    assert len(A3_OPS) == 30
    assert len(A5_ONLY_OPS) == 20
    assert len(ALL_OPS) == 50
    assert set(A3_OPS).isdisjoint(A5_ONLY_OPS)
    assert set(ALL_OPS) == set(A3_OPS) | set(A5_ONLY_OPS)
    assert A3_OPS == ('Add', 'Sub', 'Mul', 'Max', 'Min', 'Div', 'Pow', 'Abs',
        'Neg', 'Sigmoid', 'Clip', 'Equal', 'And', 'Or', 'Where', 'ReduceSum',
        'ReduceMean', 'ReduceMin', 'ReduceMax', 'Reshape', 'Transpose', 'Squeeze',
        'Concat', 'Cast', 'Slice', 'ExpandLast1', 'ExpandLast2', 'ExpandLast3',
        'ExpandLast4', 'Sqrt')
    assert A5_ONLY_OPS == ('Greater', 'Less', 'Ceil', 'Floor', 'Round', 'Sin',
        'Cos', 'Tan', 'Atan', 'Asin', 'Acos', 'GELU', 'LeakyReLU', 'Tril',
        'Triu', 'IndexSelect', 'Gather', 'LayerNorm', 'MatMul', 'Linear')


def test_sampled_protocol_uses_pre_refactor_default_a3_scope():
    from autofusedebugger.fuzz.generation.nnsmith_sampled import OPS
    from autofusedebugger.fuzz.generation.operator_catalog import HISTORICAL_OPS

    assert OPS == HISTORICAL_OPS
    assert len(OPS) == 30


def test_generation_operator_catalog_has_stable_graph_names():
    from autofusedebugger.fuzz.generation.operator_catalog import ALL_OPS, NAMES

    assert set(ALL_OPS) <= set(NAMES)
    assert len(set(NAMES.values())) == len(NAMES)
    assert NAMES['ReduceSum'] == 'sum'
    assert NAMES['ReduceMean'] == 'mean'
    assert NAMES['ReduceMin'] == 'amin'
    assert NAMES['ReduceMax'] == 'amax'


def test_sampling_factors_match_pre_refactor_generation_ranges():
    """Audit every configurable factor against the old GenerationConfig."""
    from autofusedebugger.fuzz.generation.sampling import (
        DTYPE_MODES, DIM_MODES, VALUE_MODES, PROFILES, SamplingConfig)
    pytest.importorskip('nnsmith')
    from autofusedebugger.nnsmith_integration.generation import (
        GenerationConfig, TENSOR_SCALE_PROFILES)

    # Shape/range/element-budget/timeout factors are copied field-for-field.
    for name, (dim_min, dim_max, budget, timeout, dim_mode) in PROFILES.items():
        old = TENSOR_SCALE_PROFILES[name]
        assert (dim_min, dim_max) == old['concr_ph_dim_rng']
        assert budget == old['max_elem_per_tensor']
        assert timeout * 1000 == old['timeout_ms']
        assert dim_mode == old['dim_dist_mode']

    # Node count, rank/axis protocol, and value modes retain their old bounds.
    assert SamplingConfig().max_nodes == GenerationConfig().max_nodes == 5
    assert 2 <= SamplingConfig().max_nodes < 256
    assert DIM_MODES == ('uniform', 'half_normal', 'log_normal')
    assert VALUE_MODES == ('uniform', 'log_scale', 'llm', 'recommendation')
    assert DTYPE_MODES[:3] == ('float32', 'float16', 'mixed')
    assert 'all' in DTYPE_MODES
    assert GenerationConfig().dynamic_variants == 3
