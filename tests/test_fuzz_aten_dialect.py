"""One modeling test per target, exercising all declared dtype signatures."""
import json
import random
import pytest

from autofusedebugger.nnsmith_integration.aten_catalog import RULES


@pytest.fixture(scope='module')
def runtime():
    torch = pytest.importorskip('torch')
    pytest.importorskip('nnsmith')
    from autofusedebugger.nnsmith_integration.aten_dialect import classes, validate_runtime
    validate_runtime()
    torch.set_num_threads(1)
    return torch, {c.rule.target: c for c in classes()}


@pytest.mark.parametrize('rule', RULES, ids=lambda r:r.target)
def test_every_overload_cpu_signatures(runtime, rule):
    torch, registry = runtime
    import z3
    from nnsmith.abstract.tensor import AbsTensor
    from autofusedebugger.nnsmith_integration.aten_dialect import materialize
    from autofusedebugger.fuzz.generation.torch_dialect import capture, decode
    cls = registry[rule.target]
    successful = 0
    for signature in cls.in_dtypes:
        for shape in ([2,3], [], [2,1,3]):
            state = random.getstate()
            try:
                random.seed(42)
                op = cls()
                if len(shape) not in op.inp_ranks[0]: continue
                current = list(shape)
                if rule.family == 'view':
                    op.extra_attrs['dim'] = 0
                    if rule.mode in {'squeeze','squeeze_all'}: current[0] = 1
                    if rule.mode == 't' and len(current) > 2: continue
                    if rule.mode == 'squeeze_all' and len(current)>2: current=[1,2,3]
                    if rule.mode == 'real_complex': current[-1] = 2
                    if rule.mode == 'unbind': current[0] = 2
                tensors = [AbsTensor(list(current), dtype) for dtype in signature]
                solver = z3.Solver(); solver.add(*op.checked_requires(tensors))
                assert solver.check() == z3.sat
                expected = op.checked_type_transfer(tensors)
                op.bind_input_like(tensors); op.bind_output_like(expected)
                # Model inference with configured metadata must not consume RNG.
                rng = random.getstate()
                assert op.checked_type_transfer(tensors) == expected
                assert random.getstate() == rng
                inputs = []
                for tensor in tensors:
                    size = __import__('math').prod(tensor.shape)
                    value = torch.arange(1,size+1).reshape(tensor.shape)
                    dtype = getattr(torch,str(tensor.dtype))
                    if dtype.is_floating_point or dtype.is_complex: value = value/8+0.25
                    inputs.append(value.to(dtype))
                try:
                    result = materialize(op)(*inputs)
                except NotImplementedError as exc:
                    # Explicit CPU dtype gaps are allowed case skips, never
                    # a reason to remove the overload from the generator.
                    assert str(signature[0]) in {'float16','bfloat16'}
                    assert rule.target.startswith('aten.special_')
                    assert 'not implemented' in str(exc)
                    continue
                successful += 1
                actual = [result] if rule.outputs==1 else result
                assert [(list(x.shape),str(x.dtype)) for x in actual] == [(x.shape,'torch.'+str(x.dtype)) for x in expected]
                loaded = cls.__new__(cls); loaded.__dict__.update(decode(capture(op,9)['state']))
                restored = materialize(loaded)(*inputs)
                restored = [restored] if rule.outputs==1 else restored
                for a,b in zip(actual,restored): torch.testing.assert_close(a,b,equal_nan=True)
            finally:
                random.setstate(state)
    assert successful > 0


def test_new_protocol_roundtrip_and_frozen_catalog(runtime):
    from autofusedebugger.fuzz.contracts import Seed
    from autofusedebugger.fuzz.generation.sampling import SamplingConfig,decode
    from autofusedebugger.fuzz.generation.torch_dialect import classes
    old = tuple(classes())
    from autofusedebugger.nnsmith_integration.aten_dialect import classes as aten_classes
    assert len(aten_classes()) == len(RULES)
    assert tuple(classes()) == old and len(old)==79
    for dynamic in [False,True]:
        s=SamplingConfig.profile('small').seed(42,dynamic=dynamic,aten=True)
        assert s.version == (10 if dynamic else 9)
        assert Seed.parse(s.short)==s==Seed.parse(str(s))
        assert decode(s)[1]==42
    assert all(not any(x in r.target for x in ['conv','matmul','addmm','bmm']) for r in RULES)


def test_new_campaign_keeps_persisted_old_campaigns(tmp_path):
    from autofusedebugger.fuzz.scheduling.ledger import TaskLedger
    from autofusedebugger.fuzz.scheduling.campaign import initialize,refill
    from autofusedebugger.fuzz.contracts import ExecutionConfig,EvaluationConfig,Seed
    l=TaskLedger(tmp_path/'queue')
    initialize(l,[ExecutionConfig('cpu','cpu',static_only=True)],EvaluationConfig(),2)
    assert refill(l)
    t=l.claim(); assert Seed.parse(t['seed']).version==9
    l.complete(t['id'],t['claim'])
    with l._connection() as db:
        p=json.loads(db.execute('select payload from campaign').fetchone()[0]);del p['aten_protocol']
        db.execute('update campaign set payload=?',(json.dumps(p),))
    assert refill(l)
    assert Seed.parse(l.claim()['seed']).version==7


def test_exact_and_family_post_filter():
    from autofusedebugger.fuzz.generation.operator_catalog import normalize_ops,matches_ops
    assert normalize_ops('aten.ge.Tensor,ge')==('aten.ge.Tensor','ge')
    assert matches_ops('aten.add.Tensor',{'add'})
    assert matches_ops('aten.transpose.int',{'transpose'})
    assert not matches_ops('aten.add.Scalar',{'aten.add.Tensor'})


def test_random_graph_replay_cpu_and_multiple_outputs(runtime):
    from autofusedebugger.fuzz.generation import materialize,nnsmith_sampled,workload_id
    from autofusedebugger.fuzz.generation.sampling import SamplingConfig
    from autofusedebugger.fuzz.backends import create_backend
    from autofusedebugger.fuzz.contracts import ExecutionConfig,EvaluationConfig
    from autofusedebugger.fuzz.execution import execute
    from autofusedebugger.fuzz.case_spec import encode,decode
    config=SamplingConfig.profile('small',dtype_mode='float32',max_nodes=4)
    successes=0
    for entropy in range(12):
        bundle=materialize(config.seed(entropy,aten=True))
        nnsmith_sampled.plan.cache_clear()
        assert materialize(bundle.spec.seed)==bundle
        assert decode(encode(bundle.spec))==bundle.spec
        graph=json.loads(bundle.spec.graph_json)
        assert all(node['op'].startswith('aten.') for node in graph['nodes'])
        adapter=create_backend(ExecutionConfig('cpu','cpu',1))
        result=execute(bundle,adapter,EvaluationConfig(correctness=False))
        assert result['status']=='ok',result
        successes+=1
    assert successes==12


def test_g_seed_fingerprints_unchanged(runtime):
    from autofusedebugger.fuzz.generation import materialize,workload_id
    from autofusedebugger.fuzz.generation.sampling import SamplingConfig
    expected={0:'524d1e4ca94e8b4a15716d39fc55bdb3f2f5cd83a2e5e228494c3e598938c152',
              1:'c9020025ed25c7b7939cbec0ba9cfd60a73aeb0881a283786645731d1ff1bb33',
              2:'c2089257446a184b0ca161a7b29b8c8a6d9884a3ffc6f94e1fab2d0487affeec',
              10:'a6ef3ece4465d063fa18a391f8d90827fff994375e26d340536c1421f305945c'}
    config=SamplingConfig.profile('small',dtype_mode='float32',max_nodes=3)
    for entropy,digest in expected.items():
        assert workload_id(materialize(config.seed(entropy,full_range=True)))==digest


def test_invalid_symbolic_shapes_are_rejected(runtime):
    import z3
    from nnsmith.abstract.tensor import AbsTensor
    from nnsmith.abstract.dtype import DType
    cls=runtime[1]
    for target,shapes in [('aten.add.Tensor',[[2,3],[4,3]]),
                          ('aten.expand_as.default',[[2,3],[2,4]]),
                          ('aten.view_as.default',[[2,3],[7]]),
                          ('aten.unbind.int',[[3,4]])]:
        op=cls[target]();op.extra_attrs['dim']=0
        tensors=[AbsTensor(s,DType.float32) for s in shapes]
        solver=z3.Solver();solver.add(*op.checked_requires(tensors))
        assert solver.check()==z3.unsat


def test_functional_pointwise_tensor_catalog_has_no_silent_gaps(runtime):
    torch,_ = runtime
    from autofusedebugger.nnsmith_integration.aten_catalog import BY_TARGET,inventory
    gaps=[]
    for name in sorted({s.name[6:] for s in torch._C._jit_get_all_schemas() if s.name.startswith('aten::')}):
        try: packet=getattr(torch.ops.aten,name)
        except AttributeError: continue
        for overload in packet.overloads():
            op=getattr(packet,overload);schema=op._schema
            if torch.Tag.pointwise not in op.tags: continue
            if any(a.alias_info is not None and a.alias_info.is_write for a in schema.arguments): continue
            target='aten.'+name+'.'+overload
            if target not in BY_TARGET: gaps.append(target)
    # aten.equal returns a Python bool, not a Tensor graph edge. Explicit gap.
    assert gaps == ['aten.equal.default']
    audit=inventory()
    assert {r['target'] for r in audit if r['status']=='modeled'} == set(BY_TARGET)
    assert any(r['target']=='aten.as_strided.default' and r['status']=='requires_storage_alias_model' for r in audit)


def test_tuple_results_feed_downstream_nodes_and_repository(runtime,tmp_path):
    from autofusedebugger.fuzz.generation import materialize,workload_id
    from autofusedebugger.fuzz.generation.sampling import SamplingConfig
    from autofusedebugger.fuzz.backends import create_backend
    from autofusedebugger.fuzz.contracts import ExecutionConfig,EvaluationConfig
    from autofusedebugger.fuzz.execution import execute
    from autofusedebugger.fuzz.storage import ResultRepository
    config=SamplingConfig.profile('small',max_nodes=5,dtype_mode='all')
    bundle=materialize(config.seed(6,aten=True))
    graph=json.loads(bundle.spec.graph_json)
    multi=[n for n in graph['nodes'] if len(n['outputs'])==2]
    assert multi
    # Consume the second return explicitly, independently of random insertion.
    from copy import deepcopy
    from nnsmith.abstract.tensor import AbsTensor
    from nnsmith.abstract.dtype import DType
    from autofusedebugger.fuzz.generation.torch_dialect import capture,execute_graph
    from autofusedebugger.nnsmith_integration.aten_dialect import classes
    original=deepcopy(graph)
    index=multi[-1]['outputs'][1]
    cls=next(c for c in classes() if c.rule.target=='aten.add.Tensor')
    op=cls()
    shape=multi[-1]['materializer']['state']['_output_like'][1]['$tensor']['shape']
    tensor=AbsTensor(shape,DType.int64)
    op.extra_attrs={'alpha':1}
    op.bind_input_like([tensor,tensor]);op.bind_output_like([tensor])
    graph['nodes'].append(dict(op='aten.add.Tensor',inputs=[index,index],
        output='index_twice',outputs=['index_twice'],attrs=op.extra_attrs,
        materializer=capture(op,9)))
    graph['outputs']=['index_twice']
    adapter=create_backend(ExecutionConfig('cpu','cpu',1))
    adapter.check(bundle,EvaluationConfig(correctness=False))
    inputs=adapter.inputs(bundle,0)
    weights={t.name:adapter._tensor(bundle,t) for t in bundle.spec.weights}
    names=[t.name for t in bundle.spec.inputs]
    expected=execute_graph(runtime[0],original,weights,names,'cpu')(*inputs)
    actual=execute_graph(runtime[0],graph,weights,names,'cpu')(*inputs)
    runtime[0].testing.assert_close(actual[0],expected[original['outputs'].index(index)]*2)
    adapter.close()
    result=execute(bundle,create_backend(ExecutionConfig('cpu','cpu',1)),EvaluationConfig(correctness=False))
    assert result['status']=='ok'
    repo=ResultRepository(tmp_path/'results')
    identity=repo.save(result)
    assert repo.detail(identity)['workload_id']==workload_id(bundle)
    import sqlite3
    with sqlite3.connect(repo.path) as db:
        stored={r[0] for r in db.execute("SELECT value FROM case_dimensions WHERE dimension='category'")}
    assert {'Reduce','View'}<=stored


def test_j_dynamic_variants_execute_and_replay(runtime):
    from autofusedebugger.fuzz.generation import materialize,nnsmith_sampled
    from autofusedebugger.fuzz.generation.sampling import SamplingConfig
    from autofusedebugger.fuzz.backends import create_backend
    from autofusedebugger.fuzz.contracts import ExecutionConfig,EvaluationConfig
    from autofusedebugger.fuzz.execution import execute
    config=SamplingConfig.profile('small',max_nodes=3,dtype_mode='float32')
    # Deterministic selection; a generation failure must not silently fall back
    # to a static graph. Specific qualified seeds are pinned after validation.
    successes=[]
    for entropy in (0,1,2,3):
        try: bundle=materialize(config.seed(entropy,dynamic=True,aten=True))
        except ValueError as exc:
            assert 'no three legal bounded shapes' in str(exc)
            continue
        assert len(bundle.spec.shape_samples)==3
        result=execute(bundle,create_backend(ExecutionConfig('cpu','cpu',1)),EvaluationConfig(correctness=False))
        assert result['status']=='ok'
        nnsmith_sampled.plan.cache_clear()
        assert materialize(bundle.spec.seed)==bundle
        successes.append(entropy)
    assert successes


def test_aten_reporting_categories():
    from types import SimpleNamespace
    from autofusedebugger.fuzz.generation.categories import categories,OP_CATEGORY
    assert set(r.target for r in RULES)<=set(OP_CATEGORY)
    graph={'nodes':[{'op':t} for t in ('aten.add.Tensor','aten.topk.default',
                                     'aten.permute.default','aten.reshape.default')]}
    assert categories(SimpleNamespace(graph_json=json.dumps(graph)))==('Elementwise','Reduce','Transpose','View')
