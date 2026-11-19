"""Semantic, dynamic and end-to-end boundaries for the sole current generator."""
import json
import random
from dataclasses import replace,asdict
import pytest
from autofusex.fuzz.client.contracts import TensorSpec,CaseSpec,canonical_json
from autofusex.fuzz.client.generation.sampling import SamplingConfig


def build_spec(count=3,nodes=1,extent=6,weight=True):
    """Independent valid Add chain; bypass search to exercise full boundary size."""
    from nnsmith.abstract.tensor import AbsTensor
    from nnsmith.abstract.dtype import DType
    from autofusex.fuzz.client.nnsmith.aten_dialect import classes
    from autofusex.fuzz.client.generation.torch_dialect import capture
    from autofusex.fuzz.client.generation.runtime import SOURCE_SHA256,Z3_VERSION
    cls=next(c for c in classes() if c.rule.target=='aten.add.Tensor')
    ins=TensorSpec('x','float32',(extent,),(1,),alias_group=0)
    # Weight has the same numeric extent but only reduces to a scalar. Equal
    # dimension values alone must not forbid changing the input dimension.
    wt=TensorSpec('w','float32',(extent,),(1,),alias_group=1)
    graph_nodes=[]
    if weight:
        red=next(c for c in classes() if c.rule.target=='aten.sum.default')()
        red.extra_attrs={'dims':[0],'keepdim':False}
        red.bind_input_like([AbsTensor([extent],DType.float32)])
        red.bind_output_like([AbsTensor([],DType.float32)])
        graph_nodes.append(dict(op=red.rule.target,inputs=['w'],output='w_sum',outputs=['w_sum'],attrs=red.extra_attrs,materializer=capture(red)))
    for i in range(nodes):
        op=cls();op.extra_attrs={'alpha':1}
        op.bind_input_like([AbsTensor([extent],DType.float32),AbsTensor([] if weight else [extent],DType.float32)])
        op.bind_output_like([AbsTensor([extent],DType.float32)])
        graph_nodes.append(dict(op=op.rule.target,inputs=['x' if i==0 else f'y{i-1}','w_sum' if weight else 'x'],output=f'y{i}',outputs=[f'y{i}'],attrs=op.extra_attrs,materializer=capture(op)))
    config=SamplingConfig.profile('small',dim_min=extent,dim_max=extent,max_nodes=len(graph_nodes),dynamic_variants=count)
    graph=dict(dialect='nnsmith-aten/2',sampling=asdict(config),nodes=graph_nodes,outputs=[f'y{nodes-1}'],reduce_axis=None,
               generator='nnsmith.graph_gen.model_gen',source_sha256=SOURCE_SHA256,z3=Z3_VERSION,
               aten_rules='elementwise-comparison-reduce-view/v1',torch='2.10.0+cpu')
    out=TensorSpec(f'y{nodes-1}','float32',(extent,),(1,))
    return CaseSpec(config.seed(7,dynamic=True),canonical_json(graph),(ins,),(wt,) if weight else (),((ins,),),((out,),))


@pytest.mark.parametrize('count',range(2,9))
def test_dynamic_counts_weight_dimensions_and_outside_initial_range(count):
    from autofusex.fuzz.client.generation.nnsmith_dynamic import variants
    from autofusex.fuzz.client.generation.nnsmith_sampled import materialize_spec
    from autofusex.fuzz.client.backends.torch import TorchAdapter
    from autofusex.fuzz.client.contracts import ExecutionConfig,EvaluationConfig
    from autofusex.fuzz.client.case_spec import encode,decode
    from autofusex.fuzz.client.generation.torch_dialect import execute_graph
    import torch
    spec=build_spec(count)
    expanded=variants(spec,65536,count,random.Random(123))
    assert expanded==variants(spec,65536,count,random.Random(123))
    assert len(expanded.shape_samples)==count
    assert len({s[0].shape for s in expanded.shape_samples})==count
    assert any(s[0].shape!=(6,) for s in expanded.shape_samples)
    assert expanded.weights==spec.weights
    assert decode(encode(expanded))==expanded
    bundle=materialize_spec(expanded)
    adapter=TorchAdapter(ExecutionConfig('cpu','cpu'));adapter.validated_shared_case=True
    adapter.check(bundle,EvaluationConfig())
    fn=adapter.prepare(bundle)
    try:
        for i,sample in enumerate(expanded.shape_samples):
            x=adapter.inputs(bundle,i)
            actual=fn(*x)[0]
            torch.testing.assert_close(actual,x[0]+adapter._tensor(bundle,expanded.weights[0]).sum())
            assert tuple(actual.shape)==expanded.output_samples[i][0].shape
    finally:adapter.close()


def test_64_nodes_survive_artifacts_database_snapshot(tmp_path):
    from autofusex.fuzz.client.generation.nnsmith_dynamic import variants
    from autofusex.fuzz.client.generation.nnsmith_sampled import materialize_spec
    from autofusex.fuzz.client.generation import workload_id
    from autofusex.fuzz.client.scheduling.artifacts import write_bundle,bundle_from
    from autofusex.fuzz.client.case_spec import encode,decode
    from autofusex.fuzz.client.storage import ResultRepository
    from autofusex.fuzz.client.storage.transfer import restore
    from autofusex.fuzz.client.storage.queries import ResultQueries
    spec=variants(build_spec(count=8,nodes=64,weight=False),65536,8,random.Random(123))
    raw=encode(spec);assert len(raw)>12000 and decode(raw)==spec
    bundle=materialize_spec(spec)
    ref=write_bundle(tmp_path,bundle);assert bundle_from(ref,tmp_path)==bundle
    result=dict(seed=str(spec.seed),workload_id=workload_id(bundle),oracle_id='test',backend='cpu',status='ok',
                execution={'case_spec':raw,'backend':'cpu','device':'cpu'},shape_results=[])
    repo=ResultRepository(tmp_path/'results')
    for i in range(50):identity=repo.save(result)
    assert ResultQueries(repo.path).detail(identity)['contexts']['execution']['case_spec']==raw
    snap=tmp_path/'snapshot';repo.snapshot(snap);restore(snap,tmp_path/'restored')
    assert ResultQueries(tmp_path/'restored').detail(identity)['contexts']['execution']['case_spec']==raw




@pytest.mark.parametrize('name',['Reshape','ExpandLast1','ExpandLast2','ExpandLast3','ExpandLast4','Slice','Squeeze'])
def test_json_roundtrip_matches_independent_nnsmith_native_semantics(name):
    """Oracle is the pre-existing NNSmith op, not a second run of our adapter."""
    import torch
    from nnsmith.abstract import op as O
    from nnsmith.abstract.dtype import DType
    from nnsmith.abstract.tensor import AbsTensor
    from nnsmith.materialize.torch.forward import forward_fn
    from autofusex.fuzz.client.nnsmith.aten_dialect import classes
    from autofusex.fuzz.client.generation.torch_dialect import capture,execute_graph
    # Only currently modeled targets are paired; no claim that the new four
    # families include LayerNorm/Gather/Cast, which need additional rules.
    if name=='Reshape':
        native=O.Reshape(6);xs=[torch.arange(6.).reshape(2,3)];target='aten.reshape.default';attrs={}
    elif name.startswith('ExpandLast'):
        k=int(name[-1]);native=getattr(O,name)(2)
        # New expand adds a leading dimension, paired with native explicit output.
        xs=[torch.ones([1]*k)];target='aten.expand.default';attrs={}
        native.bind_output_like([AbsTensor([2]+[1]*k,DType.float32)])
    elif name=='Slice':
        native=O.Slice(0,5,2);native.extra_attrs={'axis':0,'ndims':1,'region':'right'}
        xs=[torch.arange(5.)];target='aten.slice.Tensor';attrs={'dim':0,'step':2}
    elif name=='Squeeze':
        native=O.Squeeze();native.extra_attrs={'reduce_dim':1};xs=[torch.ones(2,1,3)]
        target='aten.squeeze.dim';attrs={'dim':1}
    native.bind_input_like([AbsTensor(list(x.shape),DType.float32) for x in xs])
    expected=forward_fn(native)(*xs)
    op=next(c for c in classes() if c.rule.target==target)();op.extra_attrs=attrs
    op.bind_input_like([AbsTensor(list(x.shape),DType.float32) for x in xs])
    op.bind_output_like([AbsTensor(list(expected.shape),DType.float32)])
    node=dict(op=target,inputs=['x'],output='y',outputs=['y'],attrs=attrs,materializer=capture(op))
    graph=json.loads(json.dumps(dict(dialect='nnsmith-aten/2',nodes=[node],outputs=['y'])))
    actual=execute_graph(torch,graph,{},['x'],'cpu')(*xs)[0]
    torch.testing.assert_close(actual,expected)


@pytest.mark.parametrize('target,attrs,shape,oracle',[
 ('aten.clamp.default',{},[3],lambda torch,x:torch.clamp(x,-2,2)),
 ('aten.sum.dim_IntList',{'dims':[0,2],'keepdim':False},[2,3,4],lambda torch,x:x.sum((0,2))),
 ('aten.sum.dim_IntList',{'dims':[0,2],'keepdim':True},[2,3,4],lambda torch,x:x.sum((0,2),keepdim=True)),
 ('aten.transpose.int',{'dim':0,'other':1},[2,3],lambda torch,x:x.transpose(0,1)),
 ('aten.slice.Tensor',{'dim':0,'step':2},[5],lambda torch,x:x[::2]),
 ('aten.permute.default',{'dims':[2,0,1]},[2,3,4],lambda torch,x:x.permute(2,0,1)),
])
def test_actual_parameters_against_independent_torch_oracle(target,attrs,shape,oracle):
    import torch,math
    from nnsmith.abstract.tensor import AbsTensor
    from nnsmith.abstract.dtype import DType
    from autofusex.fuzz.client.nnsmith.aten_dialect import classes
    from autofusex.fuzz.client.generation.torch_dialect import capture,execute_graph
    x=torch.arange(math.prod(shape),dtype=torch.float32).reshape(shape)*3-4
    op=next(c for c in classes() if c.rule.target==target)();op.extra_attrs=attrs
    tensors=[AbsTensor(shape,DType.float32)]
    op.bind_input_like(tensors);op.bind_output_like(op.checked_type_transfer(tensors))
    node=dict(op=target,inputs=['x'],output='y',outputs=['y'],attrs=op.extra_attrs,materializer=capture(op))
    graph=json.loads(json.dumps(dict(dialect='nnsmith-aten/2',nodes=[node],outputs=['y'])))
    actual=execute_graph(torch,graph,{},['x'],'cpu')(x)[0]
    torch.testing.assert_close(actual,oracle(torch,x))


def test_current_golden_fingerprints():
    from autofusex.fuzz.client.generation import materialize,workload_id
    # Materializer class names belong to the autofusex.fuzz.client.nnsmith namespace.
    # Re-pinned after commit 0154365 ("normalize operator names and expose tool
    # categories"), which changed embedded display_name strings in graph_json for
    # several ops and so, intentionally, the workload_id of any graph containing
    # them; that commit landed without updating this golden set. Re-pinning here
    # does not undo the drift it already caused for workload_ids computed before
    # it on any production database -- those are unaffected and unrecoverable by
    # this test; it only stops the *next* unrelated change from going unnoticed.
    expected=[(False,3,0,3,'81105c13c3dc953c34d8f81cc1cd695e9a3ea92e8748634e251f286a675f9cac'),
              (False,3,1,3,'ec11059a04f6dca93375b3ed2f0e29fa998f03606f95d960f4e1376838b68ca2'),
              (True,2,3,1,'8cae18ec587d71dbec5caaf138f3a932d2aea3445cf9d1104128ec6e6dfcab51'),
              (True,8,3,1,'b2b00bbc2a98eb14a92f98e32141de7790da51b50503205a94f6d8f03d2ae04a')]
    for dynamic,count,entropy,nodes,digest in expected:
        config=SamplingConfig.profile('small',max_nodes=nodes,dtype_mode='float32',dynamic_variants=count)
        assert workload_id(materialize(config.seed(entropy,dynamic)))==digest
