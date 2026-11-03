"""Dynamic shape workloads: legal variants, deterministic replay and real symbolic compile."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from autofusedebugger.nnsmith_integration.dynamic_shapes import (
    build_workload,workload_metadata,workload_fingerprint,verify_workload,DynamicShapeError)


def example():
    class Net(torch.nn.Module):
        def forward(self,x):return (torch.sin(x)+2,)
    net=Net()
    x=np.arange(16,dtype=np.float32).reshape(4,4)/16
    y=net(torch.from_numpy(x))[0].numpy()
    return SimpleNamespace(model=SimpleNamespace(native_model=net,input_like={'x':None}),
        oracle=SimpleNamespace(input={'x':x},output={'y':y}))


def test_variants_are_distinct_deterministic_and_eager_correct():
    a,axes=build_workload(example(),seed=42,variants=3,max_elements=256)
    b,axes2=build_workload(example(),seed=42,variants=3,max_elements=256)
    assert len({tuple(s.inputs['x'].shape) for s in a})==3
    assert axes==axes2 and workload_fingerprint(a,axes)==workload_fingerprint(b,axes2)
    for s in a:
        np.testing.assert_allclose(s.outputs['y'],np.sin(s.inputs['x'])+2,rtol=1e-6)
        assert s.inputs['x'].size<=256


def test_one_compiled_callable_accepts_all_shapes_with_symbolic_dimension():
    graphs=[];calls=[]
    def backend(gm,example_inputs):
        graphs.append(gm)
        return gm.forward
    def compile_fn(model,**kwargs):
        calls.append(kwargs)
        return torch.compile(model,backend=backend,fullgraph=True,dynamic=kwargs['dynamic'])
    metadata=verify_workload(example(),target='cpu',seed=42,variants=3,max_elements=256,
        rtol=1e-5,atol=1e-6,compile_fn=compile_fn)
    assert len(calls)==1 and calls[0]['dynamic'] is True
    assert len(graphs)==1
    assert any('SymInt' in str(n.meta.get('example_value')) or 's0' in str(n.meta.get('example_value'))
               for n in graphs[0].graph.nodes) or any(isinstance(n.meta.get('example_value'),torch.SymInt) for n in graphs[0].graph.nodes)
    assert len(metadata['shape_variants'])==3


def test_scalar_graph_cannot_be_reported_as_dynamic():
    c=example();c.oracle.input={'x':np.array(3,dtype=np.float32)}
    with pytest.raises(DynamicShapeError,match='No input dimension'):build_workload(c,seed=42)


def test_fixed_reshape_constraint_is_not_silently_accepted():
    c=example()
    class Fixed(torch.nn.Module):
        def forward(self,x):return (x.reshape(16),)
    c.model.native_model=Fixed()
    with pytest.raises(DynamicShapeError,match='legal distinct'):build_workload(c,seed=42,variants=3)


def test_cli_dynamic_flags_and_config_validation():
    from autofusedebugger.cli import build_parser
    from autofusedebugger.nnsmith_integration.generation import GenerationConfig
    args=build_parser().parse_args(['af-fuzz','--out-dir','/tmp/dyn','--dynamic-shapes','--dynamic-variants','4'])
    assert args.dynamic_shapes and args.dynamic_variants==4
    with pytest.raises(ValueError):GenerationConfig(dynamic_variants=1)


def test_generated_nnsmith_case_dynamic_fingerprint_replays():
    import autofusedebugger
    from nnsmith.materialize import Model
    from nnsmith.abstract.op import Neg
    from nnsmith.abstract.dtype import DType
    from autofusedebugger.nnsmith_integration.generation import generate_case,GenerationConfig
    from autofusedebugger.nnsmith_integration.performance.cases import case_fingerprint
    cls=Model.init('torch',backend_target='cpu');cls.add_seed_setter()
    old_in,old_out=Neg.in_dtypes,Neg.out_dtypes
    try:
        Neg.in_dtypes=[(DType.float32,)];Neg.out_dtypes=[(DType.float32,)]
        config=GenerationConfig(max_nodes=3,tensor_scale='small',concr_ph_dim_rng=(4,8),
            max_elem_per_tensor=4096,dynamic_shapes=True)
        a=generate_case(opset=[Neg],model_cls=cls,case_seed=42,config=config)
        b=generate_case(opset=[Neg],model_cls=cls,case_seed=42,config=config)
        assert len(a.dynamic_samples)==3
        assert case_fingerprint(a)==case_fingerprint(b)
        assert a.dynamic_metadata['shape_mode']=='dynamic'
    finally:Neg.in_dtypes,Neg.out_dtypes=old_in,old_out


def test_dynamic_performance_profiles_every_shape_and_keeps_one_compilation(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from autofusedebugger.nnsmith_integration.performance import dynamic_worker
    from autofusedebugger.nnsmith_integration.performance.schema import BackendResult,ProfileConfig
    c=example();samples,axes=build_workload(c,seed=42,variants=3,max_elements=256)
    c.dynamic_samples,c.dynamic_axes=samples,axes
    c.model.native_model.ir=SimpleNamespace(vars={})
    monkeypatch.setattr(torch,'npu',SimpleNamespace(synchronize=lambda device:None),raising=False)
    calls=[]
    def compile_fn(model,**kwargs):calls.append(kwargs);return model
    monkeypatch.setattr(torch,'compile',compile_fn)
    def profile(compiled,inputs,**kwargs):
        directory=kwargs['output_dir'];directory.mkdir(parents=True)
        p=directory/'op_statistic.csv';p.write_text('OP Type,Count,Total Time(us)\ntriton_test,10,20\n')
        return dict(total_kernel_us=20,mean_kernel_us=2,measured_steps=10,total_kernel_count=10,
            mean_kernel_count=1,csv_path=str(p),op_statistics=[])
    monkeypatch.setattr(dynamic_worker,'profile_kernels',profile)
    result=dynamic_worker.measure_dynamic(c,BackendResult('ascendc'),
        dict(profile=ProfileConfig().__dict__,profile_dir=str(tmp_path/'run'/'profiling')),
        SimpleNamespace(name='ascendc'),'cpu',None,dict(dynamic=True))
    assert len(calls)==1 and len(result.shape_results)==3
    assert result.status=='ok' and result.correctness=='pass'
    assert result.mean_kernel_us==2 and result.measured_steps==30
    from autofusedebugger.nnsmith_integration.performance.artifacts import finalize_artifacts
    run=tmp_path/'run'
    finalize_artifacts(result,run,tmp_path/'cache')
    assert not (run/'profiling').exists()
    assert len(list(run.glob('op_statistic_shape_*.csv')))==3
    assert all(__import__('pathlib').Path(s['csv_path']).exists() for s in result.shape_results)


def test_dynamic_accuracy_path_executes_all_shapes_with_same_seed(tmp_path,monkeypatch):
    from nnsmith.materialize import Model
    from nnsmith.abstract.op import Neg
    from nnsmith.abstract.dtype import DType
    from autofusedebugger.nnsmith_integration import fuzz_loop,dynamic_shapes
    cls=Model.init('torch',backend_target='cpu');cls.add_seed_setter()
    old_in,old_out=Neg.in_dtypes,Neg.out_dtypes
    real=dynamic_shapes.verify_workload
    compiled=[]
    def verify(case,**kwargs):
        def compile_fn(model,**options):compiled.append(options);return model
        return real(case,compile_fn=compile_fn,**kwargs)
    monkeypatch.setattr(dynamic_shapes,'verify_workload',verify)
    try:
        Neg.in_dtypes=[(DType.float32,)];Neg.out_dtypes=[(DType.float32,)]
        result=fuzz_loop.run_single_case(factory=SimpleNamespace(target='cpu'),opset=[Neg],model_cls=cls,
            case_seed=42,tensor_scale='small',max_nodes=3,concr_ph_dim_rng=(4,8),
            dynamic_shapes=True,out_dir=tmp_path)
        assert result.status=='pass',result.error_summary
        assert len(result.shape_variants)==3 and result.dynamic_fingerprint
        assert len(compiled)==1 and compiled[0]['dynamic'] is True
    finally:Neg.in_dtypes,Neg.out_dtypes=old_in,old_out
