from autofusedebugger.nnsmith_integration.generation import GenerationConfig
from autofusedebugger.nnsmith_integration.performance.shape_modes import resolve_generation
from autofusedebugger.nnsmith_integration.performance.shape_summary import shape_summary
import pytest


def test_mixed_seed_replay_and_conflicting_flags():
    config=GenerationConfig(mixed_shapes=True)
    assert not resolve_generation(config,42).dynamic_shapes
    assert resolve_generation(config,43).dynamic_shapes
    assert resolve_generation(config,43)==resolve_generation(config,43)
    assert not resolve_generation(config,43).mixed_shapes
    with pytest.raises(ValueError):GenerationConfig(mixed_shapes=True,dynamic_shapes=True)


def test_shape_summary_excludes_unavailable_times():
    def c(mode,status,a=2,t=4):
        return dict(shape_mode=mode,performance_status=status,ascendc_speedup=t/a,
            backends={'ascendc':{'mean_kernel_us':a},'triton-ascend':{'mean_kernel_us':t}})
    rows=shape_summary([c('static','pass'),c('static','error'),c('dynamic','fail',4,2)])
    assert rows[0]['total']==2 and rows[0]['pass']==1 and rows[0]['error']==1
    assert rows[0]['ascendc']==2 and rows[0]['pass_rate']==1
    assert rows[1]['geomean_speedup']==.5
