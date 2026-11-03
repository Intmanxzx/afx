"""Compile once per backend, verify/profile each deterministic shape separately."""
from pathlib import Path
import traceback

from ..dynamic_shapes import prepare_inputs, trace_model
from ..generation import _tolerance_for_ir
from .profiling import profile_kernels
from .schema import ProfileConfig


def measure_dynamic(case, result, request, spec, device, config, compile_kwargs):
    import torch
    from nnsmith.difftest import assert_allclose
    profile=ProfileConfig(**request['profile'])
    model=trace_model(case.model.native_model.eval()).to(device)
    compiled=torch.compile(model,**compile_kwargs)
    rtol,atol=_tolerance_for_ir(case.model.native_model.ir)
    result.correctness='pass'
    for index,sample in enumerate(case.dynamic_samples):
        shape={'index':index,'input_shapes':{n:list(v.shape) for n,v in sample.inputs.items()},
               'output_shapes':{n:list(v.shape) for n,v in sample.outputs.items()},'correctness':'not_checked','status':'error'}
        result.shape_results.append(shape)
        inputs=prepare_inputs(sample.inputs,case.dynamic_axes,device)
        try:
            with torch.no_grad():
                output=compiled(*inputs)
                torch.npu.synchronize(device)
                output=list(output) if isinstance(output,(list,tuple)) else [output]
                actual={n:t.detach().cpu().numpy() for n,t in zip(sample.outputs,output)}
                try:
                    assert_allclose(actual,sample.outputs,spec.name,'CPU dynamic oracle',equal_nan=True,rtol=rtol,atol=atol)
                    shape['correctness']='pass'
                except (AssertionError,KeyError,TypeError) as exc:
                    shape['correctness']='fail';shape['correctness_error']=str(exc)[-2000:]
                    result.correctness='fail'
                    result.correctness_error=f'shape {index}: {exc}'[-2000:]
                del output,actual
                for _ in range(profile.warmup_steps):compiled(*inputs)
                torch.npu.synchronize(device)
                shape.update(profile_kernels(compiled,inputs,device=device,
                    output_dir=Path(request['profile_dir'])/f'shape-{index}',config=profile))
                shape['status']='ok' if shape['total_kernel_us']>0 else 'no_device_time'
        except Exception:
            shape['error']=traceback.format_exc()[-3000:]
            raise
    if any(s['status']!='ok' for s in result.shape_results):
        result.status='no_device_time'
    else:result.status='ok'
    # Equal weight per tested shape; steps is the total across all captures.
    result.measured_steps=sum(s['measured_steps'] for s in result.shape_results)
    result.total_kernel_us=sum(s['total_kernel_us'] for s in result.shape_results)
    result.mean_kernel_us=result.total_kernel_us/result.measured_steps
    counts=[s.get('total_kernel_count') for s in result.shape_results]
    if all(n is not None for n in counts):
        result.total_kernel_count=sum(counts)
        result.mean_kernel_count=result.total_kernel_count/result.measured_steps
    result.op_statistics=[dict(row,shape_index=s['index']) for s in result.shape_results for row in s['op_statistics']]
    return result
