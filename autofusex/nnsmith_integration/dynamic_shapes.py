"""Deterministic multi-shape workloads and strict symbolic-input marking."""
from dataclasses import dataclass
import hashlib
import math
import random


class DynamicShapeError(ValueError):
    pass


@dataclass
class ShapeSample:
    inputs: dict
    outputs: dict


def trace_model(model):
    import torch
    from nnsmith.materialize.torch.symbolnet import FxTracing
    with FxTracing():
        return torch.fx.symbolic_trace(model)


def build_workload(case, *, seed, variants=3, max_elements=524288):
    """Vary equal-sized input axes together; keep graph attributes/weights fixed.

    Candidate shapes are accepted only when eager executes with finite outputs.
    The first supported dimension group is selected deterministically. This
    intentionally does not claim support for arbitrary shape-dependent graphs.
    """
    import numpy as np
    import torch
    if variants < 2 or variants > 8:
        raise DynamicShapeError('dynamic variants must be between 2 and 8')
    model = trace_model(case.model.native_model.eval())
    original = case.oracle.input
    groups = sorted({int(d) for value in original.values() for d in value.shape if d > 1})
    if not groups:
        raise DynamicShapeError('No input dimension greater than 1 can vary (scalar/constant/unit-only graph)')
    rng = random.Random(seed ^ 0xD1A5E)
    rng.shuffle(groups)
    for extent in groups[:16]:
        samples = [ShapeSample(original, case.oracle.output)]
        changes = {name: [axis for axis,d in enumerate(value.shape) if d == extent]
                   for name,value in original.items()}
        seen = {extent}
        sizes = sorted({2,3,4,8,16,max(2,extent//2),max(2,extent-1),extent+1,extent+2})
        rng.shuffle(sizes)
        for size in sizes:
            if size in seen: continue
            shapes = {n:tuple(size if axis in changes[n] else d for axis,d in enumerate(v.shape))
                      for n,v in original.items()}
            if any(math.prod(shape) > max_elements for shape in shapes.values()): continue
            inputs = {n:np.ascontiguousarray(np.resize(v, shapes[n])) if shapes[n] else np.asarray(v).reshape(())
                      for n,v in original.items()}
            try:
                with torch.no_grad():
                    output = model(*[torch.from_numpy(inputs[n]) for n in case.model.input_like])
                output = list(output) if isinstance(output,(list,tuple)) else [output]
                if len(output) != len(case.oracle.output): continue
                outputs = {n:t.detach().cpu().numpy() for n,t in zip(case.oracle.output,output)}
                if not all(np.isfinite(v).all() for v in outputs.values()): continue
            except (RuntimeError,ValueError,IndexError,AssertionError):
                continue
            samples.append(ShapeSample(inputs,outputs));seen.add(size)
            if len(samples)==variants:
                return samples, changes
    raise DynamicShapeError(f'Cannot generate {variants} legal distinct shapes for this fixed graph/weights')


def prepare_inputs(inputs, changes, device):
    import torch
    tensors = [torch.from_numpy(v).to(device) for v in inputs.values()]
    for (name,_),tensor in zip(inputs.items(),tensors):
        for axis in changes.get(name,[]):
            # mark_dynamic is strict: shape specialization becomes a visible failure.
            torch._dynamo.mark_dynamic(tensor,axis)
    return tensors


def workload_metadata(samples, changes):
    return {'shape_mode':'dynamic','dynamic_axes':changes,
        'shape_variants':[{'input_shapes':{n:list(v.shape) for n,v in s.inputs.items()},
                           'output_shapes':{n:list(v.shape) for n,v in s.outputs.items()}} for s in samples]}


def workload_fingerprint(samples, changes):
    import json
    h=hashlib.sha256(json.dumps(workload_metadata(samples,changes),sort_keys=True).encode())
    for sample in samples:
        for values in (sample.inputs,sample.outputs):
            for name,value in sorted(values.items()):
                h.update(name.encode());h.update(str(value.dtype).encode());h.update(value.tobytes())
    return h.hexdigest()


def verify_workload(case, *, target, seed, variants, max_elements, rtol, atol, compile_fn=None):
    import torch
    from nnsmith.difftest import assert_allclose
    if hasattr(case, "dynamic_samples"):
        samples,changes=case.dynamic_samples,case.dynamic_axes
    else:
        samples,changes=build_workload(case,seed=seed,variants=variants,max_elements=max_elements)
    model=trace_model(case.model.native_model.eval()).to(target)
    compiled=(compile_fn or torch.compile)(model,backend='inductor',fullgraph=True,dynamic=True,
                                         options={'npu_backend':'ascendc'})
    for sample in samples:
        tensors=prepare_inputs(sample.inputs,changes,target)
        with torch.no_grad(): output=compiled(*tensors)
        output=list(output) if isinstance(output,(list,tuple)) else [output]
        actual={n:t.detach().cpu().numpy() for n,t in zip(sample.outputs,output)}
        assert_allclose(actual,sample.outputs,'ascendc dynamic','CPU oracle',equal_nan=True,rtol=rtol,atol=atol)
    return workload_metadata(samples,changes)
