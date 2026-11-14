"""Captured NNSmith metadata for unit tests, never a runtime replay cache."""
from pathlib import Path
from dataclasses import replace
from autofusedebugger.fuzz.case_spec import decode
from autofusedebugger.fuzz.contracts import CaseBundle, Seed
from autofusedebugger.fuzz.generation.sampling import SamplingConfig
import math

CURRENT_SEED = str(SamplingConfig.profile('small', max_nodes=1, dtype_mode='float32').seed(3))
SECOND_CURRENT_SEED = str(SamplingConfig.profile('small', max_nodes=1, dtype_mode='float32').seed(4))


def bundle(seed='C16'):
    spec=decode((Path(__file__).parent/'fixtures/nnsmith-c16.json').read_text())
    tensors=(*spec.inputs,*spec.weights)
    raw=[b'']*len(tensors)
    for tensor in tensors:raw[tensor.alias_group]=bytes(4*math.prod(tensor.shape))
    return CaseBundle(spec,tuple(raw))


def three_shapes_spec():
    spec=bundle().spec
    # Query-only fixture: distinct shapes test projection, not graph validity.
    samples=tuple(tuple(replace(t,shape=(i+1,*t.shape[1:])) if t.shape else t for t in spec.inputs) for i in range(3))
    return replace(spec,shape_samples=samples,output_samples=spec.output_samples*3)
