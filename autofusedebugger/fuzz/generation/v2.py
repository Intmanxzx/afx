"""Frozen Elementwise + explicit Transpose + optional Reduce protocol.

v1 seed meaning stays unchanged. v2 reuses v1's byte materialization for entropy,
then explicitly transposes the activated rank-2 tensor before optional reduction.
"""
from dataclasses import replace
import json
import struct

from . import v1
from ..contracts import Seed, CaseBundle, canonical_json
from ..tensors import DTYPES

ORACLE_ID='af-etr-v2-exact-dyadic'


def plan(token):
    seed=Seed.parse(token)
    base=v1.plan(str(seed.entropy))
    graph=json.loads(base.graph_json)
    graph['dialect']='af-etr/2'
    graph['transpose']=[1,0]
    graph['nodes'].append({'op':'transpose','inputs':['activated'],'output':'transposed','axes':[1,0]})
    outputs=[]
    for sample in base.shape_samples:
        shape=sample[0].shape[::-1]
        axis=graph['reduce_axis']
        out=shape if axis is None else (shape[1-axis],)
        outputs.append((replace(base.output_samples[0][0],shape=out,stride=(out[1],1) if len(out)==2 else (1,)),))
    return replace(base,seed=seed,graph_json=canonical_json(graph),output_samples=tuple(outputs))


def materialize(token):
    spec=plan(token)
    return CaseBundle(spec,v1.materialize(str(spec.seed.entropy)).storages)


content_manifest=v1.content_manifest
workload_id=v1.workload_id


def reference(bundle,index):
    graph=json.loads(bundle.spec.graph_json)
    base=v1.materialize(str(bundle.spec.seed.entropy))
    original=json.loads(base.spec.graph_json)
    original['reduce_axis']=None
    input_spec=base.spec.shape_samples[index][0]
    rows,cols=input_spec.shape
    output=replace(base.spec.output_samples[index][0],shape=(rows,cols),stride=(cols,1))
    outputs=list(base.spec.output_samples);outputs[index]=(output,)
    unreduced=CaseBundle(replace(base.spec,graph_json=canonical_json(original),output_samples=tuple(outputs)),base.storages)
    raw=v1.reference(unreduced,index)[0][1]
    fmt,width=DTYPES[input_spec.dtype]
    values=struct.unpack('<'+fmt*(len(raw)//width),raw)
    transposed=[values[r*cols+c] for c in range(cols) for r in range(rows)]
    axis=graph['reduce_axis']
    if axis==0:
        transposed=[sum(transposed[r*rows+c] for r in range(cols)) for c in range(rows)]
    elif axis==1:
        transposed=[sum(transposed[r*rows:(r+1)*rows]) for r in range(cols)]
    return ((bundle.spec.output_samples[index][0],struct.pack('<'+fmt*len(transposed),*transposed)),)
