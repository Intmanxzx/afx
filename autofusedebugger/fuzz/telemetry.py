"""Versioned, bounded scalar timing payload, carried by immutable result hashes."""
import json

STAGES = {'generation','capability','prepare','input_binding','first_call_combined',
          'correctness','warmup','measurement','execution_total','compile','kernel_mean','shape_processing','output_transfer','dependency_wait','resource_wait'}


def validate(raw):
    if not isinstance(raw,str) or len(raw)>12000:
        raise ValueError('invalid telemetry size')
    value=json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('invalid telemetry object')
    if set(value)-{'version','stages','shapes'} or value.get('version') not in {1,2}:
        raise ValueError('invalid telemetry version')
    def metrics(items):
        if not isinstance(items,dict) or set(items)-STAGES:
            raise ValueError('invalid timing stage')
        if any(type(v) is not int or not 0<=v<2**63 for v in items.values()):
            raise ValueError('invalid timing value')
    metrics(value.get('stages',{}))
    shapes=value.get('shapes',{})
    if not isinstance(shapes,dict) or len(shapes)>1024:raise ValueError('invalid timing shapes')
    for index,items in shapes.items():
        if not index.isdecimal() or len(index)>4:raise ValueError('invalid shape timing index')
        metrics(items)
    return json.dumps(value,sort_keys=True,separators=(',',':'))
