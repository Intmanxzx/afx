"""Seed-derived metadata for the SQLite reporting index; no model imports."""
import json
import math
def spec_metadata(spec):
    graph = json.loads(spec.graph_json)
    from ...client.nnsmith.aten_catalog import rules
    display = {rule.target: rule.display_name for rule in rules()}
    for node in graph['nodes']:
        node.setdefault('display_name', display.get(node['op'], node['op']))
    ops = [node.get('display_name', node['op']) for node in graph['nodes']]
    if graph['reduce_axis'] is not None:
        ops.append('sum')
    return {'graph': graph, 'weights': [{'name': t.name, 'shape': list(t.shape), 'dtype': t.dtype} for t in spec.weights], 'op_types': ops, 'n_nodes': len(ops),
            'input_shapes': {t.name: list(t.shape) for t in spec.inputs},
            'input_dtypes': {t.name: t.dtype for t in spec.inputs},
            'output_shapes': {t.name: list(t.shape) for t in spec.output_samples[0]},
            'output_dtypes': {t.name: t.dtype for t in spec.output_samples[0]},
            'shape_variants': [{'max_dim': max((max(t.shape, default=0) for t in (*sample,*outputs)), default=0),
                                'max_elements': max((math.prod(t.shape) for t in (*sample,*outputs)), default=0),
                                'input_dtypes': {t.name: t.dtype for t in sample}, 'output_dtypes': {t.name: t.dtype for t in outputs}, 'input_shapes': {t.name: list(t.shape) for t in sample},
                                'output_shapes': {t.name: list(t.shape) for t in outputs}}
                               for sample, outputs in zip(spec.shape_samples, spec.output_samples)],
            'weight_broadcast_axes': graph['nodes'][0].get('weight_broadcast_axes', []) if graph['nodes'] else []}
