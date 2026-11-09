"""Operator categories are explicit; Transpose is not View."""
OP_CATEGORY={'add':'Elementwise','relu':'Elementwise','neg':'Elementwise','mul':'Elementwise',
             'sum':'Reduce','transpose':'Transpose'}
SUPPORTED_CATEGORIES=('Elementwise','Reduce','Transpose')


def categories(spec):
    import json
    graph=json.loads(spec.graph_json)
    result={OP_CATEGORY[node['op']] for node in graph['nodes']}
    if graph['reduce_axis'] is not None:result.add('Reduce')
    return tuple(sorted(result))
