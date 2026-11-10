"""Conservative paired result comparison; never invent speedups across contexts."""
from .queries import ResultQueries


def compare(database, left_id, right_id):
    query=ResultQueries(database)
    left,right=query.detail(left_id),query.detail(right_id)
    identity_fields=('seed','workload_id','oracle_id')
    same_workload=all(left[key]==right[key] for key in identity_fields)
    reasons=[]
    if not same_workload:reasons.append('different_workload_or_oracle')
    for kind in ('environment','evaluation'):
        if left['contexts'][kind]!=right['contexts'][kind]:reasons.append('different_'+kind)
    # backend is the compared dimension; device/static/compile budgets must match.
    execution=[]
    for value in (left,right):
        execution.append({k:v for k,v in value['contexts']['execution'].items() if k not in {'backend','telemetry','batch_id'}})
    if execution[0]!=execution[1]:reasons.append('different_execution_configuration')
    a={row['shape_index']:row for row in left['shape_results']}
    b={row['shape_index']:row for row in right['shape_results']}
    if a.keys()!=b.keys():reasons.append('different_shape_selection')
    shapes=[]
    for index in sorted(a.keys()|b.keys()):
        x,y=a.get(index),b.get(index)
        row={'shape_index':index,'left_correctness':x['correctness'] if x else None,
             'right_correctness':y['correctness'] if y else None,'speedup':None}
        if not reasons and x and y and x['correctness']==y['correctness']=='pass' and x['performance']==y['performance']=='measured' and x['metric']==y['metric'] and x['mean_wall_ns'] and y['mean_wall_ns']:
            row['speedup']=x['mean_wall_ns']/y['mean_wall_ns']
        shapes.append(row)
    return {'left':left_id,'right':right_id,'same_workload':same_workload,
            'performance_comparable':not reasons,'incomparability_reasons':reasons,'shapes':shapes}
