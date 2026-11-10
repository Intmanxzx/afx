"""Case-level rollup: CPU is a reference, not another tested case.

Use the same provenance/batch groups as the records table. A retry replaces the
backend's current outcome, but its immutable attempt remains available in history.
AF-MLIR is authoritative when present; CPU-only/incomplete cases are unknown.
"""
from ..storage.queries import filters


def rollup(seed=None, status=None, user_id=None, q=None):
    clauses, values = filters(seed, None, user_id, None)
    where = ' WHERE ' + ' AND '.join('a.' + c for c in clauses) if clauses else ''
    sql = """WITH selected AS (
        SELECT DISTINCT f.group_id FROM attempts a
        JOIN attempt_facts f ON f.attempt_id=a.id""" + where + """
    ), ranked AS (
        SELECT a.*, a.rowid AS attempt_cursor, f.group_id, f.execution,
        row_number() OVER (PARTITION BY f.group_id,a.backend ORDER BY a.rowid DESC) AS rank
        FROM attempts a JOIN attempt_facts f ON f.attempt_id=a.id
        JOIN selected g ON g.group_id=f.group_id
    ), backend_latest AS (SELECT * FROM ranked WHERE rank=1),
    latest AS (
        SELECT b.id,b.group_id,b.workload_id,b.execution,b.backend,b.attempt_cursor,
        coalesce(s.shape_index,-1) AS shape_index,
        CASE WHEN s.correctness='error' THEN 'error'
             WHEN s.correctness='fail' THEN 'fail'
             WHEN s.correctness='pass' THEN 'ok'
             WHEN s.shape_index IS NULL AND b.status IN ('error','unsupported') THEN b.status
             WHEN s.shape_index IS NULL THEN b.status ELSE 'unknown' END AS status
        FROM backend_latest b LEFT JOIN shapes s ON s.attempt_id=b.id
    ),
    unfiltered_cases AS (
        SELECT group_id, shape_index, max(attempt_cursor)*2048+1024-shape_index AS cursor, min(workload_id) AS workload_id,
        min(execution) AS execution,
        CASE
          WHEN max(CASE WHEN backend='af-mlir' THEN status END) IS NOT NULL
            THEN max(CASE WHEN backend='af-mlir' THEN status END)
          WHEN sum(CASE WHEN backend NOT IN ('cpu','cpu-eager') THEN 1 ELSE 0 END)=0 THEN 'unknown'
          WHEN max(CASE WHEN backend NOT IN ('cpu','cpu-eager') AND status IN ('error','unsupported') THEN 1 ELSE 0 END)=1 THEN 'error'
          WHEN max(CASE WHEN backend NOT IN ('cpu','cpu-eager') AND status='fail' THEN 1 ELSE 0 END)=1 THEN 'fail'
          WHEN max(CASE WHEN backend NOT IN ('cpu','cpu-eager') AND status='unknown' THEN 1 ELSE 0 END)=1 THEN 'unknown'
          ELSE 'ok' END AS status
        FROM latest GROUP BY group_id,shape_index
    ), projection AS (
        SELECT c.group_id,c.shape_index,b.backend,b.seed,b.created_at,f.device_name AS user,
        CASE c.status WHEN 'ok' THEN 'Pass' WHEN 'fail' THEN 'accuracy_failed'
             WHEN 'error' THEN 'function_failed' WHEN 'unsupported' THEN 'function_failed' ELSE 'Unknown' END AS status,
        CASE l.status WHEN 'ok' THEN 'Pass' WHEN 'fail' THEN 'Pass'
             WHEN 'error' THEN 'Failed' WHEN 'unsupported' THEN 'Failed' ELSE 'Unknown' END AS function_status,
        CASE l.status WHEN 'ok' THEN 'Pass' WHEN 'fail' THEN 'Failed' ELSE 'Unknown' END AS accuracy_status,
        json_extract(cf.metadata,'$.op_types') AS op,
        CASE WHEN c.shape_index<0 THEN NULL ELSE json_extract(cf.metadata,'$.shape_variants[' || c.shape_index || '].input_shapes') END AS input_shape,
        json_extract(cf.metadata,'$.input_dtypes') AS input_dtype,
        CASE WHEN json_extract(f.execution,'$.static_only') THEN 'static' ELSE 'dynamic' END AS mode,
        json_extract(f.versions,'$.cann') AS cann, json_extract(f.versions,'$.torch') AS torch,
        json_extract(f.versions,'$.torch_npu') AS torch_npu,
        json_extract(f.versions,'$.autofuse_commit') AS autofuse_commit,
        b.seed || ' ' || coalesce(f.device_name,'') || ' ' || b.backend || ' ' ||
        f.versions || ' ' || b.created_at || ' ' ||
        coalesce(json_extract(cf.metadata,'$.op_types'),'') || ' ' ||
        coalesce(json_extract(cf.metadata,'$.weights'),'') || ' ' ||
        CASE WHEN c.shape_index<0 THEN '' ELSE coalesce(json_extract(cf.metadata,
          '$.shape_variants[' || c.shape_index || ']'),'') END || ' ' ||
        coalesce((SELECT error_summary FROM shapes s WHERE s.attempt_id=b.id AND s.shape_index=c.shape_index),'') AS search
        FROM unfiltered_cases c JOIN latest l ON l.group_id=c.group_id AND l.shape_index=c.shape_index
        JOIN backend_latest b ON b.id=l.id JOIN attempt_facts f ON f.attempt_id=b.id
        JOIN case_facts cf ON cf.workload_id=c.workload_id
    ), cases AS (
        SELECT c.* FROM unfiltered_cases c WHERE EXISTS (
        SELECT 1 FROM projection p WHERE p.group_id=c.group_id AND p.shape_index=c.shape_index AND ("""
    from .search import condition
    predicate, query_values = condition(q)
    sql += predicate + '))'
    values.extend(query_values)
    if status:
        if status not in {'ok','fail','error','unsupported','unknown'}:
            raise ValueError('invalid status filter')
        sql += ' AND c.status=?'
        values.append(status)
    sql += ') '
    return sql, values


def summary(database, **params):
    from ..storage.queries import reading
    sql, values = rollup(**params)
    with reading(database) as conn:
        counts = {r['status']:r['count'] for r in conn.execute(
            sql + 'SELECT status,count(*) AS count FROM cases GROUP BY status', values)}
    return {'total':sum(counts.values()), 'by_status':counts, 'unit':'shape_case'}
