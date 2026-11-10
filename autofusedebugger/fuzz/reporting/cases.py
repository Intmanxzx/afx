"""Case-level rollup: CPU is a reference, not another tested case.

Use the same provenance/batch groups as the records table. A retry replaces the
backend's current outcome, but its immutable attempt remains available in history.
AF-MLIR is authoritative when present; CPU-only/incomplete cases are unknown.
"""
from ..storage.queries import filters


def rollup(seed=None, status=None, user_id=None, q=None):
    clauses, values = filters(seed, status, user_id, q)
    where = ' WHERE ' + ' AND '.join('a.' + c for c in clauses) if clauses else ''
    sql = '''WITH selected AS (
        SELECT DISTINCT f.group_id FROM attempts a
        JOIN attempt_facts f ON f.attempt_id=a.id''' + where + '''
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
    cases AS (
        SELECT group_id, shape_index, max(attempt_cursor)*2048+1024-shape_index AS cursor, min(workload_id) AS workload_id,
        min(execution) AS execution,
        CASE
          WHEN max(CASE WHEN backend='af-mlir' THEN status END) IS NOT NULL
            THEN max(CASE WHEN backend='af-mlir' THEN status END)
          WHEN sum(CASE WHEN backend NOT IN ('cpu','cpu-eager') THEN 1 ELSE 0 END)=0 THEN 'unknown'
          WHEN max(CASE WHEN backend NOT IN ('cpu','cpu-eager') AND status IN ('error','unsupported') THEN 1 ELSE 0 END)=1 THEN 'error'
          WHEN max(CASE WHEN backend NOT IN ('cpu','cpu-eager') AND status='fail' THEN 1 ELSE 0 END)=1 THEN 'fail'
          ELSE 'ok' END AS status
        FROM latest GROUP BY group_id,shape_index
    ) '''
    return sql, values


def summary(database, **params):
    from ..storage.queries import reading
    sql, values = rollup(**params)
    with reading(database) as conn:
        counts = {r['status']:r['count'] for r in conn.execute(
            sql + 'SELECT status,count(*) AS count FROM cases GROUP BY status', values)}
    return {'total':sum(counts.values()), 'by_status':counts, 'unit':'shape_case'}
