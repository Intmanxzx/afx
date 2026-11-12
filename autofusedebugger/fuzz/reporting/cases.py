"""Case-level rollup: CPU is a reference, not another tested case.

Use the same provenance/batch groups as the records table. A retry replaces the
backend's current outcome, but its immutable attempt remains available in history.
Only completed backend groups are public; CPU-only/incomplete groups are hidden.
"""
from ..storage.queries import filters


def rollup(seed=None, status=None, user_id=None, q=None, *, rowid_window=None):
    clauses, values = filters(seed, None, user_id, None)
    if rowid_window is not None:
        clauses.append('rowid BETWEEN ? AND ?')
        values.extend(rowid_window)
    where = ' WHERE ' + ' AND '.join('a.' + c for c in clauses) if clauses else ''
    # Rank only narrow keys. Sorting execution/diagnostic JSON for every
    # historical attempt makes even a small first page proportional to log size.
    selected = ("""selected AS (
        SELECT DISTINCT f.group_id FROM attempts a
        JOIN attempt_facts f ON f.attempt_id=a.id""" + where + """
    ), """) if clauses else ''
    selection = ' JOIN selected g ON g.group_id=f.group_id' if clauses else ''
    sql = 'WITH ' + selected + """backend_keys AS (
        SELECT f.group_id,a.backend,max(a.rowid) AS attempt_cursor
        FROM attempts a JOIN attempt_facts f ON f.attempt_id=a.id""" + selection + """
        GROUP BY f.group_id,a.backend
    ), backend_latest AS MATERIALIZED (
        SELECT a.id,a.workload_id,a.backend,a.seed,a.created_at,a.status,a.error_stage,a.error_summary,
               k.attempt_cursor,k.group_id,
               json_object('static_only',json_extract(f.execution,'$.static_only')) AS execution
        FROM backend_keys k JOIN attempts a ON a.rowid=k.attempt_cursor
        JOIN attempt_facts f ON f.attempt_id=a.id
    ),
    required_backends AS (
        SELECT DISTINCT b.group_id,expected.value AS backend
        FROM backend_latest b JOIN attempt_facts f ON f.attempt_id=b.id,
             json_each(json_extract(f.execution,'$.expected_backends')) expected
    ),
    incomplete_groups AS (
        SELECT DISTINCT required.group_id FROM required_backends required
        LEFT JOIN backend_keys done ON done.group_id=required.group_id AND done.backend=required.backend
        WHERE done.attempt_cursor IS NULL
    ),
    ready_groups AS (
        SELECT b.group_id FROM backend_keys b
        WHERE b.backend NOT IN ('cpu','cpu-eager')
        AND b.group_id NOT IN (SELECT group_id FROM incomplete_groups)
        GROUP BY b.group_id
    ),
    latest AS MATERIALIZED (
        SELECT b.id,b.group_id,b.workload_id,b.execution,b.backend,b.attempt_cursor,
        coalesce(s.shape_index,-1) AS shape_index,
        CASE WHEN s.correctness='error' THEN failure_status(b.backend,s.error_stage,s.error_summary,(SELECT json_extract(f.execution,'$.diagnostics') FROM attempt_facts f WHERE f.attempt_id=b.id),s.shape_index)
             WHEN s.correctness='fail' THEN 'fail'
             WHEN s.correctness='pass' THEN 'ok'
             WHEN s.shape_index IS NULL AND b.status IN ('error','unsupported')
                  THEN failure_status(b.backend,b.error_stage,b.error_summary,(SELECT json_extract(f.execution,'$.diagnostics') FROM attempt_facts f WHERE f.attempt_id=b.id),-1)
             WHEN s.shape_index IS NULL THEN b.status ELSE 'unknown' END AS status,
        CASE WHEN s.correctness='error' OR (s.shape_index IS NULL AND b.status IN ('error','unsupported'))
             THEN failure_path(b.backend,'error',coalesce(s.error_stage,b.error_stage),
                  coalesce(s.error_summary,b.error_summary),
                  (SELECT json_extract(f.execution,'$.diagnostics') FROM attempt_facts f WHERE f.attempt_id=b.id),
                  coalesce(s.shape_index,-1),
                  (SELECT count(*) FROM shapes all_shapes WHERE all_shapes.attempt_id=b.id)) ELSE 'not_applicable' END AS failure_path
        FROM backend_latest b JOIN ready_groups ready ON ready.group_id=b.group_id
        LEFT JOIN shapes s ON s.attempt_id=b.id
    ),
    unfiltered_cases AS (
        SELECT group_id, shape_index, max(attempt_cursor)*2048+1024-shape_index AS cursor, min(workload_id) AS workload_id,
        min(execution) AS execution,
        CASE WHEN max(CASE WHEN backend='af-mlir' THEN failure_path END) IS NOT NULL
             THEN max(CASE WHEN backend='af-mlir' THEN failure_path END)
             WHEN max(CASE WHEN failure_path='mlir_failed' THEN 1 ELSE 0 END)=1 THEN 'mlir_failed'
             WHEN max(CASE WHEN failure_path='unconfirmed' THEN 1 ELSE 0 END)=1 THEN 'unconfirmed'
             ELSE 'not_applicable' END AS failure_path,
        CASE
          WHEN max(CASE WHEN backend='af-mlir' THEN status END) IS NOT NULL
            THEN max(CASE WHEN backend='af-mlir' THEN status END)
          WHEN sum(CASE WHEN backend NOT IN ('cpu','cpu-eager') THEN 1 ELSE 0 END)=0 THEN 'unknown'
          WHEN max(CASE WHEN backend NOT IN ('cpu','cpu-eager') AND status='gen_failed' THEN 1 ELSE 0 END)=1 THEN 'gen_failed'
          WHEN max(CASE WHEN backend NOT IN ('cpu','cpu-eager') AND status IN ('error','unsupported') THEN 1 ELSE 0 END)=1 THEN 'error'
          WHEN max(CASE WHEN backend NOT IN ('cpu','cpu-eager') AND status='fail' THEN 1 ELSE 0 END)=1 THEN 'fail'
          WHEN max(CASE WHEN backend NOT IN ('cpu','cpu-eager') AND status='unknown' THEN 1 ELSE 0 END)=1 THEN 'unknown'
          ELSE 'ok' END AS status
        FROM latest GROUP BY group_id,shape_index
    ), projection AS (
        SELECT c.group_id,c.shape_index,c.failure_path,b.backend,b.seed,b.created_at,f.device_name AS user,
        CASE c.status WHEN 'ok' THEN 'Pass' WHEN 'fail' THEN 'accuracy_failed'
             WHEN 'gen_failed' THEN 'gen_failed'
             WHEN 'error' THEN 'function_failed' WHEN 'unsupported' THEN 'function_failed' ELSE 'Unknown' END AS status,
        CASE l.status WHEN 'ok' THEN 'Pass' WHEN 'fail' THEN 'Pass'
             WHEN 'gen_failed' THEN '构图Failed'
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
        SELECT c.* FROM unfiltered_cases c WHERE (c.group_id,c.shape_index) IN (
        SELECT p.group_id,p.shape_index FROM projection p WHERE ("""
    from .search import condition
    predicate, query_values = condition(q)
    state_queries = {
        'pass': "c.status='ok'",
        'failed': "c.status IN ('error','unsupported','gen_failed','fail')",
        '功能failed': "c.status IN ('error','unsupported')",
        'function_failed': "c.status IN ('error','unsupported')",
        '构图failed': "c.status='gen_failed'",
        'gen_failed': "c.status='gen_failed'",
        '精度failed': "c.status='fail'",
        'accuracy_failed': "c.status='fail'",
        'unknown': "c.status='unknown'",
    }
    state_predicate = state_queries.get(q.strip().lower()) if q else None
    if not q or state_predicate:
        sql = sql[:sql.index('    ), cases AS (')] + '    ), cases AS (SELECT c.* FROM unfiltered_cases c WHERE ' + (state_predicate or '1=1')
        query_values = []
    else:
        sql += predicate + '))'
    values.extend(query_values)
    if status:
        if status not in {'ok','fail','error','unsupported','gen_failed','unknown'}:
            raise ValueError('invalid status filter')
        sql += ' AND c.status=?'
        values.append(status)
    sql += ') '
    return sql, values


def summary(database, **params):
    from ..storage.queries import reading
    sql, values = rollup(**params)
    with reading(database, timeout=10) as conn:
        rows = conn.execute(sql + "SELECT status,count(*) AS count,"
            "sum(CASE WHEN status IN ('error','unsupported') AND failure_path='mlir_failed' THEN 1 ELSE 0 END) AS mlir "
            "FROM cases GROUP BY status", values).fetchall()
        counts = {r['status']:r['count'] for r in rows}
    return {'total':sum(counts.values()), 'by_status':counts, 'mlir_failed':sum(r['mlir'] for r in rows), 'unit':'shape_case'}


def page_keys(conn, *, limit, after=0, seed=None, status=None, user_id=None, q=None):
    """Bound unfiltered pages by rowid, without truncating backend history.

    A case cursor's integer quotient by 2048 is its newest contributing
    attempt rowid (shape indices are -1..1023). Selecting groups with any
    attempt in this window therefore includes every case in the cursor range.
    Expand for retries, sparse/deleted rows, and partial backend shape sets.
    """
    upper = (after // 2048 if after else
             conn.execute('SELECT coalesce(max(rowid),0) FROM attempts').fetchone()[0])
    width = max(128, limit * 4)
    while True:
        lower = max(0, upper - width)
        bounded = not any((seed, status, user_id, q))
        cte, values = rollup(seed, status, user_id, q,
                            rowid_window=(lower, upper) if bounded else None)
        rows = conn.execute(cte +
            'SELECT * FROM cases WHERE cursor<? ORDER BY cursor DESC LIMIT ?',
            (*values, after or 2**63-1, limit+1)).fetchall()
        # Cases below this range can come from older peers of a selected group.
        # They are not yet safe to return: an unselected group may precede them.
        if not bounded or lower == 0:
            return rows
        rows = [row for row in rows if row['cursor'] >= lower * 2048]
        if len(rows) > limit:
            return rows
        width *= 2


def reuse_latest(sql):
    """Use the connection-local latest snapshot for subsequent chart rollups."""
    start = sql.index('    latest AS MATERIALIZED (')
    end = sql.index('    unfiltered_cases AS (', start)
    return sql[:start] + '    latest AS (SELECT * FROM selected_latest),\n' + sql[end:]
