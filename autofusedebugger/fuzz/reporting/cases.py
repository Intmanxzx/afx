"""Global seed/Shape heads over complete multi-backend test rounds.

Backend is a member of a round, never part of the outer grouping key. Newest
means first database arrival, not client timestamps or retest submission order.
"""
PRIMARY_BACKEND = 'af-mlir'


def rollup(seed=None, status=None, user_id=None, q=None, *, rowid_window=None, current=False,
           history=False):
    from ..storage.queries import filters
    # Validate public inputs, but filter only AFTER selecting global heads.
    filters(seed, None, user_id)
    from .search import exact_seed
    if seed is None:
        seed = exact_seed(q)
    values=[]
    # Never materialize execution/diagnostic blobs in backend_latest: they can
    # exceed the VM memory budget and force repeated temporary-disk spills.
    # Read them from the indexed facts table only while classifying each shape.
    sql="""WITH ordered AS (
      SELECT a.id,a.backend,a.seed,f.group_id,json_extract(f.execution,'$.expected_backends') AS expected_backends,f.user_id,
        r.sequence AS arrival,
        row_number() OVER (PARTITION BY f.group_id,a.backend ORDER BY r.sequence) AS round_no
      FROM attempts a JOIN attempt_facts f ON f.attempt_id=a.id JOIN attempt_arrivals_v2 r ON r.attempt_id=a.id
      WHERE NOT EXISTS (SELECT 1 FROM invalid_replays_v1 x WHERE x.attempt_id=a.id)
    ), rounds AS MATERIALIZED (
      SELECT group_id,round_no,
        CASE WHEN max(expected_backends) IS NULL
          THEN max(CASE WHEN backend='af-mlir' THEN arrival END) ELSE max(arrival) END AS round_cursor,
        json_group_array(backend) AS members,
        json_group_array(expected_backends) AS requirements
      FROM ordered GROUP BY group_id,round_no
    ), complete_rounds AS (
      SELECT r.* FROM rounds r WHERE EXISTS
        (SELECT 1 FROM json_each(r.members) WHERE value='af-mlir')
      AND NOT EXISTS (
        SELECT 1 FROM json_each(r.requirements) req,json_each(req.value) expected
        WHERE NOT EXISTS (SELECT 1 FROM json_each(r.members) member WHERE member.value=expected.value)
      )
    ), backend_latest AS MATERIALIZED (
      SELECT a.id,a.workload_id,a.backend,a.seed,a.created_at,a.status,a.error_stage,a.error_summary,
        r.round_cursor AS attempt_cursor,o.group_id,o.round_no,
        o.group_id || ':' || o.round_no AS record_id
      FROM complete_rounds r JOIN ordered o ON o.group_id=r.group_id AND o.round_no=r.round_no
      JOIN attempts a ON a.id=o.id
    ), all_results AS MATERIALIZED (
      SELECT b.id,b.group_id,b.round_no,b.record_id,b.workload_id,json_object('static_only',json_extract(f.execution,'$.static_only')) AS execution,b.backend,b.seed,b.attempt_cursor,
        coalesce(s.shape_index,-1) AS shape_index,
        CASE WHEN s.correctness='pass' THEN 'ok' WHEN s.correctness='fail' THEN 'fail'
          WHEN s.correctness='error' OR (s.shape_index IS NULL AND b.status IN ('error','unsupported'))
          THEN failure_status(b.backend,coalesce(s.error_stage,b.error_stage),coalesce(s.error_summary,b.error_summary),
            json_extract(f.execution,'$.diagnostics'),coalesce(s.shape_index,-1))
          ELSE 'unknown' END AS status,
        CASE WHEN s.correctness='error' OR (s.shape_index IS NULL AND b.status IN ('error','unsupported'))
          THEN failure_path(b.backend,'error',coalesce(s.error_stage,b.error_stage),coalesce(s.error_summary,b.error_summary),
            json_extract(f.execution,'$.diagnostics'),coalesce(s.shape_index,-1),(SELECT count(*) FROM shapes WHERE attempt_id=b.id))
          ELSE 'not_applicable' END AS failure_path
      FROM backend_latest b JOIN attempt_facts f ON f.attempt_id=b.id
      LEFT JOIN shapes s ON s.attempt_id=b.id
    ), shape_bounds AS MATERIALIZED (
      SELECT seed,max(shape_index)+1 AS shape_count FROM all_results WHERE backend='af-mlir' GROUP BY seed
    ), full_rounds AS MATERIALIZED (
      SELECT a.seed,max(a.attempt_cursor) AS cursor FROM backend_latest a
      JOIN shape_bounds bounds ON bounds.seed=a.seed
      LEFT JOIN case_facts cf ON cf.workload_id=a.workload_id
      WHERE a.backend='af-mlir' AND
        (SELECT count(*) FROM shapes WHERE attempt_id=a.id)>=max(1,coalesce(cf.shape_count,1),coalesce(bounds.shape_count,1))
      GROUP BY a.seed
    ), round_cases AS (
      SELECT a.*,a.attempt_cursor*2048+1024-a.shape_index AS cursor,
        row_number() OVER (PARTITION BY a.seed,a.shape_index ORDER BY a.attempt_cursor DESC) AS head_rank,
        count(*) OVER (PARTITION BY a.seed,a.shape_index) AS history_count
      FROM all_results a LEFT JOIN full_rounds newer ON newer.seed=a.seed
      WHERE a.backend='af-mlir' AND (HISTORY_MODE OR a.shape_index>=0 OR newer.cursor IS NULL OR newer.cursor<=a.attempt_cursor)
    ), unfiltered_cases AS (
      SELECT * FROM round_cases WHERE HEAD_PREDICATE
    ), latest AS MATERIALIZED (
      SELECT a.* FROM all_results a JOIN unfiltered_cases c
        ON c.record_id=a.record_id AND c.shape_index=a.shape_index
    ), projection AS (
      SELECT c.seed,c.shape_index,c.group_id,c.record_id,c.failure_path,b.backend,b.created_at,f.device_name AS user,
        CASE c.status WHEN 'ok' THEN 'Pass' WHEN 'fail' THEN 'accuracy_failed' WHEN 'gen_failed' THEN 'gen_failed'
          WHEN 'error' THEN 'function_failed' ELSE 'Unknown' END AS status,
        CASE l.status WHEN 'ok' THEN 'Pass' WHEN 'fail' THEN 'Pass' WHEN 'gen_failed' THEN '构图Failed'
          WHEN 'error' THEN 'Failed' ELSE 'Unknown' END AS function_status,
        CASE l.status WHEN 'ok' THEN 'Pass' WHEN 'fail' THEN 'Failed' ELSE 'Unknown' END AS accuracy_status,
        json_extract(cf.metadata,'$.op_types') AS op,
        CASE WHEN c.shape_index<0 THEN NULL ELSE json_extract(cf.metadata,'$.shape_variants['||c.shape_index||'].input_shapes') END AS input_shape,
        json_extract(cf.metadata,'$.input_dtypes') AS input_dtype,
        CASE WHEN json_extract(f.execution,'$.static_only') THEN 'static' ELSE 'dynamic' END AS mode,
        json_extract(f.versions,'$.cann') AS cann,json_extract(f.versions,'$.torch') AS torch,
        json_extract(f.versions,'$.torch_npu') AS torch_npu,json_extract(f.versions,'$.autofuse_commit') AS autofuse_commit,
        b.seed || ' ' || coalesce(f.device_name,'') || ' ' || b.backend || ' ' || f.versions || ' ' || b.created_at || ' ' ||
          coalesce(json_extract(cf.metadata,'$.op_types'),'') || ' ' || coalesce(json_extract(cf.metadata,'$.input_dtypes'),'') || ' ' || CASE WHEN c.shape_index<0 THEN '' ELSE coalesce(json_extract(cf.metadata,'$.shape_variants['||c.shape_index||']'),'') END || ' ' || coalesce(s.error_summary,b.error_summary,'') AS search
      FROM unfiltered_cases c JOIN latest l ON l.record_id=c.record_id AND l.shape_index=c.shape_index
      JOIN attempts b ON b.id=l.id JOIN attempt_facts f ON f.attempt_id=b.id
      LEFT JOIN case_facts cf ON cf.workload_id=c.workload_id
      LEFT JOIN shapes s ON s.attempt_id=b.id AND s.shape_index=c.shape_index
    ), cases AS (
      SELECT c.* FROM unfiltered_cases c WHERE 1=1""".replace('HEAD_PREDICATE','1=1' if history else 'head_rank=1').replace('HISTORY_MODE','1' if history else '0')
    if seed is not None:
        from ..contracts import Seed
        sql=sql.replace('WHERE NOT EXISTS (SELECT 1 FROM invalid_replays_v1',
            'WHERE a.seed=? AND NOT EXISTS (SELECT 1 FROM invalid_replays_v1',1)
        values.append(str(Seed.parse(seed)))
    if user_id is not None:
        sql+=' AND EXISTS (SELECT 1 FROM attempt_facts f WHERE f.attempt_id=c.id AND f.user_id=?)';values.append(user_id)
    if status:
        if status not in {'ok','fail','error','unsupported','gen_failed','unknown'}:
            raise ValueError('invalid status')
        sql+=' AND c.status=?';values.append(status)
    from .search import condition, case_condition
    shortcut=case_condition(q)
    if shortcut:
        sql+=' AND ('+shortcut[0]+')';values.extend(shortcut[1])
    elif q:
        aliases={'pass':'ok','failed':None,'精度failed':'fail','accuracy_failed':'fail','unknown':'unknown','构图failed':'gen_failed','gen_failed':'gen_failed'}
        term=q.strip().lower()
        if term in aliases:
            if term=='failed':sql+=" AND c.status IN ('fail','error','gen_failed')"
            else:sql+=' AND c.status=?';values.append(aliases[term])
        else:
            predicate,args=condition(q)
            sql+=' AND EXISTS (SELECT 1 FROM projection p WHERE p.record_id=c.record_id AND p.shape_index=c.shape_index AND ('+predicate+'))'
            values.extend(args)
    return sql+') ',values


def summary(database, **params):
    from ..storage.queries import reading
    with reading(database,timeout=30) as conn:
        conn.execute('BEGIN')
        from .materialized import rollup as indexed_rollup
        sql,values=indexed_rollup(conn, **params)
        rows=conn.execute(sql+"SELECT status,count(*) AS count,sum(CASE WHEN failure_path='mlir_failed' THEN 1 ELSE 0 END) AS mlir FROM cases GROUP BY status",values).fetchall()
    counts={r['status']:r['count'] for r in rows}
    return {'total':sum(counts.values()),'by_status':counts,'mlir_failed':sum(r['mlir'] for r in rows),'unit':'shape_case'}


def page_keys(conn, *, limit, after=0, seed=None, status=None, user_id=None, q=None, history=False, shape=None, snapshot=None):
    from .materialized import rollup as indexed_rollup
    sql,values=indexed_rollup(conn,seed=seed,status=status,user_id=user_id,q=q,history=history)
    where='cursor<?';values.append(after or 2**63-1)
    if shape is not None:where+=' AND shape_index=?';values.append(shape)
    if snapshot is not None:where+=' AND attempt_cursor<=?';values.append(snapshot)
    return conn.execute(sql+'SELECT * FROM cases WHERE '+where+' ORDER BY cursor DESC LIMIT ?',(*values,limit+1)).fetchall()


def reuse_latest(sql):
    start=sql.index('latest AS MATERIALIZED (',sql.index('unfiltered_cases AS'));end=sql.index('), projection AS (',start)
    return sql[:start]+'latest AS MATERIALIZED (SELECT * FROM selected_latest'+sql[end:]


def has_retests(conn):
    return False  # No retest-specific projection: every uploaded round is equal.
