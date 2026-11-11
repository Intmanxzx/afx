"""SQL aggregation over persisted facts; no graph generation or tensor imports."""
from contextlib import nullcontext

from ..storage.queries import reading
from ..backend_names import display_backend


def statistics(database, *, seed=None, status=None, user_id=None, q=None, _timeout=10, _conn=None, _group=None):
    from .cases import rollup, reuse_latest
    cte, values = rollup(seed, status, user_id, q)
    where = ' WHERE EXISTS (SELECT 1 FROM cases selected WHERE selected.group_id=a.group_id)'
    shape_where = ' WHERE EXISTS (SELECT 1 FROM cases selected WHERE selected.group_id=a.group_id AND selected.shape_index=s.shape_index)'
    stage_where = ' WHERE EXISTS (SELECT 1 FROM cases selected WHERE selected.group_id=a.group_id AND (m.shape_index=-1 OR selected.shape_index=m.shape_index))'
    with (nullcontext(_conn) if _conn is not None else reading(database, timeout=_timeout)) as conn:
        if _conn is None:
            conn.execute('BEGIN')
        # Materialize the narrow rollup once per read snapshot, not once per
        # chart. Main database is opened mode=ro; only connection-local temp
        # tables are writable. No persistent derived cache can become stale.
        conn.execute('PRAGMA query_only=OFF')
        for table in ('selected_latest', 'selected_cases', 'selected_attempts'):
            conn.execute('DROP TABLE IF EXISTS temp.' + table)
        if _group is not None:
            # Scope before ranking/materialization; never scan unrelated groups.
            cte = cte.replace('GROUP BY f.group_id,a.backend',
                              'WHERE f.group_id=? GROUP BY f.group_id,a.backend')
            values = [_group, *values]
        conn.execute('CREATE TEMP TABLE selected_latest AS '+cte+'SELECT * FROM latest',values)
        conn.execute('CREATE TEMP TABLE selected_cases AS '+reuse_latest(cte)+'SELECT * FROM cases',values)
        conn.execute('CREATE INDEX selected_case_keys ON selected_cases(group_id,shape_index)')
        # Keep wide execution/diagnostic blobs out of every telemetry join.
        # Mode is per attempt (including retries), not borrowed from latest.
        conn.execute("""CREATE TEMP TABLE selected_attempts AS
            SELECT a.id,a.backend,a.error_stage,f.group_id,
                   CASE WHEN json_extract(f.execution,'$.static_only') THEN 'static' ELSE 'dynamic' END AS mode
            FROM (SELECT DISTINCT group_id FROM selected_cases) g
            CROSS JOIN attempt_facts f ON f.group_id=g.group_id
            JOIN attempts a ON a.id=f.attempt_id
            JOIN case_facts c ON c.workload_id=a.workload_id""")
        conn.execute('PRAGMA query_only=ON')
        cte='WITH cases AS (SELECT * FROM selected_cases), latest AS (SELECT * FROM selected_latest) '
        values=[]
        base = ' FROM selected_attempts a'
        result = {}
        result['stages']=[dict(r) for r in conn.execute(cte+'SELECT a.backend,m.stage,count(*) AS samples,sum(m.duration_ns) AS total_ns,avg(m.duration_ns) AS mean_ns,max(m.duration_ns) AS max_ns'+base+' CROSS JOIN stage_metrics m ON m.attempt_id=a.id'+stage_where+' GROUP BY a.backend,m.stage',values)]
        result['performance']=[dict(r) for r in conn.execute(cte+"SELECT a.backend,a.mode,s.correctness,s.performance,s.metric,count(*) AS samples,avg(s.mean_wall_ns) AS mean_ns,sum(s.mean_wall_ns) AS _sum_ns,count(s.mean_wall_ns) AS _measured,min(s.mean_wall_ns) AS min_ns,max(s.mean_wall_ns) AS max_ns"+base+' CROSS JOIN shapes s ON s.attempt_id=a.id'+shape_where+' GROUP BY a.backend,mode,s.correctness,s.performance,s.metric',values)]
        result['tool_efficiency'] = {'version': 1, 'scope': 'attempt_shape_processing',
            'processing': [r for r in result['stages'] if r['stage']=='shape_processing'],
            'success_shapes_per_min': None, 'completed_shapes_per_min': None,
            'cpu_pool_utilization': None, 'npu_slot_utilization': None,
            'note': ''}
        from .efficiency import statistics as efficiency
        selected_ids={r[0] for r in conn.execute(cte+'SELECT a.id'+base+where,values)} if any((seed,status,user_id,q)) else None
        result['tool_efficiency']['windows']=efficiency(conn,selected_ids,use_cached=True) if _group is None else []
        result['precise_metrics']=[r for r in result['stages'] if r['stage'] in {'compile','kernel_mean'} and not (r['stage']=='kernel_mean' and r['backend'] in {'cpu','cpu-eager'})]
        result['errors']=[dict(r) for r in conn.execute(cte+"SELECT a.backend,a.error_stage,count(*) AS count"+base+where+' GROUP BY a.backend,a.error_stage',values) if r['error_stage']]
        result['case_groups']=conn.execute(cte+'SELECT count(DISTINCT a.group_id)'+base+where,values).fetchone()[0]
        result['attempts']=conn.execute(cte+'SELECT count(*)'+base+where,values).fetchone()[0]
        # User-facing coverage/quality uses cases, not the number of CPU peers
        # or backend retries. Stage/performance telemetry above remains per-run.
        case_values = values
        def case_grouped(expression, joins=''):
            return [dict(r) for r in conn.execute(cte+'SELECT '+expression+
                ' AS label,a.status,count(*) AS count FROM cases a JOIN case_facts c ON c.workload_id=a.workload_id'+
                joins+' GROUP BY label,a.status', case_values)]
        result['overall']=case_grouped("'全部'")
        result['failure_path'] = [dict(r) for r in conn.execute(cte +
            "SELECT failure_path AS label,status,count(*) AS count FROM cases "
            "WHERE status IN ('error','unsupported') GROUP BY failure_path,status", values)]
        result['backend']=[dict(r) for r in conn.execute(cte+"SELECT backend AS label,status,count(*) AS count FROM latest l WHERE EXISTS (SELECT 1 FROM cases c WHERE c.group_id=l.group_id AND c.shape_index=l.shape_index) AND backend NOT IN ('cpu','cpu-eager') GROUP BY backend,status",case_values)]
        if _group is None:
            for row in result['backend']:row['label']=display_backend(row['label'])
        for name in ('category','dtype','op'):
            result[name]=case_grouped('d.value'," JOIN case_dimensions d ON d.workload_id=a.workload_id AND d.dimension='"+name+"'")
        for name, column, bounds in [('dimension','max_dim',[(8,'≤8'),(64,'≤64'),(512,'≤512'),(4096,'≤4096'),(16384,'≤16384'),(65536,'≤65536'),(131072,'≤131072')]),('elements','max_elements',[(1000,'≤1K'),(16000,'≤16K'),(64000,'≤64K'),(256000,'≤256K'),(1000000,'≤1M'),(16000000,'≤16M'),(64000000,'≤64M')])]:
            value="CASE WHEN a.shape_index<0 THEN NULL ELSE json_extract(c.metadata,'$.shape_variants[' || a.shape_index || ']."+column+"') END"
            expression='CASE WHEN '+value+" IS NULL THEN '未采集' "+' '.join(f"WHEN {value}<={bound} THEN '{label}'" for bound,label in bounds)+f" ELSE '>{bounds[-1][1][1:]}' END"
            result[name]=case_grouped(expression)
        result['mode']=case_grouped("CASE WHEN json_extract(a.execution,'$.static_only') THEN 'static' ELSE 'dynamic' END")
        result['unit']='shape_case'
        # Kernel profiler isn't enabled by the executor. Null is intentional,
        # not a count of zero kernels or a zero-time measurement.
        result['kernel_profiler']={'collected':False,'kernel_count':None,'ascendc_kernel_count':None,'triton_kernel_count':None,'kernel_time_ns':None}
        if _group is None:
            for row in result['performance']:
                row.pop('_sum_ns'); row.pop('_measured')
        return result
