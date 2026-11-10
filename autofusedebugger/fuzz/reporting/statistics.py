"""SQL aggregation over persisted facts; no graph generation or tensor imports."""
from ..storage.queries import reading
from ..backend_names import display_backend


def statistics(database, *, seed=None, status=None, user_id=None, q=None):
    from .cases import rollup
    cte, values = rollup(seed, status, user_id, q)
    where = ' WHERE EXISTS (SELECT 1 FROM cases selected WHERE selected.group_id=f.group_id)'
    shape_where = ' WHERE EXISTS (SELECT 1 FROM cases selected WHERE selected.group_id=f.group_id AND selected.shape_index=s.shape_index)'
    stage_where = ' WHERE EXISTS (SELECT 1 FROM cases selected WHERE selected.group_id=f.group_id AND (m.shape_index=-1 OR selected.shape_index=m.shape_index))'
    with reading(database) as conn:
        conn.execute('BEGIN')
        base = ' FROM attempts a JOIN attempt_facts f ON f.attempt_id=a.id JOIN case_facts c ON c.workload_id=a.workload_id'
        result = {}
        result['stages']=[dict(r) for r in conn.execute(cte+'SELECT a.backend,m.stage,count(*) AS samples,sum(m.duration_ns) AS total_ns,avg(m.duration_ns) AS mean_ns,max(m.duration_ns) AS max_ns'+base+' JOIN stage_metrics m ON m.attempt_id=a.id'+stage_where+' GROUP BY a.backend,m.stage',values)]
        result['performance']=[dict(r) for r in conn.execute(cte+"SELECT a.backend,CASE WHEN json_extract(f.execution,'$.static_only') THEN 'static' ELSE 'dynamic' END AS mode,s.correctness,s.performance,s.metric,count(*) AS samples,avg(s.mean_wall_ns) AS mean_ns,min(s.mean_wall_ns) AS min_ns,max(s.mean_wall_ns) AS max_ns"+base+' JOIN shapes s ON s.attempt_id=a.id'+shape_where+' GROUP BY a.backend,mode,s.correctness,s.performance,s.metric',values)]
        result['precise_metrics']=[r for r in result['stages'] if r['stage'] in {'compile','kernel_mean'} and not (r['stage']=='kernel_mean' and r['backend'] in {'cpu','cpu-eager'})]
        result['errors']=[dict(r) for r in conn.execute(cte+"SELECT a.backend,a.error_stage,count(*) AS count"+base+where+' GROUP BY a.backend,a.error_stage',values) if r['error_stage']]
        result['case_groups']=conn.execute(cte+'SELECT count(DISTINCT f.group_id)'+base+where,values).fetchone()[0]
        result['attempts']=conn.execute(cte+'SELECT count(*)'+base+where,values).fetchone()[0]
        # User-facing coverage/quality uses cases, not the number of CPU peers
        # or backend retries. Stage/performance telemetry above remains per-run.
        case_values = values
        def case_grouped(expression, joins=''):
            return [dict(r) for r in conn.execute(cte+'SELECT '+expression+
                ' AS label,a.status,count(*) AS count FROM cases a JOIN case_facts c ON c.workload_id=a.workload_id'+
                joins+' GROUP BY label,a.status', case_values)]
        result['overall']=case_grouped("'全部'")
        result['backend']=[dict(r) for r in conn.execute(cte+"SELECT backend AS label,status,count(*) AS count FROM latest l WHERE EXISTS (SELECT 1 FROM cases c WHERE c.group_id=l.group_id AND c.shape_index=l.shape_index) AND backend NOT IN ('cpu','cpu-eager') GROUP BY backend,status",case_values)]
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
        return result
