"""SQL aggregation over persisted facts; no graph generation or tensor imports."""
from ..storage.queries import reading, filters
from ..backend_names import display_backend


def statistics(database, *, seed=None, status=None, user_id=None):
    clauses, values = filters(seed, status, user_id)
    where = ' WHERE ' + ' AND '.join('a.'+c for c in clauses) if clauses else ''
    with reading(database) as conn:
        conn.execute('BEGIN')
        base = ' FROM attempts a JOIN attempt_facts f ON f.attempt_id=a.id JOIN case_facts c ON c.workload_id=a.workload_id'
        def grouped(expression, joins=''):
            return [dict(r) for r in conn.execute('SELECT '+expression+' AS label,a.status,count(*) AS count'+base+joins+where+' GROUP BY label,a.status',values)]
        result = {'overall':grouped("'全部'"), 'backend':grouped('a.backend')}
        for row in result['backend']:row['label']=display_backend(row['label'])
        for name in ('category','dtype','op'):
            result[name]=grouped('d.value'," JOIN case_dimensions d ON d.workload_id=a.workload_id AND d.dimension='"+name+"'")
        for name, column, bounds in [('dimension','max_dim',[(8,'≤8'),(64,'≤64'),(512,'≤512'),(4096,'≤4096'),(16384,'≤16384'),(65536,'≤65536'),(131072,'≤131072')]),('elements','max_elements',[(1000,'≤1K'),(16000,'≤16K'),(64000,'≤64K'),(256000,'≤256K'),(1000000,'≤1M'),(16000000,'≤16M'),(64000000,'≤64M')])]:
            expression='CASE WHEN c.'+column+" IS NULL THEN '未采集' "+' '.join(f"WHEN c.{column}<={bound} THEN '{label}'" for bound,label in bounds)+f" ELSE '>{bounds[-1][1][1:]}' END"
            result[name]=grouped(expression)
        result['mode']=grouped("CASE WHEN json_extract(f.execution,'$.static_only') THEN 'static' ELSE 'dynamic' END")
        result['stages']=[dict(r) for r in conn.execute('SELECT a.backend,m.stage,count(*) AS samples,sum(m.duration_ns) AS total_ns,avg(m.duration_ns) AS mean_ns,max(m.duration_ns) AS max_ns'+base+' JOIN stage_metrics m ON m.attempt_id=a.id'+where+' GROUP BY a.backend,m.stage',values)]
        result['performance']=[dict(r) for r in conn.execute("SELECT a.backend,CASE WHEN json_extract(f.execution,'$.static_only') THEN 'static' ELSE 'dynamic' END AS mode,s.correctness,s.performance,s.metric,count(*) AS samples,avg(s.mean_wall_ns) AS mean_ns,min(s.mean_wall_ns) AS min_ns,max(s.mean_wall_ns) AS max_ns"+base+' JOIN shapes s ON s.attempt_id=a.id'+where+' GROUP BY a.backend,mode,s.correctness,s.performance,s.metric',values)]
        result['errors']=[dict(r) for r in conn.execute("SELECT a.backend,a.error_stage,count(*) AS count"+base+where+' GROUP BY a.backend,a.error_stage',values) if r['error_stage']]
        result['case_groups']=conn.execute('SELECT count(DISTINCT f.group_id)'+base+where,values).fetchone()[0]
        result['attempts']=sum(r['count'] for r in result['overall'])
        # Kernel profiler isn't enabled by the executor. Null is intentional,
        # not a count of zero kernels or a zero-time measurement.
        result['kernel_profiler']={'collected':False,'kernel_count':None,'ascendc_kernel_count':None,'triton_kernel_count':None,'kernel_time_ns':None}
        return result
