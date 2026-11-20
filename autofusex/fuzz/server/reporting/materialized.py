"""Rebuildable structured read model; raw attempts/JSON remain authoritative.

Publication and dirty-journal draining share the writer transaction. Readers use
this model only when its version matches and no source changes are pending;
otherwise they fall back to the authoritative query, never stale heads.
"""
import json
from .shared_content import intern_text

from .cases import compatible_ctes
from .failure_origin import VERSION as CLASSIFICATION_VERSION

VERSION = 6
TABLES = ('case_results_v1', 'case_history_v1', 'case_projection_v1')


def version():
    return f'{VERSION}:{CLASSIFICATION_VERSION}'


def installed(conn):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name='case_model_meta_v1'").fetchone() is not None


def install(conn):
    conn.execute('CREATE TABLE IF NOT EXISTS case_model_meta_v1 (id INTEGER PRIMARY KEY, version TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS case_model_dirty_v1 (seed TEXT PRIMARY KEY)')
    conn.execute('CREATE TABLE IF NOT EXISTS display_text_v2 (id INTEGER PRIMARY KEY, digest BLOB NOT NULL UNIQUE, content TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS case_display_refs_v2 (attempt_id TEXT, shape_index INTEGER, seed TEXT, function_status TEXT, failure_path TEXT, evidence_scope TEXT, evidence_id INTEGER REFERENCES display_text_v2(id), core_error_id INTEGER REFERENCES display_text_v2(id), run_log_id INTEGER REFERENCES display_text_v2(id), PRIMARY KEY(attempt_id,shape_index))')
    conn.execute('CREATE INDEX IF NOT EXISTS case_display_seed_v2 ON case_display_refs_v2(seed)')
    conn.execute('CREATE VIEW IF NOT EXISTS case_display_v1 AS SELECT d.attempt_id,d.shape_index,d.seed,d.function_status,d.failure_path,d.evidence_scope,e.content AS evidence,c.content AS core_error,r.content AS run_log FROM case_display_refs_v2 d JOIN display_text_v2 e ON e.id=d.evidence_id JOIN display_text_v2 c ON c.id=d.core_error_id JOIN display_text_v2 r ON r.id=d.run_log_id')
    conn.execute('CREATE TABLE IF NOT EXISTS case_dimensions_flat_v1 (workload_id TEXT, shape_index INTEGER, max_dim INTEGER, max_elements INTEGER, PRIMARY KEY(workload_id,shape_index))')
    conn.execute('CREATE TABLE IF NOT EXISTS case_dtypes_flat_v1 (workload_id TEXT, shape_index INTEGER, dtype TEXT, PRIMARY KEY(workload_id,shape_index,dtype))')
    conn.execute('CREATE TABLE IF NOT EXISTS case_preview_v2 (id INTEGER PRIMARY KEY, digest BLOB NOT NULL UNIQUE, payload TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS case_attempt_refs_v2 (attempt_id TEXT PRIMARY KEY, seed TEXT, mode TEXT, device TEXT, has_diagnostics INTEGER, preview_id INTEGER REFERENCES case_preview_v2(id))')
    conn.execute('CREATE INDEX IF NOT EXISTS case_attempt_seed_v2 ON case_attempt_refs_v2(seed)')
    conn.execute('CREATE VIEW IF NOT EXISTS case_attempt_flat_v1 AS SELECT a.attempt_id,a.seed,a.mode,a.device,a.has_diagnostics,p.payload AS execution_preview FROM case_attempt_refs_v2 a JOIN case_preview_v2 p ON p.id=a.preview_id')
    for sql in trigger_sql():
        conn.execute(sql.replace('CREATE TRIGGER ', 'CREATE TRIGGER IF NOT EXISTS ', 1))

def trigger_sql():
    # AFTER DELETE still has OLD.seed for attempts; dependent rows use the
    # parent before cascading. UPDATE invalidates both old and new populations.
    for table in ('attempts', 'attempt_provenance', 'shapes', 'attempt_arrivals_v2', 'invalid_replays_v1', 'case_facts', 'case_dimensions'):
        for event in ('INSERT', 'UPDATE', 'DELETE'):
            rows = ('OLD', 'NEW') if event == 'UPDATE' else ('OLD',) if event == 'DELETE' else ('NEW',)
            body = []
            for row in rows:
                if table in ('attempts', 'case_facts'):
                    select = f'SELECT {row}.seed'
                elif table == 'case_dimensions':
                    select = f'SELECT seed FROM case_facts WHERE workload_id={row}.workload_id'
                else:
                    select = f'SELECT seed FROM attempts WHERE id={row}.attempt_id'
                body.append('INSERT OR IGNORE INTO case_model_dirty_v1(seed) '+select+';')
            timing = 'BEFORE' if event == 'DELETE' else 'AFTER'
            yield f'CREATE TRIGGER case_model_{table}_{event.lower()} {timing} {event} ON {table} BEGIN '+''.join(body)+' END'


def ready(conn):
    if not installed(conn):
        return False
    row = conn.execute('SELECT version FROM case_model_meta_v1 WHERE id=1').fetchone()
    return bool(row and row[0] == version() and not conn.execute('SELECT 1 FROM case_model_dirty_v1 LIMIT 1').fetchone())


def _cached_sql(sql, *, history=False):
    """Replace expensive CTEs, retaining the exact public filter semantics."""
    tail = sql[sql.index('), cases AS ('):]
    source = 'case_history_v1' if history else 'case_heads_v1'
    return f'''WITH unfiltered_cases AS (SELECT * FROM {source} WHERE SEED_PREDICATE),
        latest AS MATERIALIZED (
            SELECT a.* FROM case_results_v1 a JOIN unfiltered_cases c
              ON c.record_id=a.record_id AND c.shape_index=a.shape_index),
        projection AS (SELECT * FROM case_projection_v1''' + tail


def rollup(conn, **params):
    from .cases import rollup as original
    sql, values = original(**params)
    if not ready(conn):
        return sql, values
    from .search import exact_seed
    seed = params.get('seed') or exact_seed(params.get('q'))
    sql = compatible_ctes(_cached_sql(sql, history=params.get('history', False)))
    return sql.replace('SEED_PREDICATE', 'seed=?' if seed is not None else '1=1'), values


def refresh(conn):
    """Call inside BEGIN IMMEDIATE, after ingestion; initial rebuild is explicit."""
    from .cases import rollup as original
    install(conn)
    row = conn.execute('SELECT version FROM case_model_meta_v1 WHERE id=1').fetchone()
    full = not row or row[0] != version()
    if full:
        kind = conn.execute("SELECT type FROM sqlite_master WHERE name='case_heads_v1'").fetchone()
        if kind:
            conn.execute('DROP '+kind[0].upper()+' case_heads_v1')
        conn.execute('DROP TABLE IF EXISTS case_head_refs_v2')
        conn.execute('CREATE TABLE case_head_refs_v2 (seed TEXT, shape_index INTEGER, cursor INTEGER PRIMARY KEY, status TEXT, head_rank INTEGER, history_count INTEGER, failure_path TEXT)')
        for table in ('case_dimensions_flat_v1','case_dtypes_flat_v1'):
            conn.execute('DELETE FROM '+table)
    seeds = [None] if full else [r[0] for r in conn.execute('SELECT seed FROM case_model_dirty_v1')]
    for seed in seeds:
        sql, values = original(seed=seed)
        conn.execute('DROP TABLE IF EXISTS temp.model_results')
        # Exactly one source classification/JSON pass. All downstream builds
        # reuse this narrow table, not the diagnostic blobs or Python UDFs.
        conn.execute('CREATE TEMP TABLE model_results AS '+sql+'SELECT * FROM all_results', values)
        compact = 'WITH all_results AS (SELECT * FROM model_results), backend_latest AS (SELECT DISTINCT id,seed,backend,workload_id,attempt_cursor FROM model_results), '
        rest = sql[sql.index('shape_bounds AS '):]
        derived = compact + rest
        # Source seed predicate was consumed when creating model_results.
        history_source = original(seed=seed, history=True)[0]
        history_sql = compact + history_source[history_source.index('shape_bounds AS '):]
        queries = ['SELECT * FROM model_results',
                   history_sql+'SELECT * FROM round_cases', history_sql+'SELECT * FROM projection']
        for table, query in zip(TABLES, queries):
            if full:
                conn.execute('DROP TABLE IF EXISTS '+table)
                conn.execute('CREATE TABLE '+table+' AS '+query)
            else:
                conn.execute('DELETE FROM '+table+' WHERE seed=?', (seed,))
                conn.execute('INSERT INTO '+table+' '+query)
        if not full:
            conn.execute('DELETE FROM case_head_refs_v2 WHERE seed=?', (seed,))
        conn.execute('INSERT INTO case_head_refs_v2 '+derived+'SELECT seed,shape_index,cursor,status,head_rank,history_count,failure_path FROM unfiltered_cases')
        conn.execute('CREATE INDEX IF NOT EXISTS case_results_workload_v1 ON case_results_v1(workload_id,shape_index)')
        _flatten(conn, seed)
        conn.execute('DROP TABLE model_results')
    if full:
        for table in TABLES:
            conn.execute(f'CREATE INDEX {table}_seed ON {table}(seed)')
            conn.execute(f'CREATE INDEX {table}_record ON {table}(record_id,shape_index)')
        conn.execute('CREATE UNIQUE INDEX case_heads_key_v1 ON case_head_refs_v2(seed,shape_index)')
        conn.execute('CREATE INDEX case_heads_status_v1 ON case_head_refs_v2(status,cursor DESC)')
        conn.execute('CREATE UNIQUE INDEX case_history_identity_v2 ON case_history_v1(cursor)')
        conn.execute('CREATE INDEX case_history_cursor_v1 ON case_history_v1(seed,shape_index,cursor DESC)')
        conn.execute('CREATE INDEX case_projection_search_v1 ON case_projection_v1(seed,shape_index,record_id)')
    if full:
        columns = [r[1] for r in conn.execute('PRAGMA table_info(case_history_v1)')]
        selection = ','.join(('h.' if c in {'seed','shape_index','cursor','status','head_rank','history_count','failure_path'} else 'r.')+c for c in columns)
        # Unary + removes INTEGER affinity from the pointer operand. History
        # cursor is a CTAS expression with no affinity; without this SQLite
        # scans history instead of using its unique cursor index. CROSS JOIN
        # keeps the bounded pointer range as the outer loop.
        conn.execute('CREATE VIEW case_heads_v1 AS SELECT '+selection+' FROM case_head_refs_v2 h CROSS JOIN case_history_v1 r ON r.cursor=+h.cursor')
    conn.execute('INSERT OR REPLACE INTO case_model_meta_v1 VALUES (1,?)', (version(),))
    conn.execute('DELETE FROM case_model_dirty_v1')
    return {'full': full, 'seeds': len(seeds)}


def refresh_if_installed(conn):
    if installed(conn):
        return refresh(conn)


def rebuild(path):
    from contextlib import closing
    from ...client.storage.repository import _connect
    with closing(_connect(path, writable=True)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        return refresh(conn)


def _flatten(conn, seed):
    """Reusable scalar fields and display attribution, computed only on writes."""
    from .failure_origin import classify, mlir_failure_evidence
    from ...client.failure_details import core_error, redact
    where = '' if seed is None else ' WHERE a.seed=?'
    args = () if seed is None else (seed,)
    for table in ('case_display_refs_v2', 'case_attempt_refs_v2'):
        conn.execute('DELETE FROM '+table+('' if seed is None else ' WHERE seed=?'), args)
    rows = conn.execute("""SELECT a.id,a.seed,a.backend,a.status,a.error_stage,a.error_summary,
        f.execution,a.workload_id FROM attempts a JOIN attempt_facts f ON f.attempt_id=a.id"""+where,args)
    workloads=set()
    for identity, own_seed, backend, status, stage, summary, raw, workload in rows:
        workloads.add(workload)
        execution=json.loads(raw)
        diagnostics=json.loads(execution.get('diagnostics') or '{}')
        preview=json.dumps({k:v for k,v in execution.items() if k not in {'diagnostics','telemetry','case_spec'}})
        preview_id = intern_text(conn, 'case_preview_v2', 'payload', preview)
        conn.execute('INSERT INTO case_attempt_refs_v2 VALUES (?,?,?,?,?,?)',
            (identity,own_seed,'static' if execution.get('static_only') else 'dynamic',execution.get('device'),int(bool(execution.get('diagnostics'))),
             preview_id))
        shapes=conn.execute('SELECT shape_index,correctness,error_stage,error_summary FROM shapes WHERE attempt_id=?',(identity,)).fetchall()
        for index, correctness, local_stage, local_summary in shapes or [(-1,None,stage,summary)]:
            state={'pass':'ok','fail':'fail','error':'error'}.get(correctness,'unknown') if shapes else status
            category=classify(backend,state,local_stage,local_summary,diagnostics,index)
            evidence=mlir_failure_evidence(backend,state,local_stage,local_summary,diagnostics,index,len(shapes))
            strings=(evidence['evidence'],core_error(diagnostics,index,local_summary),redact(diagnostics.get('error','')))
            text_ids=[]
            for value in strings:
                text_ids.append(intern_text(conn, 'display_text_v2', 'content', value))
            conn.execute('INSERT INTO case_display_refs_v2 VALUES (?,?,?,?,?,?,?,?,?)',
                (identity,index,own_seed,category,evidence['path'],evidence['scope'],
                 *text_ids))
    for workload in workloads:
        row=conn.execute('SELECT metadata FROM case_facts WHERE workload_id=?',(workload,)).fetchone()
        if not row:continue
        meta=json.loads(row[0])
        conn.execute('DELETE FROM case_dimensions_flat_v1 WHERE workload_id=?',(workload,))
        conn.execute('DELETE FROM case_dtypes_flat_v1 WHERE workload_id=?',(workload,))
        indices={r[0] for r in conn.execute('SELECT DISTINCT shape_index FROM case_results_v1 WHERE workload_id=?',(workload,))}
        variants=meta.get('shape_variants',[])
        for index in indices:
            variant=variants[index] if 0<=index<len(variants) else None
            dtype_meta=variant if variant is not None else meta
            conn.execute('INSERT INTO case_dimensions_flat_v1 VALUES (?,?,?,?)',
                (workload,index,variant.get('max_dim') if variant else None,variant.get('max_elements') if variant else None))
            dtypes=set((dtype_meta.get('input_dtypes') or {}).values()) | set((dtype_meta.get('output_dtypes') or {}).values())
            conn.executemany('INSERT INTO case_dtypes_flat_v1 VALUES (?,?,?)',[(workload,index,dtype) for dtype in dtypes])
