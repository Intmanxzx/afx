"""Versioned materialized reporting index, stored alongside immutable result rows.

Additive extension: core result hashes and replication IDs never change. All facts
are derived from frozen case protocol or recorded execution data, never today's
machine versions. Rebuild only on explicit rebuild or insertion, not on reads.
"""
import json
from contextlib import closing
from autofusex.execution.storage.records import canonical, digest

DDL = '''
CREATE TABLE IF NOT EXISTS report_search(attempt_id TEXT PRIMARY KEY REFERENCES attempts(id), content TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS report_schema(version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS case_facts(
 workload_id TEXT PRIMARY KEY, seed TEXT NOT NULL, metadata TEXT NOT NULL,
 nodes INTEGER, max_dim INTEGER, max_elements INTEGER, shape_count INTEGER);
CREATE TABLE IF NOT EXISTS report_versions(
 id INTEGER PRIMARY KEY, payload TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS attempt_provenance(
 attempt_id TEXT PRIMARY KEY REFERENCES attempts(id), group_id TEXT NOT NULL,
 source_id TEXT NOT NULL, device_name TEXT, user_id TEXT, batch_id TEXT,
 version_id INTEGER NOT NULL REFERENCES report_versions(id),
 total_ns INTEGER, duration_scope TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS report_group_attempt ON attempt_provenance(group_id,attempt_id);
CREATE INDEX IF NOT EXISTS agent_result_batch ON attempt_provenance(batch_id,attempt_id);
CREATE TABLE IF NOT EXISTS stage_metrics(
 attempt_id TEXT NOT NULL REFERENCES attempts(id), shape_index INTEGER NOT NULL,
 stage TEXT NOT NULL, duration_ns INTEGER NOT NULL,
 PRIMARY KEY(attempt_id,shape_index,stage));
CREATE TABLE IF NOT EXISTS case_dimensions(
 workload_id TEXT NOT NULL, dimension TEXT NOT NULL, value TEXT NOT NULL,
 PRIMARY KEY(workload_id,dimension,value));
CREATE TABLE IF NOT EXISTS report_devices(
 source_id TEXT PRIMARY KEY, user_id TEXT, device_name TEXT);
'''


ARRIVAL_TRIGGER = """CREATE TRIGGER attempt_first_arrival_v2 AFTER INSERT ON attempts BEGIN
        INSERT OR IGNORE INTO attempt_arrivals_v2(attempt_id,received_at) VALUES (NEW.id,strftime('%Y-%m-%dT%H:%M:%f+00:00','now')); END"""


def initialize(conn):
    # execute individually: executescript would commit caller's ingestion transaction.
    for statement in DDL.split(';'):
        if statement.strip(): conn.execute(statement)
    # Join shared immutable contexts without copying large JSON per attempt.
    conn.execute('''CREATE VIEW IF NOT EXISTS attempt_facts AS
        SELECT p.attempt_id,p.group_id,p.source_id,p.device_name,p.user_id,p.batch_id,
               v.payload AS versions,e.payload AS execution,q.payload AS evaluation,
               p.total_ns,p.duration_scope
        FROM attempt_provenance p JOIN attempts a ON a.id=p.attempt_id
        JOIN report_versions v ON v.id=p.version_id
        JOIN contexts e ON e.id=a.execution_id JOIN contexts q ON q.id=a.evaluation_id''')
    conn.execute('''CREATE TRIGGER IF NOT EXISTS report_facts_insert INSTEAD OF INSERT ON attempt_facts BEGIN
        INSERT INTO report_versions(payload) VALUES (NEW.versions) ON CONFLICT(payload) DO NOTHING;
        INSERT OR REPLACE INTO attempt_provenance VALUES
        (NEW.attempt_id,NEW.group_id,NEW.source_id,NEW.device_name,NEW.user_id,NEW.batch_id,
         (SELECT id FROM report_versions WHERE payload=NEW.versions),NEW.total_ns,NEW.duration_scope);
        END''')
    conn.execute('''CREATE TRIGGER IF NOT EXISTS report_facts_delete INSTEAD OF DELETE ON attempt_facts BEGIN
        DELETE FROM attempt_provenance WHERE attempt_id=OLD.attempt_id; END''')
    conn.execute('''CREATE TRIGGER IF NOT EXISTS report_facts_update INSTEAD OF UPDATE ON attempt_facts BEGIN
        SELECT CASE WHEN NEW.execution IS NOT OLD.execution OR NEW.evaluation IS NOT OLD.evaluation
          OR NEW.attempt_id IS NOT OLD.attempt_id THEN RAISE(ABORT,'immutable facts require original context') END;
        INSERT INTO report_versions(payload) VALUES (NEW.versions) ON CONFLICT(payload) DO NOTHING;
        UPDATE attempt_provenance SET group_id=NEW.group_id,source_id=NEW.source_id,
        device_name=NEW.device_name,user_id=NEW.user_id,batch_id=NEW.batch_id,
        version_id=(SELECT id FROM report_versions WHERE payload=NEW.versions),
        total_ns=NEW.total_ns,duration_scope=NEW.duration_scope WHERE attempt_id=OLD.attempt_id; END''')
    conn.execute('CREATE TABLE IF NOT EXISTS attempt_arrivals_v2 (sequence INTEGER PRIMARY KEY AUTOINCREMENT, attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(id) ON DELETE CASCADE, received_at TEXT NOT NULL)')
    conn.execute(ARRIVAL_TRIGGER.replace('CREATE TRIGGER ', 'CREATE TRIGGER IF NOT EXISTS ',1))
    conn.execute('CREATE TABLE IF NOT EXISTS invalid_replays_v1 (attempt_id TEXT PRIMARY KEY REFERENCES attempts(id) ON DELETE CASCADE, reason TEXT NOT NULL)')
    row=conn.execute('SELECT version FROM report_schema').fetchone()
    if row is None: conn.execute('INSERT INTO report_schema VALUES (2)')
    elif row[0]!=2: raise ValueError('unsupported reporting index version')


def index_attempt(conn, identity, source_id=None, device_name=None, user_id=None):
    initialize(conn)
    columns=[r[1] for r in conn.execute('PRAGMA table_info(attempts)')]
    row=dict(zip(columns,conn.execute('SELECT * FROM attempts WHERE id=?',(identity,)).fetchone()))
    contexts={k:json.loads(conn.execute('SELECT payload FROM contexts WHERE id=?',(row[k+'_id'],)).fetchone()[0])
              for k in ('environment','execution','evaluation')}
    env,ex,ev=(contexts[k] for k in ('environment','execution','evaluation'))
    existing=conn.execute('SELECT source_id,device_name,user_id FROM attempt_facts WHERE attempt_id=?',(identity,)).fetchone()
    source_id=source_id or (existing[0] if existing else None) or env.get('installation_id') or 'unrecorded'
    device_name=device_name or (existing[1] if existing else None) or env.get('device_name') or '设备未记录'
    user_id=user_id or (existing[2] if existing else None) or env.get('user_id')
    conn.execute('INSERT OR REPLACE INTO report_devices VALUES (?,?,?)',(source_id,user_id,device_name))
    versions={key:env.get(key) for key in ('python','platform','machine','torch','torch_npu','triton','package_version','cann','autofuse_commit','autofuse_binary')}
    config={key:value for key,value in ex.items() if key not in {'backend','device','telemetry','diagnostics','batch_id','expected_backends'}}
    batch=ex.get('batch_id') or 'unbatched'
    group=digest({'workload':row['workload_id'],'oracle':row['oracle_id'],'source':source_id,
                  'versions':versions,'config':config,'evaluation':ev,'batch':batch,'environment':{k:v for k,v in env.items() if k not in {'device_uuid','device_model'}}})
    telemetry=json.loads(ex['telemetry']) if ex.get('telemetry') else {}
    stages=telemetry.get('stages',{})
    total=stages.get('execution_total')
    if total is not None: total+=stages.get('generation',0)
    scope='generation+execution' if total is not None else 'partial_recorded_only'
    if total is None:
        first=conn.execute('SELECT sum(first_call_combined_ns) FROM shapes WHERE attempt_id=?',(identity,)).fetchone()[0]
        total=(row['prepare_ns'] or 0)+(first or 0) if row['prepare_ns'] is not None or first is not None else None
    conn.execute('INSERT OR REPLACE INTO attempt_facts VALUES (?,?,?,?,?,?,?,?,?,?,?)',
        (identity,group,source_id,device_name,user_id,batch,canonical(versions),canonical(ex),canonical(ev),total,scope))
    conn.execute('DELETE FROM stage_metrics WHERE attempt_id=?',(identity,))
    pairs=[(-1,k,v) for k,v in stages.items()]
    pairs += [(int(index),key,value) for index,values in telemetry.get('shapes',{}).items() for key,value in values.items()]
    if not telemetry:
        if row['prepare_ns'] is not None:pairs.append((-1,'prepare',row['prepare_ns']))
        pairs += [(r[0],'first_call_combined',r[1]) for r in conn.execute('SELECT shape_index,first_call_combined_ns FROM shapes WHERE attempt_id=? AND first_call_combined_ns IS NOT NULL',(identity,))]
    conn.executemany('INSERT INTO stage_metrics VALUES (?,?,?,?)',[(identity,*pair) for pair in pairs])
    metadata = conn.execute('SELECT metadata FROM case_facts WHERE workload_id=?',(row['workload_id'],)).fetchone()
    from autofusex.workloads.fuzz.contracts import Seed
    from autofusex.execution.backend_names import display_backend
    from datetime import datetime, timezone, timedelta
    text = ' '.join(str(v) for v in [Seed.parse(row['seed']).short, display_backend(row['backend']), device_name, user_id, source_id, batch, row['created_at'], datetime.fromisoformat(row['created_at']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'), canonical(versions), row['error_stage'], row['error_summary'], ('Pass' if row['status']=='ok' else 'Failed' if row['status'] in {'fail','error'} else 'Unkown'), 'static 静态' if ex.get('static_only') else 'dynamic 动态', str(round(total/1e9,1))+' s' if total is not None else 'Unkown'])
    # Include displayed precise metric values and independent correctness states.
    for _, stage, value in pairs:
        if stage in {'compile','kernel_mean'}:
            text += ' '+format(value/1e6,'.3f')+' ms'
    states=[r[0] for r in conn.execute('SELECT correctness FROM shapes WHERE attempt_id=?',(identity,))]
    if 'pass' in states:text+=' Pass'
    if 'fail' in states or 'error' in states:text+=' Failed'
    if any(value is None for value in versions.values()):text+=' Unkown'
    if metadata:
        conn.execute('INSERT OR REPLACE INTO report_search VALUES (?,?)',(identity,text))
        return
    try:
        if ex.get('case_spec'):
            from autofusex.workloads.fuzz.case_spec import decode
            from autofusex.execution.reporting.report import spec_metadata
            spec=decode(ex['case_spec'])
            metadata=spec_metadata(spec)
        else:
            raise ValueError('record has no captured graph specification')
        shapes=[t.shape for sample in (*spec.shape_samples,*spec.output_samples) for t in sample]
        import math
        max_dim=max((max(shape,default=0) for shape in shapes),default=0)
        max_elements=max((math.prod(shape) for shape in shapes),default=0)
    except (ValueError,KeyError):
        metadata={};max_dim=max_elements=None
    conn.execute('INSERT INTO case_facts VALUES (?,?,?,?,?,?,?)',(row['workload_id'],row['seed'],canonical(metadata),metadata.get('n_nodes'),max_dim,max_elements,len(metadata.get('shape_variants',[]))))
    conn.execute('INSERT OR REPLACE INTO report_search VALUES (?,?)',(identity,text))
    dimensions={'op':sorted(set(metadata.get('op_types',[]))),'dtype':sorted(set(metadata.get('input_dtypes',{}).values())|set(metadata.get('output_dtypes',{}).values()))}
    from autofusex.workloads.fuzz.generation.categories import OP_CATEGORY
    # Reporting names can combine overloads; classify the captured ATen targets
    # so display aliases never discard or misclassify an operator category.
    graph = metadata.get('graph', {})
    targets = [node['op'] for node in graph.get('nodes', [])]
    dimensions['category'] = sorted({OP_CATEGORY[op] for op in targets if op in OP_CATEGORY} |
        ({'Reduce'} if graph.get('reduce_axis') is not None else set()))
    for dimension,values in dimensions.items():
        conn.executemany('INSERT INTO case_dimensions VALUES (?,?,?)',[(row['workload_id'],dimension,value) for value in values])


def rebuild(path):
    """Rebuild derived reporting facts atomically from current records."""
    from autofusex.execution.storage.repository import _connect
    with closing(_connect(path,writable=True)) as conn,conn:
        conn.execute('BEGIN IMMEDIATE');initialize(conn)
        # Derived metadata is reproducible; rebuild it to include graph edges and
        # per-input dtypes added by newer viewers. Immutable attempts stay intact.
        conn.execute('DELETE FROM case_dimensions')
        conn.execute('DELETE FROM case_facts')
        for identity, in conn.execute('SELECT id FROM attempts ORDER BY rowid').fetchall():
            index_attempt(conn, identity)
        from autofusex.execution.reporting.materialized import refresh_if_installed
        refresh_if_installed(conn)
        return {'indexed':conn.execute('SELECT count(*) FROM attempt_facts').fetchone()[0],'report_schema':2}
