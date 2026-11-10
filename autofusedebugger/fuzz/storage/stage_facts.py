"""Versioned scheduling facts, separate from immutable execution records."""
from contextlib import closing
import json
import time
from .repository import _connect
from .lifecycle import stamp, read, elapsed

DDL = '''
CREATE TABLE IF NOT EXISTS stage_facts_v1 (
 stage_id TEXT PRIMARY KEY, window_id TEXT NOT NULL, case_id TEXT NOT NULL,
 session_id TEXT, attempt_id TEXT, payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS stage_window_v1 ON stage_facts_v1(window_id);
CREATE TABLE IF NOT EXISTS stage_windows_v1 (window_id TEXT PRIMARY KEY,payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS stage_attempts_v1 (attempt_id TEXT PRIMARY KEY,case_id TEXT NOT NULL,window_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fact_outbox_v1 (revision INTEGER PRIMARY KEY AUTOINCREMENT,kind TEXT NOT NULL,identity TEXT NOT NULL,payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS stage_window_aggregates_v1(window_id TEXT PRIMARY KEY,payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fact_published_v1(kind TEXT,identity TEXT,PRIMARY KEY(kind,identity));
CREATE TABLE IF NOT EXISTS fact_ack_v1 (id INTEGER PRIMARY KEY,ack INTEGER NOT NULL);
'''


def initialize(conn):
    for sql in DDL.split(';'):
        if sql.strip():conn.execute(sql)
    conn.execute('INSERT OR IGNORE INTO fact_ack_v1 VALUES (1,0)')


def publish(conn,kind,identity,payload):
    inserted=conn.execute('INSERT OR IGNORE INTO fact_published_v1 VALUES (?,?)',(kind,identity)).rowcount
    if inserted:
        conn.execute('INSERT INTO fact_outbox_v1(kind,identity,payload) VALUES (?,?,?)',(kind,identity,json.dumps(payload,sort_keys=True,separators=(',',':'))))


def save_case(database,case,sessions,window_id,start,pool):
    saved=[t for t in case.tasks if json.loads(t['execution']).get('save_results',True)]
    if not saved:return
    with closing(_connect(database,writable=True)) as conn,conn:
        initialize(conn)
        committed={t['attempt_id'] for t in saved if conn.execute('SELECT 1 FROM attempts WHERE id=?',(t['attempt_id'],)).fetchone()}
        if not committed:return
        for t in saved:
            # Record failures without a committed result cannot be exported yet.
            if not conn.execute('SELECT 1 FROM attempts WHERE id=?',(t['attempt_id'],)).fetchone():continue
            conn.execute('INSERT OR IGNORE INTO stage_attempts_v1 VALUES (?,?,?)',(t['attempt_id'],case.identity,window_id))
            publish(conn,'attempt',t['attempt_id'],{'case_id':case.identity,'window_id':window_id})
        for s in case.stages.values():
            if s.attempt_id and s.attempt_id not in committed:continue
            payload={'version':1,'kind':s.kind.value,'state':s.state,'dependencies':list(s.dependencies),
                     'admitted':case.admitted['monotonic_ns'],'domain':case.admitted['domain'],**s.facts}
            row=(s.identity,window_id,case.identity,s.session_id,s.attempt_id,json.dumps(payload))
            conn.execute('INSERT OR IGNORE INTO stage_facts_v1 VALUES (?,?,?,?,?,?)',row)
            publish(conn,'stage',s.identity,{'window_id':window_id,'case_id':case.identity,'session_id':s.session_id,'attempt_id':s.attempt_id,'facts':payload})
        # Provisional windows expose progress but are not finite throughput.
        value={'version':1,'start':start,'finished':None,'cpu_tokens':pool.cpu_capacity,'devices':[d.uuid for d in pool.devices]}
        conn.execute('INSERT OR IGNORE INTO stage_windows_v1 VALUES (?,?)',(window_id,json.dumps(value)))


def finish_window(database,identity,start,pool,summary=None):
    if not database.exists():return
    value={'version':1,'start':start,'finished':stamp(),'cpu_tokens':pool.cpu_capacity,'devices':[d.uuid for d in pool.devices],'interrupted_attempts':(summary or {}).get('interrupted',0),'cancelled_attempts':(summary or {}).get('cancelled',0),'record_failed_attempts':(summary or {}).get('record_failed',0)}
    with closing(_connect(database,writable=True)) as conn,conn:
        initialize(conn)
        # End at the last observed result commit, not at process teardown or upload.
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='lifecycle_v1'").fetchone():
            row=conn.execute("SELECT l.domain,l.monotonic_ns,l.wall_time FROM lifecycle_v1 l JOIN stage_attempts_v1 a ON a.attempt_id=l.attempt_id WHERE a.window_id=? AND l.event='persisted' AND l.domain=? ORDER BY l.monotonic_ns DESC LIMIT 1",(identity,start['domain'])).fetchone()
            if row:value['finished']={'domain':row[0],'monotonic_ns':row[1],'wall_time':row[2]}
        conn.execute('INSERT OR REPLACE INTO stage_windows_v1 VALUES (?,?)',(identity,json.dumps(value)))
        if conn.execute('SELECT 1 FROM stage_attempts_v1 WHERE window_id=?',(identity,)).fetchone():
            publish(conn,'window',identity,value)
        from ..reporting.efficiency import statistics
        aggregates=[r for r in statistics(conn) if r['window_id']==identity]
        conn.execute('INSERT OR REPLACE INTO stage_window_aggregates_v1 VALUES (?,?)',(identity,json.dumps(aggregates)))
        retain(conn)


def timeline(conn,identity):
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='stage_attempts_v1'").fetchone():return []
    row=conn.execute('SELECT case_id FROM stage_attempts_v1 WHERE attempt_id=?',(identity,)).fetchone()
    if not row:return []
    return [dict(stage_id=r[0],**json.loads(r[1])) for r in conn.execute('SELECT stage_id,payload FROM stage_facts_v1 WHERE case_id=? AND (attempt_id=? OR attempt_id IS NULL)',(row[0],identity))]


def copy_public(source,target):
    """Copy only bounded public observations; never replication cursors/receipts."""
    from .lifecycle import initialize as init_lifecycle
    if source.execute("SELECT 1 FROM sqlite_master WHERE name='lifecycle_v1'").fetchone():
        init_lifecycle(target)
        for row in source.execute('SELECT * FROM lifecycle_v1 WHERE attempt_id IN (SELECT id FROM attempts)'):
            old=target.execute('SELECT * FROM lifecycle_v1 WHERE attempt_id=? AND event=?',tuple(row)[:2]).fetchone()
            if old and tuple(old)!=tuple(row):raise ValueError('lifecycle import conflict')
            target.execute('INSERT OR IGNORE INTO lifecycle_v1 VALUES (?,?,?,?,?)',tuple(row))
    if not source.execute("SELECT 1 FROM sqlite_master WHERE name='stage_attempts_v1'").fetchone():return
    for sql in DDL.split(';'):
        if sql.strip() and 'fact_outbox_v1' not in sql and 'fact_ack_v1' not in sql and 'fact_published_v1' not in sql:target.execute(sql)
    for table,key in [('stage_attempts_v1','attempt_id'),('stage_facts_v1','stage_id'),('stage_windows_v1','window_id'),('stage_window_aggregates_v1','window_id')]:
        for row in source.execute('SELECT * FROM '+table):
            old=target.execute('SELECT * FROM '+table+' WHERE '+key+'=?',(row[0],)).fetchone()
            if old and tuple(old)!=tuple(row):raise ValueError('stage fact import conflict')
            target.execute('INSERT OR IGNORE INTO '+table+' VALUES ('+','.join('?' for _ in row)+')',tuple(row))


def retain(conn, max_windows=256):
    """Bound raw events by completed windows; preserve long-term aggregates.

    Only ACKed local windows (or central received windows) are compacted. Offline
    outbox remains durable; user disk reserve/backpressure prevents silent loss.
    """
    initialize(conn)
    rows=conn.execute('SELECT window_id FROM stage_windows_v1 ORDER BY rowid DESC LIMIT -1 OFFSET ?',(max_windows,)).fetchall()
    for (identity,) in rows:
        if conn.execute("SELECT 1 FROM fact_outbox_v1 WHERE kind='window' AND identity=?",(identity,)).fetchone():continue
        if not conn.execute('SELECT 1 FROM stage_window_aggregates_v1 WHERE window_id=?',(identity,)).fetchone():continue
        conn.execute('DELETE FROM stage_facts_v1 WHERE window_id=?',(identity,))
        conn.execute("DELETE FROM lifecycle_v1 WHERE attempt_id IN (SELECT attempt_id FROM stage_attempts_v1 WHERE window_id=?)",(identity,))
