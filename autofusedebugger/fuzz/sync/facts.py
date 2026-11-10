"""Independent version-1 scheduling/lifecycle replication, after result ACKs.

Result protocol and hashes are untouched. Fact revisions have their own ACK;
retransmitting a lost ACK cannot duplicate a stage or change an observation.
"""
from contextlib import closing
import json
import uuid
from ..storage.repository import _connect
from ..storage.stage_facts import initialize
from ..storage.records import canonical, digest


def rebuild_outbox(conn):
    from ..storage.stage_facts import publish
    initialize(conn)
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name='lifecycle_v1'").fetchone():
        for identity,event,domain,mono,wall in conn.execute('SELECT l.* FROM lifecycle_v1 l JOIN attempts a ON a.id=l.attempt_id'):
            publish(conn,'lifecycle',identity+':'+event,{'domain':domain,'monotonic_ns':mono,'wall_time':wall})
    for identity,case_id,window in conn.execute('SELECT * FROM stage_attempts_v1 WHERE attempt_id IN (SELECT id FROM attempts)'):
        publish(conn,'attempt',identity,{'case_id':case_id,'window_id':window})
    for identity,window,case_id,session,attempt,payload in conn.execute('SELECT * FROM stage_facts_v1'):
        publish(conn,'stage',identity,{'window_id':window,'case_id':case_id,'session_id':session,'attempt_id':attempt,'facts':json.loads(payload)})
    for identity,payload in conn.execute('SELECT * FROM stage_windows_v1'):
        p=json.loads(payload)
        if p.get('finished') and conn.execute('SELECT 1 FROM stage_attempts_v1 WHERE window_id=?',(identity,)).fetchone():publish(conn,'window',identity,p)


def pending(path):
    with closing(_connect(path,writable=True)) as conn,conn:
        needs_rebuild=not conn.execute("SELECT 1 FROM sqlite_master WHERE name='fact_published_v1'").fetchone()
        initialize(conn)
        if needs_rebuild:rebuild_outbox(conn)
        source=conn.execute('SELECT source_id FROM sync_identity WHERE id=1').fetchone()
        if not source:return None
        rows=conn.execute('SELECT revision,kind,identity,payload FROM fact_outbox_v1 WHERE revision>(SELECT ack FROM fact_ack_v1) ORDER BY revision LIMIT 25').fetchall()
        ready=[]
        for row in rows:
            identity=row[2].split(':',1)[0] if row[1]=='lifecycle' else row[2] if row[1]=='attempt' else None
            if identity and not conn.execute('SELECT 1 FROM sync_outbox WHERE attempt_id=? AND revision<=(SELECT ack FROM sync_identity)',(identity,)).fetchone():
                break
            ready.append(row)
        rows=ready
        return {'version':1,'source_id':source[0],'items':[{'revision':r[0],'kind':r[1],'identity':r[2],'payload':json.loads(r[3])} for r in rows]}


def acknowledge(path,batch,ack):
    if ack.get('source_id')!=batch['source_id'] or ack.get('ack')!=batch['items'][-1]['revision']:
        raise ValueError('unexpected fact ACK')
    with closing(_connect(path,writable=True)) as conn,conn:
        conn.execute('UPDATE fact_ack_v1 SET ack=max(ack,?)',(ack['ack'],))
        # Retain facts, not redundant acknowledged outbound payloads.
        conn.execute('DELETE FROM fact_outbox_v1 WHERE revision<=?',(ack['ack'],))
        from ..storage.stage_facts import retain
        retain(conn)


def upload(home,path,opener=None):
    from pathlib import Path
    from urllib.request import Request, build_opener
    from .client import NoRedirect
    batch=pending(path)
    if not batch or not batch['items']:return {'fact_uploaded':0}
    config=json.loads((Path(home)/'sync.json').read_text())
    key=Path(config['key_file']).read_text().strip()
    body=canonical(batch).encode()
    if len(body)>2*1024*1024:raise ValueError('fact batch too large')
    request=Request(config['endpoint']+'/api/facts',body,headers={'Content-Type':'application/json','Authorization':'Bearer '+key,'User-Agent':'AutofuseDebugger/0.1.2 (+https://esay.app)'})
    with (opener or build_opener(NoRedirect).open)(request,timeout=15) as response:
        raw=response.read(8193)
        if len(raw)>8192:raise ValueError('oversized fact ACK')
        ack=json.loads(raw)
    acknowledge(path,batch,ack)
    return {'fact_uploaded':len(batch['items'])}


def validate_stage(facts):
    from ..scheduling.stages import Kind
    allowed={'version','kind','state','dependencies','admitted','domain','ready','dispatched','started','finished','worker_pid','temperature','worker_id','rss_bytes','lease','retry_index'}
    if not isinstance(facts,dict) or set(facts)-allowed or facts.get('version')!=1 or facts.get('kind') not in {k.value for k in Kind}:
        raise ValueError('invalid stage facts')
    if facts.get('state') not in {'succeeded','failed','skipped','cancelled'}:raise ValueError('unfinished stage fact')
    if not isinstance(facts.get('domain'),str) or len(facts['domain'])!=64:raise ValueError('invalid clock domain')
    for k in ('admitted','ready','dispatched','started','finished','worker_pid','rss_bytes','retry_index'):
        if k in facts and (type(facts[k]) is not int or not 0<=facts[k]<2**63):raise ValueError('invalid stage clock')
    if not isinstance(facts.get('dependencies'),list) or len(facts['dependencies'])>16:raise ValueError('invalid dependencies')
    for d in facts['dependencies']:str(uuid.UUID(d))
    if 'worker_id' in facts:str(uuid.UUID(facts['worker_id']))
    if 'temperature' in facts and facts['temperature'] not in {'cold','warm'}:raise ValueError('invalid temperature')
    lease=facts.get('lease')
    if lease is not None:
        if set(lease)!={'cpu_tokens','memory_bytes','cache_bytes','device_uuid','device','device_bytes','acquired','released'}:raise ValueError('invalid lease')
        for k in ('cpu_tokens','memory_bytes','cache_bytes','device_bytes','acquired','released'):
            if type(lease[k]) is not int or not 0<=lease[k]<2**63:raise ValueError('invalid lease value')
        if lease['released']<lease['acquired'] or not isinstance(lease['device'],str) or len(lease['device'])>64:raise ValueError('invalid lease interval')
        if lease['device_uuid'] is not None and (not isinstance(lease['device_uuid'],str) or len(lease['device_uuid'])>128):raise ValueError('invalid device UUID')


def validate_stamp(value):
    from datetime import datetime
    if not isinstance(value,dict) or set(value)!={'domain','monotonic_ns','wall_time'}:raise ValueError('invalid stamp')
    if not isinstance(value['domain'],str) or len(value['domain'])!=64 or type(value['monotonic_ns']) is not int or not 0<=value['monotonic_ns']<2**63:raise ValueError('invalid stamp clock')
    if not isinstance(value['wall_time'],str) or len(value['wall_time'])>64 or datetime.fromisoformat(value['wall_time']).tzinfo is None:raise ValueError('invalid wall clock')


def ingest(path,key,batch):
    from ..central.repository import authenticate
    from ..storage.lifecycle import initialize as init_lifecycle, EVENTS
    if not isinstance(batch,dict) or batch.get('version')!=1 or not isinstance(batch.get('items'),list) or not 1<=len(batch['items'])<=25:
        raise ValueError('invalid fact protocol')
    source=str(uuid.UUID(batch['source_id']))
    with closing(_connect(path,writable=True)) as conn,conn:
        conn.execute('BEGIN IMMEDIATE')
        user=authenticate(conn,key)
        if conn.execute('SELECT user_id FROM sources WHERE id=?',(source,)).fetchone()!=(user,):
            raise PermissionError('source ownership mismatch')
        initialize(conn);init_lifecycle(conn)
        conn.execute('CREATE TABLE IF NOT EXISTS fact_receipts_v1(source_id TEXT,revision INTEGER,payload_hash TEXT,PRIMARY KEY(source_id,revision))')
        conn.execute('CREATE TABLE IF NOT EXISTS fact_owners_v1(kind TEXT,identity TEXT,source_id TEXT,PRIMARY KEY(kind,identity))')
        previous=0
        def owned_attempt(identity):
            if conn.execute('SELECT user_id FROM owners WHERE attempt_id=?',(identity,)).fetchone()!=(user,):
                raise PermissionError('fact attempt ownership mismatch')
        def own(kind,identity):
            row=conn.execute('SELECT source_id FROM fact_owners_v1 WHERE kind=? AND identity=?',(kind,identity)).fetchone()
            if row and row!=(source,) and conn.execute('SELECT user_id FROM sources WHERE id=?',row).fetchone()!=(user,):raise PermissionError('fact ownership conflict')
            conn.execute('INSERT OR IGNORE INTO fact_owners_v1 VALUES (?,?,?)',(kind,identity,source))
        for item in batch['items']:
            if not isinstance(item,dict) or set(item)!={'revision','kind','identity','payload'}:raise ValueError('invalid fact item')
            rev=item['revision'];kind=item['kind'];identity=item['identity'];p=item['payload']
            if not isinstance(p,dict) or not isinstance(identity,str) or len(identity)>100:raise ValueError('invalid fact payload')
            if type(rev) is not int or rev<=previous:raise ValueError('fact revisions must increase')
            previous=rev
            h=digest(item);receipt=conn.execute('SELECT payload_hash FROM fact_receipts_v1 WHERE source_id=? AND revision=?',(source,rev)).fetchone()
            if receipt:
                if receipt!=(h,):raise ValueError('fact revision conflict')
                continue
            if len(canonical(p))>65536:raise ValueError('fact too large')
            if kind=='lifecycle':
                attempt,event=identity.split(':',1);owned_attempt(attempt)
                if event not in EVENTS or set(p)!={'domain','monotonic_ns','wall_time'}:raise ValueError('invalid lifecycle fact')
                if type(p['monotonic_ns']) is not int or not 0<=p['monotonic_ns']<2**63 or len(p['domain'])!=64 or len(p['wall_time'])>64:raise ValueError('invalid observation')
                validate_stamp(p)
                row=(attempt,event,p['domain'],p['monotonic_ns'],p['wall_time'])
                old=conn.execute('SELECT * FROM lifecycle_v1 WHERE attempt_id=? AND event=?',(attempt,event)).fetchone()
                if old and old!=row:raise ValueError('immutable lifecycle conflict')
                conn.execute('INSERT OR IGNORE INTO lifecycle_v1 VALUES (?,?,?,?,?)',row)
            elif kind=='attempt':
                owned_attempt(identity)
                if set(p)!={'case_id','window_id'}:raise ValueError('invalid association')
                for k in ('case_id','window_id'):str(uuid.UUID(p[k]));own(k,p[k])
                old=conn.execute('SELECT case_id,window_id FROM stage_attempts_v1 WHERE attempt_id=?',(identity,)).fetchone()
                if old and old!=(p['case_id'],p['window_id']):raise ValueError('association conflict')
                conn.execute('INSERT OR IGNORE INTO stage_attempts_v1 VALUES (?,?,?)',(identity,p['case_id'],p['window_id']))
            elif kind=='stage':
                if set(p)!={'window_id','case_id','session_id','attempt_id','facts'}:raise ValueError('invalid stage')
                for k in ('case_id','window_id'):
                    if conn.execute('SELECT s.user_id FROM fact_owners_v1 f JOIN sources s ON s.id=f.source_id WHERE f.kind=? AND f.identity=?',(k,p[k])).fetchone()!=(user,):raise PermissionError('unknown case/window')
                if p['attempt_id']:owned_attempt(p['attempt_id'])
                own('stage',identity)
                facts=p['facts'];validate_stage(facts)
                row=(identity,p['window_id'],p['case_id'],p['session_id'],p['attempt_id'],json.dumps(facts))
                old=conn.execute('SELECT * FROM stage_facts_v1 WHERE stage_id=?',(identity,)).fetchone()
                if old and (tuple(old[:-1])!=row[:-1] or json.loads(old[-1])!=facts):raise ValueError('immutable stage conflict')
                conn.execute('INSERT OR IGNORE INTO stage_facts_v1 VALUES (?,?,?,?,?,?)',row)
            elif kind=='window':
                if set(p)-{'version','start','finished','cpu_tokens','devices','interrupted_attempts','cancelled_attempts','record_failed_attempts'} or not {'version','start','finished','cpu_tokens','devices'}<=set(p) or p['version']!=1 or type(p['cpu_tokens']) is not int or not 1<=p['cpu_tokens']<=65536:raise ValueError('invalid window')
                validate_stamp(p['start']);validate_stamp(p['finished'])
                if p['start']['domain']!=p['finished']['domain'] or p['finished']['monotonic_ns']<p['start']['monotonic_ns']:raise ValueError('invalid window clock')
                if not isinstance(p['devices'],list) or len(p['devices'])>64 or any(not isinstance(d,str) or len(d)>128 for d in p['devices']):raise ValueError('invalid devices')
                if conn.execute('SELECT s.user_id FROM fact_owners_v1 f JOIN sources s ON s.id=f.source_id WHERE f.kind=? AND f.identity=?',('window_id',identity)).fetchone()!=(user,):raise PermissionError('unknown window')
                old=conn.execute('SELECT payload FROM stage_windows_v1 WHERE window_id=?',(identity,)).fetchone()
                if old and json.loads(old[0])!=p:raise ValueError('immutable window conflict')
                conn.execute('INSERT OR IGNORE INTO stage_windows_v1 VALUES (?,?)',(identity,json.dumps(p)))
                from ..reporting.efficiency import statistics
                aggregates=[r for r in statistics(conn) if r['window_id']==identity]
                conn.execute('INSERT OR REPLACE INTO stage_window_aggregates_v1 VALUES (?,?)',(identity,json.dumps(aggregates)))
                from ..storage.stage_facts import retain
                retain(conn)
            else:raise ValueError('unknown fact kind')
            conn.execute('INSERT INTO fact_receipts_v1 VALUES (?,?,?)',(source,rev,h))
        return {'source_id':source,'ack':previous}
