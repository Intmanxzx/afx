from contextlib import closing
from datetime import datetime
import hashlib
import secrets
import sqlite3
import uuid

from ...client.storage.repository import _connect
from ...client.storage.records import canonical, digest, normalize
from ...client.storage.schema import TABLE_COLUMNS


def initialize(path):
    with closing(_connect(path,writable=True)) as conn,conn:
        from ..reporting.index import initialize as initialize_reporting
        initialize_reporting(conn)
        conn.execute('CREATE TABLE IF NOT EXISTS managed_sources(source_id TEXT PRIMARY KEY, device_id TEXT NOT NULL)')
        conn.execute('CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, display_name TEXT NOT NULL UNIQUE, key_hash TEXT NOT NULL UNIQUE, active INTEGER NOT NULL DEFAULT 1)')
        conn.execute('CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), ack INTEGER NOT NULL DEFAULT 0)')
        conn.execute('CREATE TABLE IF NOT EXISTS receipts (source_id TEXT NOT NULL, revision INTEGER NOT NULL, payload_hash TEXT NOT NULL, PRIMARY KEY(source_id,revision))')
        conn.execute('CREATE TABLE IF NOT EXISTS deleted_attempts (id TEXT PRIMARY KEY, result_hash TEXT NOT NULL, user_id TEXT NOT NULL)')
        conn.execute('CREATE TABLE IF NOT EXISTS owners (attempt_id TEXT PRIMARY KEY REFERENCES attempts(id), user_id TEXT NOT NULL REFERENCES users(id))')
        from .issue_groups import initialize as initialize_issues
        initialize_issues(conn)
        from ..reporting.materialized import installed, refresh
        if not installed(conn):
            refresh(conn)


def create_user(path, display_name, key_file=None):
    from pathlib import Path
    if not isinstance(display_name,str) or not display_name.strip() or len(display_name)>80 or any(ord(c)<32 for c in display_name):
        raise ValueError('invalid display name')
    initialize(path)
    identity=str(uuid.uuid4()); key=secrets.token_urlsafe(40)
    if key_file is None:
        with closing(_connect(path,writable=True)) as conn,conn:
            conn.execute('INSERT INTO users(id,display_name,key_hash) VALUES (?,?,?)',
                         (identity,display_name,hashlib.sha256(key.encode()).hexdigest()))
        # Offline admin explicitly requests a credential: print exactly once.
        return {'user_id':identity,'display_name':display_name,'token':key}
    target=Path(key_file).expanduser().resolve()
    # Never print or store plaintext key in database. Offline admin distributes file.
    import os
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'w') as out:
            out.write(key+'\n');out.flush();os.fsync(out.fileno())
        with closing(_connect(path,writable=True)) as conn,conn:
            conn.execute('INSERT INTO users(id,display_name,key_hash) VALUES (?,?,?)',(identity,display_name,hashlib.sha256(key.encode()).hexdigest()))
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return {'user_id':identity,'display_name':display_name,'key_file':str(target)}


def authenticate(conn, key):
    row=conn.execute('SELECT id FROM users WHERE key_hash=? AND active=1',(hashlib.sha256(key.encode()).hexdigest(),)).fetchone()
    if row is None:
        raise PermissionError('invalid upload credential')
    return row[0]



def authenticate_upload(conn, key, source=None):
    """Device credentials are accepted only by result/fact upload, never browser login."""
    try:
        return authenticate(conn, key)
    except PermissionError:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='managed_devices'").fetchone():
            raise
    row = conn.execute('SELECT d.id,d.user_id FROM managed_devices d JOIN users u ON u.id=d.user_id '
                       'WHERE d.token_hash=? AND d.active=1 AND u.active=1',
                       (hashlib.sha256(key.encode()).hexdigest(),)).fetchone()
    if not row:
        raise PermissionError('invalid device upload credential')
    if source is not None:
        source = str(uuid.UUID(source))
        owner = conn.execute('SELECT device_id FROM managed_sources WHERE source_id=?',(source,)).fetchone()
        if owner and owner[0] != row[0]:
            raise PermissionError('source belongs to another device')
        if not owner:
            if conn.execute('SELECT 1 FROM sources WHERE id=?',(source,)).fetchone():
                raise PermissionError('cannot adopt a pre-existing result source')
            conn.execute('INSERT INTO managed_sources VALUES (?,?)',(source,row[0]))
    return row[1]

def ingest(path, key, payload):
    if not isinstance(payload,dict) or payload.get('protocol')!=1:
        raise ValueError('unsupported sync protocol')
    source=str(uuid.UUID(payload['source_id']))
    items=payload['items']; after=payload['after']
    if type(after) is not int or after<0 or not isinstance(items,list) or not 1<=len(items)<=100:
        raise ValueError('invalid sync batch')
    with closing(_connect(path,writable=True)) as conn,conn:
        conn.execute('BEGIN IMMEDIATE')
        user=authenticate_upload(conn,key,source)
        conn.execute('INSERT OR IGNORE INTO sources(id,user_id) VALUES (?,?)',(source,user))
        owner,ack=conn.execute('SELECT user_id,ack FROM sources WHERE id=?',(source,)).fetchone()
        if owner!=user:
            raise PermissionError('source belongs to another identity')
        if after>ack:
            raise ValueError('client ACK ahead of server; restore matching server backup before resuming')
        previous=after
        refreshed_seeds=set()
        for item in items:
            revision=item['revision']
            if type(revision) is not int or revision<=previous:
                raise ValueError('revisions must increase')
            previous=revision
            identity=str(uuid.UUID(item['attempt_id']))
            created=item['created_at']
            if not isinstance(created,str) or len(created)>64 or datetime.fromisoformat(created).tzinfo is None:
                raise ValueError('invalid creation time')
            raw=item['record']; contexts=raw['contexts']
            record=normalize(dict(raw['attempt'],execution=contexts['execution'],evaluation=contexts['evaluation'],shape_results=raw['shapes']),contexts['environment'])
            if canonical(record)!=canonical(raw):
                raise ValueError('record violates lightweight whitelist')
            receipt=digest(item)
            existing=conn.execute('SELECT payload_hash FROM receipts WHERE source_id=? AND revision=?',(source,revision)).fetchone()
            if existing:
                if existing[0]!=receipt:
                    raise ValueError('revision conflict')
                continue
            if revision<=ack:
                raise ValueError('old revision without receipt')
            deleted=conn.execute('SELECT result_hash,user_id FROM deleted_attempts WHERE id=?',(identity,)).fetchone()
            if deleted:
                if deleted!=(digest(record),user):
                    raise ValueError('deleted attempt ownership/content conflict')
                conn.execute('INSERT INTO receipts VALUES (?,?,?)',(source,revision,receipt))
                ack=revision
                continue
            existing=conn.execute('SELECT result_hash,created_at FROM attempts WHERE id=?',(identity,)).fetchone()
            if existing:
                owned=conn.execute('SELECT user_id FROM owners WHERE attempt_id=?',(identity,)).fetchone()
                if existing!=(digest(record),created) or owned!=(user,):
                    raise ValueError('attempt ownership/content conflict')
            else:
                ids={}
                for kind,value in contexts.items():
                    ids[kind]=digest({'kind':kind,'payload':value})
                    conn.execute('INSERT OR IGNORE INTO contexts VALUES (?,?,?)',(ids[kind],kind,canonical(value)))
                row=dict(record['attempt'],id=identity,created_at=created,result_hash=digest(record),**{k+'_id':v for k,v in ids.items()})
                conn.execute('INSERT INTO attempts VALUES ('+','.join('?' for _ in TABLE_COLUMNS['attempts'])+')',tuple(row[k] for k in TABLE_COLUMNS['attempts']))
                for shape in record['shapes']:
                    row=dict(shape,attempt_id=identity)
                    conn.execute('INSERT INTO shapes VALUES ('+','.join('?' for _ in TABLE_COLUMNS['shapes'])+')',tuple(row[k] for k in TABLE_COLUMNS['shapes']))
                conn.execute('INSERT INTO owners VALUES (?,?)',(identity,user))
            # Store evidence and ACK even an invalid replay, but do not let a
            # known fingerprint/config mismatch become a current test result.
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='retest_lease_history_v1'").fetchone():
                import json
                lease=contexts['execution'].get('batch_id')
                bound=conn.execute('SELECT t.spec FROM retest_lease_history_v1 h JOIN retest_tasks_v1 t ON t.id=h.task_id WHERE h.lease_id=?',(lease,)).fetchone()
                if bound:
                    spec=json.loads(bound[0]);a=record['attempt'];ex=contexts['execution']
                    if (a['seed']!=spec['seed'] or a['workload_id']!=spec['workload_id'] or
                        a['oracle_id']!=spec['oracle_id'] or a['backend'] not in {'cpu',spec['backend']} or
                        ex.get('static_only')!=spec['execution']['static_only'] or contexts['evaluation']!=spec['evaluation']):
                        conn.execute('INSERT OR IGNORE INTO invalid_replays_v1 VALUES (?,?)',(identity,'reproduction_mismatch'))
            from ..reporting.index import index_attempt
            display=conn.execute('SELECT display_name FROM users WHERE id=?',(user,)).fetchone()[0]
            index_attempt(conn,identity,source,display,user)
            refreshed_seeds.add(record['attempt']['seed'])
            conn.execute('INSERT INTO receipts VALUES (?,?,?)',(source,revision,receipt))
            ack=revision
        from ..reporting.materialized import refresh_if_installed
        refresh_if_installed(conn)
        from .issue_groups import refresh_index
        refresh_index(conn, refreshed_seeds)
        conn.execute('UPDATE sources SET ack=? WHERE id=?',(ack,source))
        return {'source_id':source,'ack':ack,'accepted':len(items)}


def rotate_key(path, user_id, key_file=None):
    """Offline administrator key rotation; identity and all data ownership persist."""
    from pathlib import Path
    import os
    identity = str(uuid.UUID(user_id))
    if key_file is None:
        key = secrets.token_urlsafe(40)
        with closing(_connect(path, writable=True)) as conn, conn:
            cursor = conn.execute('UPDATE users SET key_hash=? WHERE id=? AND active=1',
                                  (hashlib.sha256(key.encode()).hexdigest(), identity))
            if cursor.rowcount != 1:
                raise ValueError('active administrator-managed user not found')
        return {'user_id': identity, 'token': key, 'rotated': True}
    target = Path(key_file).expanduser().resolve()
    key = secrets.token_urlsafe(40)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w') as out:
            out.write(key+'\n'); out.flush(); os.fsync(out.fileno())
        with closing(_connect(path, writable=True)) as conn, conn:
            cursor = conn.execute('UPDATE users SET key_hash=? WHERE id=? AND active=1',
                                  (hashlib.sha256(key.encode()).hexdigest(), identity))
            if cursor.rowcount != 1:
                raise ValueError('active administrator-managed user not found')
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return {'user_id': identity, 'key_file': str(target), 'rotated': True}
