from contextlib import closing
from datetime import datetime
import hashlib
import secrets
import sqlite3
import uuid

from ..storage.repository import _connect
from ..storage.records import canonical, digest, normalize
from ..storage.schema import TABLE_COLUMNS


def initialize(path):
    with closing(_connect(path,writable=True)) as conn,conn:
        conn.execute('CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, display_name TEXT NOT NULL UNIQUE, key_hash TEXT NOT NULL UNIQUE, active INTEGER NOT NULL DEFAULT 1)')
        conn.execute('CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), ack INTEGER NOT NULL DEFAULT 0)')
        conn.execute('CREATE TABLE IF NOT EXISTS receipts (source_id TEXT NOT NULL, revision INTEGER NOT NULL, payload_hash TEXT NOT NULL, PRIMARY KEY(source_id,revision))')
        conn.execute('CREATE TABLE IF NOT EXISTS owners (attempt_id TEXT PRIMARY KEY REFERENCES attempts(id), user_id TEXT NOT NULL REFERENCES users(id))')


def create_user(path, display_name, key_file):
    from pathlib import Path
    if not isinstance(display_name,str) or not display_name.strip() or len(display_name)>80 or any(ord(c)<32 for c in display_name):
        raise ValueError('invalid display name')
    initialize(path)
    identity=str(uuid.uuid4()); key=secrets.token_urlsafe(40)
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


def ingest(path, key, payload):
    if not isinstance(payload,dict) or payload.get('protocol')!=1:
        raise ValueError('unsupported sync protocol')
    source=str(uuid.UUID(payload['source_id']))
    items=payload['items']; after=payload['after']
    if type(after) is not int or after<0 or not isinstance(items,list) or not 1<=len(items)<=100:
        raise ValueError('invalid sync batch')
    with closing(_connect(path,writable=True)) as conn,conn:
        conn.execute('BEGIN IMMEDIATE')
        user=authenticate(conn,key)
        conn.execute('INSERT OR IGNORE INTO sources(id,user_id) VALUES (?,?)',(source,user))
        owner,ack=conn.execute('SELECT user_id,ack FROM sources WHERE id=?',(source,)).fetchone()
        if owner!=user:
            raise PermissionError('source belongs to another identity')
        if after>ack:
            raise ValueError('client ACK ahead of server; restore matching server backup before resuming')
        previous=after
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
            conn.execute('INSERT INTO receipts VALUES (?,?,?)',(source,revision,receipt))
            ack=revision
        conn.execute('UPDATE sources SET ack=? WHERE id=?',(ack,source))
        return {'source_id':source,'ack':ack,'accepted':len(items)}


def rotate_key(path, user_id, key_file):
    """Offline administrator key rotation; identity and all data ownership persist."""
    from pathlib import Path
    import os
    identity = str(uuid.UUID(user_id))
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
