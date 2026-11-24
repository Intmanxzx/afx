"""Owner-scoped durable control plane, independent of execution dependencies."""
from contextlib import contextmanager
import hashlib
import json
import secrets
import sqlite3
import time
import uuid
from autofusex.contracts.configuration import canonical, fingerprint
from autofusex.client.configuration import validate


class Conflict(ValueError):
    pass


def digest(secret):
    if not isinstance(secret,str) or not 1<=len(secret)<=256:raise ValueError('invalid secret format')
    return hashlib.sha256(secret.encode()).hexdigest()


class ControlStore:
    def __init__(self, path):
        self.path = path
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS managed_devices(id TEXT PRIMARY KEY,user_id TEXT NOT NULL,
              token_hash TEXT UNIQUE NOT NULL,name TEXT NOT NULL,active INTEGER NOT NULL DEFAULT 1,
              seen REAL NOT NULL DEFAULT 0,status TEXT NOT NULL DEFAULT '{}',revision INTEGER NOT NULL DEFAULT 0,
              config TEXT NOT NULL DEFAULT '{}');
            CREATE TABLE IF NOT EXISTS managed_pairings(id TEXT PRIMARY KEY,secret_hash TEXT NOT NULL,
              code TEXT UNIQUE NOT NULL,name TEXT NOT NULL,expires REAL NOT NULL,user_id TEXT,
              device_id TEXT,token TEXT);
            CREATE TABLE IF NOT EXISTS managed_operations(id TEXT PRIMARY KEY,device_id TEXT NOT NULL,
              request_key TEXT NOT NULL,hash TEXT NOT NULL,kind TEXT NOT NULL,payload TEXT NOT NULL,
              state TEXT NOT NULL,generation INTEGER NOT NULL DEFAULT 0,expires REAL NOT NULL,
              result TEXT,updated REAL NOT NULL,UNIQUE(device_id,request_key));
            CREATE TABLE IF NOT EXISTS managed_sources(source_id TEXT PRIMARY KEY, device_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS managed_templates(user_id TEXT NOT NULL,name TEXT NOT NULL,revision INTEGER NOT NULL,spec TEXT NOT NULL,PRIMARY KEY(user_id,name));
            CREATE TABLE IF NOT EXISTS managed_audit(seq INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id TEXT NOT NULL,device_id TEXT NOT NULL,action TEXT NOT NULL,details TEXT NOT NULL,time REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS managed_config_history(device_id TEXT NOT NULL,revision INTEGER NOT NULL,
              config TEXT NOT NULL,PRIMARY KEY(device_id,revision));
            ''')

        from autofusex.client.agent_api import initialize
        with self.connect() as db:
            initialize(db)

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=10);db.row_factory=sqlite3.Row
        try:
            with db: yield db
        finally: db.close()

    def pairing(self, name):
        if not isinstance(name,str) or not 1 <= len(name) <= 80: raise ValueError('invalid device name')
        identity=str(uuid.uuid4()); secret=secrets.token_urlsafe(32); code=secrets.token_hex(5).upper()
        with self.connect() as db:
            db.execute('DELETE FROM managed_pairings WHERE expires<?',(time.time(),))
            if db.execute('SELECT count(*) FROM managed_pairings').fetchone()[0] >= 1000: raise ValueError('pairing capacity reached')
            db.execute('INSERT INTO managed_pairings(id,secret_hash,code,name,expires) VALUES (?,?,?,?,?)',
                       (identity,digest(secret),code,name,time.time()+600))
        return dict(id=identity,secret=secret,code=code,expires_in=600)

    def approve(self, user, code):
        if not isinstance(user, str) or not isinstance(code, str) or len(code) != 10 or any(c not in '0123456789ABCDEF' for c in code):
            raise ValueError('invalid pairing code')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM managed_pairings WHERE code=? AND expires>?',(code,time.time())).fetchone()
            if not row or row['user_id']: raise ValueError('pairing unavailable')
            identity=str(uuid.uuid4())
            db.execute('UPDATE managed_pairings SET user_id=?,device_id=? WHERE id=?',
                       (user,identity,row['id']))
            self.audit(db,user,identity,'paired',{})
        return dict(device_id=identity)

    def exchange(self, identity, secret):
        try:
            uuid.UUID(identity)
        except (ValueError, TypeError, AttributeError):
            raise PermissionError('invalid pairing') from None
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM managed_pairings WHERE id=? AND expires>?',(identity,time.time())).fetchone()
            if not row or not secrets.compare_digest(row['secret_hash'],digest(secret)): raise PermissionError('invalid pairing')
            if not row['user_id']: return dict(state='pending')
            token=digest('managed-device-v1:'+identity+':'+secret)
            db.execute('UPDATE managed_pairings SET expires=expires WHERE id=?',(identity,))
            db.execute('INSERT OR IGNORE INTO managed_devices(id,user_id,token_hash,name) VALUES (?,?,?,?)',
                       (row['device_id'],row['user_id'],digest(token),row['name']))
            return dict(state='paired',device_id=row['device_id'],token=token)

    def authenticate(self, token):
        with self.connect() as db:
            row=db.execute('SELECT * FROM managed_devices WHERE token_hash=? AND active=1',(digest(token),)).fetchone()
            if not row: raise PermissionError('invalid device credential')
            # User deactivation also revokes control, independently of per-device state.
            user=db.execute('SELECT active FROM users WHERE id=?',(row['user_id'],)).fetchone()
            if not user or not user[0]: raise PermissionError('inactive owner')
            return dict(row)

    @staticmethod
    def owner(db,user,device):
        row=db.execute('SELECT * FROM managed_devices WHERE id=? AND user_id=? AND active=1',(device,user)).fetchone()
        if not row: raise PermissionError('device unavailable')
        return row

    @staticmethod
    def audit(db,user,device,action,details):
        db.execute('INSERT INTO managed_audit(user_id,device_id,action,details,time) VALUES (?,?,?,?,?)',
                   (user,device,action,canonical(details),time.time()))

    def devices(self,user):
        with self.connect() as db:
            return [dict(id=r['id'],name=r['name'],seen=r['seen'],status=json.loads(r['status']),
                         revision=r['revision'],config=json.loads(r['config']),active=bool(r['active']))
                    for r in db.execute('SELECT * FROM managed_devices WHERE user_id=? ORDER BY name',(user,))]

    def configure(self,user,device,config,base):
        config=validate(config)
        if config['kind']!='client_policy': raise ValueError('client policy required')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE'); row=self.owner(db,user,device)
            if type(base) is not int or row['revision']!=base: raise Conflict('configuration revision changed')
            revision=base+1
            db.execute('UPDATE managed_devices SET revision=?,config=? WHERE id=?',(revision,canonical(config),device))
            db.execute('INSERT INTO managed_config_history VALUES (?,?,?)',(device,revision,canonical(config)))
            self.audit(db,user,device,'config',dict(revision=revision,hash=fingerprint(config)))
        return dict(revision=revision,state='published')

    def update_client(self, user, device, version, request_key, index_url=None):
        """Publish an exact PyPI client version to one managed device."""
        payload = dict(version=version, package='autofusex')
        if index_url:
            payload['index_url'] = index_url
        return self.submit(user, device, 'client.update', payload, request_key)

    def submit(self,user,device,kind,payload,key):
        from autofusex.client.operations import validate_operation
        payload=validate_operation(kind,payload)
        if not isinstance(key,str) or not 1<=len(key)<=128: raise ValueError('request key required')
        hashed=fingerprint(dict(kind=kind,payload=payload))
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE');self.owner(db,user,device)
            old=db.execute('SELECT * FROM managed_operations WHERE device_id=? AND request_key=?',(device,key)).fetchone()
            if old:
                if old['hash']!=hashed: raise Conflict('request key conflict')
                return dict(id=old['id'],state=old['state'])
            if db.execute("SELECT count(*) FROM managed_operations WHERE device_id=? AND state IN ('queued','delivered','running')",(device,)).fetchone()[0]>=100:
                raise ValueError('device operation queue full')
            identity=str(uuid.uuid4());now=time.time()
            db.execute('INSERT INTO managed_operations VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                       (identity,device,key,hashed,kind,canonical(payload),'queued',0,now+86400,None,now))
            self.audit(db,user,device,'submit',dict(id=identity,kind=kind,hash=hashed))
            return dict(id=identity,state='queued')

    def poll(self,device,status):
        if not isinstance(status,dict) or len(canonical(status))>16384: raise ValueError('invalid status')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            from autofusex.client.agent_api import observe
            observe(db, device['id'], status)
            db.execute('UPDATE managed_devices SET seen=?,status=? WHERE id=?',(time.time(),canonical(status),device['id']))
            db.execute("UPDATE managed_operations SET state='expired' WHERE device_id=? AND state='queued' AND expires<?",(device['id'],time.time()))
            # Delivered operations keep one generation: no auto retry of uncertain external side effects.
            rows=db.execute("SELECT * FROM managed_operations WHERE device_id=? AND state IN ('queued','delivered') ORDER BY updated LIMIT 8",(device['id'],)).fetchall()
            operations=[]
            for row in rows:
                generation=row['generation'] or 1
                db.execute("UPDATE managed_operations SET state='delivered',generation=? WHERE id=?",(generation,row['id']))
                operations.append(dict(id=row['id'],kind=row['kind'],payload=json.loads(row['payload']),
                                       request_key=row['request_key'],generation=generation,expires=row['expires']))
            row=db.execute('SELECT revision,config FROM managed_devices WHERE id=?',(device['id'],)).fetchone()
            return dict(operations=operations,revision=row['revision'],config=json.loads(row['config']))

    def ack(self,device,identity,generation,state,result):
        if state not in {'running','succeeded','failed','cancelled','needs_attention'}: raise ValueError('invalid operation state')
        if len(canonical(result))>32768: raise ValueError('result too large')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM managed_operations WHERE id=? AND device_id=?',(identity,device['id'])).fetchone()
            if not row or row['generation']!=generation: raise Conflict('stale operation generation')
            if row['state'] in {'succeeded','failed','cancelled','needs_attention'}:
                if row['state']!=state or row['result']!=canonical(result): raise Conflict('terminal operation is immutable')
            else:
                db.execute('UPDATE managed_operations SET state=?,result=?,updated=? WHERE id=?',
                           (state,canonical(result),time.time(),identity))
        return dict(accepted=True)

    def operations(self,user,device):
        with self.connect() as db:
            self.owner(db,user,device)
            return [dict(id=r['id'],kind=r['kind'],state=r['state'],updated=r['updated'],result=json.loads(r['result']) if r['result'] else None)
                    for r in db.execute('SELECT * FROM managed_operations WHERE device_id=? ORDER BY updated DESC LIMIT 100',(device,))]

# ControlAPI (the /api/manage/* HTTP dispatch for this control plane) was removed with the
# old central server: it only ever ran mounted inside autofusex/fuzz/server/central/server.py,
# which supplied its session/CSRF/rate-limit collaborator (the removed `existing` argument) and
# imported sessions.identify, maintenance.MaintenanceBlocked/reject and repository.authenticate
# from autofusex/fuzz/server/central/ -- all deleted, with no equivalent in autofusex/platform/.
# ControlStore above (the device pairing/config/operations data model) is self-contained and kept
# for whoever next decides to retire or integrate remote device management into platform v3; see
# docs/fuzz/fuzz_design.md and fuzz_usage.md.
