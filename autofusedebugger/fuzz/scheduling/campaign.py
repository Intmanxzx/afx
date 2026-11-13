"""Bounded campaign feeder; no preallocation of an infinite task queue."""
import json
import secrets
import uuid
from dataclasses import asdict
from ..generation.sampling import SamplingConfig
from ..contracts import canonical_json


def initialize(ledger, configs, evaluation, count, devices=None, exclude_ops=None, sampling=None, ops=None):
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError('campaign count must be nonnegative or unlimited')
    with ledger._connection() as conn:
        if not configs or len(configs) > conn.execute('SELECT max_active FROM queue_config').fetchone()[0]:
            raise ValueError('campaign backend matrix exceeds task capacity')
        conn.execute('CREATE TABLE IF NOT EXISTS campaign (id INTEGER PRIMARY KEY, payload TEXT NOT NULL, remaining INTEGER, stopped INTEGER NOT NULL)')
        conn.execute('INSERT OR REPLACE INTO campaign VALUES (1,?,?,0)',
                     (canonical_json({'configs':[asdict(c) for c in configs],'evaluation':asdict(evaluation),'devices':devices or [],'next_device':0,'exclude_ops':exclude_ops or [],'ops':ops or [],'sampling':sampling if sampling is not None else asdict(SamplingConfig())}),count))


def stop(ledger):
    with ledger._connection() as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='campaign'").fetchone():
            conn.execute('UPDATE campaign SET stopped=1')
        conn.execute("UPDATE tasks SET state=CASE WHEN state='running' THEN 'stop_requested' ELSE 'cancelled' END,reason='user_stop',revision=revision+1 WHERE state IN ('pending','running','interrupted')")


def refill(ledger, target_pending=1):
    with ledger._connection() as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='campaign'").fetchone():return False
        row=conn.execute('SELECT payload,remaining,stopped FROM campaign WHERE id=1').fetchone()
        if not row or row[2] or row[1]==0:return False
        if target_pending==1 and conn.execute("SELECT 1 FROM tasks WHERE state IN ('pending','running','stop_requested')").fetchone():return False
        config=json.loads(row[0])
        pending=[json.loads(r[0]) for r in conn.execute("SELECT execution FROM tasks WHERE state='pending'")]
        # Resource-aware refill, still bounded independently of total history.
        # A backlog for one resource must not starve the other class.
        required={c['device']=='cpu' for c in config['configs']}
        covered=all(sum((p['device']=='cpu')==cpu for p in pending)>=target_pending for cpu in required)
        prefetch_limit=target_pending * max(1,len(config['configs'])) * max(1,len(config.get('devices',[])))
        if covered or len(pending)>=prefetch_limit:return False
        entropy=secrets.randbelow(36**9)
        dynamic=not all(c.get('static_only') for c in config['configs']) and bool(entropy & 1)
        if 'sampling' in config:
            from ..generation.sampling import SamplingConfig
            seed = str(SamplingConfig(**config['sampling']).seed(entropy % 2**32, dynamic))
        else:
            # An already persisted pre-upgrade campaign retains C/D semantics.
            seed=str((3 if dynamic else 2)*2**128+entropy)
        active = conn.execute("SELECT count(*) FROM tasks WHERE state NOT IN ('completed','cancelled')").fetchone()[0]
        capacity = conn.execute('SELECT max_active FROM queue_config').fetchone()[0]
        if active + len(config['configs']) > capacity:
            return False
        from .placement import initialize as initialize_placement
        initialize_placement(conn)
        identities = []
        for template in config['configs']:
            execution=dict(template,static_only=not dynamic)
            if execution['device']!='cpu' and config.get('devices'):
                execution['device']=config['devices'][config.get('next_device',0)%len(config['devices'])]
            identity = str(uuid.uuid4())
            identities.append(identity)
            if execution['device'] != 'cpu' and config.get('devices'):
                conn.execute('INSERT INTO task_devices VALUES (?,?)', (identity, canonical_json(config['devices'])))
            conn.execute('INSERT INTO tasks(id,seed,execution,evaluation,state,revision,reason) VALUES (?,?,?,?,?,0,?)',
                         (identity,seed,canonical_json(execution),canonical_json(config['evaluation']),'pending',''))
        from .placement import admit
        admit(conn, identities)
        config['next_device']=config.get('next_device',0)+1
        conn.execute('UPDATE campaign SET payload=? WHERE id=1',(canonical_json(config),))
        if row[1] is not None:conn.execute('UPDATE campaign SET remaining=remaining-1 WHERE id=1')
        return True
