"""Authenticated live observations, separate from immutable results and ACKs.

Server receipt time drives presence. Old/retried samples do not refresh a run.
Rates use only adjacent fresh cumulative counters, never a reconnect catch-up.
"""
from contextlib import closing
import json
import math
import sqlite3
from pathlib import Path
import time
import uuid

from ..storage.repository import _connect
from .repository import authenticate


def initialize(conn):
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS runtime_hosts_v1 (
      user_id TEXT, machine_id TEXT, source_id TEXT, name TEXT, seen REAL,
      PRIMARY KEY(user_id,machine_id,source_id));
    CREATE TABLE IF NOT EXISTS runtime_machine_names_v1 (
      user_id TEXT NOT NULL, machine_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
      PRIMARY KEY(user_id,machine_id), UNIQUE(user_id,ordinal));
    CREATE TABLE IF NOT EXISTS runtime_runs_v1 (
      run_id TEXT PRIMARY KEY, user_id TEXT, machine_id TEXT, source_id TEXT,
      sequence INTEGER, seen REAL, payload TEXT);
    CREATE TABLE IF NOT EXISTS runtime_cards_v1 (
      user_id TEXT, machine_id TEXT, uuid TEXT, model TEXT, logical TEXT, seen REAL,
      PRIMARY KEY(user_id,machine_id,uuid));
    CREATE TABLE IF NOT EXISTS runtime_rates_v1 (
      run_id TEXT, sequence INTEGER, time REAL, span REAL, completed INTEGER,
      success INTEGER, case_errors INTEGER, PRIMARY KEY(run_id,sequence));
    CREATE INDEX IF NOT EXISTS runtime_rates_time ON runtime_rates_v1(time);
    ''')
    # Backfill old hosts once, deterministically. Heartbeat names/timestamps can
    # change; the persisted ordinal must never be recomputed or reused.
    conn.execute('BEGIN IMMEDIATE')
    for user, machine in conn.execute(
            'SELECT user_id,machine_id FROM runtime_hosts_v1 '
            'GROUP BY user_id,machine_id ORDER BY user_id,machine_id').fetchall():
        ensure_machine_name(conn, user, machine)


def ensure_machine_name(conn, user, machine):
    # Called inside the writer transaction, serializing concurrent registration.
    conn.execute(
        'INSERT INTO runtime_machine_names_v1(user_id,machine_id,ordinal) '
        'SELECT ?,?,coalesce(max(ordinal)+1,0) FROM runtime_machine_names_v1 '
        'WHERE user_id=? ON CONFLICT(user_id,machine_id) DO NOTHING',
        (user, machine, user))


def connect(database):
    # Runtime writes must not wait behind result ingestion or historical rollups.
    path=Path(str(database)+'.runtime.sqlite')
    conn=sqlite3.connect(path,timeout=5)
    conn.execute('PRAGMA journal_mode=WAL')
    return conn


def text(value, limit=128):
    if not isinstance(value, str) or not value or len(value)>limit or any(ord(c)<32 for c in value):
        raise ValueError('invalid runtime text')
    return value


def number(value, maximum=10**15):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=maximum:
        raise ValueError('invalid runtime number')
    return value


def validate(payload):
    if not isinstance(payload,dict) or set(payload)!={'version','machine_id','machine_name','source_id','runs'} or payload['version']!=1:
        raise ValueError('invalid runtime protocol')
    text(payload['machine_id']);text(payload['machine_name'],80);uuid.UUID(payload['source_id'])
    runs=payload['runs']
    if not isinstance(runs,list) or len(runs)>32:raise ValueError('too many runtime sessions')
    ids=set()
    for r in runs:
        required={'run_id','sequence','age','elapsed','state','devices','stages','queued','active_cases','completed','success','case_errors','record_failed','backend','versions'}
        if not isinstance(r,dict) or set(r)!=required:raise ValueError('invalid runtime session')
        uuid.UUID(r['run_id'])
        if r['run_id'] in ids:raise ValueError('duplicate session')
        ids.add(r['run_id'])
        if type(r['sequence']) is not int or r['sequence']<0:raise ValueError('invalid sequence')
        for k in ('age','elapsed','queued','active_cases','completed','success','case_errors','record_failed'):number(r[k])
        if r['success']>r['completed']:raise ValueError('invalid counters')
        if r['state'] not in {'running','idle','error'}:raise ValueError('invalid state')
        if not isinstance(r['backend'],list) or len(r['backend'])>16:raise ValueError('invalid backend')
        for b in r['backend']:text(b,64)
        if not isinstance(r['versions'],dict) or set(r['versions'])-{'package_version','autofuse_binary','cann','torch','torch_npu'}:raise ValueError('invalid versions')
        for v in r['versions'].values():text(v,128)
        if not isinstance(r['devices'],list) or len(r['devices'])>64:raise ValueError('invalid devices')
        seen=set()
        for d in r['devices']:
            if set(d)!={'uuid','model','logical','busy'} or type(d['busy']) is not bool:raise ValueError('invalid device')
            for k in ('uuid','model','logical'):text(d[k])
            if d['uuid'] in seen:raise ValueError('duplicate device')
            seen.add(d['uuid'])
        if not isinstance(r['stages'],dict) or set(r['stages'])-{'generate','reference','first_call_combined','compare','record'}:raise ValueError('invalid stages')
        for v in r['stages'].values():number(v,128)


def ingest(database, key, payload, *, now=None):
    validate(payload);now=time.time() if now is None else now
    with closing(_connect(database)) as auth:
        user=authenticate(auth,key)
        owner=auth.execute('SELECT user_id FROM sources WHERE id=?',(payload['source_id'],)).fetchone()
        if owner and owner[0]!=user:raise PermissionError('source ownership mismatch')
    with closing(connect(database)) as conn,conn:
        conn.execute('BEGIN IMMEDIATE')
        source=payload['source_id'];machine=payload['machine_id']
        owner=conn.execute('SELECT user_id FROM runtime_hosts_v1 WHERE source_id=?',(source,)).fetchone()
        if owner and owner[0]!=user:raise PermissionError('source ownership mismatch')
        ensure_machine_name(conn,user,machine)
        conn.execute('INSERT OR REPLACE INTO runtime_hosts_v1 VALUES (?,?,?,?,?)',(user,machine,source,payload['machine_name'],now))
        for r in payload['runs']:
            old=conn.execute('SELECT user_id,machine_id,source_id,sequence,seen,payload FROM runtime_runs_v1 WHERE run_id=?',(r['run_id'],)).fetchone()
            if old and tuple(old[:3])!=(user,machine,source):raise PermissionError('runtime owner mismatch')
            if old and r['sequence']<=old[3]:continue
            observed=now-r['age']
            for d in r['devices']:
                conn.execute('INSERT INTO runtime_cards_v1 VALUES (?,?,?,?,?,?) ON CONFLICT(user_id,machine_id,uuid) DO UPDATE SET model=excluded.model,logical=excluded.logical,seen=max(seen,excluded.seen)',(user,machine,d['uuid'],d['model'],d['logical'],observed))
            if old:
                previous=json.loads(old[5]);span=r['elapsed']-previous['elapsed']
                deltas=[r[k]-previous[k] for k in ('completed','success','case_errors')]
                if any(n<0 for n in deltas) or span<0:raise ValueError('counter regression')
                # Drop reconnect bursts and stale/spooled snapshots from live rates.
                if 0<span<=30 and 0<observed-old[4]<=30 and r['age']<=20:
                    conn.execute('INSERT INTO runtime_rates_v1 VALUES (?,?,?,?,?,?,?)',(r['run_id'],r['sequence'],observed,span,*deltas))
            conn.execute('INSERT OR REPLACE INTO runtime_runs_v1 VALUES (?,?,?,?,?,?,?)',(r['run_id'],user,machine,source,r['sequence'],observed,json.dumps(r)))
        conn.execute('DELETE FROM runtime_rates_v1 WHERE time<?',(now-86400,))
        conn.execute('DELETE FROM runtime_runs_v1 WHERE seen<?',(now-7*86400,))
    return {'accepted':True,'server_time':now}


def snapshot(database, *, user_id='', machine_id='', model='', backend='', version='', minutes='5', now=None):
    now=time.time() if now is None else now
    minutes=int(minutes)
    if minutes not in {1,5,15,60}:raise ValueError('invalid time range')
    with closing(_connect(database)) as conn:
        users=dict(conn.execute('SELECT id,display_name FROM users WHERE active=1'))
    with closing(connect(database)) as conn:
        conn.execute('BEGIN')  # Names and hosts must come from the same read snapshot.
        ordinals={(u,m):n for u,m,n in conn.execute('SELECT user_id,machine_id,ordinal FROM runtime_machine_names_v1')}
        hosts=conn.execute('SELECT user_id,machine_id,source_id,name,seen FROM runtime_hosts_v1').fetchall()
        runs=conn.execute('SELECT run_id,user_id,machine_id,source_id,seen,payload FROM runtime_runs_v1').fetchall()
        inventory=conn.execute('SELECT user_id,machine_id,uuid,model,logical,seen FROM runtime_cards_v1').fetchall()
        rates=conn.execute('SELECT run_id,time,span,completed,success,case_errors FROM runtime_rates_v1 WHERE time>?',(now-3600,)).fetchall()
    machines={};run_map={};all_models=set();all_backends=set();all_versions=set()
    for identity,user,machine,source,seen,raw in runs:
        r=json.loads(raw);all_models.update(d['model'] for d in r['devices']);all_backends.update(r['backend']);all_versions.update(r['versions'].values())
        if user not in users or (user_id and user!=user_id) or (machine_id and machine!=machine_id):continue
        if model and not any(d['model']==model for d in r['devices']):continue
        if backend and backend not in r['backend']:continue
        if version and version not in r['versions'].values():continue
        run_map[identity]=(user,machine)
        m=machines.setdefault((user,machine),dict(user_id=user,user_name=users[user],machine_id=machine,name=machine[:12],seen=0,cards={},runs=[],queued=0,active_cases=0,completed=0,success=0,case_errors=0,record_failed=0))
        age=max(0,now-seen);fresh=age<=30;live=fresh and r['state']=='running'
        m['runs'].append(dict(run_id=identity,age_seconds=round(age),state=r['state'] if fresh else 'unknown',stages=r['stages'] if live else {},backend=r['backend'],versions=r['versions']))
        if live:m['queued']+=r['queued'];m['active_cases']+=r['active_cases']
        m['record_failed']+=r['record_failed']
        for d in r['devices']:
            if model and d['model']!=model:continue
            card=m['cards'].setdefault(d['uuid'],dict(uuid=d['uuid'],model=d['model'],logical=d['logical'],busy=False,observed=0))
            card['busy']|=bool(live and d['busy']);card['observed']=max(card['observed'],seen)
    for user,machine,source,name,seen in hosts:
        if user not in users or (user_id and user!=user_id) or (machine_id and machine!=machine_id):continue
        key=(user,machine)
        if key not in machines:
            if model or backend or version:continue
            machines[key]=dict(user_id=user,user_name=users[user],machine_id=machine,name=name,seen=0,cards={},runs=[],queued=0,active_cases=0,completed=0,success=0,case_errors=0,record_failed=0)
        m=machines[key]
        if seen>m['seen']:m.update(seen=seen,name=name)
    for user,machine,card,card_model,logical,seen in inventory:
        m=machines.get((user,machine))
        if m is not None and not (backend or version) and (not model or model==card_model):
            m['cards'].setdefault(card,dict(uuid=card,model=card_model,logical=logical,busy=False,observed=0))
    trend={int((now-i*60)//60)*60:{'time':int((now-i*60)//60)*60,'completed':0,'success':0} for i in range(60)}
    for run,t,span,completed,success,errors in rates:
        if run not in run_map:continue
        m=machines[run_map[run]]
        if t>now-minutes*60:
            m['completed']+=completed;m['success']+=success;m['case_errors']+=errors
        bucket=int(t//60)*60
        if bucket in trend:trend[bucket]['completed']+=completed;trend[bucket]['success']+=success
    physical={};online_machines=set();active_users=set()
    for m in machines.values():
        m['display_name']=f"{m['user_name']}-npu{ordinals[(m['user_id'],m['machine_id'])]}"
        m['age_seconds']=round(max(0,now-m.pop('seen')));m['status']='online' if m['age_seconds']<=30 else 'delayed' if m['age_seconds']<=60 else 'offline'
        m['cards']=list(m['cards'].values())
        if m['status']=='online':online_machines.add(m['machine_id'])
        for d in m['cards']:
            d['status']='busy' if d['busy'] else 'idle' if now-d['observed']<=30 else 'unknown';d.pop('observed')
            key=(m['machine_id'],d['uuid']);physical[key]=max(physical.get(key,0),2 if d['busy'] else 1 if d['status']=='idle' else 0)
        if m['active_cases']:active_users.add(m['user_id'])
        m['completed_per_min']=m['completed']/minutes;m['success_per_min']=m['success']/minutes
    values=list(machines.values())
    # A physical machine can belong to multiple users. Keep one filter value
    # per machine ID and include all applicable aliases instead of hostnames.
    options={}
    for (user,machine),ordinal in sorted(ordinals.items(),key=lambda item:(users.get(item[0][0],''),item[1],item[0])):
        if user in users and (not user_id or user==user_id):
            options.setdefault(machine,[]).append(f'{users[user]}-npu{ordinal}')
    return dict(generated_at=now,minutes=minutes,unit='backend_shape_execution',users=[dict(id=k,name=v) for k,v in users.items()],models=sorted(all_models),backends=sorted(all_backends),versions=sorted(all_versions),machines=values,trend=sorted(trend.values(),key=lambda t:t['time']),
                machine_options=[dict(id=m,name=' / '.join(names)) for m,names in options.items()],summary=dict(active_users=len(active_users),registered_machines=len({m['machine_id'] for m in values}),online_machines=len(online_machines),known_cards=len(physical),online_cards=sum(v>0 for v in physical.values()),busy_cards=sum(v==2 for v in physical.values()),active_cases=sum(m['active_cases'] for m in values),queued=sum(m['queued'] for m in values),completed_per_min=sum(m['completed_per_min'] for m in values),success_per_min=sum(m['success_per_min'] for m in values)))
