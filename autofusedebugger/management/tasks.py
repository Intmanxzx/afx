"""Managed task admission; execution remains in the existing staged scheduler."""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from .artifacts import atomic
from .config import canonical, fingerprint, quantity, validate
from .environments import resolve


def prepare(store, spec):
    spec=validate(spec)
    if spec['kind']!='fuzz_task': raise ValueError('task required')
    manifest=resolve(store,spec['environment'])
    from .artifacts import trusted
    local=trusted(store.home).get('resource_limits',{})
    policy=store.get('effective_policy',validate(dict(schema_version=1,kind='client_policy')))
    limits=policy['resources']; requested=spec['execution']['resources']
    if any(backend!='cpu' for backend in spec['execution']['backends']) and not set(spec['execution']['devices']).issubset(local.get('allowed_devices',[])):
        raise PermissionError('NPU execution requires explicit local device authorization')
    if not set(spec['execution']['devices']).issubset(limits['allowed_devices']):
        raise PermissionError('task requests devices outside local policy')
    for name, cap in [('workers','max_workers'),('compile_threads','max_compile_threads')]:
        if requested.get(name,1)>limits[cap]: raise ValueError(name+' exceeds device limit')
    if requested.get('memory_budget') and limits.get('memory_budget') and quantity(requested['memory_budget'])>quantity(limits['memory_budget']):
        raise ValueError('task memory exceeds device limit')
    if spec['results']['upload'] and not store.get('credential'):
        raise ValueError('result upload requires a paired device')
    if requested['policy']=='auto':
        from ..fuzz.resources import adaptive_defaults
        defaults=adaptive_defaults(store.home)
        requested.update(workers=min(defaults['cpu_workers'],limits['max_workers'],local.get('max_workers',1)),
                         compile_threads=min(defaults['compile_threads'],limits['max_compile_threads'],local.get('max_compile_threads',1)))
    for name, cap in [('workers','max_workers'),('compile_threads','max_compile_threads')]:
        if requested.get(name,1)>local.get(cap,1): raise PermissionError('task exceeds local resource authorization')
    spec['environment']=manifest['id']
    return dict(spec=spec,environment=manifest,policy=policy,hash=fingerprint(spec))


def start(store,spec,operation_id,processes):
    prepared=prepare(store,spec)
    from ..fuzz.scheduling.hosting import status
    from ..fuzz.scheduling.ledger import TaskLedger
    legacy=store.home/'tasks/tasks.sqlite'
    if legacy.exists() and status(TaskLedger(legacy)).get('ownership')=='owned_by_supervisor_or_worker':
        raise ValueError('legacy fuzz/retest supervisor owns the shared device')
    with store.connect() as db:
        if db.execute("SELECT 1 FROM runs WHERE state IN ('running','paused','pausing','draining','preparing','cancelling','needs_attention')").fetchone():
            raise ValueError('another managed campaign owns this device')
    import fcntl
    ownership=(store.home/'managed-task.lock').open('a')
    try: fcntl.flock(ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        ownership.close();raise ValueError('a previous managed task still owns this device') from None
    identity=str(uuid.uuid4()); directory=store.home/'runs'/identity;directory.mkdir(parents=True,mode=0o700)
    prepared['run_id']=identity;prepared['directory']=str(directory)
    atomic(directory/'spec.json',prepared)
    atomic(directory/'control.json',dict(action='run'))
    manifest=prepared['environment'];env=os.environ.copy();env.update(manifest['env'])
    env.pop('PYTHONPATH',None)
    try:
        child=subprocess.Popen([manifest['python'],'-m','autofusedebugger.management.task_worker',str(directory/'spec.json'),str(ownership.fileno())],
                               env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,pass_fds=(ownership.fileno(),))
    finally: ownership.close()
    processes[identity]=child
    with store.connect() as db:
        db.execute('INSERT INTO runs VALUES (?,?,?,?,?)',(identity,canonical(prepared),'preparing',child.pid,None))
    return dict(run_id=identity,state='preparing',config_hash=prepared['hash'])


def action(store,identity,action,processes):
    if action not in {'pause','resume','stop','force-stop'}: raise ValueError('invalid task action')
    with store.connect() as db:
        row=db.execute('SELECT * FROM runs WHERE id=?',(identity,)).fetchone()
        if not row: raise LookupError('task not found')
        if identity not in processes:
            if row['state']!='needs_attention' or action not in {'stop','force-stop'}:
                raise ValueError('task ownership not held; only cooperative stop of interrupted task is permitted')
            # Never signal a recovered PID. The original worker reads its private control file.
            if not (store.home/'runs'/identity/'spec.json').is_file():
                raise ValueError('missing private task manifest')
        state={'pause':'pausing','resume':'running','stop':'draining','force-stop':'cancelling'}[action]
        directory=store.home/'runs'/identity
        atomic(directory/'control.json',dict(action={'resume':'run','force-stop':'force-stop'}.get(action,action)))
        db.execute('UPDATE runs SET state=? WHERE id=?',(state,identity))
    # Force-stop is handled inside the owned scheduler, including its separately grouped workers.
    return dict(run_id=identity,state=state)


def refresh(store,processes):
    import fcntl
    with (store.home/'managed-task.lock').open('a') as ownership:
        try:
            fcntl.flock(ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:
            with store.connect() as db:
                rows=db.execute("SELECT id FROM runs WHERE state IN ('needs_attention','draining','cancelling')").fetchall()
                for row in rows:
                    if row['id'] not in processes:
                        db.execute("UPDATE runs SET state='interrupted' WHERE id=?",(row['id'],))
    for identity,child in list(processes.items()):
        directory=store.home/'runs'/identity
        statefile=directory/'state.json'
        state=json.loads(statefile.read_text()) if statefile.exists() else dict(state='preparing')
        code=child.poll()
        if code is not None:
            requested=json.loads((directory/'control.json').read_text()).get('action')
            terminal='cancelled' if requested=='force-stop' else 'stopped' if requested=='stop' else 'completed' if code==0 else 'failed'
            state=dict(state=terminal,exit_code=code,detail=state)
            processes.pop(identity)
        if (directory/'spec.json').exists():
            with store.connect() as db:db.execute('UPDATE runs SET spec=? WHERE id=?',((directory/'spec.json').read_text(),identity))
        with store.connect() as db:
            db.execute('UPDATE runs SET state=?,result=? WHERE id=?',(state['state'],canonical(state),identity))


def reconfigure(store, identity, spec, processes):
    if identity not in processes: raise ValueError('task is not owned by this daemon')
    with store.connect() as db:
        row=db.execute('SELECT spec FROM runs WHERE id=?',(identity,)).fetchone()
    if not row: raise LookupError('task not found')
    current=json.loads(row[0]);prepared=prepare(store,spec)
    if current['spec'].get('termination',{}).get('mode')!='continuous' or prepared['spec'].get('termination',{}).get('mode')!='continuous':
        raise ValueError('live reconfiguration requires a continuous campaign')
    if current['environment']['id']!=prepared['environment']['id']:
        raise ValueError('environment changes require stopping and submitting a new task')
    directory=store.home/'runs'/identity
    if (directory/'next-spec.json').exists():raise ValueError('a configuration switch is already pending')
    prepared.update(run_id=identity,directory=str(directory),epoch=current.get('epoch',0)+1)
    atomic(directory/'next-spec.json',prepared)
    atomic(directory/'control.json',dict(action='reconfigure'))
    return dict(run_id=identity,state='waiting_safe_point',config_hash=prepared['hash'],epoch=prepared['epoch'])
