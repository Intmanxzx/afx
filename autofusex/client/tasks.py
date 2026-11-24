"""Managed task admission; execution remains in the existing staged scheduler."""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from autofusex.client.artifacts import atomic
from autofusex.contracts.configuration import canonical, fingerprint, quantity
from autofusex.client.configuration import validate
from autofusex.client.environments import resolve


def _expand_devices(devices):
    """Resolve the user-facing ``all`` device wildcard to logical NPUs.

    Device names in task and policy files are logical ``npu:<index>`` names;
    physical IDs reported by ``npu-smi`` must not leak into the task ledger.
    """
    values = list(dict.fromkeys(devices or []))
    if "all" not in values:
        return values
    from autofusex.execution.scheduling.devices import select_devices
    discovered = select_devices(None)
    return list(dict.fromkeys([value for value in values if value != "all"] + discovered))


def prepare(store, spec):
    spec=validate(spec)
    if spec['kind'] not in {'fuzz_task', 'imported_task', 'model_task'}: raise ValueError('task required')
    if spec['kind'] != 'fuzz_task':
        from autofusex.workloads.imported.registry import inspect
        inspect(store.home, spec['workload']['case_key'], spec['workload']['revision'])
    manifest=resolve(store,spec['environment'])
    from autofusex.client.artifacts import trusted
    local=trusted(store.home).get('resource_limits',{})
    policy=store.get('effective_policy',validate(dict(schema_version=1,kind='client_policy')))
    limits=policy['resources']; requested=spec['execution']['resources']
    local_devices = local.get('allowed_devices', [])
    if 'all' in local_devices and spec['execution']['devices'] != ['all']:
        from autofusex.client.execution_runtime import facts
        observed = facts(store)
        local_allowed = observed['inventory']['devices'] if observed else _expand_devices(local_devices)
        if local_allowed is None: raise ValueError('execution inventory unavailable')
    else:
        local_allowed = _expand_devices(local_devices)
    policy_allowed = _expand_devices(limits.get('allowed_devices', []))
    if spec['execution']['devices'] == ['all']:
        available = local_allowed or policy_allowed
        if not available:
            raise PermissionError('no authorized NPU devices available')
        spec['execution']['devices'] = list(available)
    if any(backend!='cpu' for backend in spec['execution']['backends']) and not set(spec['execution']['devices']).issubset(local_allowed):
        raise PermissionError('NPU execution requires explicit local device authorization')
    if not set(spec['execution']['devices']).issubset(policy_allowed):
        raise PermissionError('task requests devices outside local policy')
    for name, cap in [('workers','max_workers'),('compile_threads','max_compile_threads')]:
        if requested.get(name,1)>limits[cap]: raise ValueError(name+' exceeds device limit')
    if requested.get('memory_budget') and limits.get('memory_budget') and quantity(requested['memory_budget'])>quantity(limits['memory_budget']):
        raise ValueError('task memory exceeds device limit')
    client_path=store.home/'client.json'
    platform_client=json.loads(client_path.read_text()) if client_path.exists() else {}
    if spec['results']['upload'] and not store.get('credential') and platform_client.get('mode')!='managed':
        raise ValueError('result upload requires a paired device')
    if requested['policy']=='auto':
        from autofusex.execution.resources import npu_resource_plan
        from autofusex.execution.scheduling.budgets import Budget
        device_count=len(spec['execution']['devices']) if any(b!='cpu' for b in spec['execution']['backends']) else 0
        configured_memory=requested.get('memory_budget') or limits.get('memory_budget')
        plan=npu_resource_plan(store.home,device_count,Budget(memory_budget_bytes=int(quantity(configured_memory)) if configured_memory else None))
        if plan['budget_limits'].get('memory_budget_bytes'):
            requested['memory_budget']=str(plan['budget_limits']['memory_budget_bytes'])+'B'
        defaults=plan['selected']
        requested.update(workers=min(defaults['cpu_workers'],limits['max_workers'],local.get('max_workers',1)),
                         compile_threads=min(defaults['compile_threads'],limits['max_compile_threads'],local.get('max_compile_threads',1)))
    for name, cap in [('workers','max_workers'),('compile_threads','max_compile_threads')]:
        if requested.get(name,1)>local.get(cap,1): raise PermissionError('task exceeds local resource authorization')
    spec['environment']=manifest['id']
    return dict(spec=spec,environment=manifest,policy=policy,hash=fingerprint(spec))


def start(store,spec,operation_id,processes,resume_run=None):
    if store.get('client_switch', {}).get('state') in {'draining', 'staging', 'requested', 'stage_failed'}:
        raise ValueError('client upgrade blocks task admission')
    # Runs written by older clients may have auto policy together with the
    # resolved worker/thread values that were added during preparation.  New
    # validation correctly rejects that combination for fresh submissions,
    # but resuming such a durable run must preserve its already selected
    # resources across an upgrade.
    if resume_run:
        spec = json.loads(json.dumps(spec))
        resources = spec.get('execution', {}).get('resources', {})
        if resources.get('policy') == 'auto' and any(k in resources for k in ('workers', 'compile_threads')):
            resources['policy'] = 'manual'
    prepared=prepare(store,spec)
    from autofusex.workloads.fuzz.scheduling.hosting import status
    from autofusex.workloads.fuzz.scheduling.ledger import TaskLedger
    standalone=store.home/'tasks/tasks.sqlite'
    if standalone.exists() and status(TaskLedger(standalone)).get('ownership')=='owned_by_supervisor_or_worker':
        raise ValueError('standalone fuzz/retest supervisor owns the shared device')
    with store.connect() as db:
        if db.execute("SELECT 1 FROM runs WHERE state IN ('running','paused','pausing','draining','preparing','cancelling','needs_attention')").fetchone():
            raise ValueError('another managed campaign owns this device')
    import fcntl
    ownership=(store.home/'managed-task.lock').open('a')
    try: fcntl.flock(ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        ownership.close();raise ValueError('a previous managed task still owns this device') from None
    identity=resume_run or str(uuid.uuid4()); directory=store.home/'runs'/identity;directory.mkdir(parents=True,mode=0o700,exist_ok=bool(resume_run))
    if resume_run:
        old=json.loads((directory/'spec.json').read_text())
        prepared.update(resume_queue=True, epoch=old.get('epoch',0))
    prepared['run_id']=identity;prepared['directory']=str(directory)
    atomic(directory/'spec.json',prepared)
    atomic(directory/'control.json',dict(action='run'))
    atomic(directory/'state.json',dict(state='preparing'))
    (directory/'worker-version.json').unlink(missing_ok=True)
    manifest=prepared['environment'];env=os.environ.copy();env.update(manifest['env'])
    env['AFX_TASK_HOME']=str(store.home)
    if manifest['id'] != 'local':
        env.pop('PYTHONPATH',None)
    try:
        # Execution uses its registered environment, never the control venv.
        worker_python = manifest['python']
        # Keep the execution worker's stdout/stderr with the run.  The
        # bootstrap daemon is deliberately quiet, but dropping worker output
        # made failed runs impossible to diagnose after the child exited.
        log = (directory / 'execution.log').open('ab', buffering=0)
        try:
            child=subprocess.Popen([worker_python,'-m','autofusex.client.task_worker',str(directory/'spec.json'),str(ownership.fileno())],
                                   env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(ownership.fileno(),))
        except BaseException:
            log.close()
            raise
        log.close()
    finally: ownership.close()
    processes[identity]=child
    with store.connect() as db:
        db.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)',(identity,canonical(prepared),'preparing',child.pid,None))
    return dict(run_id=identity,state='preparing',config_hash=prepared['hash'])


def action(store,identity,action,processes):
    if action not in {'pause','resume','stop','force-stop'}: raise ValueError('invalid task action')
    with store.connect() as db:
        row=db.execute('SELECT * FROM runs WHERE id=?',(identity,)).fetchone()
        if not row: raise LookupError('task not found')
        spec = json.loads(row['spec']).get('spec', {})
        if spec.get('kind', 'fuzz_task') != 'fuzz_task' and action in {'pause','resume'}:
            raise ValueError('this workload supports stop and force-stop; pause/resume are unavailable')
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
    if store.get('client_switch', {}).get('state') in {'draining','staging','requested','activating'}:
        raise ValueError('client upgrade blocks reconfiguration')
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
