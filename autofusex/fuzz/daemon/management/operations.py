"""Explicit operation allowlist shared by both ingress paths."""
import json
import platform
import sys
import re
from .config import validate

KINDS={'environment.plan','environment.apply','environment.inspect','build.submit','task.validate','task.submit','task.action','config.apply','tool.stage','tool.activate','client.update','task.reconfigure','daemon.control','operation.cancel'}


def validate_operation(kind,payload):
    if kind not in KINDS or not isinstance(payload,dict): raise ValueError('unsupported operation')
    if kind=='operation.cancel':
        if set(payload)!={'operation_id'} or not isinstance(payload['operation_id'],str):raise ValueError('operation ID required')
        return payload
    if kind == 'daemon.control':
        if set(payload) != {'state'} or payload['state'] not in {'running', 'stopped'}:
            raise ValueError('daemon control requires running or stopped state')
        return payload
    if kind=='task.reconfigure':
        if set(payload)!={'run_id','spec'} or not isinstance(payload['run_id'],str):raise ValueError('run ID and task spec required')
        payload=dict(run_id=payload['run_id'],spec=validate(payload['spec']))
        if payload['spec']['kind']!='fuzz_task':raise ValueError('task spec required')
        return payload
    if kind=='tool.stage':
        if set(payload)!={'manifest','signature','key_id'}: raise ValueError('signed release required')
        return payload
    if kind=='tool.activate':
        if set(payload)!={'release_id'} or not isinstance(payload['release_id'],str): raise ValueError('release ID required')
        return payload
    if kind=='client.update':
        allowed={'version','package','index_url','extra_args'}
        if set(payload)-allowed or not isinstance(payload.get('version'), str) or not re.fullmatch(r'[0-9]+(?:\.[0-9]+)*(?:[.-][0-9A-Za-z.-]+)?', payload['version']):
            raise ValueError('client update requires an exact version')
        if payload.get('package','autofusex') != 'autofusex':
            raise ValueError('only the autofusex client package may be updated')
        if payload.get('index_url') is not None and not isinstance(payload['index_url'], str):
            raise ValueError('invalid package index')
        if payload.get('extra_args') is not None:
            if not isinstance(payload['extra_args'], list) or not all(isinstance(x,str) for x in payload['extra_args']):
                raise ValueError('invalid pip arguments')
        return dict(payload, package='autofusex')
    if kind in {'environment.plan','task.validate','task.submit','config.apply'}:
        value=validate(payload)
        expected='environment' if kind.startswith('environment.') else 'client_policy' if kind=='config.apply' else 'fuzz_task'
        if value['kind']!=expected: raise ValueError('wrong specification kind')
        return value
    fields={'environment.apply':{'plan_id'},'environment.inspect':set(),'build.submit':{'source_dir','profile','mode','jobs'},'task.action':{'run_id','action'}}[kind]
    if set(payload)-fields: raise ValueError('unknown operation field')
    if kind=='environment.apply' and (set(payload)!=fields or not isinstance(payload['plan_id'],str)): raise ValueError('plan_id required')
    if kind=='task.action' and (set(payload)!=fields or payload['action'] not in {'pause','resume','stop','force-stop'}): raise ValueError('invalid task action')
    if kind=='build.submit':
        if not {'source_dir','profile'}<=set(payload) or payload.get('mode','incremental') not in {'incremental','snapshot'}: raise ValueError('invalid build')
        if type(payload.get('jobs',1)) is not int or not 1<=payload.get('jobs',1)<=1024: raise ValueError('invalid build parallelism')
    return payload


def execute(store,operation,processes):
    from . import environments,tasks
    kind=operation['kind'];payload=validate_operation(kind,operation['payload']);identity=operation['id']
    if kind=='operation.cancel':
        target=store.operation(payload['operation_id'])
        if target['state']=='queued':store.state(target['id'],'cancelled',dict(code='cancelled_before_start'))
        elif target['state']=='running':store.put('cancel:'+target['id'],True)
        return dict(id=target['id'],state='cancel_requested')
    if kind == 'daemon.control':
        state = payload['state']
        if state == 'stopped':
            for identity in list(processes):
                tasks.action(store, identity, 'stop', processes)
            store.put('shutdown_when_idle', True)
            return {'state': 'stopping'}
        store.put('shutdown_when_idle', False)
        return {'state': 'running'}
    if kind in {'environment.apply','tool.activate'}:
        with store.connect() as db:
            if db.execute("SELECT 1 FROM runs WHERE state IN ('preparing','running','pausing','paused','draining','cancelling','needs_attention')").fetchone():
                raise ValueError('environment changes require idle device with resolved task ownership')
    if kind=='tool.stage':
        from .updates import stage
        return stage(store,payload,identity)
    if kind=='tool.activate':
        from .updates import activate
        return activate(store,payload['release_id'],processes)
    if kind=='client.update':
        from .client_updates import install
        return install(store, payload, processes, identity)
    if kind=='environment.plan': return environments.plan(store,payload)
    if kind=='environment.apply': return environments.apply(store,payload['plan_id'],identity)
    if kind=='environment.inspect':
        with store.connect() as db:
            items=[dict(id=r['id'],name=r['name'],state=r['state']) for r in db.execute('SELECT * FROM environments')]
        return dict(python=sys.version.split()[0],architecture=platform.machine(),environments=items)
    if kind=='build.submit': return environments.build_run(store,payload,identity)
    if kind=='task.validate': return tasks.prepare(store,payload)
    if kind=='task.submit': return tasks.start(store,payload,identity,processes)
    if kind=='task.reconfigure': return tasks.reconfigure(store,payload['run_id'],payload['spec'],processes)
    if kind=='task.action': return tasks.action(store,payload['run_id'],payload['action'],processes)
    if kind=='config.apply':
        from .artifacts import trusted
        ceilings=trusted(store.home).get('resource_limits',{})
        limits=payload['resources']
        for key in ('max_workers','max_compile_threads'):
            if limits[key]>ceilings.get(key,1): raise PermissionError('policy exceeds locally authorized '+key)
        if not set(limits['allowed_devices'])<=set(ceilings.get('allowed_devices',[])):
            raise PermissionError('policy requests unauthorized devices')
        from .config import quantity
        if limits.get('memory_budget') and ceilings.get('memory_budget') and quantity(limits['memory_budget'])>quantity(ceilings['memory_budget']):
            raise PermissionError('memory policy exceeds local authority')
        store.put('effective_policy',payload)
        return dict(applied=True,scope='new_tasks',existing_tasks_unchanged=True)
    raise ValueError('unsupported operation')
