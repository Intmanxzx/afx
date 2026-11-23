"""Receive fixed-source jobs through the same owned managed-task executor."""
import json
from autofusex.client.artifacts import atomic
from autofusex.workloads.imported.registry import receive


def synchronize(manager, api, machine):
    store=manager.store
    config=store.home/'client.json'
    if not config.exists() or json.loads(config.read_text()).get('mode')!='managed':return
    if store.get('client_switch',{}).get('state') in {'draining','staging','requested','activating'}:return
    saved=store.get('source_job')
    if saved:
        operation=store.operation(saved['operation_id'])
        outcome=operation.get('result') or {}
        run_id=outcome.get('run_id')
        if operation['state'] in {'failed','cancelled','needs_attention'}:
            result=dict(case=saved['case'],status='admission_failed',message=outcome,artifacts=[])
        elif not run_id:
            return
        else:
            with store.connect() as db:
                run=db.execute('SELECT state FROM runs WHERE id=?',(run_id,)).fetchone()
            directory=store.home/'runs'/run_id
            atomic(directory/'source-upload.json', {'central_job':saved['job_id']})
            if run and run['state'] not in {'completed','failed','stopped','cancelled','interrupted'}:return
            result_path=directory/'result.json'
            result=json.loads(result_path.read_text()) if result_path.exists() else dict(case=saved['case'],status=run['state'] if run else 'interrupted',artifacts=[])
        api.call('POST','/api/source-jobs/'+saved['job_id']+'/complete',dict(lease_id=saved['lease_id'],result=result))
        if run_id:
            upload_artifacts(api,directory,saved['job_id'],saved['lease_id'],result)
        store.put('source_job',None)
        return
    if manager.active:return
    # Respect explicit local stop and do not replace a running task's configuration.
    command=store.home/'daemon-control.json'
    if command.exists() and json.loads(command.read_text()).get('state')=='stopped':return
    job=api.call('POST','/api/source-jobs/claim',dict(machine_id=machine))['job']
    if not job:return
    record=receive(store.home,job['bundle'])
    spec=job['spec'];spec['results']={'save':True,'upload':True}
    # Durable idempotent operation admission survives a crash before the next heartbeat.
    operation=store.submit('task.submit',spec,'source-job:'+job['job_id'])
    store.put('source_job',dict(job_id=job['job_id'],lease_id=job['lease_id'],operation_id=operation['id'],case=record))


def upload_artifacts(api,directory,job_id,lease,result):
    import base64
    from pathlib import Path
    directory=Path(directory).resolve()
    for item in result.get('artifacts',[]):
        if item['size']>1024*1024:continue
        path=(directory/item['path']).resolve()
        if not path.is_relative_to(directory):raise ValueError('artifact escaped run directory')
        if path.suffix not in {'.html','.json','.txt','.csv','.svg'}:continue
        api.call('POST','/api/source-jobs/'+job_id+'/artifacts',dict(lease_id=lease,path=item['path'],data=base64.b64encode(path.read_bytes()).decode('ascii')))
