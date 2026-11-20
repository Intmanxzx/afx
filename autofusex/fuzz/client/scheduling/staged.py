"""Bounded common Case -> Stage DAG coordinator. No Torch in supervisor.

Single generated Case fans out to CPU reference and backend sessions. All
shapes of a session stay in one worker. Compare and record are CPU stages after
safe host output publication. Failed stages do not synthesize Shape results.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
import json
import shutil
from pathlib import Path
import tempfile
import time
import uuid

from .stages import Case, StageTask, BackendSession, Kind, Requirements
from .pool import Pool, discover
from ..contracts import ExecutionConfig, EvaluationConfig, fingerprint
from ..storage.lifecycle import stamp


def drain(ledger, database, *, timeout=120, _owned_fd=None, _ready=None):
    from .supervisor import _bind_results, _reconcile, _tasks, _validate_task
    from .ownership import own_queue
    from .budgets import read_policy, Budget, group_rss, scratch_size
    from .placement import candidates
    from .campaign import refill
    database = Path(database).resolve()
    budget = read_policy(ledger) or Budget()
    ownership = own_queue(ledger) if _owned_fd is None else nullcontext(_owned_fd)
    with ownership as fd, tempfile.TemporaryDirectory(prefix='.stage-window-',dir=ledger.path.parent) as scratch:
        _bind_results(ledger,database)
        summary = dict(_reconcile(ledger,database),completed=0,failed=0,cancelled=0,record_failed=0,skipped=0)
        names = set()
        exclude_ops = []
        ops = []
        for task in _tasks(ledger):
            if task['state']=='pending':
                _validate_task(task)
                names.update(d for d in candidates(ledger,task) if d!='cpu')
        with ledger._connection() as conn:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='campaign'").fetchone():
                row=conn.execute('SELECT payload FROM campaign WHERE id=1').fetchone()
                if row:
                    policy=json.loads(row[0]);names.update(policy.get('devices',[]))
                    exclude_ops=[] if policy.get('ops') else policy.get('exclude_ops',[])
                    ops=policy.get('ops',[])
        pool = Pool(budget,scratch,fd,discover(sorted(names)),max_tasks=budget.worker_max_tasks)
        if budget.device_budget_bytes:
            for device in pool.devices: device.capacity=min(device.capacity,budget.device_budget_bytes)
        cases, sessions, stages = {}, {}, {}
        window_id = str(uuid.uuid4())
        window_start = stamp()
        completed_attempts = []
        window_summary = dict(summary)
        from ..runtime import Reporter
        reporter=Reporter(ledger.path.parent.parent,pool.devices)
        finalizer = ThreadPoolExecutor(max_workers=1,thread_name_prefix="stage-facts")
        finalized = {}
        if _ready: _ready()

        def stage(kind,case,dependencies=(),session=None):
            devices = tuple(candidates(ledger,session.task)) if session and kind==Kind.COMBINED and json.loads(session.task['execution'])['device']!='cpu' else ()
            req=Requirements(cpu_tokens=budget.compile_threads if kind in {Kind.COMBINED,Kind.REFERENCE} else 1,
                memory_bytes=min(pool.memory_capacity//budget.cpu_workers,512*1024**2),
                cache_bytes=min(pool.cache_capacity//budget.cpu_workers,128*1024**2),devices=devices,
                device_bytes=256*1024**2 if devices else 0)
            value=StageTask(kind,case.identity,tuple(d.identity for d in dependencies),session.identity if session else None,
                            session.task['attempt_id'] if session else None,req)
            stages[value.identity]=value;case.stages[value.identity]=value
            return value

        def failure(task, kind, error, bundle=None):
            from ..failure_details import excerpt
            execution=json.loads(task['execution'])
            execution['diagnostics']=json.dumps({'version':1,'error':excerpt(RuntimeError(str(error))),'shapes':{}})
            if bundle: execution['case_spec']=bundle['spec']
            return {'seed':task['seed'],'workload_id':bundle['workload_id'] if bundle else fingerprint({'generation_failed':task['seed']}),
                'oracle_id':'torch-eager-cpu/1' if bundle else 'generation-failed/1','backend':execution['backend'],
                'execution':execution,'evaluation':json.loads(task['evaluation']),'status':'error',
                'error_stage':kind if bundle else 'generation','error_summary':str(error)[-500:],'shape_results':[]}

        def make_record(case,session,result,dependencies):
            result['execution']['round_id']=case.identity
            result['execution']['expected_backends']=json.dumps(sorted({json.loads(t['execution'])['backend'] for t in case.tasks}))
            s=stage(Kind.RECORD,case,dependencies,session)
            s.output={'result':result}
            s.facts['retry_index']=max(0,int(session.task.get('try_count',1))-1)
            return s

        def has_record(case, session_id):
            return any(x.kind == Kind.RECORD and x.session_id == session_id
                       for x in case.stages.values())

        def skip_case(case, reason):
            """Drop an un-evaluable case without creating an attempt row."""
            if case.skip_reason is None:
                case.skip_reason = str(reason)[:500]
            for other in list(case.stages.values()):
                other.facts.setdefault('skip_reason', case.skip_reason)
                if other.state in {'ready', 'waiting_dependencies'}:
                    other.state = 'cancelled'

        def interrupt_case(case, reason):
            # A broken worker/environment is not evidence of an invalid input.
            skip_case(case, reason)
            for task in case.tasks:
                if not task.get('done'):
                    ledger.interrupt(task['id'], task['claim'], reason)
                    task['done'] = True
                    summary['interrupted'] += 1
            case.skip_finalized = True

        def finalize_case_skip(case):
            if case.skip_reason is None or case.skip_finalized:
                return
            if any(x.state == 'running' for x in case.stages.values()):
                return
            for task in case.tasks:
                if task.get('done'):
                    continue
                ledger._finish_claim(task['id'], task['claim'], 'cancelled', case.skip_reason)
                task['done'] = True
                summary['skipped'] += 1
            case.skip_finalized = True

        def flush_deferred_targets(case):
            """Publish outcomes only after the CPU oracle and AF verdict."""
            if case.skip_reason is not None or case.reference is None:
                return
            combined = [x for x in case.stages.values() if x.kind == Kind.COMBINED]
            compares = [x for x in case.stages.values() if x.kind == Kind.COMPARE]
            # A target with a non-empty result still needs its independent
            # compare stage.  Do not publish the CPU peer while that decision
            # could still turn into a no-coverage skip.
            if any(x.state not in {'succeeded', 'failed', 'cancelled'}
                   for x in (*combined, *compares)):
                return
            for other in combined:
                if has_record(case, other.session_id):
                    continue
                session = sessions[other.session_id]
                compare = next((x for x in compares if x.session_id == other.session_id), None)
                if other.state == 'failed':
                    result = failure(session.task, 'worker',
                                     other.facts.get('error', 'target execution failed'), case.bundle)
                elif compare is not None and compare.output is not None:
                    result = compare.output['result']
                elif other.state == 'succeeded' and other.output and not other.output['result']['shape_results']:
                    result = other.output['result']
                else:
                    continue
                make_record(case, session, result, ())
            # The CPU reference is shared by all target sessions and is
            # recorded exactly once, after the target verdict is durable in the
            # stage DAG (the record workers may run in parallel afterwards).
            if any(json.loads(t['execution'])['backend'] in {'cpu'}
                   for t in case.tasks):
                cpu_task = next(t for t in case.tasks
                                if json.loads(t['execution'])['backend'] in {'cpu'})
                cpu_session = next(x for x in sessions.values() if x.task['id'] == cpu_task['id'])
                if not has_record(case, cpu_session.identity):
                    result = json.loads(json.dumps(case.reference['result']))
                    result['execution'].update(json.loads(cpu_task['execution']))
                    result['backend'] = json.loads(cpu_task['execution'])['backend']
                    make_record(case, cpu_session, result, ())

        def jobs(s,lease):
            case=cases[s.case_id];session=sessions.get(s.session_id)
            config=json.loads(session.task['execution']) if session else asdict(ExecutionConfig('cpu','cpu',budget.compile_threads,scheduling='staged'))
            ev=json.loads(session.task['evaluation']) if session else json.loads(case.tasks[0]['evaluation'])
            job={'kind':s.kind.value,'directory':case.directory,'bundle':case.bundle,'config':config,'evaluation':ev}
            if s.kind==Kind.GENERATE:
                is_replay = any(ledger.priorities().get(t["id"], 0) == 100 for t in case.tasks)
                job.update(seed=case.seed,exclude_ops=[] if is_replay else exclude_ops,ops=[] if is_replay else ops)
            elif s.kind==Kind.REFERENCE:
                job['output_file']='reference.npz'
                job['config']['static_only']=all(json.loads(t['execution'])['static_only'] for t in case.tasks)
            elif s.kind==Kind.COMBINED:
                session.device=lease['device'];session.uuid=lease['device_uuid']
                config['device']=session.device
                # This task is already claimed; update its actual placement only
                # under the claim fence, before any result is produced.
                session.task['execution']=json.dumps(config,sort_keys=True,separators=(',',':'))
                with ledger._connection() as conn:
                    changed=conn.execute("UPDATE tasks SET execution=? WHERE id=? AND claim=? AND state='running'",(session.task['execution'],session.task['id'],session.task['claim'])).rowcount
                    if not changed:raise ValueError('task cancelled before dispatch')
                job['output_file']=session.identity+'.npz'
                job['device_budget_bytes']=next((d.capacity for d in pool.devices if d.uuid==session.uuid),None)
            elif s.kind==Kind.COMPARE:
                combined=stages[s.dependencies[0]].output
                job.update(combined)
                job['reference']=case.reference['outputs']
            elif s.kind==Kind.RECORD:
                # Only publish observed transitions. A generation failure has no
                # backend start; missing events must not be fabricated at admission.
                events = {'admitted': case.admitted}
                own = [x for x in case.stages.values()
                       if (x.session_id == s.session_id or x.session_id is None)
                       and x.kind != Kind.RECORD]
                for name in ('ready', 'dispatched', 'started', 'finished'):
                    observed = [x.facts[name] for x in own if name in x.facts]
                    if observed:
                        mono = (max if name == 'finished' else min)(observed)
                        from datetime import datetime, timedelta
                        wall = datetime.fromisoformat(case.admitted['wall_time'])
                        wall += timedelta(microseconds=(mono-case.admitted['monotonic_ns'])/1000)
                        events[name] = dict(case.admitted, monotonic_ns=mono, wall_time=wall.isoformat())
                job['provenance']={'device_uuid':session.uuid,'device_model':next((d.model for d in pool.devices if d.uuid==session.uuid),None)}
                job.update(result=s.output['result'],save=json.loads(session.task['execution'])['save_results'],
                           database=str(database),attempt_id=session.task['attempt_id'],task=session.task,events=events,home=str(ledger.path.parent.parent))
            return job

        def finish(s,response):
            case=cases[s.case_id];session=sessions.get(s.session_id)
            s.state='succeeded' if response['ok'] else 'failed'
            s.facts.update({k:v for k,v in response.items() if k not in {'value','error','ok'}})
            s.facts['finished']=response.get('finished',time.monotonic_ns())
            if s.kind==Kind.GENERATE:
                if not response['ok']:
                    s.facts['error'] = response.get('error', '')
                    interrupt_case(case, 'generation_worker_failed')
                    return
                if response['value'].get('skipped'):
                    s.facts['error'] = response['value'].get('error', '')
                    skip_case(case, response['value'].get('reason', 'generation_failed'))
                    return
                if response['value'].get('excluded'):
                    for t in case.tasks:
                        ledger._finish_claim(t['id'],t['claim'],'cancelled','excluded_op')
                        t['done']=True;summary['skipped']+=1
                    return
                case.bundle=response['value'];s.output=case.bundle
                ref=stage(Kind.REFERENCE,case,(s,))
                for t in case.tasks:
                    ss=next(x for x in sessions.values() if x.task['id']==t['id'])
                    # CPU is the once-per-Case reference, not a second target
                    # self-comparison. CPU correctness remains not_checked;
                    # target sessions compare against this shared artifact.
                    if json.loads(t['execution'])['backend'] in {'cpu'}:continue
                    combined=stage(Kind.COMBINED,case,(s,),ss)
                    stage(Kind.COMPARE,case,(combined,ref),ss)
            elif s.kind==Kind.REFERENCE:
                if not response['ok']:
                    s.facts['error'] = response.get('error', '')
                    interrupt_case(case, 'reference_worker_failed')
                    return
                if response['ok'] and response['value']['result']['status']!='ok':
                    s.facts['error'] = response['value']['result'].get('error_summary', '') or json.dumps(response['value']['result'].get('shape_results', []))[-1600:]
                    response={'ok':False,'error':response['value']['result'].get('error_summary','CPU reference failed')}
                    s.state='failed'
                if response['ok']:
                    case.reference=response['value'];s.output=response['value']
                    # Hold the CPU row until every target has crossed the
                    # coverage/compare decision.  This prevents a later
                    # no-Autofuse-coverage skip from leaving an orphan CPU
                    # attempt in the effective denominator.
                    flush_deferred_targets(case)
                else:
                    # CPU is the oracle. Its failure makes this input
                    # un-evaluable, not an Autofuse failure.
                    skip_case(case, 'cpu_reference_failed')
            elif s.kind==Kind.COMBINED:
                if response['ok']:
                    s.output=response['value']
                    if response['value'].get('case_skipped'):
                        skip_case(case, response['value'].get('skip_reason') or 'target_un-evaluable')
                        return
                    # Empty results cannot be compared, even when the target
                    # finishes before the shared reference. The publish barrier
                    # below still waits for every other target's AF verdict.
                    if not s.output['result']['shape_results']:
                        for other in case.stages.values():
                            if other.kind == Kind.COMPARE and other.session_id == s.session_id:
                                other.state = 'cancelled'
                else:
                    s.facts['error'] = response['error']
                    for other in case.stages.values():
                        if other.kind == Kind.COMPARE and other.session_id == s.session_id:
                            other.state = 'cancelled'
            elif s.kind==Kind.COMPARE:
                s.output=response.get('value')
                if response['ok']:
                    result=s.output['result']
                else:
                    result=json.loads(json.dumps(stages[s.dependencies[0]].output['result']))
                    result['status']='error'
                    for row in result['shape_results']:
                        if not row.get('error_stage'):row.update(correctness='error',error_stage='compare',error_summary=str(response['error'])[-500:])
                s.output = {'result': result}
            elif s.kind==Kind.RECORD:
                if response['ok']:
                    ledger.complete(session.task['id'],session.task['claim']);summary['completed']+=1
                    completed_attempts.append(session.task['attempt_id'])
                    reporter.record(s.output['result'],json.loads(session.task['execution'])['backend'])
                    if response['value'].get('status')!='ok':summary['failed']+=1
                else:
                    ledger.interrupt(session.task['id'],session.task['claim'],'record_failed');summary['interrupted']+=1;summary['record_failed']+=1
                    from ..runtime import atomic
                    atomic(ledger.path.parent/'last-record-error.json',{'error':str(response.get('error','unknown'))[-2000:]})
                session.task['done']=True

        try:
            while True:
                control_path = ledger.path.parent / 'control.json'
                control = json.loads(control_path.read_text()).get('action', 'run') if control_path.exists() else 'run'
                if control == 'force-stop':
                    from .campaign import stop
                    stop(ledger)
                if control in {'pause', 'stop', 'force-stop'} and not cases:
                    if control == 'pause':
                        from ..runtime import atomic
                        atomic(ledger.path.parent / 'state.json', {'state': 'paused'})
                        time.sleep(.2)
                        continue
                    from .campaign import stop
                    stop(ledger)
                    break
                if control == 'run' and control_path.exists() and (ledger.path.parent / 'state.json').exists() and json.loads((ledger.path.parent / 'state.json').read_text()).get('state') == 'paused':
                    from ..runtime import atomic
                    atomic(ledger.path.parent / 'state.json', {'state': 'running'})
                # Bounded Case window; no infinite queue or unbounded outputs.
                while control in {'run', 'reconfigure'} and len(cases)<budget.cpu_workers and len(completed_attempts)<100:
                    budget.admit(ledger.path.parent)
                    budget.admit(database.parent)
                    if control == 'run':
                        refill(ledger,target_pending=budget.cpu_workers)
                    pending=[t for t in _tasks(ledger) if t['state']=='pending']
                    if not pending:break
                    # Keep a pinned-card backlog from filling the entire Case
                    # window while another compatible device has no work.
                    pressure = {}
                    for active in cases.values():
                        for task in active.tasks:
                            if not task.get('done'):
                                for device in candidates(ledger, task):
                                    pressure[device] = pressure.get(device, 0) + 1
                    priorities = ledger.priorities()
                    first = min(pending, key=lambda t: (-priorities.get(t["id"], 0), min(
                        (pressure.get(d, 0) for d in candidates(ledger, t)), default=0)))
                    ex=json.loads(first['execution'])
                    peers=[t for t in pending if t['seed']==first['seed'] and json.loads(t['execution']).get('batch_id')==ex.get('batch_id') and t['evaluation']==first['evaluation']]
                    # Repeated same-backend seed executions are distinct Cases.
                    selected=[];seen=set()
                    for t in peers:
                        backend=json.loads(t['execution'])['backend']
                        if backend not in seen:selected.append(t);seen.add(backend)
                    claimed=[ledger.claim(t['id']) for t in selected];claimed=[t for t in claimed if t]
                    if not claimed:continue
                    with ledger._connection() as conn:
                        for t in claimed:t['try_count']=conn.execute('SELECT count FROM task_tries_v1 WHERE task_id=?',(t['id'],)).fetchone()[0]
                    identity=str(uuid.uuid4());directory=Path(scratch)/identity;directory.mkdir(mode=0o700)
                    admitted=stamp()
                    if not cases and not completed_attempts:window_start=admitted
                    case=Case(identity,first['seed'],admitted,claimed,str(directory));cases[identity]=case
                    for t in claimed:
                        ss=BackendSession(str(uuid.uuid4()),t);sessions[ss.identity]=ss
                    stage(Kind.GENERATE,case)
                for case in list(cases.values()):
                    stopped=any(ledger.get(t['id'])['state']=='stop_requested' for t in case.tasks if not t.get('done'))
                    if stopped:
                        for s in case.stages.values():
                            if s.state=='running':pool.cancel(s.identity)
                            elif s.state in {'ready','waiting_dependencies'}:s.state='cancelled'
                        if not any(s.state=='running' for s in case.stages.values()):
                            for t in case.tasks:
                                if not t.get('done'):
                                    ledger.interrupt(t['id'],t['claim'],'user_stop');summary['interrupted']+=1;summary['cancelled']+=1;t['done']=True
                    for s in list(case.stages.values()):
                        if s.state=='running' and s.future.done():
                            lease=pool.release(s.identity);s.facts['lease']=lease
                            try:response=s.future.result()
                            except Exception as exc:response={'ok':False,'error':type(exc).__name__+': '+str(exc)}
                            if stopped or case.skip_reason is not None:
                                s.state='cancelled'
                            else:finish(s,response)
                    flush_deferred_targets(case)
                    finalize_case_skip(case)
                    if all(t.get('done') for t in case.tasks) and not any(s.state=='running' for s in case.stages.values()):
                        from ..storage.stage_facts import save_case
                        if case.identity not in finalized:
                            for unused in case.stages.values():
                                if unused.state in {'ready','waiting_dependencies'}:unused.state='cancelled'
                            finalized[case.identity]=finalizer.submit(save_case,database,case,dict(sessions),window_id,window_start,pool)
                        if not finalized[case.identity].done():continue
                        finalized.pop(case.identity).result()
                        shutil.rmtree(case.directory)
                        for sid in list(case.stages):stages.pop(sid,None)
                        for sid,ss in list(sessions.items()):
                            if ss.task in case.tasks:sessions.pop(sid)
                        del cases[case.identity]
                now=time.monotonic_ns()
                ready=[s for s in stages.values() if s.ready(stages,now)]
                priorities = ledger.priorities()
                ready.sort(key=lambda s:(-max((priorities.get(t["id"],0) for t in cases[s.case_id].tasks),default=0), 0 if s.kind==Kind.COMBINED else 1 if s.kind in {Kind.COMPARE,Kind.RECORD} else 2,s.ready_ns))
                for s in ready:
                    try:lease=pool.reserve(s)
                    except ValueError as exc:
                        finish(s,{'ok':False,'error':str(exc)})
                        continue
                    if lease is None:continue
                    s.facts.update(ready=s.ready_ns,dispatched=time.monotonic_ns())
                    try:
                        job=jobs(s,lease)
                        s.future=pool.submit(s,job,timeout);s.state='running'
                    except Exception as exc:
                        s.facts['lease']=pool.release(s.identity)
                        finish(s,{'ok':False,'error':str(exc)})
                if reporter.enabled and time.monotonic()-reporter.last>=5:
                    with ledger._connection() as conn:queued=conn.execute("SELECT count(*) FROM tasks WHERE state='pending'").fetchone()[0]
                    reporter.publish(cases,stages,pool,queued,summary)
                # Count idle RSS too. Kill active groups on measured overflow;
                # response failure becomes an explicit result, never a retry.
                rss=sum(group_rss(w['child'].pid) for w in list(pool.workers))
                if rss>pool.memory_capacity or scratch_size(scratch)>pool.cache_capacity:
                    for identity in list(pool.leases):pool.cancel(identity)
                if not cases and control == 'reconfigure':
                    if not any(t['state'] == 'pending' for t in _tasks(ledger)):
                        break
                if not cases and control == 'run':
                    if len(completed_attempts)>=100:
                        from ..storage.stage_facts import finish_window
                        finish_window(database,window_id,window_start,pool,{k:v-window_summary.get(k,0) for k,v in summary.items()})
                        completed_attempts.clear();window_id=str(uuid.uuid4());window_start=stamp();window_summary=dict(summary)
                    if not refill(ledger,target_pending=budget.cpu_workers) and not any(t["state"] == "pending" for t in _tasks(ledger)):break
                time.sleep(.02)
            from ..storage.stage_facts import finish_window
            finish_window(database,window_id,window_start,pool,{k:v-window_summary.get(k,0) for k,v in summary.items()})
            return summary
        finally:
            reporter.publish({}, {}, pool, 0, summary, state='idle' if not cases else 'error',force=True)
            pool.close()
            finalizer.shutdown(wait=True)
            for case in cases.values():
                for t in case.tasks:
                    if not t.get('done'):
                        try:ledger.interrupt(t['id'],t['claim'],'stage_supervisor_interrupted')
                        except ValueError:pass


def replay(args):
    """Replay feeds the same scheduler, using a private ephemeral task ledger."""
    from ..config import data_home,load_config
    from ..resources import adaptive_defaults
    from .ledger import TaskLedger
    from .budgets import Budget,write_policy
    from ..contracts import Seed
    home=data_home();home.mkdir(parents=True,exist_ok=True)
    budget=Budget.configured(load_config(home) or adaptive_defaults(home),args.compile_threads)
    with tempfile.TemporaryDirectory(prefix='.replay-stages-',dir=home) as directory:
        ledger=TaskLedger(Path(directory)/'tasks.sqlite')
        database=Path(args.database) if args.database else (home/'results/results.sqlite' if args.save else Path(directory)/'results.sqlite')
        write_policy(ledger,budget)
        batch=str(uuid.uuid4());seed=Seed.parse(args.seed)
        for backend in ([args.backend] if args.backend else ['cpu','af-mlir']):
            config=ExecutionConfig(backend,'cpu' if backend=='cpu' else args.device or 'npu:0',budget.compile_threads,
                scheduling='staged',static_only=args.static_only or not seed.dynamic,batch_id=batch,save_results=args.save)
            ledger.enqueue(seed,config,EvaluationConfig(performance=args.performance,warmup_steps=args.warmup_steps,measured_steps=args.measured_steps))
        result=drain(ledger,database,timeout=120)
        print(json.dumps(result))
        if args.save:
            from ..storage.queries import ResultQueries
            query=ResultQueries(database)
            results=[query.detail(t['attempt_id']) for t in ledger.page()['items'] if t['state']=='completed']
            print(json.dumps({'attempts':[{'attempt_id':r['id'],'status':r['status'],'shape_count':len(r['shape_results'])} for r in results]}))
            return 0 if results and all(r['status']=='ok' for r in results) else 1
        return 0 if not result['interrupted'] and not result.get('failed',0) else 1
