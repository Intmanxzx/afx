"""One disposable worker; uses the SAME execution loop as replay.

Parent owns lifecycle/acknowledgement; worker only persists its assigned attempt.
"""
import json
import platform
import sys


def run(queue, database, identity, claim):
    from .ledger import TaskLedger
    from ..contracts import ExecutionConfig, EvaluationConfig
    from ..generation import materialize
    from ..backends import create_backend
    from ..execution import execute
    from ..storage import ResultRepository
    task = TaskLedger(queue).get(identity)
    if task['state'] not in {'running', 'stop_requested'} or task['claim'] != claim:
        raise ValueError('worker claim is stale')
    from ..storage.lifecycle import append
    if json.loads(task['execution']).get('save_results', True):
        append(database, task['attempt_id'], 'started')
    config = ExecutionConfig(**json.loads(task['execution']))
    evaluation = EvaluationConfig(**json.loads(task['evaluation']))
    from .supervisor import _validate_task
    _validate_task(task)
    import time
    generation_start=time.perf_counter_ns()
    try:
        bundle = materialize(task['seed'])
    except Exception as exc:
        from dataclasses import asdict
        from ..contracts import fingerprint
        from ..failure_details import excerpt
        from ..environment import collect
        from pathlib import Path
        message=excerpt(exc)
        result={'seed':task['seed'],'workload_id':fingerprint({'generation_failed':task['seed']}),
                'oracle_id':'generation-failed/1','backend':config.backend,'status':'error',
                'error_stage':'generation','error_summary':message[-500:],'shape_results':[],
                'execution':dict(asdict(config),diagnostics=json.dumps({'version':1,'error':message,'shapes':{}})),
                'evaluation':asdict(evaluation)}
        if config.save_results:
            append(database, task['attempt_id'], 'finished')
            ResultRepository(database).save(result,environment=collect(Path(queue).parent.parent),attempt_id=task['attempt_id'])
        else:
            print(json.dumps({'seed':task['seed'],'status':'error','stage':'generation','saved':False}))
        return 0
    generation_ns=time.perf_counter_ns()-generation_start
    from ..core_logs import capture_tail
    with capture_tail(forward=True) as core_log:
        result = execute(bundle, create_backend(config), evaluation)
    if result['status'] in {'error', 'unsupported'} and core_log():
        diagnostics = json.loads(result['execution']['diagnostics'])
        diagnostics['error'] = core_log()
        result['execution']['diagnostics'] = json.dumps(diagnostics)
    telemetry=json.loads(result['execution']['telemetry'])
    telemetry['stages']['generation']=generation_ns
    result['execution']['telemetry']=json.dumps(telemetry)
    if not config.save_results:
        print(json.dumps({"seed": task["seed"], "backend": config.backend, "status": result["status"], "saved": False}))
        return 0
    from ..environment import collect
    from pathlib import Path
    append(database, task['attempt_id'], 'finished')
    environment=collect(Path(queue).parent.parent)
    ResultRepository(database).save(result, environment=environment, attempt_id=task['attempt_id'])
    return 0  # A failed correctness result is still a durably completed attempt.


if __name__ == '__main__':
    raise SystemExit(run(*sys.argv[1:]))
