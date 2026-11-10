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
    config = ExecutionConfig(**json.loads(task['execution']))
    evaluation = EvaluationConfig(**json.loads(task['evaluation']))
    from .supervisor import _validate_task
    _validate_task(task)
    import time
    generation_start=time.perf_counter_ns()
    bundle = materialize(task['seed'])
    generation_ns=time.perf_counter_ns()-generation_start
    result = execute(bundle, create_backend(config), evaluation)
    telemetry=json.loads(result['execution']['telemetry'])
    telemetry['stages']['generation']=generation_ns
    result['execution']['telemetry']=json.dumps(telemetry)
    from ..environment import collect
    from pathlib import Path
    environment=collect(Path(queue).parent.parent)
    ResultRepository(database).save(result, environment=environment, attempt_id=task['attempt_id'])
    return 0  # A failed correctness result is still a durably completed attempt.


if __name__ == '__main__':
    raise SystemExit(run(*sys.argv[1:]))
