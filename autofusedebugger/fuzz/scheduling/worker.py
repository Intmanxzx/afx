"""One disposable CPU worker; uses the SAME execution loop as replay.

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
    if config.backend not in {'cpu-eager', 'cpu-compile'} or config.device != 'cpu':
        raise ValueError('managed NPU execution requires validated device leases')
    bundle = materialize(task['seed'])
    result = execute(bundle, create_backend(config), evaluation)
    from importlib.metadata import version, PackageNotFoundError
    environment = {'python': platform.python_version(), 'platform': sys.platform,
                   'machine': platform.machine()}
    for package, key in [('torch', 'torch'), ('autofusedebugger', 'package_version')]:
        try:
            environment[key] = version(package)
        except PackageNotFoundError:
            pass
    ResultRepository(database).save(result, environment=environment, attempt_id=task['attempt_id'])
    return 0  # A failed correctness result is still a durably completed attempt.


if __name__ == '__main__':
    raise SystemExit(run(*sys.argv[1:]))
