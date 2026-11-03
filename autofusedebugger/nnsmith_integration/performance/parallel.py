"""Dynamic case scheduling across distinct NPUs with durable in-flight seeds."""
from dataclasses import asdict
import os
from pathlib import Path
import signal

from ..task_pool import dynamic_results
from .artifacts import compact_pending_case
from .lifecycle import SessionStopped, defer_stop_signals, guard_worker_parent
from .schema import ProfileConfig


def worker_targets(jobs, devices, target):
    if jobs < 1:
        raise ValueError('--jobs must be positive')
    if not devices:
        # Never benchmark two cases on the same card, even with the CLI default jobs=2.
        return [target]
    if len(set(devices)) != len(devices) or any(d < 0 for d in devices):
        raise ValueError('Performance --devices must contain distinct nonnegative device indices')
    return [f'npu:{d}' for d in devices[:jobs]]


def initialize_worker(index, config):
    guard_worker_parent(config['parent_pid'])
    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise SessionStopped(signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    from nnsmith.materialize import Model
    from .cases import restore_opset
    model_cls = Model.init('torch', backend_target='cpu')
    model_cls.add_seed_setter()
    return model_cls, restore_opset(config['opset'], model_cls), config['targets'][index]


def execute_case(runtime, task, config):
    from ..generation import GenerationConfig
    from .session import _measure_case
    model_cls, opset, target = runtime
    case = _measure_case(index=task['index'], case_seed=task['seed'],
        out=Path(config['out_dir']), generation=GenerationConfig(**config['generation']),
        profile=ProfileConfig(**config['profile']), opset=opset, model_cls=model_cls,
        target=target, devices=None)
    case['target'] = target
    return case


def run_parallel(*, summary, out, generation, profile, rng, limit, targets, resume):
    from .session import _save_checkpoint, _commit_case
    checkpoint = summary['checkpoint']
    if 'pending_tasks' not in checkpoint:
        pending_seed = checkpoint.get('pending_seed')
        checkpoint['pending_tasks'] = ([] if pending_seed is None else
            [dict(index=summary['total'], seed=pending_seed)])
        checkpoint['next_index'] = summary['total'] + (pending_seed is not None)
        checkpoint['pending_seed'] = None
    pending = checkpoint['pending_tasks']
    if resume and not profile.keep_artifacts:
        for task in pending:
            compact_pending_case(out, f"{task['index']:05d}-{task['seed']}")
    if limit is not None and limit < checkpoint['next_index']:
        raise ValueError('--n-cases cannot discard previously assigned pending cases')
    # Retrying in-flight tasks precedes any new seeds. Completed higher indices stay saved.
    retries = list(sorted(pending, key=lambda task: task['index']))

    def tasks():
        for task in retries:
            yield dict(task)
        while limit is None or checkpoint['next_index'] < limit:
            with defer_stop_signals():
                index = checkpoint['next_index']
                seed = summary['repro_seed'] if summary['repro_seed'] is not None else rng.getrandbits(32)
                task = dict(index=index, seed=seed)
                pending.append(task)
                checkpoint.update(next_index=index + 1, rng_state=rng.getstate())
                _save_checkpoint(summary, out)
            yield dict(task)

    config = dict(parent_pid=os.getpid(), out_dir=str(out), targets=targets,
        generation=asdict(generation), profile=asdict(profile), opset=checkpoint['opset'])
    stream = dynamic_results(tasks(), workers=len(targets), config=config,
        initialize=initialize_worker, execute=execute_case)
    try:
        for case in stream:
            with defer_stop_signals():
                task = next(t for t in pending if t['index'] == case['index'])
                if task['seed'] != case['seed']:
                    raise RuntimeError('Worker returned a different case seed')
                pending.remove(task)
                _commit_case(summary, case, out)
    finally:
        stream.close()
