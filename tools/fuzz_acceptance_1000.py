"""Remote acceptance runner using the unified execution and lightweight results.

Ranges can run on separate devices. --resume skips only fully executed matching
seed/backend/workload records; it never substitutes cached tensors for generation.
"""
import argparse
import gc
import json
import platform
import signal
import time

from autofusedebugger.fuzz.contracts import ExecutionConfig, EvaluationConfig
from autofusedebugger.fuzz.generation import materialize, workload_id
from autofusedebugger.fuzz.generation.categories import categories
from autofusedebugger.fuzz.backends import create_backend
from autofusedebugger.fuzz.execution import execute
from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.storage.queries import reading


def completed_pairs(database):
    with reading(database) as conn:
        return {(row['seed'], row['backend'], row['workload_id']) for row in conn.execute(
            "SELECT a.seed,a.backend,a.workload_id FROM attempts a "
            "JOIN shapes s ON s.attempt_id=a.id WHERE a.status IN ('ok','fail') "
            "GROUP BY a.id HAVING count(*)=3 AND min(s.shape_index)=0 AND max(s.shape_index)=2")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('database')
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--backends', nargs='+', default=['cpu-eager', 'ascendc'])
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.start < 0 or args.count < 1 or args.start + args.count > 2**128:
        parser.error('invalid v2 entropy range')
    repo = ResultRepository(args.database)
    existing = completed_pairs(repo.path) if args.resume else set()
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True  # finish the current backend and persist before stopping

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    counts = {}; started = time.monotonic(); completed = skipped = 0
    for index in range(args.start, args.start + args.count):
        if stopped:
            break
        seed = str(2**128 + index); bundle = materialize(seed)
        fingerprint = workload_id(bundle)
        for name in args.backends:
            if stopped:
                break
            if (seed, name, fingerprint) in existing:
                skipped += 1
                continue
            config = ExecutionConfig(name, args.device if name in {'ascendc', 'triton'} else 'cpu', 1)
            result = execute(bundle, create_backend(config), EvaluationConfig())
            identity = repo.save(result, environment={
                'machine': platform.machine(), 'python': platform.python_version(), 'package_version': '0.1.2'})
            counts[name + '/' + result['status']] = counts.get(name + '/' + result['status'], 0) + 1
            print(json.dumps({'index': index, 'seed': seed, 'backend': name, 'status': result['status'],
                              'shapes': len(result['shape_results']), 'categories': categories(bundle.spec),
                              'attempt_id': identity, 'workload_id': fingerprint,
                              'elapsed': round(time.monotonic() - started, 2)}), flush=True)
            import torch
            torch._dynamo.reset()
            gc.collect()
        else:
            completed += 1
    print(json.dumps({'completed_seeds': completed, 'interrupted': stopped, 'skipped_attempts': skipped,
                      'counts': counts, 'elapsed': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()
