"""Bounded dual-card A/B/A runtime reuse probe; never uses production credentials.

Run with the installed NPU Python environment:
  python tools/fuzz/fuzz_mlir_reuse_probe.py NEW_OUTPUT_DIRECTORY
Run baseline and candidate separately, with no competing managed NPU jobs.
Timings include compilation/dispatch and are NOT kernel-only measurements.
"""
from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
import argparse
import json
import time
from pathlib import Path

from autofusex.contracts.execution import ExecutionConfig, EvaluationConfig
from autofusex.workloads.fuzz.contracts import Seed
from autofusex.execution.scheduling.ledger import TaskLedger
from autofusex.execution.scheduling.budgets import Budget, write_policy
from autofusex.execution.scheduling.staged import drain
from autofusex.execution.storage.queries import ResultQueries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--compile-threads', type=int, default=1)
    parser.add_argument('--worker-max-tasks', type=int, default=16)
    parser.add_argument('--memory-gib', type=int, default=16)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'tasks').mkdir()
    (root / 'results').mkdir()
    ledger = TaskLedger(root / 'tasks/tasks.sqlite')
    database = root / 'results/results.sqlite'
    budget = Budget(cpu_workers=args.workers, compile_threads=args.compile_threads,
                    memory_budget_bytes=args.memory_gib * 1024**3,
                    cache_budget_bytes=8 * 1024**3, worker_max_tasks=args.worker_max_tasks)
    budget.admit(root)
    write_policy(ledger, budget)
    target = 'E9SQECC3GJ1J8KJBR869MZ1XHC'
    for short in [target, SamplingConfig.profile('small',max_nodes=3,dtype_mode='float32').seed(42).short, SamplingConfig.profile('small',max_nodes=3,dtype_mode='float32').seed(42,True).short, target]:
        seed = Seed.parse(short)
        for device in ['npu:0', 'npu:1']:
            ledger.enqueue(seed, ExecutionConfig('af-mlir', device, args.compile_threads,
                scheduling='staged', static_only=seed.version in {3, 5},
                save_results=True), EvaluationConfig())
    started = time.monotonic()
    summary = drain(ledger, database, timeout=180)
    elapsed = time.monotonic() - started
    queries = ResultQueries(database)
    attempts = []
    for task in ledger.page(limit=100)['items']:
        if task['state'] != 'completed':
            attempts.append({'state': task['state'], 'seed': task['seed']})
            continue
        result = queries.detail(task['attempt_id'])
        execution = result['contexts']['execution']
        attempts.append(dict(seed=result['seed_short'], status=result['status'],
            device=execution['device'], shapes=result['shape_results'],
            telemetry=json.loads(execution['telemetry']),
            stages=result['stage_timeline']))
    report = dict(summary=summary, wall_seconds=elapsed, attempts=attempts)
    (root / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    assert len(attempts) == 8 and all(a.get('status') == 'ok' for a in attempts)
    assert sum(len(a['shapes']) for a in attempts) == 12


if __name__ == '__main__':
    main()
