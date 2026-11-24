"""Resolve replay settings from history and enqueue atomically ahead of generation."""
from autofusex.client.cleanup import reader
import copy
import json
import sqlite3
import uuid
from contextlib import closing

from autofusex.contracts.execution import ExecutionConfig, EvaluationConfig
from autofusex.workloads.fuzz.contracts import Seed
from autofusex.workloads.fuzz.scheduling.ledger import TaskLedger
from autofusex.execution.scheduling.budgets import read_policy
from autofusex.client.configuration import validate


@reader
def submit(store, seeds, processes):
    seeds = [str(Seed.parse(seed)) for seed in seeds]
    if not 1 <= len(seeds) <= 1000:
        raise ValueError('replay requires 1..1000 seeds')
    with store.connect() as db:
        runs = list(db.execute('SELECT id,spec FROM runs ORDER BY rowid DESC'))
    resolved = []
    for seed in seeds:
        for run in runs:
            database = store.home / 'runs' / run['id'] / 'results.sqlite'
            if not database.exists():
                continue
            with closing(sqlite3.connect(database.as_uri()+'?mode=ro', uri=True)) as db:
                found = db.execute('SELECT 1 FROM attempts WHERE seed=? LIMIT 1', (seed,)).fetchone()
            if found:
                resolved.append((seed, json.loads(run['spec'])))
                break
        else:
            raise LookupError('no saved execution configuration for seed '+seed)
    active = list(processes)
    if active:
        if len(active) != 1:
            raise ValueError('multiple active runs; cannot choose replay owner')
        control=store.home/'runs'/active[0]/'control.json'
        if control.exists() and json.loads(control.read_text()).get('action')!='run':
            raise ValueError('active run is pausing or stopping; retry after it stops')
        current = next(json.loads(r['spec']) for r in runs if r['id'] == active[0])
        if any(old['environment']['id'] != current['environment']['id'] for _, old in resolved):
            raise ValueError('replay environment differs from active run; stop it before replay')
        ledger = TaskLedger(store.home/'runs'/active[0]/'tasks.sqlite')
        budget = read_policy(ledger)
        if budget is None:
            raise ValueError('run is preparing; retry replay once running')
        tasks=[]
        batch=str(uuid.uuid4())
        for seed, old in resolved:
            ex=old['spec']['execution']; perf=old['spec']['evaluation']['performance']
            if not set(ex['devices']).issubset(current['spec']['execution']['devices']):
                raise ValueError('historical devices unavailable in active run')
            evaluation=EvaluationConfig(performance=perf['enabled'],warmup_steps=perf['warmup_steps'],measured_steps=perf['measured_steps'])
            for backend in ex['backends']:
                execution=ExecutionConfig(backend, 'cpu' if backend=='cpu' else ex['devices'][0],
                    budget.compile_threads, scheduling='staged', static_only=ex['static_only'],
                    batch_id=batch, save_results=old['spec']['results']['save'])
                tasks.append((seed,execution,evaluation,None))
        ids=ledger.enqueue_many(tasks, priority=100)
        return dict(run_id=active[0],task_ids=ids,priority=100,state='queued')
    specs=[]
    for seed, old in resolved:
        spec=copy.deepcopy(old['spec'])
        spec['workload']={'mode':'replay','seeds':[seed]};spec.pop('termination',None)
        resources=spec['execution']['resources']
        resources['policy']='auto';resources.pop('workers',None);resources.pop('compile_threads',None)
        specs.append(validate(spec))
    # Separate histories can have different environments/evaluation settings.
    # Preserve each snapshot; the daemon serializes their task submissions.
    operations=[store.submit('task.submit',spec,'replay:'+str(uuid.uuid4())) for spec in specs]
    return dict(operation_ids=[op['id'] for op in operations],priority=100,state='queued')
