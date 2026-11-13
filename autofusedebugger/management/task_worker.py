"""Execution-only entry point. Never imported by the bootstrap daemon."""
import json
from pathlib import Path
import sys


def main(path):
    from dataclasses import replace
    from .artifacts import atomic
    from .config import quantity
    from ..fuzz.contracts import ExecutionConfig, EvaluationConfig, Seed
    from ..fuzz.generation.nnsmith_v3 import source_evidence
    from ..fuzz.generation.sampling import SamplingConfig
    from ..fuzz.scheduling.budgets import Budget
    from ..fuzz.scheduling.hosting import launch
    from ..fuzz.scheduling.ledger import TaskLedger
    prepared=json.loads(Path(path).read_text());spec=prepared['spec'];directory=Path(prepared['directory'])
    state=directory/'state.json'
    try:
        source_evidence()
        ex=spec['execution'];resources=ex['resources'];limits=prepared['policy']['resources']
        budget=Budget(cpu_workers=resources.get('workers',limits['max_workers']),
                      compile_threads=resources.get('compile_threads',limits['max_compile_threads']),
                      worker_max_tasks=resources['worker_max_tasks'],
                      memory_budget_bytes=int(quantity(resources.get('memory_budget',limits.get('memory_budget')))) if resources.get('memory_budget',limits.get('memory_budget')) else None)
        ledger=TaskLedger(directory/'tasks.sqlite');database=directory/'results.sqlite'
        performance=spec['evaluation']['performance']
        evaluation=EvaluationConfig(performance=performance['enabled'],warmup_steps=performance['warmup_steps'],measured_steps=performance['measured_steps'])
        configs=[ExecutionConfig(backend,'cpu' if backend=='cpu' else ex['devices'][0],budget.compile_threads,
                                 scheduling='staged',static_only=ex['static_only'],batch_id=prepared['run_id']+':'+str(prepared.get('epoch',0)),save_results=spec['results']['save']) for backend in ex['backends']]
        workload=spec['workload'];campaign=None;tasks=[]
        if workload['mode']=='replay':
            tasks=[(str(Seed.parse(seed)),replace(config,static_only=ex['static_only'] or Seed.parse(seed).version in {3,5}),evaluation,None)
                   for seed in workload['seeds'] for config in configs]
        else:
            from dataclasses import asdict
            generation=workload.get('generation',{})
            sampling=SamplingConfig.profile(generation.get('tensor_scale','default'),max_nodes=generation.get('max_nodes'),
                    max_elem_per_tensor=generation.get('max_elements_per_tensor'),timeout_seconds=int(quantity(generation['timeout'])) if generation.get('timeout') else None)
            campaign=(configs,evaluation,spec['termination'].get('max_cases'),ex['devices'],generation.get('exclude_ops',[]),asdict(sampling))
        atomic(state,dict(state='running'))
        result=launch(ledger,database,tasks,foreground=True,budget=budget,devices=ex['devices'],campaign=campaign,timeout=quantity(ex['timeout']))
        if json.loads((directory/'control.json').read_text()).get('action')=='reconfigure':
            return 75
        atomic(state,dict(state='completed',result=result))
        return 0
    except Exception as exc:
        atomic(state,dict(state='failed',error=type(exc).__name__))
        return 1


if __name__=='__main__':
    from .artifacts import atomic
    path=Path(sys.argv[1])
    while True:
        code=main(path)
        if code!=75:raise SystemExit(code)
        pending=path.parent/'next-spec.json'
        value=json.loads(pending.read_text())
        history=path.parent/'epochs';history.mkdir(exist_ok=True)
        old=json.loads(path.read_text());atomic(history/(str(old.get('epoch',0))+'.json'),old)
        atomic(path,value);pending.unlink()
        atomic(path.parent/'control.json',dict(action='run'))
