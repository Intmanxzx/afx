"""Execution-only entry point. Never imported by the bootstrap daemon."""
import json
from pathlib import Path
import sys
import traceback


def main(path):
    prepared = json.loads(Path(path).read_text())
    if prepared['spec'].get('kind', 'fuzz_task') != 'fuzz_task':
        if prepared.get('resume_queue') and (Path(prepared['directory'])/'result.json').exists():
            from autofusex.client.artifacts import atomic
            import autofusex
            atomic(Path(prepared['directory'])/'worker-version.json',dict(python=sys.executable,version=autofusex.__version__))
            result=json.loads((Path(prepared['directory'])/'result.json').read_text())
            atomic(Path(prepared['directory'])/'state.json',dict(state='completed',result=result))
            return 0 if result['status'] in {'passed','accuracy_failed'} else 1
        from autofusex.execution.source_task import run
        from autofusex.client.artifacts import atomic
        import autofusex
        atomic(Path(prepared['directory'])/'worker-version.json', dict(python=sys.executable, version=autofusex.__version__))
        return run(path, prepared, int(sys.argv[2]) if len(sys.argv)>2 else None)
    from dataclasses import replace
    from autofusex.client.artifacts import atomic
    from autofusex.contracts.configuration import quantity
    from autofusex.contracts.execution import ExecutionConfig, EvaluationConfig
    from autofusex.workloads.fuzz.contracts import Seed
    from autofusex.workloads.fuzz.generation.runtime import source_evidence
    from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
    from autofusex.execution.scheduling.budgets import Budget
    from autofusex.workloads.fuzz.scheduling.hosting import launch
    from autofusex.workloads.fuzz.scheduling.ledger import TaskLedger
    prepared=json.loads(Path(path).read_text());spec=prepared['spec'];directory=Path(prepared['directory'])
    state=directory/'state.json'
    import autofusex
    atomic(directory/'worker-version.json',dict(python=sys.executable,version=autofusex.__version__))
    try:
        if not {'execution', 'evaluation', 'workload'} <= spec.keys() or 'policy' not in prepared:
            raise KeyError('incomplete task spec')
        # Fail before scheduling work if the installation identity was lost.
        from autofusex.execution.environment import collect
        collect()
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
            sampling=SamplingConfig.profile(generation.get('tensor_scale','default'),
                    **{k:generation[k] for k in ('dim_min','dim_max','dim_dist_mode','value_dist_mode','dtype_mode','dynamic_variants','method') if k in generation},max_nodes=generation.get('max_nodes'),
                    max_elem_per_tensor=generation.get('max_elements_per_tensor'),timeout_seconds=int(quantity(generation['timeout'])) if generation.get('timeout') else None)
            campaign=(configs,evaluation,spec['termination'].get('max_cases'),ex['devices'] if any(c.backend!='cpu' for c in configs) else [],generation.get('exclude_ops',[]),asdict(sampling),generation.get('ops',[]),None,generation.get('shape_mode','mixed'))
        if prepared.get('resume_queue'):
            tasks=[]; campaign=None
        atomic(state,dict(state='running'))
        result=launch(ledger,database,tasks,foreground=True,budget=budget,devices=ex['devices'] if any(c.backend!='cpu' for c in configs) else [],campaign=campaign,preserve_campaign=bool(prepared.get('resume_queue')),timeout=quantity(ex['timeout']))
        if json.loads((directory/'control.json').read_text()).get('action')=='reconfigure':
            return 75
        atomic(state,dict(state='completed',result=result))
        return 0
    except Exception as exc:
        # The parent persists the terminal state, while execution.log contains
        # the complete traceback because the worker's stderr is redirected
        # there by tasks.start().
        atomic(state,dict(state='failed',error=type(exc).__name__,message=str(exc)[:1000]))
        traceback.print_exc()
        return 1


if __name__=='__main__':
    from autofusex.client.artifacts import atomic
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
