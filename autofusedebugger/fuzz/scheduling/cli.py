"""Managed task command surface; detached by default, explicit foreground option."""
from ..config import data_home, load_config
from ..contracts import Seed, ExecutionConfig, EvaluationConfig
from .ledger import TaskLedger


def register(sub, maintenance):
    upgrade = maintenance.add_parser('upgrade-queue', help='offline task schema 1 -> 2 with a required new backup')
    upgrade.set_defaults(command='upgrade-queue')
    upgrade.add_argument('queue')
    upgrade.add_argument('backup')
    for name in ('run', 'status', 'stop', 'resume'):
        parser = (maintenance if name=='resume' else sub).add_parser(name, help='managed case/backend tasks')
        parser.set_defaults(command=name)
        parser.add_argument('--data-dir', help=__import__('argparse').SUPPRESS)
        parser.add_argument('--queue', help=__import__('argparse').SUPPRESS)
        if name == 'run':
            parser.add_argument('seeds', nargs='*', help='self-contained short seeds')
            from ..generation.sampling import PROFILES, DIM_MODES, VALUE_MODES, DTYPE_MODES
            parser.add_argument('--tensor-scale', choices=tuple(PROFILES))
            parser.add_argument('--dim-range', nargs=2, type=int, metavar=('MIN', 'MAX'))
            parser.add_argument('--dim-dist-mode', choices=DIM_MODES)
            parser.add_argument('--dtype-mode', choices=DTYPE_MODES)
            parser.add_argument('--value-dist-mode', choices=VALUE_MODES)
            parser.add_argument('--max-elem-per-tensor', type=int)
            parser.add_argument('--max-nodes', type=int)
            parser.add_argument('--generation-timeout', type=int, help='solver timeout in seconds; recorded in E/F seed')
            parser.add_argument('--exclude-op', action='append', choices=('transpose',), help='skip random candidates containing this op; seed semantics are unchanged')
            resources = parser.add_mutually_exclusive_group()
            resources.add_argument('--auto-resources', action='store_true', default=None, help='recompute resource budget at startup (default for new setups)')
            resources.add_argument('--fixed-resources', dest='auto_resources', action='store_false', help='use saved fixed resource budget')
            parser.add_argument('--count', type=int, help='number of cases; omitted means continuous until fuzz stop')
            parser.add_argument('--fg', dest='foreground', action='store_true',
                                help='wait and supervise in this terminal instead of detaching')
            parser.add_argument('--backend', action='append', choices=('cpu', 'eager-npu', 'triton-npu', 'af-legacy', 'af-mlir'))
            parser.add_argument('--device', action='append', help='logical npu:index; repeat for multiple cards; default: all visible NPUs')
            parser.add_argument('--device-memory-bytes',type=int,help='per physical NPU admission/peak memory budget')
            parser.add_argument('--worker-max-tasks',type=int,default=16,help='recycle a stage runtime after N stages; 1 disables reuse')
            parser.add_argument('--scheduling', choices=('combined','staged'), default='staged', help='stage DAG or rollback to legacy whole-worker execution')
            parser.add_argument('--workers', type=int, help='maximum concurrent workers (CPU and NPU combined)')
            parser.add_argument('--compile-threads', type=int, help=__import__('argparse').SUPPRESS)
            parser.add_argument('--static-only', action='store_true')
            parser.add_argument('--perf', dest='performance', action='store_true')
            parser.add_argument('--save', action='store_true', help='save results and enqueue upload; default: neither')
            parser.add_argument('--timeout', type=float, default=120)
        elif name == 'status':
            parser.add_argument('task', nargs='?')
            parser.add_argument('--limit', type=int, default=50)
            parser.add_argument('--after', type=int, default=0)
        else:
            parser.add_argument('task', nargs='?' if name=='stop' else None)
        if name in {'run', 'resume'}:
            parser.add_argument('--database', help=__import__('argparse').SUPPRESS)


def dispatch(args):
    if args.command == 'upgrade-queue':
        from .migrations import upgrade_queue
        return upgrade_queue(args.queue, args.backup)
    home = data_home(args.data_dir)
    ledger = TaskLedger(args.queue or home/'tasks/tasks.sqlite')
    if args.command == 'status':
        if args.task:
            return ledger.get(args.task)
        from .hosting import status
        return dict(ledger.page(limit=args.limit, after=args.after), supervisor=status(ledger))
    if args.command == 'stop':
        if args.task:
            ledger.stop(args.task)
            return ledger.get(args.task)
        from .campaign import stop
        stop(ledger)
        return {'state':'stop_requested'}
    database = args.database or home/'results/results.sqlite'
    from .supervisor import resume, _validate_task
    if args.command == 'resume':
        return resume(ledger, database, args.task)
    from dataclasses import asdict
    from ..contracts import canonical_json
    import math
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    parsed = [Seed.parse(s) for s in args.seeds]
    if any(seed.version not in {3,4,5,6} for seed in parsed):
        raise ValueError('only NNSmith C/D/E/F seeds can run; retired template seeds are not regenerated')
    from ..generation.sampling import SamplingConfig
    sampling_keys = ('tensor_scale', 'dim_range', 'dim_dist_mode', 'value_dist_mode',
                     'max_elem_per_tensor', 'max_nodes', 'generation_timeout', 'dtype_mode')
    if parsed and any(getattr(args, key, None) is not None for key in sampling_keys):
        raise ValueError('sampling options only apply to new random cases; replay uses seed configuration')
    dim_range = getattr(args, 'dim_range', None)
    sampling = SamplingConfig.profile(getattr(args, 'tensor_scale', None) or 'default',
        dim_min=dim_range[0] if dim_range else None, dim_max=dim_range[1] if dim_range else None,
        dim_dist_mode=getattr(args, 'dim_dist_mode', None),
        dtype_mode=getattr(args, 'dtype_mode', None),
        value_dist_mode=getattr(args, 'value_dist_mode', None),
        max_elem_per_tensor=getattr(args, 'max_elem_per_tensor', None),
        max_nodes=getattr(args, 'max_nodes', None),
        timeout_seconds=getattr(args, 'generation_timeout', None))
    from ..generation.nnsmith_v3 import source_evidence
    source_evidence()  # Fail before creating a campaign if the dependency is absent/wrong.
    seeds = [str(seed) for seed in parsed]
    count = getattr(args, 'count', None)
    if count is not None:
        if seeds or count < 1:
            raise ValueError('--count requires a positive count and no explicit seeds')

    if getattr(args,'exclude_op',None) and (seeds or args.scheduling!='staged'):
        raise ValueError('--exclude-op requires a random staged campaign')
    evaluation = EvaluationConfig(performance=args.performance)
    from .budgets import Budget
    from ..resources import adaptive_defaults
    config = load_config(home)
    budget = Budget.configured(config or adaptive_defaults(home), args.compile_threads)
    from dataclasses import replace
    if args.workers is not None: budget=replace(budget,cpu_workers=args.workers)
    budget=replace(budget,worker_max_tasks=getattr(args,'worker_max_tasks',16))
    if getattr(args,'device_memory_bytes',None) is not None:budget=replace(budget,device_budget_bytes=args.device_memory_bytes)
    from .devices import select_devices
    devices=select_devices(args.device)
    from ..resources import npu_resource_plan, use_auto_resources
    resource_plan = None
    if use_auto_resources(config, getattr(args, 'auto_resources', None), args.workers, args.compile_threads):
        resource_plan = npu_resource_plan(home, len(devices), budget)
        budget = replace(budget, **resource_plan['selected'], **resource_plan['budget_limits'])
    import uuid
    batch_id = str(uuid.uuid4())
    configs = [ExecutionConfig(backend, devices[0] if backend in {'ascendc','triton','eager-npu','triton-npu','af-legacy','af-mlir'} else 'cpu', budget.compile_threads, scheduling=getattr(args,'scheduling','staged'), static_only=args.static_only, batch_id=batch_id, save_results=args.save)
               for backend in dict.fromkeys(['cpu','af-mlir']+(args.backend or []))]
    for config in configs:
        _validate_task({'execution': canonical_json(asdict(config)), 'evaluation': canonical_json(asdict(evaluation))})
    from .hosting import launch
    if resource_plan is not None and args.foreground:
        # Continuous foreground runs may never return. Explain the plan before
        # entering launch, on stderr so stdout retains its final JSON contract.
        import json
        import sys
        print(json.dumps({'resource_plan': resource_plan}), file=sys.stderr, flush=True)
    result = launch(ledger, database, ((seed, replace(config,static_only=args.static_only or Seed.parse(seed).version in {3,5},device='cpu' if config.device=='cpu' else devices[i%len(devices)]), evaluation, None) for i,seed in enumerate(seeds) for config in configs),
                  timeout=args.timeout, foreground=args.foreground, budget=budget, devices=devices,
                  campaign=(configs,evaluation,count,devices,getattr(args,'exclude_op',None),asdict(sampling)) if not seeds else None)

    if resource_plan is not None:
        result["resource_plan"] = resource_plan
    return result
