"""Managed task command surface; detached by default, explicit foreground option."""
from ..config import data_home, load_config
from ..contracts import Seed, ExecutionConfig, EvaluationConfig
from .ledger import TaskLedger


def register(sub, maintenance):
    for name in ('run', 'status', 'stop', 'resume'):
        parser = (maintenance if name=='resume' else sub).add_parser(name, help='managed case/backend tasks')
        parser.set_defaults(command=name)
        parser.add_argument('--data-dir', help=__import__('argparse').SUPPRESS)
        parser.add_argument('--queue', help=__import__('argparse').SUPPRESS)
        if name == 'run':
            parser.add_argument('seeds', nargs='*', help='self-contained short seeds')
            parser.add_argument('--continue', dest='continue_campaign', action='store_true',
                                help="relaunch the saved random campaign exactly as it last ran (backends, devices, "
                                     "sampling, resource budget); use after a restart instead of retyping flags from "
                                     "memory. Conflicts with seeds and any other generation/backend/device/resource flag; "
                                     "see `afx fuzz status` for what is saved")
            from ..generation.sampling import PROFILES, DIM_MODES, VALUE_MODES, DTYPE_MODES
            parser.add_argument('--tensor-scale', choices=tuple(PROFILES))
            parser.add_argument('--dim-range', nargs=2, type=int, metavar=('MIN', 'MAX'))
            parser.add_argument('--dim-dist-mode', choices=DIM_MODES)
            parser.add_argument('--dtype-mode', choices=DTYPE_MODES)
            parser.add_argument('--value-dist-mode', choices=VALUE_MODES)
            parser.add_argument('--max-elem-per-tensor', type=int)
            parser.add_argument('--max-nodes', type=int)
            parser.add_argument('--generation-timeout', type=int, help='solver timeout in seconds; embedded in seed')
            parser.add_argument('--dynamic-variants', type=int, choices=range(2,9))
            from ..generation.sampling import METHODS
            parser.add_argument('--method', choices=METHODS)
            parser.add_argument('--exclude-op', action='append', choices=('transpose',), help='skip random candidates containing this op; seed semantics are unchanged')
            parser.add_argument('--ops', help='comma-separated OP targets, display names or categories (Elementwise, Broadcast, Reduce, Reshape, Transpose, Split); generation remains full-range and filtering happens after seed replay')
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
            parser.add_argument('--scheduling', choices=('combined','staged'), default='staged', help='stage DAG or rollback to combined whole-worker execution')
            parser.add_argument('--workers', type=int, help='maximum concurrent workers (CPU and NPU combined)')
            parser.add_argument('--compile-threads', type=int, help=__import__('argparse').SUPPRESS)
            parser.add_argument('--static-only', action='store_true')
            parser.add_argument('--perf', dest='performance', action='store_true')
            parser.add_argument('--save', action='store_true', help='save results to the local database; default: neither')
            parser.add_argument('--timeout', type=float, default=120)
        elif name == 'status':
            parser.add_argument('task', nargs='?')
            parser.add_argument('--limit', type=int, default=50)
            parser.add_argument('--after', type=int, default=0)
        else:
            parser.add_argument('task', nargs='?' if name=='stop' else None)
        if name in {'run', 'resume'}:
            parser.add_argument('--database', help=__import__('argparse').SUPPRESS)


# Flags a saved campaign already pins down; `--continue` refuses to guess which
# one the operator actually meant to change, so it allows none of them at all.
# default=None here must match each flag's own argparse default above.
_CONTINUE_CONFLICTS = {
    'tensor_scale': None, 'dim_range': None, 'dim_dist_mode': None, 'dtype_mode': None,
    'value_dist_mode': None, 'max_elem_per_tensor': None, 'max_nodes': None, 'generation_timeout': None,
    'dynamic_variants': None, 'method': None, 'exclude_op': None, 'ops': None, 'backend': None,
    'device': None, 'auto_resources': None, 'count': None, 'static_only': False, 'performance': False,
    'worker_max_tasks': 16, 'workers': None, 'compile_threads': None, 'device_memory_bytes': None,
    'scheduling': 'staged', 'save': False,
}


def _continue_campaign(args, ledger, database):
    changed = [name for name, default in _CONTINUE_CONFLICTS.items() if getattr(args, name, default) != default]
    if args.seeds or changed:
        raise ValueError('--continue reuses the saved campaign exactly; it takes no seeds and no other '
                          'generation/backend/device/resource flag (set: ' +
                          ', '.join((['seeds'] if args.seeds else []) + changed) + ')')
    from .campaign import read as read_campaign
    saved = read_campaign(ledger)
    if saved is None:
        raise ValueError('no saved campaign in this queue to continue (--queue/--data-dir ' + str(ledger.path) +
                          '); this host has never run `afx fuzz run` without --continue here')
    import sys
    import uuid
    from dataclasses import asdict
    from ..contracts import canonical_json, ExecutionConfig, EvaluationConfig
    from ..generation.sampling import SamplingConfig
    from ..generation.runtime import source_evidence
    from .supervisor import _validate_task
    source_evidence()
    configs = [ExecutionConfig(**{**c, 'batch_id': str(uuid.uuid4())}) for c in saved['configs']]
    evaluation = EvaluationConfig(**saved['evaluation'])
    devices = saved['devices']
    sampling = SamplingConfig(**saved['sampling'])
    ops = saved['ops'] or None
    exclude_ops = saved['exclude_ops'] or None
    from .budgets import Budget
    if saved.get('budget'):
        budget = Budget(**saved['budget'])
    else:
        from ..config import load_config, data_home
        from ..resources import adaptive_defaults
        home = data_home(args.data_dir)
        budget = Budget.configured(load_config(home) or adaptive_defaults(home), None)
        print('warning: saved campaign predates budget persistence; resuming with the current default '
              'worker/compile-thread settings, not necessarily the ones it actually ran with', file=sys.stderr)
    for config in configs:
        _validate_task({'execution': canonical_json(asdict(config)), 'evaluation': canonical_json(asdict(evaluation))})
    from .hosting import launch
    return launch(ledger, database, [], timeout=args.timeout, foreground=args.foreground, budget=budget, devices=devices,
                  campaign=(configs, evaluation, saved['remaining'], devices, exclude_ops, asdict(sampling), ops, asdict(budget)))


def dispatch(args):
    home = data_home(args.data_dir)
    ledger = TaskLedger(args.queue or home/'tasks/tasks.sqlite')
    if args.command == 'status':
        if args.task:
            return ledger.get(args.task)
        from .hosting import status
        from .campaign import read as read_campaign
        return dict(ledger.page(limit=args.limit, after=args.after), supervisor=status(ledger), campaign=read_campaign(ledger))
    if args.command == 'stop':
        if args.task:
            ledger.stop(args.task)
            return ledger.get(args.task)
        from .campaign import stop
        stop(ledger)
        return {'state':'stop_requested'}
    managed_lock = home / 'managed-task.lock'
    if managed_lock.exists():
        import fcntl
        with managed_lock.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('a managed campaign owns this device; control it through task CLI/Web') from None
    database = args.database or home/'results/results.sqlite'
    from .supervisor import resume, _validate_task
    if args.command == 'resume':
        return resume(ledger, database, args.task)
    from dataclasses import asdict
    from ..contracts import canonical_json
    if args.command == 'run' and getattr(args, 'continue_campaign', False):
        return _continue_campaign(args, ledger, database)
    import math
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    parsed = [Seed.parse(s) for s in args.seeds]
    from ..generation.sampling import SamplingConfig
    sampling_keys = ('tensor_scale', 'dim_range', 'dim_dist_mode', 'value_dist_mode',
                     'max_elem_per_tensor', 'max_nodes', 'generation_timeout', 'dtype_mode', 'dynamic_variants', 'method')
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
        timeout_seconds=getattr(args, 'generation_timeout', None),
        dynamic_variants=getattr(args,'dynamic_variants',None), method=getattr(args,'method',None))
    from ..generation.runtime import source_evidence
    source_evidence()  # Fail before creating a campaign if the dependency is absent/wrong.
    # Keep planning import-light; the execution worker validates the Torch/NNSmith
    # runtime immediately before materializing a Case.
    seeds = [str(seed) for seed in parsed]
    count = getattr(args, 'count', None)
    if count is not None:
        if seeds or count < 1:
            raise ValueError('--count requires a positive count and no explicit seeds')

    from ..generation.operator_catalog import normalize_ops
    ops = normalize_ops(getattr(args, 'ops', None))
    if ops and args.scheduling != 'staged':
        raise ValueError('--ops requires staged scheduling')
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
    configs = [ExecutionConfig(backend, devices[0] if backend in {'eager-npu','triton-npu','af-legacy','af-mlir'} else 'cpu', budget.compile_threads, scheduling=getattr(args,'scheduling','staged'), static_only=args.static_only, batch_id=batch_id, save_results=args.save)
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
    if not seeds or ops:
        from .campaign import read as read_campaign
        previous = read_campaign(ledger)
        if previous is not None and not previous['stopped']:
            # Silent clobbering of a still-live campaign is exactly the footgun `--continue` exists to
            # avoid; anyone who skips --continue and types fresh flags still gets told what they are
            # replacing, in case that was not the intent (e.g. after a host restart, typed from memory).
            import sys
            print(f"note: replacing the saved campaign started at {previous.get('started_at', 'unknown time')} "
                  f"(backends {sorted({c['backend'] for c in previous['configs']})}, devices {previous['devices']}); "
                  "use `afx fuzz run --continue` to resume it unchanged instead", file=sys.stderr)
    result = launch(ledger, database, ((seed, replace(config,static_only=args.static_only or not Seed.parse(seed).dynamic,device='cpu' if config.device=='cpu' else devices[i%len(devices)]), evaluation, None) for i,seed in enumerate(seeds) for config in configs),
                  timeout=args.timeout, foreground=args.foreground, budget=budget, devices=devices,
                  campaign=(configs,evaluation,0 if seeds and ops else count,devices,
                            getattr(args,'exclude_op',None),asdict(sampling),ops,asdict(budget)) if (not seeds or ops) else None)

    if resource_plan is not None:
        result["resource_plan"] = resource_plan
    return result
