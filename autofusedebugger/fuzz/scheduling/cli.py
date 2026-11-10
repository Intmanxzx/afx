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
            parser.add_argument('--count', type=int, help='number of cases; omitted means continuous until fuzz stop')
            parser.add_argument('--fg', dest='foreground', action='store_true',
                                help='wait and supervise in this terminal instead of detaching')
            parser.add_argument('--backend', action='append', choices=('cpu', 'eager-npu', 'triton-npu', 'af-legacy', 'af-mlir'))
            parser.add_argument('--device', help='NPU logical device (default npu:0); does not alter generated case')
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
    seeds = [str(Seed.parse(s)) for s in args.seeds]
    count = getattr(args, 'count', None)
    if count is not None:
        if seeds or count < 1:
            raise ValueError('--count requires a positive count and no explicit seeds')

    evaluation = EvaluationConfig(performance=args.performance)
    from .budgets import Budget
    from ..resources import adaptive_defaults
    budget = Budget.configured(load_config(home) or adaptive_defaults(home), args.compile_threads)
    import uuid
    batch_id = str(uuid.uuid4())
    configs = [ExecutionConfig(backend, (args.device or 'npu:0') if backend in {'ascendc','triton','eager-npu','triton-npu','af-legacy','af-mlir'} else 'cpu', budget.compile_threads, static_only=args.static_only, batch_id=batch_id, save_results=args.save)
               for backend in dict.fromkeys(['cpu','af-mlir']+(args.backend or []))]
    for config in configs:
        _validate_task({'execution': canonical_json(asdict(config)), 'evaluation': canonical_json(asdict(evaluation))})
    from .hosting import launch
    return launch(ledger, database, ((seed, config, evaluation, None) for seed in seeds for config in configs),
                  timeout=args.timeout, foreground=args.foreground, budget=budget,
                  campaign=(configs,evaluation,count) if not seeds else None)
