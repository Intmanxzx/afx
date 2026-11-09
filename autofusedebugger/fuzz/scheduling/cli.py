"""Managed task command surface; detached by default, explicit foreground option."""
from ..config import data_home, load_config
from ..contracts import Seed, ExecutionConfig, EvaluationConfig
from .ledger import TaskLedger


def register(sub):
    upgrade = sub.add_parser('upgrade-queue', help='offline task schema 1 -> 2 with a required new backup')
    upgrade.add_argument('queue')
    upgrade.add_argument('backup')
    for name in ('run', 'status', 'stop', 'resume'):
        parser = sub.add_parser(name, help='managed case/backend tasks')
        parser.add_argument('--data-dir')
        parser.add_argument('--queue', help='operational task database')
        if name == 'run':
            parser.add_argument('seeds', nargs='*', help='numeric seeds; omit to drain existing pending tasks')
            parser.add_argument('--count', type=int, help='generate 1..1000 new self-contained ETR v2 numeric seeds')
            parser.add_argument('--foreground', action='store_true',
                                help='wait and supervise in this terminal instead of detaching')
            parser.add_argument('--backend', action='append', choices=('cpu-eager', 'cpu-compile', 'ascendc', 'triton'))
            parser.add_argument('--device', help='NPU logical device (default npu:0); does not alter generated case')
            parser.add_argument('--compile-threads', type=int)
            parser.add_argument('--static-only', action='store_true')
            parser.add_argument('--performance', action='store_true')
            parser.add_argument('--timeout', type=float, default=120)
        elif name == 'status':
            parser.add_argument('task', nargs='?')
            parser.add_argument('--limit', type=int, default=50)
            parser.add_argument('--after', type=int, default=0)
        else:
            parser.add_argument('task')
        if name in {'run', 'resume'}:
            parser.add_argument('--database', help='result archive; bound on first run')


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
        ledger.stop(args.task)
        return ledger.get(args.task)
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
        if seeds or not 1 <= count <= 1000:
            raise ValueError('--count requires 1..1000 cases and no explicit seeds')
        import secrets
        seeds = [str(2**128 + secrets.randbits(128)) for _ in range(count)]
    evaluation = EvaluationConfig(performance=args.performance)
    from .budgets import Budget
    budget = Budget.configured(load_config(home), args.compile_threads)
    configs = [ExecutionConfig(backend, (args.device or 'npu:0') if backend in {'ascendc','triton'} else 'cpu', budget.compile_threads, static_only=args.static_only)
               for backend in dict.fromkeys(args.backend or ['cpu-eager'])]
    for config in configs:
        _validate_task({'execution': canonical_json(asdict(config)), 'evaluation': canonical_json(asdict(evaluation))})
    from .hosting import launch
    return launch(ledger, database, ((seed, config, evaluation, None) for seed in seeds for config in configs),
                  timeout=args.timeout, foreground=args.foreground, budget=budget)
