"""Result database maintenance without a generator dependency."""
import argparse
import json
import sqlite3
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog='afx results')
    sub = parser.add_subparsers(dest='command', required=True)
    pair = sub.add_parser("compare", help="compare two result attempts with explicit comparability checks")
    pair.add_argument("database")
    pair.add_argument("left")
    pair.add_argument("right")
    snapshot = sub.add_parser("snapshot", help="export a consistent lightweight SQLite snapshot")
    snapshot.add_argument("source")
    snapshot.add_argument("destination")
    for command in ("restore", "merge"):
        transfer = sub.add_parser(command, help="restore to a new database" if command == "restore" else "merge into an existing database atomically")
        transfer.set_defaults(command=command)
        transfer.add_argument("source")
        transfer.add_argument("destination")
    check = sub.add_parser("check", help="audit a trusted result database without modification")
    check.add_argument("database")
    results = sub.add_parser("list", help="read lightweight results without model dependencies")
    results.add_argument("database")
    results.add_argument("--limit", type=int, default=50)
    results.add_argument("--after", type=int)
    results.add_argument("--seed")
    results.add_argument("--attempt")
    index = sub.add_parser('rebuild-index', help='rebuild reporting indexes')
    index.add_argument('database')
    index.add_argument('--kind', choices=('results', 'cases', 'all'), default='all')
    args = parser.parse_args(argv)
    try:
        from autofusex.execution.storage import ResultRepository
        if args.command == 'compare':
            from autofusex.execution.storage.comparison import compare
            result = compare(args.database, args.left, args.right)
        elif args.command == 'snapshot':
            result = {'snapshot': ResultRepository(args.source).snapshot(args.destination)}
        elif args.command in {'restore', 'merge', 'check'}:
            from autofusex.execution.storage.transfer import restore, merge, check_database
            result = check_database(args.database) if args.command == 'check' else (restore if args.command == 'restore' else merge)(args.source, args.destination)
        elif args.command == 'list':
            repo = ResultRepository(args.database)
            result = repo.detail(args.attempt) if args.attempt else repo.list(limit=args.limit, after=args.after, seed=args.seed)
        else:
            result = {}
            if args.kind in {'results', 'all'}:
                from autofusex.execution.reporting.index import rebuild
                result['results'] = rebuild(args.database)
            if args.kind in {'cases', 'all'}:
                from autofusex.execution.reporting.materialized import rebuild
                result['cases'] = rebuild(args.database)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
