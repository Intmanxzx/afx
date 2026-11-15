"""Read a managed run's saved results in the CLI, outside the daemon loop."""
from pathlib import Path


def read(home, run, *, limit=50, after=0, status=None, seed=None, attempt=None):
    from ..storage.queries import ResultQueries, filters
    if type(limit) is not int or not 1 <= limit <= 200 or type(after) is not int or not 0 <= after < 2**63:
        raise ValueError('invalid pagination')
    filters(seed=seed, status=status)
    if attempt and (seed is not None or status is not None or after != 0):
        raise ValueError('--attempt cannot be combined with filters or a cursor')
    root = (Path(home) / 'runs').resolve()
    directory = (root / run['id']).resolve()
    if directory.parent != root:
        raise ValueError('invalid run directory')
    database = directory / 'results.sqlite'
    if database.is_symlink():
        raise ValueError('result database must not be a symlink')
    saved = run['spec']['spec']['results']['save']
    result = dict(run_id=run['id'], run_state=run['state'], saved=saved,
                  available=database.is_file(), unit='attempt')
    if not result['available']:
        if attempt:
            raise LookupError('attempt not found; no saved result database')
        return dict(result, items=[], next_cursor=None)
    queries = ResultQueries(database)
    if attempt:
        return dict(result, attempt=queries.detail(attempt))
    return dict(result, **queries.page(limit=limit, after=after, status=status, seed=seed))
