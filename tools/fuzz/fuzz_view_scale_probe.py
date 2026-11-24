"""Remote-only reproducible synthetic viewer scale probe; creates a NEW database.

Run: python tools/fuzz/fuzz_view_scale_probe.py /isolated/path/scale.sqlite
This measures storage/query overhead, not NPU execution or production distributions.
"""
import json
from pathlib import Path
import sqlite3
import sys
import time
import uuid

from autofusex.execution.storage.repository import _initialize
from autofusex.execution.storage.queries import ResultQueries


def main(path):
    path = Path(path)
    # Fail closed rather than overwrite any previous results.
    with path.open('xb'):
        pass
    with sqlite3.connect(path) as conn:
        _initialize(conn)
        conn.executemany('INSERT INTO contexts VALUES (?,?,?)',
                         [(kind, kind, '{}') for kind in ('execution', 'evaluation', 'environment')])
        conn.executemany('INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            ((str(uuid.UUID(int=i+1)), str(i), 'a'*64, 'ref/1', 'cpu-eager',
              ('ok','fail','error','unsupported')[i%4], '2026-11-09T00:00:00+00:00',
              'execution', 'evaluation', 'environment', 1000, None, None, 'b'*64)
             for i in range(100000)))
        conn.executemany('INSERT INTO shapes VALUES (?,?,?,?,?,?,?,?,?,?)',
            ((str(uuid.UUID(int=i+1)), shape, 'pass', 'not_requested', 1000, None, None, None, None, None)
             for i in range(100000) for shape in range(3)))
    queries = ResultQueries(path)
    stats = {'attempts': 100000, 'shapes': 300000, 'bytes': path.stat().st_size,
             'bytes_per_attempt': path.stat().st_size/100000, 'queries': {}}
    for name, operation in [('page', lambda: queries.page()),
                            ('deep_page', lambda: queries.page(after=99900)),
                            ('seed_filter', lambda: queries.page(seed='42')),
                            ('status_filter', lambda: queries.page(status='fail')),
                            ('summary', queries.summary)]:
        start = time.perf_counter(); result = operation(); elapsed = time.perf_counter()-start
        size = len(json.dumps(result).encode())
        stats['queries'][name] = {'seconds': elapsed, 'response_bytes': size}
        if 'items' in result:
            assert len(result['items']) <= 50 and size < 32768
        assert elapsed < 3
    assert queries.summary()['total'] == 100000
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
