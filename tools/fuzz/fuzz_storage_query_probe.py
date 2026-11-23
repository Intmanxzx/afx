"""Read-only distribution/plan measurements on an existing capacity fixture."""
import argparse
import json
import platform
import sqlite3
import statistics
import time
from pathlib import Path
from autofusex.execution.storage.queries import ResultQueries
from autofusex.execution.reporting.statistics import statistics as charts

p = argparse.ArgumentParser()
p.add_argument('database', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--samples', type=int, default=30)
a = p.parse_args()
q = ResultQueries(a.database)
with sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro', uri=True) as c:
    seed = c.execute('SELECT seed FROM attempts LIMIT 1').fetchone()[0]
    result = dict(environment=platform.platform(), attempts=c.execute('SELECT count(*) FROM attempts').fetchone()[0],
                  source='existing fixture; does not measure ingestion or NPU execution', queries={})
    result['tables'] = c.execute('SELECT name,sum(pgsize) FROM dbstat GROUP BY name ORDER BY 2 DESC').fetchall()
    result['plans'] = c.execute('EXPLAIN QUERY PLAN SELECT * FROM case_heads_v1 WHERE cursor<? ORDER BY cursor DESC LIMIT 51',(2**63-1,)).fetchall()
for name, fn in [('page',q.case_page),('seed',lambda:q.case_page(seed=seed)),('summary',q.case_summary),('charts',lambda:charts(a.database))]:
    samples, errors = [], []
    for i in range(a.samples):
        started = time.perf_counter()
        try:
            fn()
        except Exception as e:
            errors.append(dict(type=type(e).__name__,message=str(e)))
        samples.append(1000*(time.perf_counter()-started))
        if errors:
            break  # Preserve failure; do not hide timeouts by retrying indefinitely.
    samples.sort()
    result['queries'][name] = dict(n=len(samples),p50_ms=statistics.median(samples),
        p95_ms=samples[min(len(samples)-1,int(len(samples)*.95))],
        p99_ms=samples[min(len(samples)-1,int(len(samples)*.99))],errors=errors)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({name:result['queries'][name]}),flush=True)
