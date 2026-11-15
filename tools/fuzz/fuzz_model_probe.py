"""Explicit private-copy migration and equivalence benchmark; no network/tokens."""
from contextlib import closing
import json
from pathlib import Path
import sys
import time

from autofusex.fuzz.reporting.materialized import migrate
from autofusex.fuzz.reporting.cases import rollup
from autofusex.fuzz.reporting.statistics import statistics
from autofusex.fuzz.storage.repository import _connect
from autofusex.fuzz.storage.queries import ResultQueries

path=Path(sys.argv[1]);report={}
with closing(_connect(path)) as c:
    c.row_factory=__import__('sqlite3').Row
    t=time.monotonic();s,v=rollup()
    legacy=[dict(r) for r in c.execute(s+'SELECT * FROM cases ORDER BY cursor DESC',v)]
    report['legacy_heads_seconds']=time.monotonic()-t
    report['attempts']=c.execute('SELECT count(*) FROM attempts').fetchone()[0]
print(json.dumps(report),flush=True)
t=time.monotonic();old=statistics(path,_timeout=600);report['legacy_statistics_seconds']=time.monotonic()-t
print(json.dumps(report),flush=True)
t=time.monotonic();report['migration']=migrate(path);report['migration_seconds']=time.monotonic()-t
print(json.dumps(report),flush=True)
with closing(_connect(path)) as c:
    c.row_factory=__import__('sqlite3').Row
    heads=[dict(r) for r in c.execute('SELECT * FROM case_heads_v1 ORDER BY cursor DESC')]
    assert heads==legacy,'current rows differ'
    assert c.execute('SELECT count(*) FROM attempts').fetchone()[0]==report['attempts']
    assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
q=ResultQueries(path)
for label,call in [('summary',q.case_summary),('page10',lambda:q.case_page(limit=10)),('page200',lambda:q.case_page(limit=200)),('statistics',lambda:statistics(path,_timeout=600))]:
    t=time.monotonic();value=call();report[label+'_seconds']=time.monotonic()-t
    if label=='statistics':assert value==old,'statistics differ'
report.update(heads=len(heads),equivalent=True)
print(json.dumps(report),flush=True)
